"""对话入口：会话锁、图执行、SSE 事件与一轮指标。"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from app.config import get_settings
from app.graph.hydrate import ahydrate_inputs_from_archive, hydrate_inputs_from_archive
from app.graph.state import make_thread_id
from app.graph.stream import events_from_stream_item
from app.llm_gateway import LLM_UNAVAILABLE_ANSWER, LlmCircuitOpenError, TenantQuotaExceededError, begin_turn_usage, take_turn_usage
from app.llm_gateway.types import TurnUsage
from app.observability import (
    get_request_id,
    inc_chat,
    log_event,
    maybe_inc_runaway_llm_calls,
    observe_chat,
    observe_llm_calls,
    observe_ttft,
    set_span_attributes,
)
from app.observability.langfuse import langfuse_callbacks, score_turn, trace_metadata
from app.observability.otel import current_trace_ids
from app.runtime import get_app_context
from app.schemas import ChatResponse
from app.security.callback import SecurityCallbackHandler
from app.security.filters import mask_pii
from app.session_cache import SessionBusyError, ahold_session
from app.tenancy import set_current_tenant, set_current_user

logger = logging.getLogger("cs.graph")

_QUOTA_ANSWER = "今日大模型额度已用完，请明日再试或联系管理员。"
_UNAVAILABLE_ANSWER = LLM_UNAVAILABLE_ANSWER


class ChatTurn:
    """一轮对话的公共上下文，供 stream / astream 共用。"""

    def __init__(
        self,
        message: str,
        user_id: str,
        tenant_id: str | None,
        session_id: str | None,
        *,
        hydrate: bool = True,
    ) -> None:
        self.tenant = tenant_id or get_settings().default_tenant
        set_current_tenant(self.tenant)
        self.uid = (user_id or "").strip() or "anonymous"
        set_current_user(self.uid)
        self.sid = (session_id or "").strip() or str(uuid.uuid4())
        self.thread_id = make_thread_id(self.tenant, self.sid, self.uid)
        self.handler = SecurityCallbackHandler(user_id=self.uid, tenant_id=self.tenant)
        self.inputs: dict[str, Any] = {
            "query": message,
            "user_id": self.uid,
            "tenant_id": self.tenant,
            "session_id": self.sid,
        }
        if hydrate:
            hydrate_inputs_from_archive(
                self.inputs,
                tenant_id=self.tenant,
                user_id=self.uid,
                session_id=self.sid,
                thread_id=self.thread_id,
            )

    async def ahydrate(self) -> None:
        """异步路径：Redis checkpoint 与 MySQL 归档不堵事件循环。"""
        await ahydrate_inputs_from_archive(
            self.inputs,
            tenant_id=self.tenant,
            user_id=self.uid,
            session_id=self.sid,
            thread_id=self.thread_id,
        )

    def run_config(self) -> dict[str, Any]:
        return {
            "callbacks": [self.handler, *langfuse_callbacks()],
            "configurable": {"thread_id": self.thread_id},
            "metadata": trace_metadata(
                user_id=self.uid,
                tenant_id=self.tenant,
                session_id=self.sid,
                thread_id=self.thread_id,
            ),
        }

    def meta_event(self) -> tuple[str, dict[str, Any]]:
        trace_id, _ = current_trace_ids()
        payload: dict[str, Any] = {
            "session_id": self.sid,
            "user_id": self.uid,
            "tenant_id": self.tenant,
            "request_id": get_request_id(),
        }
        if trace_id:
            payload["trace_id"] = trace_id
        return ("meta", payload)


def _quota_done(session_id: str, usage: TurnUsage | None = None) -> dict[str, Any]:
    del usage
    return ChatResponse(
        answer=_QUOTA_ANSWER,
        intent="chitchat",
        blocked=False,
        session_id=session_id,
    ).model_dump()


def _unavailable_done(session_id: str, usage: TurnUsage | None = None) -> dict[str, Any]:
    del usage
    return ChatResponse(
        answer=_UNAVAILABLE_ANSWER,
        intent="chitchat",
        blocked=False,
        session_id=session_id,
    ).model_dump()


def _done_payload(
    accumulated: dict[str, Any],
    session_id: str,
    usage: TurnUsage | None = None,
) -> dict[str, Any]:
    usage = usage or TurnUsage()
    usage_dict = usage.to_dict() if usage.steps or usage.total_tokens else None
    answer = accumulated.get("answer") or ""
    if not str(answer).strip() and not accumulated.get("blocked"):
        answer = _UNAVAILABLE_ANSWER
    payload = ChatResponse(
        answer=answer,
        intent=accumulated.get("intent") or "chitchat",
        blocked=bool(accumulated.get("blocked")),
        structured=accumulated.get("structured"),
        visualization=accumulated.get("visualization"),
        oss_url=accumulated.get("oss_url"),
        session_id=session_id,
        usage=usage_dict,
        request_id=get_request_id() or None,
        trace_id=(current_trace_ids()[0] or None),
    ).model_dump()
    return payload


def _attach_trace_ids(payload: dict[str, Any]) -> dict[str, Any]:
    """配额 / 熔断等早退 done 也带上 request_id / trace_id，便于前端赞踩。"""
    out = dict(payload)
    rid = get_request_id()
    if rid and not out.get("request_id"):
        out["request_id"] = rid
    trace_id, _ = current_trace_ids()
    if trace_id and not out.get("trace_id"):
        out["trace_id"] = trace_id
    return out


def _maybe_observe_ttft(
    event: str,
    *,
    started: float,
    accumulated: dict[str, Any],
    ttft_recorded: list[bool],
) -> None:
    """第一条 SSE token 打 TTFT；无 token 路径不记。"""
    if event != "token" or ttft_recorded[0]:
        return
    ttft_recorded[0] = True
    intent = str(accumulated.get("intent") or "unknown")
    observe_ttft(intent, time.monotonic() - started)


def _record_turn(
    turn: ChatTurn,
    accumulated: dict[str, Any],
    started: float,
    nodes: list[str],
    outcome: str,
    usage: TurnUsage | None = None,
) -> None:
    """一轮结束时打结构化日志并记 metrics。"""
    elapsed_ms = int((time.monotonic() - started) * 1000)
    intent = str(accumulated.get("intent") or "chitchat")
    blocked = bool(accumulated.get("blocked"))
    usage = usage or TurnUsage()
    raw_query = str(turn.inputs.get("query") or "")
    query, _ = mask_pii(raw_query[:200])
    log_event(
        logger,
        "chat_turn",
        session_id=turn.sid,
        thread_id=turn.thread_id,
        intent=intent,
        blocked=blocked,
        nodes=nodes,
        elapsed_ms=elapsed_ms,
        outcome=outcome,
        query=query,
        query_chars=len(raw_query),
        llm_calls=len(usage.steps),
    )
    inc_chat(turn.tenant, intent, blocked, outcome)
    observe_chat(intent, elapsed_ms / 1000.0)
    observe_llm_calls(intent, len(usage.steps))
    maybe_inc_runaway_llm_calls(len(usage.steps))
    set_span_attributes(
        **{
            "cs.intent": intent,
            "cs.outcome": outcome,
            "cs.blocked": "true" if blocked else "false",
            "session.id": turn.sid,
            "tenant.id": turn.tenant,
        }
    )
    structured = accumulated.get("structured") or {}
    rag_empty = (
        intent == "course_consult"
        and isinstance(structured, dict)
        and structured.get("course_name") == "未找到课程"
    )
    score_turn(blocked=blocked, intent=intent, rag_empty=rag_empty, outcome=outcome)

    trace_id, _ = current_trace_ids()
    if trace_id:
        from app.agents.rag import get_last_search_query_context
        from app.evals.harvest import TurnSnapshot, cache_turn_snapshot, record_retrieval_miss

        search_ctx = get_last_search_query_context()
        sources = []
        course_name = ""
        if isinstance(structured, dict):
            sources = list(structured.get("sources") or [])
            course_name = str(structured.get("course_name") or "")
        answer_preview = str(accumulated.get("answer") or "")[:200]
        cache_turn_snapshot(
            TurnSnapshot(
                trace_id=trace_id,
                tenant_id=turn.tenant,
                original_query=raw_query,
                search_query=(search_ctx.search_query if search_ctx else raw_query),
                intent=intent,
                rag_empty=rag_empty,
                sources=sources,
                course_name=course_name,
                rewrite_reason=(search_ctx.rewrite_reason if search_ctx else ""),
                answer_preview=answer_preview,
                session_id=turn.sid,
                request_id=get_request_id() or "",
            )
        )
        if rag_empty:
            record_retrieval_miss(
                tenant_id=turn.tenant,
                query=raw_query,
                search_query=(search_ctx.search_query if search_ctx else raw_query),
                reason="rag_empty_turn",
                trace_id=trace_id,
                rewrite_reason=(search_ctx.rewrite_reason if search_ctx else ""),
            )


async def _chat_turn_stream(
    message: str,
    user_id: str = "anonymous",
    tenant_id: str | None = None,
    session_id: str | None = None,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """一轮对话的共享流式实现：会话锁、图 astream、指标与归档。"""
    begin_turn_usage()
    turn = ChatTurn(message, user_id, tenant_id, session_id, hydrate=False)
    await turn.ahydrate()
    accumulated: dict[str, Any] = {}
    nodes: list[str] = []
    started = time.monotonic()
    outcome = "ok"
    usage = TurnUsage()
    ttft_recorded = [False]
    try:
        ctx = get_app_context()
        async with ahold_session(ctx.session_lock, turn.thread_id):
            yield turn.meta_event()
            try:
                async for item in ctx.graph.astream(
                    turn.inputs,
                    config=turn.run_config(),
                    stream_mode=["updates", "custom"],
                ):
                    for event, data in events_from_stream_item(item, accumulated, nodes):
                        _maybe_observe_ttft(
                            event,
                            started=started,
                            accumulated=accumulated,
                            ttft_recorded=ttft_recorded,
                        )
                        yield event, data
            except TenantQuotaExceededError:
                outcome = "quota"
                logger.warning("租户 %s 当日 LLM token 额度已用尽", turn.tenant)
                usage = take_turn_usage()
                yield ("done", _attach_trace_ids(_quota_done(turn.sid, usage)))
                return
            except LlmCircuitOpenError:
                outcome = "llm_unavailable"
                logger.warning("LLM 熔断打开，租户 %s 回退模板话术", turn.tenant)
                usage = take_turn_usage()
                yield ("done", _attach_trace_ids(_unavailable_done(turn.sid, usage)))
                return
            usage = take_turn_usage()
            yield ("done", _done_payload(accumulated, turn.sid, usage))
    except SessionBusyError:
        outcome = "session_busy"
        raise
    except Exception:
        if outcome == "ok":
            outcome = "error"
        raise
    finally:
        if outcome not in {"ok", "quota", "llm_unavailable"}:
            usage = take_turn_usage()
        _record_turn(turn, accumulated, started, nodes, outcome, usage)
        set_current_user("anonymous")


async def chat_astream(
    message: str,
    user_id: str = "anonymous",
    tenant_id: str | None = None,
    session_id: str | None = None,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """异步流式对话：updates + custom token，供 HTTP SSE 与 CLI 使用。

    全异步路径：节点已 async 化，不再有 ``asyncio.run`` 桥接或同步 ``chat``/``chat_stream``。
    同步调用方（CLI / 测试）请用 ``asyncio.run(achat(...))`` 或 ``asyncio.run`` 收集本流。
    """
    async for event in _chat_turn_stream(
        message,
        user_id=user_id,
        tenant_id=tenant_id,
        session_id=session_id,
    ):
        yield event
