"""回合收尾：脱敏、课纲压缩、分级压缩、消息补丁与冷归档。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage

from app.config import Settings, get_settings
from app.context.slots import SessionSlots
from app.context.summary import CompressResult, compact_rag_history, compress_by_importance
from app.db.archive import upsert_session_archive
from app.llm_gateway.exceptions import LLM_UNAVAILABLE_ANSWER
from app.thread_id import make_thread_id
from app.security.filters import inspect_output

if TYPE_CHECKING:
    from app.graph.state import CSState


@dataclass
class RespondTurnUpdate:
    """单轮 respond 产出的状态补丁（不含 DB 写入）。"""

    answer: str
    stored_answer: str
    message_updates: list[BaseMessage]
    session_summary: str
    kept_messages: list[BaseMessage]


def build_stored_answer(answer: str) -> tuple[str, str]:
    """输出脱敏后，再压课纲得到 checkpoint 用内容；返回 (full_answer, stored_answer)。"""
    full = inspect_output(answer or "")
    stored = compact_rag_history(full)
    return full, stored


def build_message_updates(
    *,
    result: CompressResult,
    stored_answer: str,
) -> list[BaseMessage]:
    """将 CompressResult 转为 LangGraph 消息补丁。"""
    to_remove = result.discarded + result.summarized
    removals = [RemoveMessage(id=m.id) for m in to_remove if getattr(m, "id", None)]
    replacements = [m for m in result.updated if getattr(m, "id", None)]
    last = result.kept[-1] if result.kept else None
    if isinstance(last, AIMessage) and not getattr(last, "id", None):
        turn_ai: BaseMessage = last
    else:
        turn_ai = AIMessage(content=stored_answer)
    return [*removals, *replacements, turn_ai]


def resolve_respond_answer(state: CSState) -> str:
    """有业务答案则沿用；空答复时回退 LLM 不可用模板话术。"""
    from app.graph.state import ResultView, RouteView

    result = ResultView.from_state(state)
    if result.answer.strip():
        return result.answer
    if RouteView.from_state(state).blocked:
        return result.answer or "请求已被安全中间件拦截。"
    return LLM_UNAVAILABLE_ANSWER


async def finalize_respond_turn(
    state: CSState,
    *,
    settings: Settings | None = None,
) -> RespondTurnUpdate:
    """脱敏 + 课纲压缩 + 分级压缩，生成消息补丁（不写归档）。"""
    from app.graph.state import IdentityView, MemoryView

    answer, stored_answer = build_stored_answer(resolve_respond_answer(state))
    existing = list(state.get("messages") or [])
    projected = existing + [AIMessage(content=stored_answer)]
    cfg = settings or get_settings()
    ident = IdentityView.from_state(state, default_tenant=cfg.default_tenant)
    memory = MemoryView.from_state(state)
    result = await compress_by_importance(
        projected,
        current_query=ident.query,
        token_budget=cfg.history_token_budget,
        slots=SessionSlots.from_state(state),
        always_keep_recent=cfg.context_always_keep_recent,
        discard_score=cfg.context_discard_score,
        summarize_score=cfg.context_summarize_score,
        max_messages=cfg.history_max_messages,
        existing_summary=memory.session_summary,
        use_llm_summary=cfg.context_summary_use_llm,
        max_summary_chars=cfg.session_summary_max_chars,
    )
    message_updates = build_message_updates(result=result, stored_answer=stored_answer)
    return RespondTurnUpdate(
        answer=answer,
        stored_answer=stored_answer,
        message_updates=message_updates,
        session_summary=result.summary,
        kept_messages=result.kept,
    )


async def persist_respond_turn(state: CSState, update: RespondTurnUpdate) -> None:
    """将会话公共状态写入 MySQL 冷归档；失败只打日志。"""
    from app.graph.state import IdentityView

    cfg = get_settings()
    ident = IdentityView.from_state(state, default_tenant=cfg.default_tenant)
    await asyncio.to_thread(
        upsert_session_archive,
        ident.tenant_id,
        ident.user_id,
        ident.session_id,
        thread_id=make_thread_id(ident.tenant_id, ident.session_id, ident.user_id)
        if ident.session_id
        else "",
        state={
            **state,
            "messages": update.kept_messages,
            "session_summary": update.session_summary,
        },
    )
