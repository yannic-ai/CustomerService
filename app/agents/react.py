"""ReACT Agent：在课程 RAG 与订单工具之间推理并调用。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, BaseMessage
from typing_extensions import override

from app.capabilities import langchain_tools
from app.capabilities.order import tool_order_query
from app.config import get_settings
from app.context import format_history_for_prompt, order_no_from_history
from app.observability import inc_agent_runaway, inc_react_limit_hit
from app.observability.langfuse import PROMPT_REACT, get_text_prompt
from app.observability.otel import node_span
from app.security.middleware import SecurityAgentMiddleware
from app.tenancy import set_current_tenant, set_current_user
from app.tool_context import tool_context_scope
from app.tool_cache import session_tool_cache_scope
from app.tool_escalation import ToolEscalationError, tool_failure_tracker_scope
from app.services.order import extract_order_no

logger = logging.getLogger("cs.react")


class ObservingToolCallLimitMiddleware(ToolCallLimitMiddleware):
    """撞 run_limit 时打 Prometheus 计数；可配置自动转人工。"""

    def _on_limit_hit(self, result: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(result, dict) or not result.get("messages"):
            return result
        inc_react_limit_hit()
        inc_agent_runaway("react_limit")
        settings = get_settings()
        if not settings.tool_escalation_on_react_limit:
            return result
        from app.tool_escalation import escalation_for_react_limit

        decision = escalation_for_react_limit()
        return {
            "messages": [AIMessage(content=decision.message or "")],
            "jump_to": "end",
        }

    @override
    def after_model(self, state, runtime):
        result = super().after_model(state, runtime)
        return self._on_limit_hit(result)

    @override
    async def aafter_model(self, state, runtime):
        result = await super().aafter_model(state, runtime)
        return self._on_limit_hit(result)


SYSTEM_PROMPT = """你是智能客服 ReACT Agent。
处理流程：思考（Reason）→ 选择工具（Act）→ 观察结果（Observe）→ 给出最终答复。
- 课程大纲、内容、学习路径：调用 retrieve_course_knowledge
- 订单、退款、支付进度：调用 tool_order_query，必须传入 order_no
- 没有订单号时不要调用 tool_order_query，先向用户确认完整编号，严禁编造
- 对话历史或「关联订单号」里已有编号时，将其作为 order_no 传入，不要再向用户索要
- 可并行或依次调用多个工具
工具返回错误时：
- 缺订单号：向用户确认完整订单编号，不要编造
- 未找到订单：告知用户并建议核对，不要用同一订单号反复调用
- 其他错误：用中文说明，不要暴露内部堆栈或系统细节
最终用中文回答，课程问题要列出模块和学习路径；订单问题要保留进度可视化。"""


def build_react_agent(user_id: str = "anonymous", tenant_id: str = "demo"):
    """构建带安全中间件的 LangChain 1.0 ReACT Agent；无 API Key 时返回 None。"""
    settings = get_settings()
    if not settings.llm_enabled:
        return None
    from app.llm import get_chat_model

    return create_agent(
        model=get_chat_model(temperature=0.1, usage_tag="react", cache_enabled=False),
        tools=langchain_tools(),
        system_prompt=get_text_prompt(PROMPT_REACT, SYSTEM_PROMPT),
        middleware=[
            SecurityAgentMiddleware(user_id=user_id, tenant_id=tenant_id),
            ObservingToolCallLimitMiddleware(run_limit=8, exit_behavior="continue"),
        ],
    )


async def _offline_react(
    query: str,
    *,
    user_id: str,
    tenant_id: str,
    history: list[BaseMessage] | None,
    order_no: str | None,
    resolved_order: str | None,
    course_query: str | None,
    last_course_name: str | None,
    search_query: str,
    session_summary: str | None,
) -> str:
    """无 LLM 或调用失败时走订单+检索离线拼接（异步）。"""
    from app.agents.rag import NOT_FOUND_COURSE
    from app.capabilities.course import retrieve_course
    from app.capabilities.order import format_order_answer, query_order

    parts: list[str] = []
    try:
        parts.append(
            format_order_answer(
                await asyncio.to_thread(
                    query_order,
                    query,
                    resolved_order,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    last_order_no=order_no or resolved_order,
                    persist=False,
                    use_order_pool=False,
                )
            )
        )
    except ValueError:
        pass
    course = await retrieve_course(
        search_query,
        tenant_id=tenant_id,
        history=history,
        last_course_query=course_query,
        last_course_name=last_course_name,
        session_summary=session_summary,
    )
    if course.course_name != NOT_FOUND_COURSE:
        from app.agents.rag import format_course_answer

        parts.append(format_course_answer(course))
    return "\n\n".join(parts) or "暂时无法处理该问题。"


async def _react_or_offline(agent: Any, runner: Callable, **offline: Any) -> str:
    """无 LLM 或调用失败时走订单+检索离线拼接（异步）。"""
    if agent is None:
        return await _offline_react(**offline)
    try:
        return await runner()
    except ToolEscalationError as exc:
        return exc.decision.message or "已为您转接人工客服。"
    except Exception as exc:
        from app.llm_gateway import is_llm_degradable

        if not is_llm_degradable(exc):
            raise
        logger.warning("ReACT LLM 不可用，回退离线路径: %s", exc)
        return await _offline_react(**offline)


async def _areact_or_offline(agent: Any, runner: Callable, **offline: Any) -> str:
    """向后兼容别名：等价于 :func:`_react_or_offline`。"""
    return await _react_or_offline(agent, runner, **offline)


def _build_react_user_content(
    query: str,
    *,
    history: list[BaseMessage] | None,
    session_summary: str | None,
    resolved_order: str | None,
    course_query: str | None,
    last_course_name: str | None = None,
) -> str:
    history_text = format_history_for_prompt(history, session_summary=session_summary)
    user_content = query
    if history_text:
        user_content = f"对话历史：\n{history_text}\n\n当前用户问题：{query}"
    if resolved_order and resolved_order not in user_content:
        user_content = f"{user_content}\n（关联订单号：{resolved_order}）"
    course_hint = (last_course_name or course_query or "").strip()
    if course_hint and course_hint not in user_content:
        user_content = f"{user_content}\n（关联课程：{course_hint}）"
    return user_content


def _message_has_tool_calls(message: Any) -> bool:
    calls = getattr(message, "tool_calls", None) or []
    if calls:
        return True
    additional = getattr(message, "additional_kwargs", None) or {}
    if isinstance(additional, dict) and additional.get("tool_calls"):
        return True
    chunk_calls = getattr(message, "tool_call_chunks", None) or []
    return bool(chunk_calls)


def _chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return ""


def _emit_token(content: str) -> None:
    if not content:
        return
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:
        return
    if writer is None:
        return
    from app.security.filters import mask_pii

    masked, _ = mask_pii(content)
    if masked:
        writer({"event": "token", "content": masked})


def _react_config(user_id: str, tenant_id: str, resolved_order: str | None) -> dict[str, Any]:
    from app.observability.langfuse import langfuse_callbacks, trace_metadata
    from app.security.callback import SecurityCallbackHandler

    thread_id = f"{tenant_id}:{user_id}:{resolved_order or 'react'}"
    return {
        "callbacks": [
            SecurityCallbackHandler(user_id=user_id, tenant_id=tenant_id),
            *langfuse_callbacks(),
        ],
        "metadata": trace_metadata(
            user_id=user_id,
            tenant_id=tenant_id,
            session_id=resolved_order or "react",
            thread_id=thread_id,
        ),
    }


async def run_react(
    query: str,
    user_id: str = "anonymous",
    tenant_id: str = "demo",
    history: list[BaseMessage] | None = None,
    order_no: str | None = None,
    course_query: str | None = None,
    last_course_name: str | None = None,
    session_summary: str | None = None,
    session_cache_state: dict[str, Any] | None = None,
) -> str:
    """执行 ReACT 循环；无 LLM 时依次尝试订单查询和课程检索（全异步）。

    内部工具 ``tool_order_query`` 与订单查询仍可能被 LangGraph 用线程池跑
    （LangGraph 1.0 在 ainvoke 路径下用 anyio 线程池跑同步 tool）；
    主调用链不再有 ``asyncio.run`` 桥接或外层课程线程池。
    """
    with node_span("react", **{"tenant.id": tenant_id}):
        return await _run_react_inner(
            query,
            user_id=user_id,
            tenant_id=tenant_id,
            history=history,
            order_no=order_no,
            course_query=course_query,
            last_course_name=last_course_name,
            session_summary=session_summary,
            session_cache_state=session_cache_state,
        )


async def arun_react(
    query: str,
    user_id: str = "anonymous",
    tenant_id: str = "demo",
    history: list[BaseMessage] | None = None,
    order_no: str | None = None,
    course_query: str | None = None,
    last_course_name: str | None = None,
    session_summary: str | None = None,
) -> str:
    """向后兼容别名：等价于 :func:`run_react`。"""
    return await run_react(
        query,
        user_id=user_id,
        tenant_id=tenant_id,
        history=history,
        order_no=order_no,
        course_query=course_query,
        last_course_name=last_course_name,
        session_summary=session_summary,
    )


async def _run_react_inner(
    query: str,
    user_id: str = "anonymous",
    tenant_id: str = "demo",
    history: list[BaseMessage] | None = None,
    order_no: str | None = None,
    course_query: str | None = None,
    last_course_name: str | None = None,
    session_summary: str | None = None,
    session_cache_state: dict[str, Any] | None = None,
) -> str:
    set_current_tenant(tenant_id)
    set_current_user(user_id)
    resolved_order = order_no or extract_order_no(query) or order_no_from_history(history)
    search_query = course_query or last_course_name or query
    agent = build_react_agent(user_id=user_id, tenant_id=tenant_id)
    offline = dict(
        query=query,
        user_id=user_id,
        tenant_id=tenant_id,
        history=history,
        order_no=order_no,
        resolved_order=resolved_order,
        course_query=course_query,
        last_course_name=last_course_name,
        search_query=search_query,
        session_summary=session_summary,
    )

    async def _ainvoke() -> str:
        user_content = _build_react_user_content(
            query,
            history=history,
            session_summary=session_summary,
            resolved_order=resolved_order,
            course_query=course_query,
            last_course_name=last_course_name,
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_content}]},
            config=_react_config(user_id, tenant_id, resolved_order),
        )
        messages = result.get("messages") or []
        last = messages[-1]
        return getattr(last, "content", str(last))

    with tool_failure_tracker_scope(), session_tool_cache_scope(session_cache_state):
        with tool_context_scope(
            history=history,
            last_course_query=course_query,
            last_course_name=last_course_name,
            session_summary=session_summary,
        ):
            return await _react_or_offline(agent, _ainvoke, **offline)


async def _arun_react_inner(
    query: str,
    user_id: str = "anonymous",
    tenant_id: str = "demo",
    history: list[BaseMessage] | None = None,
    order_no: str | None = None,
    course_query: str | None = None,
    last_course_name: str | None = None,
    session_summary: str | None = None,
) -> str:
    """向后兼容别名：等价于 :func:`_run_react_inner`。"""
    return await _run_react_inner(
        query,
        user_id=user_id,
        tenant_id=tenant_id,
        history=history,
        order_no=order_no,
        course_query=course_query,
        last_course_name=last_course_name,
        session_summary=session_summary,
    )


async def stream_react(
    query: str,
    user_id: str = "anonymous",
    tenant_id: str = "demo",
    history: list[BaseMessage] | None = None,
    order_no: str | None = None,
    course_query: str | None = None,
    last_course_name: str | None = None,
    session_summary: str | None = None,
    session_cache_state: dict[str, Any] | None = None,
    emit_token: Callable[[str], None] | None = None,
) -> str:
    """流式 ReACT：async for agent.astream，只推送无 tool_calls 的最终文本 delta。"""
    with node_span("react", **{"tenant.id": tenant_id}):
        return await _stream_react_inner(
            query,
            user_id=user_id,
            tenant_id=tenant_id,
            history=history,
            order_no=order_no,
            course_query=course_query,
            last_course_name=last_course_name,
            session_summary=session_summary,
            session_cache_state=session_cache_state,
            emit_token=emit_token,
        )


async def astream_react(
    query: str,
    user_id: str = "anonymous",
    tenant_id: str = "demo",
    history: list[BaseMessage] | None = None,
    order_no: str | None = None,
    course_query: str | None = None,
    last_course_name: str | None = None,
    session_summary: str | None = None,
    emit_token: Callable[[str], None] | None = None,
) -> str:
    """向后兼容别名：等价于 :func:`stream_react`。"""
    return await stream_react(
        query,
        user_id=user_id,
        tenant_id=tenant_id,
        history=history,
        order_no=order_no,
        course_query=course_query,
        last_course_name=last_course_name,
        session_summary=session_summary,
        emit_token=emit_token,
    )


async def _stream_react_inner(
    query: str,
    user_id: str = "anonymous",
    tenant_id: str = "demo",
    history: list[BaseMessage] | None = None,
    order_no: str | None = None,
    course_query: str | None = None,
    last_course_name: str | None = None,
    session_summary: str | None = None,
    session_cache_state: dict[str, Any] | None = None,
    emit_token: Callable[[str], None] | None = None,
) -> str:
    set_current_tenant(tenant_id)
    set_current_user(user_id)
    resolved_order = order_no or extract_order_no(query) or order_no_from_history(history)
    search_query = course_query or last_course_name or query
    agent = build_react_agent(user_id=user_id, tenant_id=tenant_id)
    offline = dict(
        query=query,
        user_id=user_id,
        tenant_id=tenant_id,
        history=history,
        order_no=order_no,
        resolved_order=resolved_order,
        course_query=course_query,
        last_course_name=last_course_name,
        search_query=search_query,
        session_summary=session_summary,
    )

    async def _astream() -> str:
        emit = emit_token or _emit_token
        user_content = _build_react_user_content(
            query,
            history=history,
            session_summary=session_summary,
            resolved_order=resolved_order,
            course_query=course_query,
            last_course_name=last_course_name,
        )
        parts: list[str] = []
        last_message: Any = None
        async for item in agent.astream(
            {"messages": [{"role": "user", "content": user_content}]},
            config=_react_config(user_id, tenant_id, resolved_order),
            stream_mode="messages",
        ):
            message = item[0] if isinstance(item, tuple) else item
            last_message = message
            if not isinstance(message, AIMessage) and type(message).__name__ not in {
                "AIMessage",
                "AIMessageChunk",
            }:
                continue
            if _message_has_tool_calls(message):
                continue
            delta = _chunk_text(message)
            if not delta:
                continue
            parts.append(delta)
            emit(delta)

        answer = "".join(parts).strip()
        if answer:
            return answer
        if last_message is not None:
            content = getattr(last_message, "content", None)
            if isinstance(content, str) and content.strip():
                return content
        return await _run_react_inner(
            query,
            user_id=user_id,
            tenant_id=tenant_id,
            history=history,
            order_no=order_no,
            course_query=course_query,
            last_course_name=last_course_name,
            session_summary=session_summary,
            session_cache_state=session_cache_state,
        )

    with tool_failure_tracker_scope(), session_tool_cache_scope(session_cache_state):
        with tool_context_scope(
            history=history,
            last_course_query=course_query,
            last_course_name=last_course_name,
            session_summary=session_summary,
        ):
            return await _react_or_offline(agent, _astream, **offline)


async def _astream_react_inner(
    query: str,
    user_id: str = "anonymous",
    tenant_id: str = "demo",
    history: list[BaseMessage] | None = None,
    order_no: str | None = None,
    course_query: str | None = None,
    last_course_name: str | None = None,
    session_summary: str | None = None,
    emit_token: Callable[[str], None] | None = None,
) -> str:
    """向后兼容别名：等价于 :func:`_stream_react_inner`。"""
    return await _stream_react_inner(
        query,
        user_id=user_id,
        tenant_id=tenant_id,
        history=history,
        order_no=order_no,
        course_query=course_query,
        last_course_name=last_course_name,
        session_summary=session_summary,
        emit_token=emit_token,
    )
