"""LangChain 1.0 AgentMiddleware：把安全回调接入 create_agent 生命周期。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.messages import AIMessage, ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.runtime import Runtime

from app.security.callback import SecurityCallbackHandler
from app.security.filters import inspect_input, inspect_output, mask_pii
from app.tool_escalation import ToolEscalationError, escalation_for_transient, is_transient_tool_failure

_TRANSIENT_TYPES = (TimeoutError, ConnectionError, OSError)
_MAX_TRANSIENT_ATTEMPTS = 2


def _is_transient_error(exc: BaseException) -> bool:
    """判断是否为可短暂重试的超时/连接类错误。"""
    if isinstance(exc, _TRANSIENT_TYPES):
        return True
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return "timeout" in name or "timeout" in message or "connection" in name


def _tool_resource_key(request: Any, tool_name: str) -> str:
    tool_call = getattr(request, "tool_call", None) or {}
    if isinstance(tool_call, dict):
        args = tool_call.get("args") or {}
        if isinstance(args, dict):
            order_no = args.get("order_no")
            if order_no:
                return str(order_no)
            question = args.get("question") or args.get("query")
            if question:
                return str(question)[:120]
    return tool_name or "default"


def _tool_call_meta(request: Any) -> tuple[str, str]:
    """从工具请求中取出名称与 call id。"""
    tool_call = getattr(request, "tool_call", None) or {}
    if isinstance(tool_call, dict):
        return str(tool_call.get("name") or ""), str(tool_call.get("id") or "")
    return str(getattr(request, "name", "") or ""), str(getattr(request, "id", "") or "")


def _sanitize_tool_result(result: Any) -> Any:
    """对 ToolMessage 内容做 PII 脱敏。"""
    if not isinstance(result, ToolMessage) or not isinstance(result.content, str):
        return result
    masked, _ = mask_pii(result.content)
    if masked == result.content:
        return result
    if hasattr(result, "model_copy"):
        return result.model_copy(update={"content": masked})
    return result


def _error_tool_message(name: str, call_id: str, exc: BaseException) -> ToolMessage:
    """把执行异常转成可回灌模型的错误 ToolMessage，不含堆栈。"""
    detail, _ = mask_pii(str(exc).replace("\n", " ").strip()[:200])
    content = (
        f"工具 {name or 'unknown'} 执行失败：{detail or type(exc).__name__}。"
        "请根据错误修正参数后重试；若是业务错误请向用户说明，不要用相同参数反复调用。"
    )
    return ToolMessage(
        content=content,
        name=name or None,
        tool_call_id=call_id or "unknown",
        status="error",
    )


class SecurityAgentMiddleware(AgentMiddleware):
    """v1.0 适配层：在 Agent 前后做输入拦截、PII 脱敏、工具审计。"""

    def __init__(self, user_id: str = "anonymous", tenant_id: str = "demo") -> None:
        """初始化中间件，并挂载同用户的安全回调。"""
        super().__init__()
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.callback = SecurityCallbackHandler(user_id=user_id, tenant_id=tenant_id)

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Agent 启动前检查用户输入，拦截注入后直接结束。"""
        messages = state.get("messages") or []
        if not messages:
            return None
        last = messages[-1]
        content = last.content if hasattr(last, "content") else str(last)
        if not isinstance(content, str):
            content = str(content)
        verdict = inspect_input(content)
        if not verdict.allowed:
            return {
                "messages": [AIMessage(f"请求已被安全中间件拦截：{verdict.reason}")],
                "jump_to": "end",
            }
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """模型输出后脱敏 PII。"""
        messages = state.get("messages") or []
        if not messages:
            return None
        last = messages[-1]
        content = getattr(last, "content", None)
        if not isinstance(content, str):
            return None
        masked = inspect_output(content)
        if masked == content:
            return None
        return {"messages": [AIMessage(content=masked)]}

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """包裹每次 LLM 调用，发送前对消息做 PII 脱敏。"""
        sanitized_messages = []
        for message in request.messages:
            content = getattr(message, "content", "")
            if isinstance(content, str):
                masked, _ = mask_pii(content)
                if masked != content and hasattr(message, "model_copy"):
                    message = message.model_copy(update={"content": masked})
            sanitized_messages.append(message)
        return handler(request.override(messages=sanitized_messages))

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        """包裹工具调用：审计、瞬态重试一次，执行失败转为错误 ToolMessage。"""
        name, call_id = _tool_call_meta(request)
        self.callback._record("middleware_tool_call", tool=name)
        last_error: BaseException | None = None
        for attempt in range(_MAX_TRANSIENT_ATTEMPTS):
            try:
                return _sanitize_tool_result(handler(request))
            except GraphBubbleUp:
                raise
            except ToolEscalationError:
                raise
            except Exception as exc:
                last_error = exc
                self.callback._record(
                    "middleware_tool_error",
                    tool=name,
                    attempt=attempt + 1,
                    error=str(exc)[:200],
                )
                if attempt + 1 < _MAX_TRANSIENT_ATTEMPTS and _is_transient_error(exc):
                    continue
                if is_transient_tool_failure(exc):
                    escalation_for_transient(name or "unknown", _tool_resource_key(request, name))
                break
        assert last_error is not None
        return _error_tool_message(name, call_id, last_error)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """wrap_model_call 的异步孪生：发送前对消息做 PII 脱敏，再 await handler。

        LangChain 1.0 在 ainvoke/astream 路径下只调 awrap_model_call；
        不实现则抛 NotImplementedError，使整条异步 ReACT 链路不可用。
        """
        sanitized_messages = []
        for message in request.messages:
            content = getattr(message, "content", "")
            if isinstance(content, str):
                masked, _ = mask_pii(content)
                if masked != content and hasattr(message, "model_copy"):
                    message = message.model_copy(update={"content": masked})
            sanitized_messages.append(message)
        return await handler(request.override(messages=sanitized_messages))

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        """wrap_tool_call 的异步孪生：审计、瞬态重试一次，失败转错误 ToolMessage。"""
        name, call_id = _tool_call_meta(request)
        self.callback._record("middleware_tool_call", tool=name)
        last_error: BaseException | None = None
        for attempt in range(_MAX_TRANSIENT_ATTEMPTS):
            try:
                return _sanitize_tool_result(await handler(request))
            except GraphBubbleUp:
                raise
            except ToolEscalationError:
                raise
            except Exception as exc:
                last_error = exc
                self.callback._record(
                    "middleware_tool_error",
                    tool=name,
                    attempt=attempt + 1,
                    error=str(exc)[:200],
                )
                if attempt + 1 < _MAX_TRANSIENT_ATTEMPTS and _is_transient_error(exc):
                    continue
                if is_transient_tool_failure(exc):
                    escalation_for_transient(name or "unknown", _tool_resource_key(request, name))
                break
        assert last_error is not None
        return _error_tool_message(name, call_id, last_error)
