"""工具失败升级：同轮瞬态失败计数与自动转人工。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Literal

from app.config import get_settings
from app.executors import ToolPoolBusyError
from app.observability import inc_tool_escalation

EscalationReason = Literal["transient_exhausted", "react_limit"]

_TRANSIENT_TYPES = (TimeoutError, ConnectionError, OSError)


def _is_transient_error(exc: BaseException) -> bool:
    if isinstance(exc, _TRANSIENT_TYPES):
        return True
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return "timeout" in name or "timeout" in message or "connection" in name


@dataclass(frozen=True)
class EscalationDecision:
    escalate: bool
    reason: EscalationReason | None = None
    message: str | None = None


class ToolEscalationError(Exception):
    """工具层判定应自动转人工。"""

    def __init__(self, decision: EscalationDecision) -> None:
        self.decision = decision
        super().__init__(decision.message or "tool escalation")


@dataclass
class ToolFailureTracker:
    """同轮内按 tool+resource_key 累计瞬态失败。"""

    counts: dict[tuple[str, str], int]

    def record_transient(self, tool: str, resource_key: str) -> int:
        key = (tool, (resource_key or "").strip() or "__default__")
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    def should_escalate(self, tool: str, resource_key: str) -> bool:
        settings = get_settings()
        if not settings.tool_escalation_enabled:
            return False
        key = (tool, (resource_key or "").strip() or "__default__")
        threshold = int(settings.tool_escalation_transient_threshold)
        return self.counts.get(key, 0) >= threshold


_tracker_ctx: ContextVar[ToolFailureTracker | None] = ContextVar(
    "tool_failure_tracker", default=None
)


def get_tool_failure_tracker() -> ToolFailureTracker | None:
    return _tracker_ctx.get()


@contextmanager
def tool_failure_tracker_scope() -> Iterator[ToolFailureTracker]:
    tracker = ToolFailureTracker(counts={})
    token = _tracker_ctx.set(tracker)
    try:
        yield tracker
    finally:
        _tracker_ctx.reset(token)


def is_transient_tool_failure(exc: BaseException | None = None, *, status: str | None = None) -> bool:
    """瞬态失败：超时/连接/线程池 busy。"""
    if status == "busy":
        return True
    if exc is None:
        return False
    if isinstance(exc, ToolPoolBusyError):
        return True
    return _is_transient_error(exc)


def build_handoff_answer(*, auto_escalated: bool = False, reason: EscalationReason | None = None) -> str:
    """转人工话术；自动升级 variant 说明多次查询失败。"""
    if auto_escalated and reason == "transient_exhausted":
        lead = (
            "多次未能获取最新查询结果，已为您转接人工客服。"
            "当前为演示环境，请稍候；工作时间（09:00-21:00）内会有专员继续跟进。"
        )
    elif auto_escalated and reason == "react_limit":
        lead = (
            "处理步骤较多且未能完成自动查询，已为您转接人工客服。"
            "当前为演示环境，请稍候；工作时间（09:00-21:00）内会有专员继续跟进。"
        )
    else:
        lead = (
            "已为您转接人工客服。当前为演示环境，请稍候；"
            "工作时间（09:00-21:00）内会有专员继续跟进您的问题。"
        )
    return f"{lead}您也可以继续描述问题，我会先记录在会话中。"


def escalation_for_transient(tool: str, resource_key: str) -> EscalationDecision | None:
    tracker = get_tool_failure_tracker()
    if tracker is None:
        return None
    count = tracker.record_transient(tool, resource_key)
    settings = get_settings()
    if not settings.tool_escalation_enabled:
        return None
    threshold = int(settings.tool_escalation_transient_threshold)
    if count < threshold:
        return None
    message = build_handoff_answer(auto_escalated=True, reason="transient_exhausted")
    inc_tool_escalation("transient_exhausted")
    return EscalationDecision(escalate=True, reason="transient_exhausted", message=message)


def escalation_for_react_limit() -> EscalationDecision:
    message = build_handoff_answer(auto_escalated=True, reason="react_limit")
    inc_tool_escalation("react_limit")
    return EscalationDecision(escalate=True, reason="react_limit", message=message)


def raise_if_escalated(decision: EscalationDecision | None) -> None:
    if decision and decision.escalate:
        raise ToolEscalationError(decision)
