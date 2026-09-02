"""ReACT 工具执行上下文：history / 课程槽位经 ContextVar 注入。

模型不传 tenant/user/history 等业务上下文；ReACT 入口在调用 Agent 前
用 :func:`tool_context_scope` 写入，LangChain 工具从 :func:`get_tool_context` 读取。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from langchain_core.messages import BaseMessage


@dataclass(frozen=True)
class ToolContext:
    """一轮 ReACT 内工具可读的会话上下文（与 RAG 热路径参数对齐）。"""

    history: tuple[BaseMessage, ...] | None = None
    last_course_query: str | None = None
    last_course_name: str | None = None
    session_summary: str | None = None
    course_code: str | None = None

    @property
    def history_list(self) -> list[BaseMessage] | None:
        if not self.history:
            return None
        return list(self.history)


_EMPTY = ToolContext()
_current: ContextVar[ToolContext] = ContextVar("tool_context", default=_EMPTY)


def get_tool_context() -> ToolContext:
    """读取当前任务的工具上下文；未设置时返回空上下文。"""
    return _current.get()


@contextmanager
def tool_context_scope(
    *,
    history: list[BaseMessage] | None = None,
    last_course_query: str | None = None,
    last_course_name: str | None = None,
    session_summary: str | None = None,
    course_code: str | None = None,
) -> Iterator[ToolContext]:
    """在 ReACT / 工具调用期间注入会话上下文。"""
    ctx = ToolContext(
        history=tuple(history) if history else None,
        last_course_query=_strip(last_course_query),
        last_course_name=_strip(last_course_name),
        session_summary=_strip(session_summary),
        course_code=_strip(course_code),
    )
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)


def _strip(value: str | None) -> str | None:
    text = (value or "").strip()
    return text or None
