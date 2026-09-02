"""ReACT 挂载的 LangChain Tool 入口（实现见 ``app.capabilities``）。"""

from __future__ import annotations

from typing import Any

__all__ = ["langchain_tools", "retrieve_course_knowledge", "tool_order_query"]


def langchain_tools() -> list[Any]:
    """ReACT Agent 使用的全部 LangChain tool。"""
    from app.capabilities.registry import langchain_tools as _langchain_tools

    return _langchain_tools()


def __getattr__(name: str) -> Any:
    if name == "retrieve_course_knowledge":
        from app.capabilities.course import retrieve_course_knowledge

        return retrieve_course_knowledge
    if name == "tool_order_query":
        from app.capabilities.order import tool_order_query

        return tool_order_query
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
