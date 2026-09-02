"""跨轮记忆：槽位、滚动摘要。不随 reset_turn_fields 清空。"""

from __future__ import annotations

from typing import Any, TypedDict

MEMORY_SLOT_KEYS = (
    "last_intent",
    "last_order_no",
    "last_course_query",
    "last_course_name",
    "handoff_pending",
    "last_order_structured",
    "last_course_structured",
    "last_course_structured_key",
)


class MemoryState(TypedDict, total=False):
    """跨轮保留的槽位与摘要。新增跨轮字段写这里，不要写进 RouteState。"""

    last_intent: str | None
    last_order_no: str | None
    last_course_query: str | None
    last_course_name: str | None
    handoff_pending: bool
    last_order_structured: dict[str, Any] | None
    last_course_structured: dict[str, Any] | None
    last_course_structured_key: str | None
    session_summary: str  # 被 token 窗口裁掉的旧对话摘要
