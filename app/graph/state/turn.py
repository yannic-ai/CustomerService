"""本轮瞬时字段：入口清空，避免 checkpointer 把旧报告带回。"""

from __future__ import annotations

from typing import Any

from app.graph.state.result import RESULT_TURN_RESET
from app.graph.state.route import ROUTE_TURN_RESET


def reset_turn_fields() -> dict[str, Any]:
    """清空 route + result；跨轮 memory 槽位保持。"""
    return {**ROUTE_TURN_RESET, **RESULT_TURN_RESET}
