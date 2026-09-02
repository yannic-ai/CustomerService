"""客服图状态：扁平 checkpoint + 按主题拆开的 TypedDict。

LangGraph / Redis 仍使用一层 key（兼容已有会话）。新增字段写到对应 mixin：
``identity`` / ``route`` / ``memory`` / ``result``，不要继续往 ``CSState`` 堆。
读分组语义用 ``IdentityView`` / ``RouteView`` / ``MemoryView`` / ``ResultView``。
"""

from __future__ import annotations

from app.config import DATA_DIR
from app.graph.state.identity import IdentityState
from app.graph.state.memory import MEMORY_SLOT_KEYS, MemoryState
from app.graph.state.result import RESULT_TURN_RESET, ResultState
from app.graph.state.route import ROUTE_TURN_RESET, RouteState
from app.graph.state.turn import reset_turn_fields
from app.graph.state.views import IdentityView, MemoryView, ResultView, RouteView
from app.thread_id import make_thread_id

GRAPH_MERMAID_PATH = DATA_DIR / "langgraph.mmd"
GRAPH_PNG_PATH = DATA_DIR / "langgraph.png"

NODE_STATUS = {
    "security": "正在做安全检查…",
    "router": "正在识别意图…",
    "rag": "正在检索课程知识库…",
    "tool": "正在查询订单…",
    "supervisor": "正在综合处理问题…",
    "chitchat": "正在整理回复…",
    "handoff": "正在转接人工…",
    "respond": "正在生成回复…",
}


class CSState(IdentityState, RouteState, MemoryState, ResultState):
    """客服图状态。字段按主题拆 mixin，序列化仍是扁平 dict。"""


__all__ = [
    "CSState",
    "GRAPH_MERMAID_PATH",
    "GRAPH_PNG_PATH",
    "IdentityState",
    "IdentityView",
    "MEMORY_SLOT_KEYS",
    "MemoryState",
    "MemoryView",
    "NODE_STATUS",
    "RESULT_TURN_RESET",
    "ROUTE_TURN_RESET",
    "ResultState",
    "ResultView",
    "RouteState",
    "RouteView",
    "make_thread_id",
    "reset_turn_fields",
]
