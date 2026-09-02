"""LangGraph 主流程：安全检查 → 路由 → RAG / 订单工具 / Supervisor。

低置信或当前问句含订单信号时不直达 RAG。若仍进入 RAG，有订单信号则不生成答案、直接升 ReACT。

实现按职责拆到子模块：
- ``state``：扁平 checkpoint；字段按 identity / route / memory / result 分组
- ``nodes``：节点实现
- ``builder``：图编译；进程单例见 ``app.runtime.AppContext``
- ``hydrate``：checkpoint miss 时从归档回填
- ``stream``：LangGraph 流 → SSE
- ``runner``：对话入口

分流规则统一走 ``app.agents.policy.RoutingPolicy``（intent 出口、图边、RAG 升级、Supervisor 澄清）。
"""

from __future__ import annotations

from typing import Any

from app.graph.state import (
    CSState,
    GRAPH_MERMAID_PATH,
    GRAPH_PNG_PATH,
    NODE_STATUS,
    make_thread_id,
    reset_turn_fields,
)

_RUNNER_EXPORTS = frozenset({"ChatTurn", "chat_astream"})
_BUILDER_EXPORTS = frozenset(
    {
        "build_graph",
        "export_graph_visual",
        "get_checkpointer",
        "get_graph",
        "graph_mermaid",
        "reset_graph",
    }
)

__all__ = [
    "CSState",
    "GRAPH_MERMAID_PATH",
    "GRAPH_PNG_PATH",
    "NODE_STATUS",
    "ChatTurn",
    "build_graph",
    "chat_astream",
    "export_graph_visual",
    "get_checkpointer",
    "get_graph",
    "graph_mermaid",
    "make_thread_id",
    "reset_graph",
    "reset_turn_fields",
]


def __getattr__(name: str) -> Any:
    if name in _RUNNER_EXPORTS:
        from app.graph import runner as _runner

        return getattr(_runner, name)
    if name in _BUILDER_EXPORTS:
        from app.graph import builder as _builder

        return getattr(_builder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
