"""编译客服主图：节点接线、checkpointer 单例与可视化导出。"""

from __future__ import annotations

import logging
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from app.agents.policy import DEFAULT_POLICY
from app.config import DATA_DIR
from app.graph.nodes import (
    chitchat_node,
    handoff_node,
    rag_node,
    respond_node,
    router_node,
    security_node,
    supervisor_node,
    tool_node,
)
from app.graph.state import CSState, GRAPH_MERMAID_PATH, GRAPH_PNG_PATH, RouteView
from app.observability.otel import node_span
from app.session_cache import DictSessionLock, build_checkpointer

logger = logging.getLogger("cs.graph")


def _after_rag(state: CSState) -> str:
    """RAG 因订单信号升到 Supervisor 时不重跑 Router。"""
    route = RouteView.from_state(state)
    return DEFAULT_POLICY.next_after_rag(
        upgraded=route.rag_upgraded_to_supervisor,
        target=route.target,
    )


def _after_router(state: CSState) -> str:
    """根据路由 target 选择下一个节点；低置信 RAG 改走 Supervisor。"""
    route = RouteView.from_state(state)
    return DEFAULT_POLICY.next_after_router(
        blocked=route.blocked,
        target=route.target,
        intent=route.intent,
        confidence=route.confidence,
    )


def graph_mermaid(compiled) -> str:
    """导出编译后客服图的 Mermaid 源码。"""
    return compiled.get_graph().draw_mermaid()


def export_graph_visual(compiled, *, png: bool = False) -> Path:
    """把流程图写成 Mermaid 文件；可选尝试导出 PNG。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    mermaid = graph_mermaid(compiled)
    GRAPH_MERMAID_PATH.write_text(mermaid, encoding="utf-8")
    logger.info("LangGraph 可视化已写入 %s", GRAPH_MERMAID_PATH)
    if png:
        try:
            compiled.get_graph().draw_mermaid_png(output_file_path=str(GRAPH_PNG_PATH))
            logger.info("LangGraph PNG 已写入 %s", GRAPH_PNG_PATH)
        except Exception as exc:  # mermaid.ink 不可用时跳过
            logger.warning("PNG 可视化跳过：%s", exc)
    return GRAPH_MERMAID_PATH


def _timed(name: str, fn):
    """图节点：Prometheus 耗时 + OTel 子 span ``cs.graph.{name}``。"""

    def wrapped(state: CSState) -> CSState:
        with node_span(
            name,
            **{"tenant.id": state.get("tenant_id"), "cs.intent": RouteView.from_state(state).intent},
        ):
            return fn(state)

    wrapped.__name__ = getattr(fn, "__name__", name)
    return wrapped


def _atimed(name: str, fn):
    """async 节点：await 原生 async 节点，并打耗时直方图与 OTel 子 span。"""

    async def wrapped(state: CSState) -> CSState:
        with node_span(
            name,
            **{"tenant.id": state.get("tenant_id"), "cs.intent": RouteView.from_state(state).intent},
        ):
            return await fn(state)

    wrapped.__name__ = getattr(fn, "__name__", name)
    return wrapped


def build_graph(
    *,
    visualize: bool = True,
    visualize_png: bool = False,
    checkpointer=None,
    store=None,
):
    """编译客服主图，并导出可视化流程图。

    LLM 节点（router/rag/supervisor/respond）为原生 async，在 ``astream`` 路径下
    直接 await，不再经线程池转发；security/tool 为同步节点（无 LLM，开销小），
    chitchat/handoff 同步出口。混合 sync/async 节点由 LangGraph 自动调度。

    ``checkpointer`` / ``store`` 缺省时取进程 ``AppContext``，便于测试注入替身。
    """
    graph = StateGraph(CSState)
    graph.add_node("security", _timed("security", security_node))
    graph.add_node("router", _atimed("router", router_node))
    graph.add_node("rag", _atimed("rag", rag_node))
    graph.add_node("tool", _timed("tool", tool_node))
    graph.add_node("supervisor", _atimed("supervisor", supervisor_node))
    graph.add_node("chitchat", _timed("chitchat", chitchat_node))
    graph.add_node("handoff", _timed("handoff", handoff_node))
    graph.add_node("respond", _atimed("respond", respond_node))

    graph.add_edge(START, "security")
    graph.add_edge("security", "router")
    graph.add_conditional_edges(
        "router",
        _after_router,
        {
            "rag": "rag",
            "tool": "tool",
            "supervisor": "supervisor",
            "chitchat": "chitchat",
            "handoff": "handoff",
            "respond": "respond",
        },
    )
    graph.add_conditional_edges(
        "rag",
        _after_rag,
        {
            "supervisor": "supervisor",
            "respond": "respond",
        },
    )
    graph.add_edge("tool", "respond")
    graph.add_edge("supervisor", "respond")
    graph.add_edge("chitchat", "respond")
    graph.add_edge("handoff", "respond")
    graph.add_edge("respond", END)
    compiled = graph.compile(
        checkpointer=checkpointer if checkpointer is not None else get_checkpointer(),
        store=store if store is not None else _user_store(),
    )
    if visualize:
        export_graph_visual(compiled, png=visualize_png)
    return compiled


def _user_store():
    from app.memory import get_user_store

    return get_user_store()


def get_checkpointer():
    """短期记忆 checkpointer（进程 AppContext）。"""
    from app.runtime import get_app_context

    return get_app_context().checkpointer


def get_graph():
    """返回编译后的图（进程 AppContext，首次访问时编译）。"""
    from app.runtime import get_app_context

    return get_app_context().graph


def reset_graph(*, checkpointer=None, session_lock=None) -> None:
    """测试用：替换 checkpointer / 会话锁并丢掉已编译图。"""
    from app.runtime import replace_app_context

    if checkpointer is not None:
        replace_app_context(
            checkpointer=checkpointer,
            session_lock=session_lock or DictSessionLock(),
            invalidate_graph=True,
        )
        return
    saver = build_checkpointer(session_lock=session_lock)
    lock = session_lock
    if lock is None:
        from app.session_cache import get_session_lock

        lock = get_session_lock()
    replace_app_context(checkpointer=saver, session_lock=lock, invalidate_graph=True)
