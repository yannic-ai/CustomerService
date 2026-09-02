"""本轮路由：意图、图分支、安全拦截、本轮订单号/课程检索词。"""

from __future__ import annotations

from typing import Any, TypedDict


class RouteState(TypedDict, total=False):
    """只活在本轮的分流字段。新增路由/拦截字段写这里。"""

    intent: str  # course_consult / order / mixed / ambiguous / chitchat / handoff
    target: str  # rag / tool / handoff / chitchat / supervisor
    confidence: float  # 路由置信度 0~1
    needs_clarification: bool
    clarification_question: str | None
    order_no: str | None  # 本轮识别出的订单号
    course_query: str | None  # 本轮用于 RAG 的课程检索 query
    blocked: bool
    block_reason: str
    rag_upgraded_to_supervisor: bool  # RAG 空召回且问句含订单信号，升到 ReACT


# checkpoint 会带回上一轮瞬时字段；入口必须清空，避免旧报告/旧意图污染本轮
ROUTE_TURN_RESET: dict[str, Any] = {
    "blocked": False,
    "block_reason": "",
    "intent": "",
    "target": "",
    "confidence": 0.0,
    "needs_clarification": False,
    "clarification_question": None,
    "order_no": None,
    "course_query": None,
    "rag_upgraded_to_supervisor": False,
}
