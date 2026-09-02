"""Agent 集合：意图识别、路由、RAG、订单工具、ReACT 与 Supervisor。"""

from app.agents.intent import recognize_intent
from app.agents.policy import DEFAULT_POLICY, RoutingPolicy
from app.agents.rag import consult_course, format_course_answer, stream_course_answer
from app.agents.react import run_react, stream_react
from app.agents.router import route_query
from app.agents.supervisor import run_supervisor
from app.agents.tool import format_order_answer, handle_order_query

__all__ = [
    "recognize_intent",
    "RoutingPolicy",
    "DEFAULT_POLICY",
    "route_query",
    "consult_course",
    "format_course_answer",
    "stream_course_answer",
    "handle_order_query",
    "format_order_answer",
    "run_react",
    "stream_react",
    "run_supervisor",
]
