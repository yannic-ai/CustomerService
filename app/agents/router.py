"""Router Agent：调用意图识别，供主图决定下游节点。"""

from app.agents.intent import recognize_intent
from app.schemas import RouteDecision


async def route_query(text: str) -> RouteDecision:
    """Router Agent 入口，将用户问题交给意图识别模块（异步）。"""
    return await recognize_intent(text)
