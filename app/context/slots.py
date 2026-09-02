"""会话槽位：跨轮保留的轻量业务指针，以及追问时如何回填意图。

与窗口/摘要的分工：
- 窗口管「带哪些消息」；摘要管「裁掉的旧轮留下什么事实」。
- 槽位管「当前话题是哪张单 / 哪门课」，供短追问和 Router 补全，不存对话原文。

``handoff_pending`` 只表示已发起转接；演示环境没有真实工单状态机。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.context.window import SHORT_QUERY_CHARS
from app.schemas import RouteDecision, intent_to_target

# 这些意图结束后才写 last_intent；闲聊 / 模糊澄清不覆盖业务话题
BUSINESS_INTENTS = frozenset({"course_consult", "order", "mixed", "handoff"})
CONTINUE_HINTS = ("继续", "还有", "然后呢", "接着", "再问", "补充一下")
DEICTIC_HINTS = ("这", "那", "它", "该", "此")
ORDER_FOLLOWUP_HINTS = ("进度", "退款", "到哪", "怎么样", "何时", "什么时候", "开课")
COURSE_FOLLOWUP_HINTS = ("难", "适合", "多少钱", "内容", "大纲", "模块", "能学")
# RAG 空召回占位名：不能写进 last_course_name，否则下一轮会检索到假课名
MISSING_COURSE_NAMES = frozenset({"", "未找到课程"})


@dataclass(frozen=True)
class SessionSlots:
    """跨轮保留的轻量槽位；不随本轮瞬时字段（answer / structured）清空。"""

    last_intent: str | None = None
    last_order_no: str | None = None
    last_course_query: str | None = None
    last_course_name: str | None = None
    handoff_pending: bool = False

    @classmethod
    def from_state(cls, state: dict) -> SessionSlots:
        """从图状态读取槽位。"""
        from app.graph.state import MemoryView

        memory = MemoryView.from_state(state)
        return cls(
            last_intent=memory.last_intent,
            last_order_no=memory.last_order_no,
            last_course_query=memory.last_course_query,
            last_course_name=memory.last_course_name,
            handoff_pending=memory.handoff_pending,
        )


def is_followup(text: str) -> bool:
    """是否像在追问上文，而不是开启全新话题。"""
    query = (text or "").strip()
    if not query:
        return False
    if any(token in query for token in CONTINUE_HINTS):
        return True
    if len(query) <= 20 and any(token in query for token in DEICTIC_HINTS):
        return True
    if len(query) <= 10 and any(token in query for token in ORDER_FOLLOWUP_HINTS):
        return True
    if len(query) <= 10 and any(token in query for token in COURSE_FOLLOWUP_HINTS):
        return True
    return False


def apply_session_slots(text: str, decision: RouteDecision, slots: SessionSlots) -> RouteDecision:
    """用会话槽位补全追问的意图、订单号与课程检索词。

    只在识别结果是闲聊/模糊、且当前句像追问时，才把 last_intent 接回来。
    明确的新话题（例如从查单改口问另一门课）不会被槽位覆盖。
    """
    intent = decision.intent
    order_no = decision.order_no
    course_query = decision.course_query
    reason = decision.reason
    confidence = decision.confidence
    needs_clarification = decision.needs_clarification
    clarification_question = decision.clarification_question

    if intent in {"chitchat", "ambiguous"} and is_followup(text) and slots.last_intent in BUSINESS_INTENTS:
        intent = slots.last_intent  # type: ignore[assignment]
        reason = "slot-followup"
        confidence = max(confidence, 0.85)
        needs_clarification = False
        clarification_question = None
        if intent in {"course_consult", "mixed"} and not course_query:
            course_query = slots.last_course_name or slots.last_course_query

    if intent in {"order", "mixed"} and not order_no:
        order_no = slots.last_order_no

    if intent in {"course_consult", "mixed"}:
        short = len((text or "").strip()) < SHORT_QUERY_CHARS
        if short or not course_query:
            course_query = course_query or slots.last_course_name or slots.last_course_query or text

    return RouteDecision(
        intent=intent,  # type: ignore[arg-type]
        target=intent_to_target(intent),
        confidence=confidence,
        order_no=order_no,
        course_query=course_query,
        needs_clarification=needs_clarification,
        clarification_question=clarification_question,
        reason=reason,
    )


def slot_updates_after_turn(
    intent: str,
    *,
    order_no: str | None = None,
    course_query: str | None = None,
    course_name: str | None = None,
) -> dict[str, Any]:
    """业务轮结束后更新跨轮槽位；未识别到的字段保持原值（不写回 None）。"""
    updates: dict[str, Any] = {"handoff_pending": intent == "handoff"}
    if intent in BUSINESS_INTENTS:
        updates["last_intent"] = intent
    if order_no:
        updates["last_order_no"] = order_no
    name = (course_name or "").strip()
    if course_query:
        updates["last_course_query"] = course_query
    if name and name not in MISSING_COURSE_NAMES:
        updates["last_course_name"] = name
    return updates
