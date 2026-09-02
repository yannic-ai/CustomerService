"""意图识别：将用户问题分类为课程咨询、订单查询、混合、模糊、转人工或闲聊。"""

from __future__ import annotations

from langchain_core.messages import BaseMessage

from app.agents.policy import (
    CONFIDENCE_RAG_MIN,
    DEFAULT_POLICY,
    apply_low_confidence_guard,
    apply_order_signal_guard,
    extract_query_order_no,
    finalize_route,
    query_has_order_signal,
)
from app.config import get_settings
from app.context import SessionSlots, apply_session_slots, format_history_for_prompt, order_no_from_history
from app.observability.langfuse import PROMPT_INTENT, get_text_prompt
from app.schemas import Intent, RouteDecision, intent_to_target

HANDOFF_HINTS = ("转人工", "人工客服", "找人工", "人工服务", "联系人工")
COURSE_HINTS = (
    "课程",
    "大纲",
    "包含",
    "学什么",
    "模块",
    "学习路径",
    "入门课",
    "讲什么",
    "新手",
    "能学",
    "适合",
    "零基础",
    "难吗",
    "售前",
    "售后",
)
ORDER_HINTS = ("订单", "退款", "物流", "进度", "支付", "开通")
GREETING_HINTS = ("你好", "您好", "在吗", "嗨", "hello", "hi", "早上好", "晚上好")
PURE_CHITCHAT = frozenset({"你好", "您好", "在吗", "嗨", "hello", "hi", "早上好", "晚上好", "谢谢", "感谢", "好的", "嗯"})

INTENT_SYSTEM_PROMPT = (
    "你是智能客服意图识别器。只做意图分类："
    "course_consult=课程咨询/大纲/学习路径/售前适学；"
    "order=订单、退款、支付进度；"
    "mixed=同时涉及课程和订单；"
    "ambiguous=信息不足、缺订单号、过短追问或问候混业务词，需要澄清；"
    "handoff=转人工/人工客服；"
    "chitchat=闲聊或其他。"
    "主要依据【当前用户问题】分类；对话历史仅用于理解指代与补全订单号。"
    "若出现订单号请提取到 order_no。"
)

VALID_INTENTS = ("course_consult", "order", "mixed", "ambiguous", "chitchat", "handoff")
SHORT_QUERY_CHARS = 4
CONFIDENCE_DIRECT = 0.9
CONFIDENCE_MIXED = 0.65
CONFIDENCE_AMBIGUOUS = 0.35
CONFIDENCE_HANDOFF = 0.99
CONFIDENCE_CHITCHAT = 0.8
# LLM 未给出置信度时，课程意图按低置信处理，避免默认 0.9 直达 RAG
CONFIDENCE_UNSET_COURSE = 0.5

GENERIC_CLARIFY = "您想咨询课程内容，还是查询订单进度？请补充一下具体问题。"


def _has_course(text: str) -> bool:
    return any(token in text for token in COURSE_HINTS)


def _has_order_keywords(text: str) -> bool:
    return any(token in text for token in ORDER_HINTS)


def _extract_order_no(text: str) -> str | None:
    return extract_query_order_no(text)


def _is_greeting_mixed(text: str, *, has_course: bool, has_order: bool) -> bool:
    if not (has_course or has_order):
        return False
    lowered = text.lower()
    return any(token in lowered for token in GREETING_HINTS)


def _is_pure_chitchat(text: str) -> bool:
    return text.strip().lower() in PURE_CHITCHAT


def _enough_context(slots: SessionSlots) -> bool:
    if slots.last_intent in {"course_consult", "order", "mixed", "handoff"}:
        return True
    if slots.last_order_no or slots.last_course_query or slots.last_course_name:
        return True
    return False


def _decision(
    *,
    intent: Intent,
    reason: str,
    confidence: float,
    order_no: str | None = None,
    course_query: str | None = None,
    needs_clarification: bool = False,
    clarification_question: str | None = None,
) -> RouteDecision:
    return finalize_route(
        RouteDecision(
            intent=intent,
            target=intent_to_target(intent),
            confidence=confidence,
            order_no=order_no,
            course_query=course_query,
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
            reason=reason,
        )
    )


def _recognize_by_keyword(text: str) -> RouteDecision:
    """无大模型时的关键词意图识别，同时尝试提取订单号。"""
    query = (text or "").strip()
    if any(token in query for token in HANDOFF_HINTS):
        return _decision(intent="handoff", reason="keyword-intent", confidence=CONFIDENCE_HANDOFF)

    order_no = _extract_order_no(query)
    has_course = _has_course(query)
    has_order = _has_order_keywords(query) or bool(order_no)

    if has_course and has_order:
        return _decision(
            intent="mixed",
            reason="keyword-mixed",
            confidence=CONFIDENCE_MIXED,
            order_no=order_no,
            course_query=query,
        )

    if _is_greeting_mixed(query, has_course=has_course, has_order=has_order):
        return _decision(
            intent="ambiguous",
            reason="ambiguous-query",
            confidence=CONFIDENCE_AMBIGUOUS,
            order_no=order_no,
            course_query=query if has_course else None,
            needs_clarification=True,
            clarification_question=GENERIC_CLARIFY,
        )

    if has_order:
        return _decision(
            intent="order",
            reason="keyword-intent",
            confidence=CONFIDENCE_DIRECT,
            order_no=order_no,
        )

    if has_course:
        return _decision(
            intent="course_consult",
            reason="keyword-intent",
            confidence=CONFIDENCE_DIRECT,
            course_query=query,
        )

    if len(query) < SHORT_QUERY_CHARS and not _is_pure_chitchat(query):
        return _decision(
            intent="ambiguous",
            reason="short-query",
            confidence=0.3,
            needs_clarification=True,
            clarification_question=GENERIC_CLARIFY,
        )

    return _decision(
        intent="chitchat",
        reason="keyword-intent",
        confidence=CONFIDENCE_CHITCHAT,
    )


def _finalize_decision(text: str, decision: RouteDecision) -> RouteDecision:
    """识别结果交给 RoutingPolicy：订单信号、低置信、缺单号。"""
    return DEFAULT_POLICY.finalize(text, decision)


async def _recognize_by_llm(
    text: str,
    history: list[BaseMessage] | None = None,
    session_summary: str | None = None,
) -> RouteDecision | None:
    """用 DeepSeek 做结构化意图路由（异步）；失败时返回 None。"""
    from app.llm import get_chat_model

    history_text = format_history_for_prompt(history, session_summary=session_summary)
    user_content = text
    if history_text:
        user_content = f"对话历史：\n{history_text}\n\n当前用户问题：{text}"

    try:
        llm = get_chat_model(temperature=0, usage_tag="intent", cache_enabled=True).with_structured_output(
            RouteDecision
        )
        decision = await llm.ainvoke(
            [
                {"role": "system", "content": get_text_prompt(PROMPT_INTENT, INTENT_SYSTEM_PROMPT)},
                {"role": "user", "content": user_content},
            ]
        )
    except Exception as exc:
        from app.llm_gateway import TenantQuotaExceededError

        if isinstance(exc, TenantQuotaExceededError):
            raise
        return None
    if not isinstance(decision, RouteDecision):
        return None
    if decision.intent not in VALID_INTENTS:
        return None
    decision.reason = decision.reason or "llm-intent"
    if not decision.target or decision.target == "chitchat" and decision.intent != "chitchat":
        decision = decision.model_copy(update={"target": intent_to_target(decision.intent)})
    if decision.confidence <= 0:
        fallback = (
            CONFIDENCE_UNSET_COURSE
            if decision.intent == "course_consult"
            else CONFIDENCE_DIRECT
        )
        decision = decision.model_copy(update={"confidence": fallback})
    return finalize_route(decision)


async def arecognize_intent(
    text: str,
    history: list[BaseMessage] | None = None,
    slots: SessionSlots | None = None,
    session_summary: str | None = None,
) -> RouteDecision:
    """先用关键字识别本轮问句；槽位/历史回填订单号；闲聊时再走 LLM（await ainvoke）。

    出口统一做订单信号强制分流：当前问句含订单号/退款等时，禁止把请求留在纯 RAG 或闲聊。

    注意：本模块已全异步化，对外标准名是 :func:`recognize_intent`；
    `arecognize_intent` 作为别名保留以兼容旧导入。
    """
    return await recognize_intent(
        text, history=history, slots=slots, session_summary=session_summary
    )


async def recognize_intent(
    text: str,
    history: list[BaseMessage] | None = None,
    slots: SessionSlots | None = None,
    session_summary: str | None = None,
) -> RouteDecision:
    """先用关键字识别本轮问句；槽位/历史回填订单号；闲聊时再走 LLM（await ainvoke）。

    出口统一做订单信号强制分流：当前问句含订单号/退款等时，禁止把请求留在纯 RAG 或闲聊。
    """
    slots = slots or SessionSlots()
    keyword = _recognize_by_keyword(text)
    if not keyword.order_no:
        keyword = keyword.model_copy(
            update={"order_no": slots.last_order_no or order_no_from_history(history)}
        )

    if keyword.intent not in {"chitchat", "ambiguous"}:
        decision = apply_session_slots(text, keyword, slots)
        return _finalize_decision(text, decision)

    # 短问/模糊问若槽位足够，优先走槽位承接，避免无谓进 supervisor
    if keyword.intent == "ambiguous" and _enough_context(slots):
        decision = apply_session_slots(
            text,
            keyword.model_copy(update={"intent": "chitchat", "reason": keyword.reason}),
            slots,
        )
        if decision.intent != "chitchat":
            return _finalize_decision(text, decision)

    settings = get_settings()
    if not settings.llm_enabled:
        decision = apply_session_slots(text, keyword, slots)
        if decision.intent == "chitchat" and keyword.intent == "ambiguous":
            decision = keyword
        return _finalize_decision(text, decision)

    if keyword.intent == "ambiguous":
        # 规则已判定模糊：不必再花一次 LLM；槽位承接失败则直接澄清
        decision = apply_session_slots(text, keyword, slots)
        if decision.intent == "ambiguous" or (
            decision.intent == "order" and not decision.order_no
        ):
            return _finalize_decision(
                text, keyword if decision.intent == "ambiguous" else decision
            )
        return _finalize_decision(text, decision)

    decision = await _recognize_by_llm(text, history=history, session_summary=session_summary)
    if decision is None:
        decision = apply_session_slots(text, keyword, slots)
        return _finalize_decision(text, decision)
    if not decision.order_no:
        decision = decision.model_copy(update={"order_no": keyword.order_no})
    decision = apply_session_slots(text, decision, slots)
    return _finalize_decision(text, decision)
