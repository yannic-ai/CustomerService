import asyncio

from app.agents.intent import apply_low_confidence_guard, apply_order_signal_guard, recognize_intent
from app.context import SessionSlots, apply_session_slots, is_followup
from app.schemas import RouteDecision


def _recognize(text, **kwargs):
    return asyncio.run(recognize_intent(text, **kwargs))


def test_keyword_order_query() -> None:
    decision = _recognize("查询订单#20251114001的退款进度")
    assert decision.intent == "order"
    assert decision.target == "tool"
    assert decision.order_no == "20251114001"
    assert decision.confidence >= 0.8
    assert decision.reason == "keyword-intent"


def test_keyword_course_consult() -> None:
    decision = _recognize("Python入门课包含哪些内容？")
    assert decision.intent == "course_consult"
    assert decision.target == "rag"
    assert decision.reason == "keyword-intent"


def test_keyword_presale_course() -> None:
    decision = _recognize("新手能学吗？")
    assert decision.intent == "course_consult"
    assert decision.reason == "keyword-intent"


def test_keyword_order_opencourse() -> None:
    decision = _recognize("查询订单 20251114001 什么时候开课？")
    assert decision.intent == "order"
    assert decision.target == "tool"
    assert decision.order_no == "20251114001"


def test_keyword_handoff() -> None:
    decision = _recognize("转人工")
    assert decision.intent == "handoff"
    assert decision.target == "handoff"
    assert decision.reason == "keyword-intent"


def test_keyword_handoff_priority_over_order() -> None:
    decision = _recognize("转人工查一下订单退款")
    assert decision.intent == "handoff"


def test_keyword_mixed() -> None:
    decision = _recognize("订单20251114001买的入门课包含哪些内容？")
    assert decision.intent == "mixed"
    assert decision.target == "supervisor"
    assert decision.order_no == "20251114001"
    assert decision.reason == "keyword-mixed"
    assert decision.confidence < 0.8


def test_order_without_order_no_becomes_ambiguous() -> None:
    decision = _recognize("退款进度怎么样")
    assert decision.intent == "ambiguous"
    assert decision.target == "supervisor"
    assert decision.needs_clarification is True
    assert decision.order_no is None
    assert decision.reason == "missing-order-no"


def test_order_without_order_no_but_slot_reuses_order() -> None:
    slots = SessionSlots(last_intent="order", last_order_no="20251114001")
    decision = _recognize("退款进度怎么样", slots=slots)
    assert decision.intent == "order"
    assert decision.target == "tool"
    assert decision.order_no == "20251114001"


def test_short_query_becomes_ambiguous_without_context() -> None:
    decision = _recognize("这个呢")
    assert decision.intent == "ambiguous"
    assert decision.target == "supervisor"
    assert decision.reason == "short-query"


def test_short_query_with_course_context_reuses_course() -> None:
    slots = SessionSlots(
        last_intent="course_consult",
        last_course_name="Python入门课",
        last_course_query="Python入门课包含哪些内容？",
    )
    decision = _recognize("这个呢", slots=slots)
    assert decision.intent == "course_consult"
    assert decision.target == "rag"
    assert decision.course_query
    assert "Python" in decision.course_query


def test_greeting_plus_business_word_becomes_ambiguous() -> None:
    decision = _recognize("你好 退款")
    assert decision.intent == "ambiguous"
    assert decision.target == "supervisor"
    assert decision.reason == "ambiguous-query"


def test_slot_fills_order_no_on_followup() -> None:
    slots = SessionSlots(last_intent="order", last_order_no="20251114001")
    decision = _recognize("进度怎么样", slots=slots)
    assert decision.intent == "order"
    assert decision.order_no == "20251114001"


def test_slot_continue_reuses_last_intent() -> None:
    slots = SessionSlots(last_intent="order", last_order_no="20251114001")
    decision = _recognize("继续", slots=slots)
    assert decision.intent == "order"
    assert decision.order_no == "20251114001"
    assert decision.reason == "slot-followup"


def test_slot_course_deictic_followup() -> None:
    slots = SessionSlots(
        last_intent="course_consult",
        last_course_name="Python入门课",
        last_course_query="Python入门课包含哪些内容？",
    )
    decision = _recognize("这个怎么样", slots=slots)
    assert decision.intent == "course_consult"
    assert decision.course_query
    assert "Python" in decision.course_query


def test_apply_slots_does_not_override_explicit_new_topic() -> None:
    slots = SessionSlots(last_intent="order", last_order_no="20251114001")
    base = RouteDecision(intent="course_consult", course_query="机器学习课大纲", reason="keyword-intent")
    decision = apply_session_slots("机器学习课大纲", base, slots)
    assert decision.intent == "course_consult"
    assert decision.order_no is None


def test_is_followup_continue() -> None:
    assert is_followup("继续")
    assert is_followup("那退款呢")
    assert not is_followup("Python入门课包含哪些内容？")


def test_order_signal_guard_upgrades_course_consult_to_mixed() -> None:
    base = RouteDecision(
        intent="course_consult",
        target="rag",
        course_query="查询订单退款进度",
        reason="llm-intent",
    )
    decision = apply_order_signal_guard("查询订单#20251114001的退款进度", base)
    assert decision.intent == "mixed"
    assert decision.target == "supervisor"
    assert decision.reason == "order-signal-force"
    assert decision.order_no == "20251114001"


def test_order_signal_guard_upgrades_chitchat_to_order() -> None:
    base = RouteDecision(intent="chitchat", target="chitchat", reason="llm-intent")
    decision = apply_order_signal_guard("20251114001 退款到哪了", base)
    assert decision.intent == "order"
    assert decision.target == "tool"
    assert decision.reason == "order-signal-force"
    assert decision.order_no == "20251114001"


def test_order_signal_guard_ignores_slot_only_order_no() -> None:
    base = RouteDecision(
        intent="course_consult",
        target="rag",
        order_no="20251114001",
        course_query="机器学习课大纲",
        reason="keyword-intent",
    )
    decision = apply_order_signal_guard("机器学习课大纲", base)
    assert decision.intent == "course_consult"
    assert decision.target == "rag"


def test_order_signal_guard_does_not_override_handoff() -> None:
    base = RouteDecision(intent="handoff", target="handoff", reason="keyword-intent")
    decision = apply_order_signal_guard("转人工查一下订单退款", base)
    assert decision.intent == "handoff"


def test_misrouted_keyword_course_is_forced_to_mixed(monkeypatch) -> None:
    """关键词误判成课程时，本轮订单信号仍强制改走 ReACT。"""
    from app.agents import intent as intent_mod

    def fake_keyword(text: str) -> RouteDecision:
        return intent_mod._decision(
            intent="course_consult",
            reason="fake-misroute",
            confidence=0.9,
            course_query=text,
        )

    monkeypatch.setattr(intent_mod, "_recognize_by_keyword", fake_keyword)
    decision = _recognize("查询订单#20251114001的退款进度")
    assert decision.intent == "mixed"
    assert decision.target == "supervisor"
    assert decision.reason == "order-signal-force"
    assert decision.order_no == "20251114001"


def test_low_confidence_course_consult_goes_supervisor() -> None:
    base = RouteDecision(
        intent="course_consult",
        target="rag",
        confidence=0.4,
        course_query="这个课怎么样",
        reason="llm-intent",
    )
    decision = apply_low_confidence_guard(base)
    assert decision.intent == "mixed"
    assert decision.target == "supervisor"
    assert decision.reason == "low-confidence-upgrade"
    assert decision.confidence == 0.4


def test_high_confidence_course_consult_stays_rag() -> None:
    base = RouteDecision(
        intent="course_consult",
        target="rag",
        confidence=0.9,
        course_query="Python入门课包含哪些内容？",
        reason="keyword-intent",
    )
    decision = apply_low_confidence_guard(base)
    assert decision.intent == "course_consult"
    assert decision.target == "rag"


def test_keyword_course_still_direct_rag() -> None:
    decision = _recognize("Python入门课包含哪些内容？")
    assert decision.intent == "course_consult"
    assert decision.target == "rag"
    assert decision.confidence >= 0.9


def test_arecognize_intent_keyword_path() -> None:
    """异步孪生在关键词分支应与同步版一致，不触达 LLM。"""
    import asyncio

    from app.agents.intent import arecognize_intent

    decision = asyncio.run(arecognize_intent("转人工"))
    assert decision.intent == "handoff"
    assert decision.target == "handoff"
    assert decision.reason == "keyword-intent"


def test_arecognize_intent_uses_llm_ainvoke(monkeypatch) -> None:
    """chitchat 分支应 await llm.ainvoke（不进线程池）。"""
    import asyncio

    from app.agents import intent as intent_mod
    from app.agents.intent import arecognize_intent
    from app.config import get_settings
    from app.schemas import RouteDecision

    real = get_settings()
    monkeypatch.setattr(
        intent_mod,
        "get_settings",
        lambda: real.model_copy(update={"deepseek_api_key": "test-key"}),
    )

    class _Structured:
        def __init__(self, decision):
            self._d = decision

        async def ainvoke(self, messages, config=None, **kwargs):
            del messages, config, kwargs
            return self._d

    class _ChatModel:
        def __init__(self, decision):
            self._d = decision

        def with_structured_output(self, schema, **kwargs):
            del schema, kwargs
            return _Structured(self._d)

    expected = RouteDecision(
        intent="chitchat", target="chitchat", confidence=0.8, reason="llm-intent"
    )
    monkeypatch.setattr("app.llm.get_chat_model", lambda **kwargs: _ChatModel(expected))
    decision = asyncio.run(arecognize_intent("今天天气真好"))
    assert decision.intent == "chitchat"
    assert decision.reason == "llm-intent"
