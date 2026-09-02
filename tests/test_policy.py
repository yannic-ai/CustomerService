import asyncio

from app.agents.policy import DEFAULT_POLICY, RoutingPolicy
from app.agents.supervisor import run_supervisor
from app.schemas import RouteDecision


def test_next_after_router_blocked_goes_respond() -> None:
    assert DEFAULT_POLICY.next_after_router(blocked=True, target="rag") == "respond"


def test_next_after_router_maps_intent_fallback() -> None:
    assert DEFAULT_POLICY.next_after_router(intent="order", confidence=0.9) == "tool"
    assert DEFAULT_POLICY.next_after_router(intent="mixed") == "supervisor"
    assert DEFAULT_POLICY.next_after_router(target="handoff") == "handoff"


def test_next_after_router_low_confidence_rag_goes_supervisor() -> None:
    assert (
        DEFAULT_POLICY.next_after_router(target="rag", confidence=0.4) == "supervisor"
    )
    assert DEFAULT_POLICY.next_after_router(target="rag", confidence=0.9) == "rag"


def test_next_after_rag_upgrade_flag() -> None:
    assert DEFAULT_POLICY.next_after_rag(upgraded=True) == "supervisor"
    assert DEFAULT_POLICY.next_after_rag(target="supervisor") == "supervisor"
    assert DEFAULT_POLICY.next_after_rag() == "respond"


def test_decide_rag_upgrade_on_order_signal() -> None:
    decision = DEFAULT_POLICY.decide_rag_upgrade(
        "查询订单#20251114001的退款进度",
        course_query="查询订单#20251114001的退款进度",
    )
    assert decision is not None
    assert decision.reason == "order_signal"
    assert decision.order_no == "20251114001"
    updates = DEFAULT_POLICY.rag_upgrade_updates(decision)
    assert updates["target"] == "supervisor"
    assert updates["intent"] == "mixed"
    assert updates["rag_upgraded_to_supervisor"] is True


def test_decide_rag_upgrade_ignores_pure_course() -> None:
    assert DEFAULT_POLICY.decide_rag_upgrade("Python入门课包含哪些内容？") is None


def test_plan_supervisor_clarifies_when_asked() -> None:
    plan = DEFAULT_POLICY.plan_supervisor(
        intent="order",
        needs_clarification=True,
        clarification_question="请提供完整订单编号",
    )
    assert plan.action == "ask_user"
    assert plan.final_intent == "ambiguous"


def test_plan_supervisor_react_when_mixed_unbound() -> None:
    plan = DEFAULT_POLICY.plan_supervisor(intent="mixed", query="那个怎么样了")
    assert plan.action == "react"
    assert plan.final_intent == "mixed"
    assert plan.reason == "unbound-fallback"


def test_plan_supervisor_serial_mixed_with_order_no() -> None:
    plan = DEFAULT_POLICY.plan_supervisor(
        intent="mixed",
        query="订单#20251114001 退款进度和这门课",
        order_no="20251114001",
        course_query="订单#20251114001 退款进度和这门课",
    )
    assert plan.action == "execute"
    assert plan.reason == "serial-order-then-course"
    assert "tool_order_query" in plan.tools_to_call
    assert "retrieve_course_knowledge" in plan.tools_to_call


def test_run_supervisor_ask_user_does_not_call_react(monkeypatch) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("澄清路径不应调用 ReACT")

    monkeypatch.setattr("app.agents.supervisor.stream_react", boom)
    decision = asyncio.run(run_supervisor(
        "退款进度怎么样",
        needs_clarification=True,
        clarification_question="请提供完整订单编号",
        intent="order",
    ))
    assert decision.action == "ask_user"
    assert decision.answer == "请提供完整订单编号"


def test_custom_policy_threshold_changes_graph_edge() -> None:
    strict = RoutingPolicy(confidence_rag_min=0.95)
    assert strict.next_after_router(target="rag", confidence=0.9) == "supervisor"
    assert DEFAULT_POLICY.next_after_router(target="rag", confidence=0.9) == "rag"


def test_finalize_is_single_pipeline() -> None:
    base = RouteDecision(
        intent="course_consult",
        target="rag",
        confidence=0.9,
        course_query="查询订单退款进度",
        reason="llm-intent",
    )
    decision = DEFAULT_POLICY.finalize("查询订单#20251114001的退款进度", base)
    assert decision.intent == "mixed"
    assert decision.target == "supervisor"
    assert decision.reason == "order-signal-force"
