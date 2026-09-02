"""Supervisor 能力编排：计划规则与执行路径。"""

from __future__ import annotations

import asyncio

from app.agents.policy import DEFAULT_POLICY
from app.agents.supervisor import run_supervisor
from app.schemas import CourseConsultResult, OrderReport


def test_plan_supervisor_clarifies_when_asked() -> None:
    plan = DEFAULT_POLICY.plan_supervisor(
        intent="order",
        needs_clarification=True,
        clarification_question="请提供完整订单编号",
    )
    assert plan.action == "ask_user"
    assert plan.final_intent == "ambiguous"
    assert plan.tools_to_call == ()


def test_plan_parallel_when_order_and_course_bound() -> None:
    plan = DEFAULT_POLICY.plan_supervisor(
        intent="mixed",
        query="订单#20251114001 的 Python入门课大纲是什么",
        order_no="20251114001",
        course_query="Python入门课",
    )
    assert plan.action == "execute"
    assert plan.reason == "parallel-bound"
    assert plan.parallel is True
    assert plan.tools_to_call == ("tool_order_query", "retrieve_course_knowledge")
    assert {s.parallel_group for s in plan.steps} == {0}


def test_plan_serial_when_course_only_in_order_product() -> None:
    query = "订单#20251114001 这门课讲什么，退款进度呢"
    plan = DEFAULT_POLICY.plan_supervisor(
        intent="mixed",
        query=query,
        order_no="20251114001",
        course_query=query,  # mixed 整句伪绑定，不算已有课名
    )
    assert plan.action == "execute"
    assert plan.reason == "serial-order-then-course"
    assert plan.parallel is False
    assert plan.steps[0].capability == "order_query"
    assert plan.steps[0].parallel_group == 0
    assert plan.steps[1].capability == "course_retrieve"
    assert plan.steps[1].parallel_group == 1
    assert plan.steps[1].course_source == "from_order_product"


def test_plan_react_when_unbound() -> None:
    plan = DEFAULT_POLICY.plan_supervisor(
        intent="mixed",
        query="那个怎么样了",
    )
    assert plan.action == "react"
    assert plan.reason == "unbound-fallback"
    assert plan.tools_to_call == ()


def test_run_supervisor_ask_user_skips_tools_and_react(monkeypatch) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("澄清路径不应调用工具或 ReACT")

    monkeypatch.setattr("app.agents.supervisor.stream_react", boom)
    monkeypatch.setattr("app.capabilities.order.query_order", boom)
    monkeypatch.setattr("app.capabilities.course.retrieve_course", boom)
    decision = asyncio.run(run_supervisor(
        "退款进度怎么样",
        needs_clarification=True,
        clarification_question="请提供完整订单编号",
        intent="order",
    ))
    assert decision.action == "ask_user"
    assert decision.answer == "请提供完整订单编号"


def test_run_supervisor_parallel_calls_both_tools(monkeypatch) -> None:
    calls: list[str] = []

    def fake_order(query, order_no=None, **kwargs):
        del query
        calls.append("order")
        return OrderReport(
            order_no=order_no or kwargs.get("order_no") or "20251114001",
            product_name="Python入门课",
            order_type="course",
            current_status="paid",
            progress_percent=80,
            summary="订单进度正常",
            visualization="进度图",
        )

    async def fake_course(query, **kwargs):
        del kwargs
        calls.append(f"course:{query}")
        return CourseConsultResult(
            course_name="Python入门课",
            summary="适合零基础。",
            modules=[],
        )

    monkeypatch.setattr("app.capabilities.order.query_order", fake_order)
    monkeypatch.setattr("app.capabilities.course.retrieve_course", fake_course)
    monkeypatch.setattr(
        "app.agents.supervisor.stream_react",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("不应 ReACT")),
    )

    decision = asyncio.run(run_supervisor(
        "订单#20251114001 的 Python入门课大纲是什么",
        intent="mixed",
        order_no="20251114001",
        course_query="Python入门课",
        user_id="u10001",
        tenant_id="demo",
    ))
    assert decision.action == "execute"
    assert decision.parallel is True
    assert "order" in calls
    assert any(c.startswith("course:") for c in calls)
    assert "订单进度正常" in (decision.answer or "")
    assert "适合零基础" in (decision.answer or "")


def test_run_supervisor_serial_uses_product_name(monkeypatch) -> None:
    course_queries: list[str] = []

    def fake_order(query, order_no=None, **kwargs):
        del query, order_no, kwargs
        return OrderReport(
            order_no="20251114001",
            product_name="Python入门课",
            course_code="python-intro",
            order_type="course",
            current_status="paid",
            progress_percent=50,
            summary="退款处理中",
            visualization="进度图",
        )

    course_codes: list[str | None] = []

    async def fake_course(query, **kwargs):
        course_queries.append(query)
        course_codes.append(kwargs.get("course_code"))
        return CourseConsultResult(
            course_name="Python入门课",
            summary="模块含基础语法。",
            modules=[],
        )

    monkeypatch.setattr("app.capabilities.order.query_order", fake_order)
    monkeypatch.setattr("app.capabilities.course.retrieve_course", fake_course)
    monkeypatch.setattr(
        "app.agents.supervisor.stream_react",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("不应 ReACT")),
    )

    query = "订单#20251114001 这门课讲什么，退款进度呢"
    decision = asyncio.run(run_supervisor(
        query,
        intent="mixed",
        order_no="20251114001",
        course_query=query,
        user_id="u10001",
        tenant_id="demo",
    ))
    assert decision.action == "execute"
    assert decision.reason == "serial-order-then-course"
    assert decision.parallel is False
    assert course_queries == ["Python入门课"]
    assert course_codes == ["python-intro"]
    assert "退款处理中" in (decision.answer or "")
    assert "模块含基础语法" in (decision.answer or "")


def test_run_supervisor_serial_skips_course_when_order_fails(monkeypatch) -> None:
    def fake_order(*_a, **_k):
        raise ValueError("请提供完整订单编号")

    async def boom_course(*_a, **_k):
        raise AssertionError("查单失败不应盲检索")

    monkeypatch.setattr("app.capabilities.order.query_order", fake_order)
    monkeypatch.setattr("app.capabilities.course.retrieve_course", boom_course)
    monkeypatch.setattr(
        "app.agents.supervisor.stream_react",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("不应 ReACT")),
    )

    query = "订单#999 这门课讲什么"
    decision = asyncio.run(run_supervisor(
        query,
        intent="mixed",
        order_no="999",
        course_query=query,
    ))
    assert decision.action == "execute"
    assert "请提供完整订单编号" in (decision.answer or "")


def test_run_supervisor_react_fallback(monkeypatch) -> None:
    called = {"react": False}

    async def fake_react(*_a, **_k):
        called["react"] = True
        return "ReACT 兜底答案"

    monkeypatch.setattr("app.agents.supervisor.stream_react", fake_react)
    monkeypatch.setattr(
        "app.capabilities.order.query_order",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("不应直接查单")),
    )

    decision = asyncio.run(run_supervisor("那个怎么样了", intent="mixed"))
    assert called["react"] is True
    assert decision.action == "react"
    assert decision.answer == "ReACT 兜底答案"
    assert decision.reason == "unbound-fallback"
