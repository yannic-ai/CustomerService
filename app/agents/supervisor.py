"""Supervisor Agent：按计划调用能力注册表（计划 + 执行）。

决定「调谁、并行还是串行、何时 ReACT」：
- ``ask_user``：澄清，不调能力
- ``execute``：按 ``PlanStep.capability`` 查注册表执行（可并行或串行）
- ``react``：绑不上或依赖不明时才走 ReACT 兜底

新增能力只需 ``register`` 并在 Policy 里加 PlanStep；本模块执行循环不用改。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from langchain_core.messages import BaseMessage

from app.agents.policy import DEFAULT_POLICY, PlanStep, SupervisorPlan, to_supervisor_decision
from app.agents.react import stream_react
from app.capabilities import ExecutionContext, get_capability
from app.schemas import SupervisorDecision
from app.tool_escalation import build_handoff_answer


def _is_auto_handoff_answer(answer: str | None) -> bool:
    text = (answer or "").strip()
    return "转接人工客服" in text and "演示环境" in text


async def run_supervisor(
    query: str,
    *,
    user_id: str = "anonymous",
    tenant_id: str = "demo",
    history: list[BaseMessage] | None = None,
    order_no: str | None = None,
    course_query: str | None = None,
    last_course_name: str | None = None,
    session_summary: str | None = None,
    session_cache_state: dict[str, Any] | None = None,
    intent: str | None = None,
    needs_clarification: bool = False,
    clarification_question: str | None = None,
    emit_token: Callable[[str], None] | None = None,
) -> SupervisorDecision:
    """澄清 / 按计划执行已注册能力 / ReACT 兜底。"""
    plan = DEFAULT_POLICY.plan_supervisor(
        intent=intent,
        needs_clarification=needs_clarification,
        clarification_question=clarification_question,
        query=query,
        order_no=order_no,
        course_query=course_query,
        last_course_name=last_course_name,
    )
    if plan.action == "ask_user":
        return to_supervisor_decision(
            plan,
            order_no=order_no,
            course_query=course_query or plan.bound_course_name,
        )

    if plan.action == "react":
        answer = await stream_react(
            query,
            user_id=user_id,
            tenant_id=tenant_id,
            history=history,
            order_no=order_no,
            course_query=course_query,
            last_course_name=last_course_name,
            session_summary=session_summary,
            session_cache_state=session_cache_state,
            emit_token=emit_token,
        )
        if _is_auto_handoff_answer(answer):
            return to_supervisor_decision(
                plan,
                answer=answer,
                order_no=order_no,
                course_query=course_query or plan.bound_course_name,
                escalated=True,
                escalation_reason="react_limit",
            )
        return to_supervisor_decision(
            plan,
            answer=answer,
            order_no=order_no,
            course_query=course_query or plan.bound_course_name,
        )

    answer, bound_course, cache_updates, escalated = await _execute_plan(
        plan,
        query=query,
        user_id=user_id,
        tenant_id=tenant_id,
        history=history,
        order_no=order_no,
        course_query=course_query,
        last_course_name=last_course_name,
        session_summary=session_summary,
        emit_token=emit_token,
    )
    return to_supervisor_decision(
        plan,
        answer=answer,
        order_no=order_no,
        course_query=bound_course or course_query or plan.bound_course_name,
        cache_state_updates=cache_updates,
        escalated=escalated,
        escalation_reason="transient_exhausted" if escalated else None,
    )


async def arun_supervisor(
    query: str,
    *,
    user_id: str = "anonymous",
    tenant_id: str = "demo",
    history: list[BaseMessage] | None = None,
    order_no: str | None = None,
    course_query: str | None = None,
    last_course_name: str | None = None,
    session_summary: str | None = None,
    session_cache_state: dict[str, Any] | None = None,
    intent: str | None = None,
    needs_clarification: bool = False,
    clarification_question: str | None = None,
    emit_token: Callable[[str], None] | None = None,
) -> SupervisorDecision:
    """向后兼容别名：等价于 :func:`run_supervisor`。"""
    return await run_supervisor(
        query,
        user_id=user_id,
        tenant_id=tenant_id,
        history=history,
        order_no=order_no,
        course_query=course_query,
        last_course_name=last_course_name,
        session_summary=session_summary,
        session_cache_state=session_cache_state,
        intent=intent,
        needs_clarification=needs_clarification,
        clarification_question=clarification_question,
        emit_token=emit_token,
    )


def _execution_context(
    *,
    query: str,
    user_id: str,
    tenant_id: str,
    history: list[BaseMessage] | None,
    order_no: str | None,
    course_query: str | None,
    last_course_name: str | None,
    session_summary: str | None,
    bound_course_name: str | None,
) -> ExecutionContext:
    return ExecutionContext(
        query=query,
        tenant_id=tenant_id,
        user_id=user_id,
        history=history,
        session_summary=session_summary,
        order_no=order_no,
        last_order_no=order_no,
        course_query=course_query,
        last_course_query=course_query,
        last_course_name=last_course_name or bound_course_name,
        bound_course_name=bound_course_name,
    )


async def _execute_plan(
    plan: SupervisorPlan,
    *,
    query: str,
    user_id: str,
    tenant_id: str,
    history: list[BaseMessage] | None,
    order_no: str | None,
    course_query: str | None,
    last_course_name: str | None,
    session_summary: str | None,
    emit_token: Callable[[str], None] | None,
) -> tuple[str, str | None, dict[str, Any], bool]:
    """按 parallel_group 执行；同组 gather，不同组串行。观察写入 observations。"""
    groups: dict[int, list[PlanStep]] = {}
    for step in plan.steps:
        groups.setdefault(step.parallel_group, []).append(step)

    parts: list[str] = []
    observations: dict[str, object] = {}
    cache_updates: dict[str, Any] = {}
    bound_course = plan.bound_course_name

    for group_id in sorted(groups):
        steps = groups[group_id]
        ctx = _execution_context(
            query=query,
            user_id=user_id,
            tenant_id=tenant_id,
            history=history,
            order_no=order_no,
            course_query=course_query,
            last_course_name=last_course_name or bound_course,
            session_summary=session_summary,
            bound_course_name=bound_course,
        )
        if len(steps) == 1:
            text, observations, escalated = await _run_step(steps[0], ctx, observations)
            cache_updates.update(_extract_cache_updates(observations))
            if escalated:
                return text or build_handoff_answer(auto_escalated=True), bound_course, cache_updates, True
            if text:
                parts.append(text)
            bound_course = str(observations["course_name"]) if observations.get("course_name") else bound_course
            continue

        results = await asyncio.gather(*(_run_step(step, ctx, observations) for step in steps))
        for text, obs, escalated in results:
            observations.update(obs)
            cache_updates.update(_extract_cache_updates(obs))
            if escalated:
                return text or build_handoff_answer(auto_escalated=True), bound_course, cache_updates, True
            if text:
                parts.append(text)
        bound_course = str(observations["course_name"]) if observations.get("course_name") else bound_course

    answer = "\n\n".join(parts) or "暂时无法处理该问题。"
    if emit_token:
        emit_token(answer)
    return answer, bound_course, cache_updates, False


def _extract_cache_updates(observations: dict[str, object]) -> dict[str, Any]:
    raw = observations.get("_cache_state_updates")
    if isinstance(raw, dict):
        return dict(raw)
    return {}


async def _run_step(
    step: PlanStep,
    ctx: ExecutionContext,
    observations: dict[str, object],
) -> tuple[str | None, dict[str, object], bool]:
    """查注册表执行一步；依赖未绑定时跳过（例如无 product_name 不盲检索）。"""
    cap = get_capability(step.capability)
    kwargs = cap.bind(step, ctx, observations)
    if kwargs is None:
        return None, observations, False
    result = await cap.ainvoke(ctx, **kwargs)
    merged = {**observations, **result.observations}
    if result.escalated:
        return result.text, merged, True
    return result.text, merged, False
