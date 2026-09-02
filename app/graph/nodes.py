"""LangGraph 节点：安全检查、路由、RAG / 订单 / Supervisor、出口归档。"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from langchain_core.messages import HumanMessage

from app.agents.policy import DEFAULT_POLICY, RagUpgradeDecision
from app.agents.rag import astream_course_answer
from app.agents.supervisor import arun_supervisor
from app.capabilities.order import cache_state_updates_for_order, format_order_answer, query_order
from app.config import get_settings
from app.context import (
    SessionSlots,
    expand_query,
    recent_messages,
    slot_updates_after_turn,
)
from app.context.turn import finalize_respond_turn, persist_respond_turn
from app.graph.state import CSState, IdentityView, MemoryView, RouteView, reset_turn_fields
from app.memory import is_persistent_user, load_user_memory, overlay_user_memory, persist_from_state
from app.observability import inc_rag_route_upgrade, log_event
from app.security.filters import inspect_input
from app.tenancy import set_current_tenant, set_current_user
from app.services.order import can_access_order
from app.tool_cache import session_tool_cache_scope
from app.tool_escalation import ToolEscalationError, build_handoff_answer, tool_failure_tracker_scope

logger = logging.getLogger("cs.graph")


def _handoff_state(answer: str) -> dict[str, Any]:
    return {
        "intent": "handoff",
        "answer": answer,
        **slot_updates_after_turn("handoff"),
    }


def _identity(state: CSState) -> IdentityView:
    return IdentityView.from_state(state, default_tenant=get_settings().default_tenant)


def security_node(state: CSState) -> CSState:
    """入口安全检查：拦截注入/空输入，并对 PII 脱敏；追加本轮用户消息。"""
    ident = _identity(state)
    set_current_tenant(ident.tenant_id)
    set_current_user(ident.user_id)
    cleared = reset_turn_fields()
    verdict = inspect_input(ident.query)
    if not verdict.allowed:
        return {
            **cleared,
            "blocked": True,
            "block_reason": verdict.reason,
            "answer": f"请求已被安全中间件拦截：{verdict.reason}",
            "intent": "chitchat",
        }
    return {
        **cleared,
        "query": verdict.sanitized_text,
        "messages": [HumanMessage(content=verdict.sanitized_text)],
    }


def _hydrate_slots(state: CSState) -> SessionSlots:
    """会话槽位优先，空缺时用长期记忆回填；登录用户不回填他人订单号。"""
    ident = _identity(state)
    slots = overlay_user_memory(
        SessionSlots.from_state(state),
        load_user_memory(ident.tenant_id, ident.user_id),
    )
    if is_persistent_user(ident.user_id) and slots.last_order_no:
        if not can_access_order(slots.last_order_no, ident.tenant_id, ident.user_id):
            slots = replace(slots, last_order_no=None)
    return slots


def _with_memory(state: CSState, updates: dict[str, Any]) -> dict[str, Any]:
    """合并本轮更新后写入用户长期记忆。"""
    persist_from_state({**state, **updates})
    return updates


async def router_node(state: CSState) -> CSState:
    """调用意图识别，判定课程咨询 / 订单查询等下游路径。"""
    from app.agents.intent import arecognize_intent

    if RouteView.from_state(state).blocked:
        return {}
    ident = _identity(state)
    memory = MemoryView.from_state(state)
    history = recent_messages(state.get("messages"))
    slots = _hydrate_slots(state)
    decision = await arecognize_intent(
        ident.query,
        history=history,
        slots=slots,
        session_summary=memory.session_summary,
    )
    return {
        "intent": decision.intent,
        "target": decision.target,
        "confidence": decision.confidence,
        "needs_clarification": decision.needs_clarification,
        "clarification_question": decision.clarification_question,
        "order_no": decision.order_no,
        "course_query": decision.course_query,
        "last_intent": slots.last_intent,
        "last_order_no": slots.last_order_no,
        "last_course_query": slots.last_course_query,
        "last_course_name": slots.last_course_name,
        "handoff_pending": slots.handoff_pending,
    }


async def rag_node(state: CSState) -> CSState:
    """课程咨询：有订单信号则不生成、升 ReACT；否则检索生成，空召回且像订单同样升级。"""
    with tool_failure_tracker_scope(), session_tool_cache_scope(state):
        ident = _identity(state)
        route = RouteView.from_state(state)
        memory = MemoryView.from_state(state)
        set_current_tenant(ident.tenant_id)
        history = recent_messages(state.get("messages"))
        query = expand_query(
            route.course_query or ident.query,
            history,
            last_course_query=memory.last_course_query,
            last_course_name=memory.last_course_name,
        )
        upgrade = DEFAULT_POLICY.decide_rag_upgrade(
            ident.query,
            order_no=route.order_no,
            course_query=query,
        )
        if upgrade:
            return _upgrade_rag_to_supervisor(state, upgrade)
        answer, result = await astream_course_answer(
            query,
            tenant_id=ident.tenant_id,
            history=history,
            last_course_query=memory.last_course_query,
            last_course_name=memory.last_course_name,
            session_summary=memory.session_summary,
        )
        updates: dict[str, Any] = {
            "structured": result.model_dump(),
            "answer": answer,
        }
        updates.update(
            slot_updates_after_turn(
                "course_consult",
                course_query=query,
                course_name=result.course_name,
            )
        )
        from app.tool_cache import (
            is_cacheable_course_result,
            normalize_course_cache_key,
            put_cached_course,
            session_updates_for_course,
        )

        if is_cacheable_course_result(result):
            resource_key = normalize_course_cache_key(query, course_code=None)
            put_cached_course(result, resource_key, tenant_id=ident.tenant_id, user_id=ident.user_id)
            updates.update(session_updates_for_course(result, resource_key))
        return _with_memory(state, updates)


def _upgrade_rag_to_supervisor(state: CSState, upgrade: RagUpgradeDecision) -> dict[str, Any]:
    """误进 RAG 时升 ReACT：不重跑 Router，也不把检索结果当成最终答案。"""
    ident = _identity(state)
    inc_rag_route_upgrade(ident.tenant_id)
    log_event(
        logger,
        "rag_upgrade_to_react",
        tenant_id=ident.tenant_id,
        order_no=upgrade.order_no,
        query_chars=len(ident.query),
        reason=upgrade.reason,
    )
    return DEFAULT_POLICY.rag_upgrade_updates(upgrade)


def tool_node(state: CSState) -> CSState:
    """订单查询：调用订单 API 并生成进度报告。"""
    with tool_failure_tracker_scope(), session_tool_cache_scope(state):
        ident = _identity(state)
        route = RouteView.from_state(state)
        memory = MemoryView.from_state(state)
        set_current_tenant(ident.tenant_id)
        set_current_user(ident.user_id)
        try:
            report = query_order(
                ident.query,
                route.order_no,
                tenant_id=ident.tenant_id,
                user_id=ident.user_id,
                last_order_no=memory.last_order_no,
            )
        except ToolEscalationError as exc:
            return _with_memory(state, _handoff_state(exc.decision.message or build_handoff_answer(auto_escalated=True)))
        except TimeoutError as exc:
            return {"answer": str(exc), "intent": "order"}
        except ValueError as exc:
            return {"answer": str(exc), "intent": "order"}
        updates: dict[str, Any] = {
            "structured": report.model_dump(),
            "visualization": report.visualization,
            "oss_url": report.oss_url,
            "answer": format_order_answer(report),
            "order_no": report.order_no,
            **cache_state_updates_for_order(report),
        }
        updates.update(slot_updates_after_turn("order", order_no=report.order_no))
        return _with_memory(state, updates)


async def supervisor_node(state: CSState) -> CSState:
    """mixed / ambiguous：澄清，或按计划编排订单/课程，绑不上才 ReACT。"""
    with tool_failure_tracker_scope(), session_tool_cache_scope(state):
        ident = _identity(state)
        route = RouteView.from_state(state)
        memory = MemoryView.from_state(state)
        history = recent_messages(state.get("messages"))
        session_cache = {
            "last_order_structured": memory.last_order_structured,
            "last_course_structured": memory.last_course_structured,
            "last_course_structured_key": memory.last_course_structured_key,
        }
        decision = await arun_supervisor(
            ident.query,
            user_id=ident.user_id,
            tenant_id=ident.tenant_id,
            history=history,
            order_no=route.order_no or memory.last_order_no,
            course_query=route.course_query or memory.last_course_query,
            last_course_name=memory.last_course_name,
            session_summary=memory.session_summary,
            intent=route.intent,
            needs_clarification=route.needs_clarification,
            clarification_question=route.clarification_question,
            session_cache_state=session_cache,
        )
        if decision.final_intent == "handoff" and decision.reason in {
            "transient_exhausted",
            "react_limit",
            "tool-escalation",
        }:
            return _with_memory(
                state,
                _handoff_state(
                    decision.answer
                    or build_handoff_answer(auto_escalated=True, reason=decision.reason)  # type: ignore[arg-type]
                ),
            )
        intent = decision.final_intent or route.intent or "mixed"
        # 澄清短路径：不写槽位、不走工具副作用
        if decision.action == "ask_user":
            return {
                "answer": decision.answer or decision.clarification_question or "",
                "intent": intent,
            }

        bound_order = decision.bound_order_no or route.order_no or memory.last_order_no
        if bound_order and not can_access_order(bound_order, ident.tenant_id, ident.user_id):
            bound_order = None
        updates: dict[str, Any] = {
            "answer": decision.answer or "",
            "intent": intent,
        }
        if decision.cache_state_updates:
            updates.update(decision.cache_state_updates)
        updates.update(
            slot_updates_after_turn(
                intent,
                order_no=bound_order,
                course_query=decision.bound_course_query
                or route.course_query
                or memory.last_course_query,
            )
        )
        return _with_memory(state, updates)


def chitchat_node(state: CSState) -> CSState:
    """非业务闲聊时给出能力说明。"""
    return {
        "answer": (
            "我是课程与订单智能客服，可以咨询课程大纲（例如「Python入门课包含哪些内容？」）"
            "，或查询订单进度（例如「查询订单#20251114001的退款进度」）。"
        )
    }


def handoff_node(state: CSState) -> CSState:
    """人工转接：演示环境返回转接确认，不建真实工单。"""
    updates: dict[str, Any] = {
        **_handoff_state(build_handoff_answer()),
    }
    return _with_memory(state, updates)


async def respond_node(state: CSState) -> CSState:
    """出口脱敏、压缩与归档；无答案时回退 LLM 不可用模板话术。"""
    update = await finalize_respond_turn(state)
    await persist_respond_turn(state, update)
    return {
        "answer": update.answer,
        "messages": update.message_updates,
        "session_summary": update.session_summary,
    }
