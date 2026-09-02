"""课程检索能力：Supervisor / ReACT 共用 ``retrieve_course``。"""

from __future__ import annotations

import inspect
import time
from typing import Any

from langchain.tools import tool
from langchain_core.messages import BaseMessage

from app.capabilities.types import CapabilityResult, CapabilitySpec, ExecutionContext
from app.schemas import CourseConsultResult
from app.tenancy import get_current_tenant, get_current_user
from app.tool_context import get_tool_context
from app.tool_cache import (
    TOOL_COURSE,
    annotate_stale_course,
    get_cached_course,
    is_cacheable_course_result,
    normalize_course_cache_key,
    put_cached_course,
    session_updates_for_course,
)
from app.tool_escalation import (
    ToolEscalationError,
    escalation_for_transient,
    raise_if_escalated,
)

NOT_FOUND_COURSE = "未找到课程"


async def retrieve_course(
    query: str,
    *,
    tenant_id: str | None = None,
    history: list[BaseMessage] | None = None,
    last_course_query: str | None = None,
    last_course_name: str | None = None,
    session_summary: str | None = None,
    course_code: str | None = None,
    user_id: str | None = None,
) -> CourseConsultResult:
    """课程检索统一入口（结构化结果）。RAG 图节点仍走流式 ``stream_course_answer``。"""
    from app.agents.rag import consult_course

    tenant = tenant_id or get_current_tenant()
    uid = user_id or get_current_user()
    resource_key = normalize_course_cache_key(query, course_code=course_code)
    try:
        result = consult_course(
            query,
            tenant_id=tenant,
            history=history,
            last_course_query=last_course_query,
            last_course_name=last_course_name,
            session_summary=session_summary,
            course_code=course_code,
        )
        if inspect.isawaitable(result):
            result = await result
    except Exception as exc:
        cached = get_cached_course(resource_key, tenant_id=tenant, user_id=uid)
        if cached:
            age = max(0.0, time.time() - cached.cached_at)
            return annotate_stale_course(cached.payload, age_seconds=age)
        raise_if_escalated(escalation_for_transient(TOOL_COURSE, resource_key))
        raise

    if is_cacheable_course_result(result):
        put_cached_course(result, resource_key, tenant_id=tenant, user_id=uid)
    return result


def _course_search_query(
    step: Any,
    ctx: ExecutionContext,
    observations: dict[str, Any],
) -> str | None:
    if getattr(step, "course_source", None) == "from_order_product":
        name = str(observations.get("product_name") or "").strip()
        code = str(observations.get("course_code") or "").strip()
        return name or code or None
    return (
        ctx.bound_course_name
        or ctx.last_course_name
        or ctx.course_query
        or ctx.query
        or None
    )


def _bound_course_code(ctx: ExecutionContext, observations: dict[str, Any], step: Any) -> str:
    code = str(observations.get("course_code") or "").strip()
    if code:
        return code
    if getattr(step, "course_source", None) != "from_order_product":
        return ""
    from app.db.courses import resolve_course_code

    return resolve_course_code(ctx.tenant_id, name=observations.get("product_name")) or ""


@tool
async def retrieve_course_knowledge(question: str) -> str:
    """检索课程知识库，返回课程模块与学习路径的结构化 JSON。

    会话 history、课名槽位、摘要由 ReACT 入口经 ContextVar 注入，
    与 RAG 热路径的 ``expand_query`` / ``format_history_for_prompt`` 对齐。
    """
    ctx = get_tool_context()
    try:
        result = await retrieve_course(
            question,
            tenant_id=get_current_tenant(),
            user_id=get_current_user(),
            history=ctx.history_list,
            last_course_query=ctx.last_course_query,
            last_course_name=ctx.last_course_name,
            session_summary=ctx.session_summary,
            course_code=ctx.course_code,
        )
    except ToolEscalationError:
        raise
    from app.agents.rag import format_course_answer

    return format_course_answer(result)


class CourseCapability:
    """课程能力：计划可绑定课名，或从订单 ``product_name`` 观察串行检索。"""

    spec = CapabilitySpec(
        id="course_retrieve",
        tool_name="retrieve_course_knowledge",
        description="检索课程大纲，返回模块与学习路径。",
        accepts=("course_query", "product_name", "course_code"),
        provides=("course_name",),
    )

    def bind(self, step: Any, ctx: ExecutionContext, observations: dict[str, Any]) -> dict[str, Any] | None:
        search_query = _course_search_query(step, ctx, observations)
        code = _bound_course_code(ctx, observations, step)
        if getattr(step, "course_source", None) == "from_order_product":
            if not search_query and not code:
                return None
            search_query = search_query or code
        elif not search_query:
            return None
        payload: dict[str, Any] = {
            "query": search_query,
            "last_course_query": ctx.course_query or ctx.bound_course_name,
            "last_course_name": ctx.last_course_name or ctx.bound_course_name or observations.get("product_name"),
        }
        if code:
            payload["course_code"] = code
        return payload

    async def ainvoke(self, ctx: ExecutionContext, **kwargs: Any) -> CapabilityResult:
        search_query = str(kwargs.get("query") or "")
        course_code = kwargs.get("course_code")
        resource_key = normalize_course_cache_key(search_query, course_code=course_code)
        try:
            course = await retrieve_course(
                search_query,
                tenant_id=ctx.tenant_id,
                user_id=ctx.user_id,
                history=ctx.history,
                last_course_query=kwargs.get("last_course_query"),
                last_course_name=kwargs.get("last_course_name"),
                session_summary=ctx.session_summary,
                course_code=course_code,
            )
        except ToolEscalationError as exc:
            return CapabilityResult(
                capability_id=self.spec.id,
                ok=False,
                text=exc.decision.message,
                error="escalated",
                escalated=True,
            )
        if course.course_name == NOT_FOUND_COURSE:
            return CapabilityResult(
                capability_id=self.spec.id,
                ok=False,
                text=None,
                data=course,
                observations={"course_name": search_query},
            )
        bound = course.course_name or search_query
        from app.agents.rag import format_course_answer

        return CapabilityResult(
            capability_id=self.spec.id,
            ok=True,
            text=format_course_answer(course),
            data=course,
            observations={
                "course_name": bound,
                "_cache_state_updates": session_updates_for_course(course, resource_key),
            },
        )

    def as_langchain_tool(self) -> Any:
        return retrieve_course_knowledge


from app.capabilities.registry import register as _register

_register(CourseCapability())
