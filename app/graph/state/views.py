"""从扁平 CSState 抽出分组视图。checkpoint 仍是一层 key，节点不必记字段属于哪一组。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class IdentityView:
    query: str
    user_id: str
    tenant_id: str
    session_id: str

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        default_tenant: str = "demo",
    ) -> IdentityView:
        user = str(state.get("user_id") or "").strip() or "anonymous"
        tenant = str(state.get("tenant_id") or "").strip() or default_tenant
        return cls(
            query=str(state.get("query") or ""),
            user_id=user,
            tenant_id=tenant,
            session_id=str(state.get("session_id") or "").strip(),
        )


@dataclass(frozen=True)
class RouteView:
    intent: str
    target: str
    confidence: float
    needs_clarification: bool
    clarification_question: str | None
    order_no: str | None
    course_query: str | None
    blocked: bool
    block_reason: str
    rag_upgraded_to_supervisor: bool

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> RouteView:
        return cls(
            intent=str(state.get("intent") or ""),
            target=str(state.get("target") or ""),
            confidence=float(state.get("confidence") or 0.0),
            needs_clarification=bool(state.get("needs_clarification")),
            clarification_question=state.get("clarification_question"),
            order_no=state.get("order_no") or None,
            course_query=state.get("course_query") or None,
            blocked=bool(state.get("blocked")),
            block_reason=str(state.get("block_reason") or ""),
            rag_upgraded_to_supervisor=bool(state.get("rag_upgraded_to_supervisor")),
        )


@dataclass(frozen=True)
class MemoryView:
    last_intent: str | None
    last_order_no: str | None
    last_course_query: str | None
    last_course_name: str | None
    handoff_pending: bool
    last_order_structured: dict[str, Any] | None
    last_course_structured: dict[str, Any] | None
    last_course_structured_key: str | None
    session_summary: str

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> MemoryView:
        order_structured = state.get("last_order_structured")
        course_structured = state.get("last_course_structured")
        return cls(
            last_intent=state.get("last_intent") or None,
            last_order_no=state.get("last_order_no") or None,
            last_course_query=state.get("last_course_query") or None,
            last_course_name=state.get("last_course_name") or None,
            handoff_pending=bool(state.get("handoff_pending")),
            last_order_structured=order_structured if isinstance(order_structured, dict) else None,
            last_course_structured=course_structured if isinstance(course_structured, dict) else None,
            last_course_structured_key=state.get("last_course_structured_key") or None,
            session_summary=str(state.get("session_summary") or ""),
        )

    def as_state(self) -> dict[str, Any]:
        """写回图状态 / 归档的扁平字段（不含 messages）。"""
        return {
            "last_intent": self.last_intent,
            "last_order_no": self.last_order_no,
            "last_course_query": self.last_course_query,
            "last_course_name": self.last_course_name,
            "handoff_pending": self.handoff_pending,
            "last_order_structured": self.last_order_structured,
            "last_course_structured": self.last_course_structured,
            "last_course_structured_key": self.last_course_structured_key,
            "session_summary": self.session_summary,
        }


@dataclass(frozen=True)
class ResultView:
    answer: str
    structured: dict[str, Any] | None
    visualization: str | None
    oss_url: str | None

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> ResultView:
        structured = state.get("structured")
        return cls(
            answer=str(state.get("answer") or ""),
            structured=structured if isinstance(structured, dict) else None,
            visualization=state.get("visualization"),
            oss_url=state.get("oss_url"),
        )
