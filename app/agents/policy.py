"""路由策略：意图护栏、图分流、RAG 升级、Supervisor 澄清的唯一权威。

识别（关键词 / LLM / 槽位）仍在 ``intent.py``；本模块只负责识别之后的分流规则，
避免新增场景时在 intent、图边、RAG 节点、Supervisor 各改一处。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from app.schemas import Intent, RouteDecision, RouteTarget, SupervisorDecision, intent_to_target

FORCE_ORDER_HINTS = ("订单", "退款", "物流", "单号", "查单")
_ORDER_NO_RE = re.compile(r"(?:订单|#|NO\.?)?\s*(\d{8,})", re.IGNORECASE)

CONFIDENCE_RAG_MIN = 0.7
CONFIDENCE_DIRECT = 0.9
CONFIDENCE_MIXED = 0.65
CONFIDENCE_AMBIGUOUS = 0.35

MISSING_ORDER_CLARIFY = "请提供完整订单编号，我再帮您查询退款或进度。"

GraphNode = Literal["rag", "tool", "supervisor", "handoff", "chitchat", "respond"]

_TARGET_BY_KEY: dict[str, GraphNode] = {
    "rag": "rag",
    "tool": "tool",
    "supervisor": "supervisor",
    "handoff": "handoff",
    "chitchat": "chitchat",
    "course_consult": "rag",
    "order": "tool",
    "order_query": "tool",
    "mixed": "supervisor",
    "ambiguous": "supervisor",
}

_SUPERVISOR_INTENTS = frozenset({"mixed", "order", "course_consult"})
_ORDER_SIGNAL_PASSTHROUGH = frozenset({"handoff", "order", "mixed", "ambiguous"})

# 明确课程信号：不能只用 course_query is not None——mixed 时常把整句塞进去
_EXPLICIT_COURSE_HINTS = (
    "课程",
    "大纲",
    "模块",
    "学习路径",
    "入门课",
    "讲什么",
    "学什么",
    "包含哪些",
    "适合",
    "零基础",
    "难吗",
)
SupervisorCapability = str
SupervisorAction = Literal["ask_user", "execute", "react"]


@dataclass(frozen=True)
class RagUpgradeDecision:
    """误进 RAG 时的升级结果：不生成课程答案，改走 Supervisor。"""

    reason: str
    intent: Intent = "mixed"
    target: RouteTarget = "supervisor"
    order_no: str | None = None
    course_query: str | None = None


@dataclass(frozen=True)
class PlanStep:
    """编排一步：同一 ``parallel_group`` 内并发，不同组号按升序串行。"""

    capability: SupervisorCapability
    parallel_group: int = 0
    # course_retrieve 专用：from_order_product 表示课名依赖订单观察（先查单）
    course_source: Literal["bound", "from_order_product"] | None = None


@dataclass(frozen=True)
class SupervisorPlan:
    """Supervisor 编排计划：澄清、按步骤执行工具，或 ReACT 兜底。

    编排层既做计划也执行；``react`` 只覆盖绑不上 / 依赖不明的场景。
    """

    action: SupervisorAction
    final_intent: Intent
    clarification_question: str | None = None
    steps: tuple[PlanStep, ...] = ()
    reason: str = ""
    resolved_from_ambiguous: bool = False
    # 执行时可直接用于课程检索的绑定课名（不含「整句当课名」的伪绑定）
    bound_course_name: str | None = None

    @property
    def tools_to_call(self) -> tuple[str, ...]:
        """可观察工具名列表（来自能力注册表）。"""
        from app.capabilities import tool_name_for

        names: list[str] = []
        for step in self.steps:
            name = tool_name_for(step.capability)
            if name and name not in names:
                names.append(name)
        return tuple(names)

    @property
    def parallel(self) -> bool:
        """是否存在同组并发步骤。"""
        if len(self.steps) < 2:
            return False
        groups = {step.parallel_group for step in self.steps}
        for group in groups:
            if sum(1 for step in self.steps if step.parallel_group == group) > 1:
                return True
        return False


def extract_query_order_no(text: str) -> str | None:
    """从当前问句提取订单号，不含槽位回填。"""
    match = _ORDER_NO_RE.search(text or "")
    return match.group(1) if match else None


def query_has_order_signal(text: str) -> bool:
    """当前问句是否含订单号或退款/物流等强信号。不看会话槽位，避免跨话题误升。"""
    raw = text or ""
    if extract_query_order_no(raw):
        return True
    return any(token in raw for token in FORCE_ORDER_HINTS)


def finalize_route(decision: RouteDecision) -> RouteDecision:
    """补齐 target / confidence 默认值，保证图分支可用。"""
    target = decision.target or intent_to_target(decision.intent)
    confidence = decision.confidence
    if confidence <= 0:
        confidence = 1.0
    return decision.model_copy(update={"target": target, "confidence": confidence})


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


def _looks_like_course(text: str, decision: RouteDecision) -> bool:
    if decision.intent == "course_consult":
        return True
    from app.agents.intent import COURSE_HINTS

    return any(token in (text or "") for token in COURSE_HINTS)


@dataclass(frozen=True)
class RoutingPolicy:
    """客服图路由规则集。默认单例 ``DEFAULT_POLICY``；测试可换阈值。"""

    confidence_rag_min: float = CONFIDENCE_RAG_MIN
    confidence_direct: float = CONFIDENCE_DIRECT
    confidence_mixed: float = CONFIDENCE_MIXED
    confidence_ambiguous: float = CONFIDENCE_AMBIGUOUS
    missing_order_clarify: str = MISSING_ORDER_CLARIFY

    def apply_order_signal(self, text: str, decision: RouteDecision) -> RouteDecision:
        """当前问句含强订单信号时，把误判的课程咨询 / 闲聊改走订单或 ReACT。

        只看本轮文本，不用槽位里的 last_order_no。课程误判升 mixed（两边工具都有），
        闲误判升 order（再由缺单号规则决定是否澄清）。
        """
        if decision.intent in _ORDER_SIGNAL_PASSTHROUGH:
            if not decision.order_no:
                found = extract_query_order_no(text)
                if found:
                    return decision.model_copy(update={"order_no": found})
            return decision
        if not query_has_order_signal(text):
            return decision
        order_no = decision.order_no or extract_query_order_no(text)
        if _looks_like_course(text, decision):
            return _decision(
                intent="mixed",
                reason="order-signal-force",
                confidence=self.confidence_mixed,
                order_no=order_no,
                course_query=decision.course_query or text,
            )
        return _decision(
            intent="order",
            reason="order-signal-force",
            confidence=self.confidence_direct,
            order_no=order_no,
            course_query=decision.course_query,
        )

    def apply_low_confidence(self, decision: RouteDecision) -> RouteDecision:
        """低置信课程咨询不直达 RAG：改走 mixed / ReACT，两边工具都在。"""
        if decision.intent != "course_consult":
            return decision
        if decision.confidence >= self.confidence_rag_min:
            return decision
        return _decision(
            intent="mixed",
            reason="low-confidence-upgrade",
            confidence=decision.confidence,
            order_no=decision.order_no,
            course_query=decision.course_query,
        )

    def downgrade_missing_order(self, decision: RouteDecision) -> RouteDecision:
        """订单意图但无单号：降为 ambiguous，交给 supervisor 澄清。"""
        if decision.intent != "order" or decision.order_no:
            return decision
        return _decision(
            intent="ambiguous",
            reason="missing-order-no",
            confidence=self.confidence_ambiguous,
            course_query=decision.course_query,
            needs_clarification=True,
            clarification_question=self.missing_order_clarify,
        )

    def finalize(self, text: str, decision: RouteDecision) -> RouteDecision:
        """意图出口：订单信号强制分流 → 低置信不直达 RAG → 缺单号澄清。"""
        guarded = self.apply_low_confidence(self.apply_order_signal(text, decision))
        return finalize_route(self.downgrade_missing_order(guarded))

    def next_after_router(
        self,
        *,
        blocked: bool = False,
        target: str | None = None,
        intent: str | None = None,
        confidence: float = 0.0,
    ) -> GraphNode:
        """Router 之后的图边；低置信 RAG 改走 Supervisor（兜住未走 finalize 的决策）。"""
        if blocked:
            return "respond"
        key = target or intent or "chitchat"
        dest = _TARGET_BY_KEY.get(key, "chitchat")
        if dest == "rag" and float(confidence or 0.0) < self.confidence_rag_min:
            return "supervisor"
        return dest

    def next_after_rag(
        self,
        *,
        upgraded: bool = False,
        target: str | None = None,
    ) -> Literal["supervisor", "respond"]:
        """RAG 因订单信号升到 Supervisor 时不重跑 Router。"""
        if upgraded or target == "supervisor":
            return "supervisor"
        return "respond"

    def decide_rag_upgrade(
        self,
        query: str,
        *,
        order_no: str | None = None,
        course_query: str | None = None,
    ) -> RagUpgradeDecision | None:
        """已进入 RAG 时是否应放弃生成、改走 Supervisor。只看本轮问句。"""
        if not query_has_order_signal(query):
            return None
        return RagUpgradeDecision(
            reason="order_signal",
            order_no=order_no or extract_query_order_no(query),
            course_query=course_query,
        )

    def rag_upgrade_updates(self, decision: RagUpgradeDecision) -> dict[str, Any]:
        """写成图状态补丁，供 rag 节点直接 return。"""
        return {
            "target": decision.target,
            "intent": decision.intent,
            "needs_clarification": False,
            "clarification_question": None,
            "order_no": decision.order_no,
            "course_query": decision.course_query,
            "rag_upgraded_to_supervisor": True,
        }

    def plan_supervisor(
        self,
        *,
        intent: str | None = None,
        needs_clarification: bool = False,
        clarification_question: str | None = None,
        query: str | None = None,
        order_no: str | None = None,
        course_query: str | None = None,
        last_course_name: str | None = None,
    ) -> SupervisorPlan:
        """澄清优先；能绑订单/课名则 ``execute``（可并行或串行）；否则 ``react``。

        不能把 ``course_query is not None`` 当成「已有课名」：mixed 识别常把整句
        塞进 ``course_query``，真课名往往在订单 ``product_name`` 观察结果里。
        """
        if needs_clarification and clarification_question:
            return SupervisorPlan(
                action="ask_user",
                final_intent="ambiguous",
                clarification_question=clarification_question,
                reason="needs-clarification",
            )

        final_intent: Intent
        if intent in _SUPERVISOR_INTENTS:
            final_intent = intent  # type: ignore[assignment]
        else:
            final_intent = "mixed"
        from_ambiguous = intent == "ambiguous"

        resolved_order = (order_no or "").strip() or None
        bound_course = _bound_course_name(
            query=query or "",
            course_query=course_query,
            last_course_name=last_course_name,
            order_no=resolved_order,
        )

        # 两边都能绑定 → 并行查单 + 检索（无数据依赖）
        if resolved_order and bound_course:
            return SupervisorPlan(
                action="execute",
                final_intent=final_intent,
                steps=(
                    PlanStep(capability="order_query", parallel_group=0),
                    PlanStep(
                        capability="course_retrieve",
                        parallel_group=0,
                        course_source="bound",
                    ),
                ),
                reason="parallel-bound",
                resolved_from_ambiguous=from_ambiguous,
                bound_course_name=bound_course,
            )

        # 有单号、mixed、课名只能从订单商品来 → 先查单再检索（数据依赖）
        # 写死 RAG→订单会错课：课名在订单观察 product_name 里，必须先查单。
        if resolved_order and final_intent == "mixed":
            return SupervisorPlan(
                action="execute",
                final_intent="mixed",
                steps=(
                    PlanStep(capability="order_query", parallel_group=0),
                    PlanStep(
                        capability="course_retrieve",
                        parallel_group=1,
                        course_source="from_order_product",
                    ),
                ),
                reason="serial-order-then-course",
                resolved_from_ambiguous=from_ambiguous,
                bound_course_name=None,
            )

        # 纯订单且已有单号：只查单（订单节点偶发落到 Supervisor 时）
        if resolved_order and final_intent == "order":
            return SupervisorPlan(
                action="execute",
                final_intent="order",
                steps=(PlanStep(capability="order_query", parallel_group=0),),
                reason="order-only",
                resolved_from_ambiguous=from_ambiguous,
            )

        # 只有明确课名、无单号 → 直接检索（不必进 ReACT）
        if bound_course and not resolved_order and final_intent in {"course_consult", "mixed"}:
            return SupervisorPlan(
                action="execute",
                final_intent="course_consult",
                steps=(
                    PlanStep(
                        capability="course_retrieve",
                        parallel_group=0,
                        course_source="bound",
                    ),
                ),
                reason="course-only",
                resolved_from_ambiguous=from_ambiguous,
                bound_course_name=bound_course,
            )

        return SupervisorPlan(
            action="react",
            final_intent=final_intent,
            reason="unbound-fallback",
            resolved_from_ambiguous=from_ambiguous,
            bound_course_name=bound_course,
        )


def _bound_course_name(
    *,
    query: str,
    course_query: str | None,
    last_course_name: str | None,
    order_no: str | None = None,
) -> str | None:
    """解析可用于检索的课名；整句伪绑定不算。

    mixed 常把整句塞进 ``course_query``（与问句相同）。若已有订单号上下文，
    更不可能是课名——课名往往在订单 ``product_name`` 观察结果里。
    """
    name = (last_course_name or "").strip()
    if name and name not in {"未找到课程"}:
        return name

    cq = (course_query or "").strip()
    q = (query or "").strip()
    if not cq:
        return None

    has_order_ctx = bool((order_no or "").strip()) or bool(extract_query_order_no(q))

    # mixed 整句伪绑定：与问句相同则默认不算已绑课名
    if cq == q:
        if has_order_ctx:
            return None
        # 纯课程短问（如「Python入门课大纲」）可当检索词
        if any(token in cq for token in _EXPLICIT_COURSE_HINTS) and len(cq) <= 40:
            return cq
        return None

    if len(cq) > 80:
        return None
    # 槽位课名本身若仍是「带单号的长句」，也不算绑定
    if extract_query_order_no(cq) and len(cq) > 24:
        return None
    return cq


DEFAULT_POLICY = RoutingPolicy()


def apply_order_signal_guard(text: str, decision: RouteDecision) -> RouteDecision:
    """兼容旧调用：委托默认策略。"""
    return DEFAULT_POLICY.apply_order_signal(text, decision)


def apply_low_confidence_guard(decision: RouteDecision) -> RouteDecision:
    """兼容旧调用：委托默认策略。"""
    return DEFAULT_POLICY.apply_low_confidence(decision)


def to_supervisor_decision(
    plan: SupervisorPlan,
    *,
    answer: str | None = None,
    order_no: str | None = None,
    course_query: str | None = None,
    cache_state_updates: dict[str, Any] | None = None,
    escalated: bool = False,
    escalation_reason: str | None = None,
) -> SupervisorDecision:
    """把策略计划转成 Supervisor 节点使用的结构化结果。"""
    bound_course = course_query or plan.bound_course_name
    cache_updates = dict(cache_state_updates or {})
    if escalated:
        return SupervisorDecision(
            action=plan.action,
            final_intent="handoff",
            tools_to_call=list(plan.tools_to_call),
            parallel=plan.parallel,
            answer=answer,
            bound_order_no=order_no,
            bound_course_query=bound_course,
            resolved_from_ambiguous=plan.resolved_from_ambiguous,
            reason=escalation_reason or "tool-escalation",
            cache_state_updates=cache_updates,
        )
    if plan.action == "ask_user":
        question = plan.clarification_question or ""
        return SupervisorDecision(
            action="ask_user",
            final_intent=plan.final_intent,
            clarification_question=plan.clarification_question,
            answer=answer or question,
            bound_order_no=order_no,
            bound_course_query=bound_course,
            resolved_from_ambiguous=False,
            reason=plan.reason,
        )
    # 图节点只分支 ask_user；execute / react 均带答案继续走 respond
    return SupervisorDecision(
        action=plan.action,
        final_intent=plan.final_intent,
        tools_to_call=list(plan.tools_to_call),
        parallel=plan.parallel,
        answer=answer,
        bound_order_no=order_no,
        bound_course_query=bound_course,
        resolved_from_ambiguous=plan.resolved_from_ambiguous,
        reason=plan.reason,
        cache_state_updates=cache_updates,
    )
