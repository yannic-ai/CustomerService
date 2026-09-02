"""对外数据结构：课程咨询结果、订单进度报告、路由决策与聊天 API。"""

from typing import Any, Literal

from pydantic import BaseModel, Field

Intent = Literal["course_consult", "order", "mixed", "ambiguous", "chitchat", "handoff"]
RouteTarget = Literal["rag", "tool", "handoff", "chitchat", "supervisor"]

_INTENT_TARGET: dict[str, RouteTarget] = {
    "course_consult": "rag",
    "order": "tool",
    "mixed": "supervisor",
    "ambiguous": "supervisor",
    "handoff": "handoff",
    "chitchat": "chitchat",
}


def intent_to_target(intent: Intent | str) -> RouteTarget:
    """意图到图分支目标的映射。"""
    return _INTENT_TARGET.get(intent, "chitchat")


class CourseModule(BaseModel):
    """课程中的一个教学模块。"""

    name: str
    topics: list[str] = Field(default_factory=list)
    duration_hours: float = 0
    outcome: str = ""


class LearningStage(BaseModel):
    """学习路径中的一个阶段（入门 / 巩固 / 实战）。"""

    stage: str
    modules: list[str] = Field(default_factory=list)
    goal: str = ""


class EvidenceSection(BaseModel):
    """检索证据：章节/模块级引用。"""

    source: str
    section_path: str = ""
    module_name: str = ""
    excerpt: str = ""


class CourseConsultResult(BaseModel):
    """RAG 课程咨询的结构化输出。"""

    course_name: str
    summary: str
    modules: list[CourseModule] = Field(default_factory=list)
    learning_path: list[LearningStage] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    evidence: list[EvidenceSection] = Field(default_factory=list)


class OrderProgressStep(BaseModel):
    """订单时间线上的单个节点。"""

    name: str
    status: Literal["completed", "in_progress", "pending", "rejected"]
    timestamp: str | None = None
    note: str | None = None


class OrderReport(BaseModel):
    """订单进度可视化报告。"""

    order_no: str
    product_name: str
    course_code: str = ""
    order_type: str
    current_status: str
    progress_percent: int
    amount: float = 0
    steps: list[OrderProgressStep] = Field(default_factory=list)
    visualization: str = ""
    summary: str = ""
    oss_url: str | None = None


class RouteDecision(BaseModel):
    """意图识别结果，供 Router 选择下游 Agent。"""

    intent: Intent
    target: RouteTarget = "chitchat"
    confidence: float = 1.0
    order_no: str | None = None
    course_query: str | None = None
    needs_clarification: bool = False
    clarification_question: str | None = None
    reason: str = ""


class SupervisorDecision(BaseModel):
    """主 Agent / supervisor 的结构化决策。

    ``action``：图节点只分支 ``ask_user`` vs 其余；编排细节在
    ``tools_to_call`` / ``parallel`` / ``reason``（parallel-bound /
    serial-order-then-course / unbound-fallback 等）。
    """

    action: Literal["answer", "ask_user", "use_tools", "execute", "react"] = "answer"
    final_intent: Intent = "ambiguous"
    tools_to_call: list[str] = Field(default_factory=list)
    parallel: bool = False
    clarification_question: str | None = None
    bound_order_no: str | None = None
    bound_course_query: str | None = None
    resolved_from_ambiguous: bool = False
    answer: str | None = None
    reason: str | None = None
    cache_state_updates: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    """聊天接口入参。"""

    message: str
    user_id: str = "anonymous"
    session_id: str | None = None


class ChatResponse(BaseModel):
    """聊天接口出参，含答案、意图与可选结构化数据。"""

    answer: str
    intent: Intent | str = "chitchat"
    blocked: bool = False
    structured: dict[str, Any] | None = None
    visualization: str | None = None
    oss_url: str | None = None
    session_id: str | None = None
    usage: dict[str, Any] | None = None
    request_id: str | None = None
    trace_id: str | None = None


class FeedbackRequest(BaseModel):
    """用户赞踩：挂到对应对话轮的 Langfuse trace（或 session）。"""

    trace_id: str = ""
    value: int  # 1=赞，0=踩
    comment: str | None = None
    request_id: str | None = None
    session_id: str | None = None


class FeedbackResponse(BaseModel):
    """赞踩受理结果。"""

    status: str = "ok"
    recorded: bool = True
    langfuse: bool = False
    reason: str | None = None


class GreetOption(BaseModel):
    """开场白预置选项。"""

    label: str
    message: str


class GreetResponse(BaseModel):
    """开场白：问候语与快捷问题。"""

    greeting: str
    options: list[GreetOption] = Field(default_factory=list)


class KnowledgeIngestResponse(BaseModel):
    """管理面 ingest 受理结果。"""

    status: str = "accepted"
    tenant_id: str
    rebuild: bool = False
    saved_path: str | None = None
    message: str = ""
