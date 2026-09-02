"""能力契约：注册表、计划绑定与执行结果。

新增发票 / 优惠券 / 工单时实现 ``Capability`` 并 ``register``，不必再改
Graph 节点、Supervisor 执行循环或 ReACT 工具列表。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.messages import BaseMessage


@dataclass(frozen=True)
class CapabilitySpec:
    """能力声明：计划层看 id / provides / bind，执行层看 tool_name。"""

    id: str
    tool_name: str
    description: str
    # 计划可绑定的输入名（order_no / course_query / product_name …）
    accepts: tuple[str, ...] = ()
    # 执行后写入 observations、供后续步骤依赖（如课名来自订单商品）
    provides: tuple[str, ...] = ()


@dataclass
class ExecutionContext:
    """一轮编排的共享上下文；身份只从这里进能力，不进 tool 参数。"""

    query: str
    tenant_id: str = "demo"
    user_id: str = "anonymous"
    history: list[BaseMessage] | None = None
    session_summary: str | None = None
    order_no: str | None = None
    last_order_no: str | None = None
    course_query: str | None = None
    last_course_query: str | None = None
    last_course_name: str | None = None
    bound_course_name: str | None = None
    persist: bool = True


@dataclass
class CapabilityResult:
    """能力执行结果。``text is None`` 表示本步跳过（例如空召回不拼进答案）。"""

    capability_id: str
    ok: bool = True
    text: str | None = None
    data: Any = None
    observations: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    escalated: bool = False


class Capability(Protocol):
    """可注册能力：``bind`` 从计划+观察取参，``ainvoke`` 真正执行。"""

    spec: CapabilitySpec

    def bind(self, step: Any, ctx: ExecutionContext, observations: dict[str, Any]) -> dict[str, Any] | None:
        """返回 ainvoke 的 kwargs；None 表示本步跳过（依赖未满足）。"""
        ...

    async def ainvoke(self, ctx: ExecutionContext, **kwargs: Any) -> CapabilityResult:
        ...

    def as_langchain_tool(self) -> Any:
        """ReACT 用的 LangChain tool；身份仍从 ContextVar 读取。"""
        ...
