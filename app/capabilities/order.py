"""订单查询能力：鉴权、超时、busy 降级、OSS 归档只在这里写一次。"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from langchain.tools import tool

from app.capabilities.types import CapabilityResult, CapabilitySpec, ExecutionContext
from app.executors import ToolPoolBusyError, run_order
from app.observability import inc_tool_call
from app.schemas import OrderReport
from app.tenancy import get_current_tenant, get_current_user
from app.services.order import (
    extract_order_no,
    fetch_order_report,
    persist_order_report,
    resolve_query_order_no,
)
from app.tool_cache import (
    TOOL_ORDER,
    annotate_stale_order,
    get_cached_order,
    is_cacheable_order_report,
    put_cached_order,
    session_updates_for_order,
)
from app.tool_escalation import (
    ToolEscalationError,
    escalation_for_transient,
    raise_if_escalated,
)

BUSY_PRODUCT = "查单服务繁忙"
MISSING_ORDER_NO = "未识别到订单号"


def format_order_answer(report: OrderReport) -> str:
    """拼接摘要与可视化 Markdown。"""
    if report.visualization:
        return f"{report.summary}\n\n{report.visualization}"
    return report.summary or ""


def usable_product_name(report: OrderReport) -> str | None:
    """可供后续课程检索绑定的商品名；繁忙 / 失败报告不算。"""
    name = (report.product_name or "").strip()
    if not name or name in {BUSY_PRODUCT}:
        return None
    if report.current_status in {"busy", "error", "not_found"}:
        return None
    return name


def usable_course_code(report: OrderReport) -> str | None:
    """可供后续课程检索绑定的课程编码；繁忙 / 失败报告不算。"""
    code = (report.course_code or "").strip()
    if not code:
        return None
    if report.current_status in {"busy", "error", "not_found"}:
        return None
    if (report.product_name or "").strip() in {BUSY_PRODUCT}:
        return None
    return code


def _busy_report(order_no: str | None, exc: BaseException) -> OrderReport:
    return OrderReport(
        order_no=order_no or "unknown",
        product_name=BUSY_PRODUCT,
        order_type="",
        current_status="busy",
        progress_percent=0,
        amount=0,
        steps=[],
        visualization="",
        summary=f"当前查询订单请求较集中，建议 1 分钟后重试。（{exc}）",
        oss_url=None,
    )


def _tool_json(error: str, hint: str) -> str:
    return json.dumps({"error": error, "hint": hint}, ensure_ascii=False)


def _resolve_order_number(
    query: str,
    order_no: str | None,
    *,
    user_id: str,
    last_order_no: str | None,
    require_explicit_order_no: bool,
) -> str | None:
    if require_explicit_order_no:
        return (extract_order_no(order_no or "") or (order_no or "").strip()) or None
    return resolve_query_order_no(
        query,
        order_no=order_no,
        user_id=user_id,
        last_order_no=last_order_no,
    )


def _handle_order_transient_failure(
    order_no: str | None,
    *,
    tenant_id: str,
    user_id: str,
    exc: BaseException | None = None,
    busy: bool = False,
) -> OrderReport:
    number = (order_no or "").strip() or "unknown"
    cached = get_cached_order(number, tenant_id=tenant_id, user_id=user_id)
    if cached:
        age = max(0.0, time.time() - cached.cached_at)
        return annotate_stale_order(cached.payload, age_seconds=age)
    raise_if_escalated(escalation_for_transient(TOOL_ORDER, number))
    if busy:
        inc_tool_call("order_query", "busy", tenant_id)
        return _busy_report(order_no, exc or Exception("busy"))
    inc_tool_call("order_query", "timeout", tenant_id)
    raise TimeoutError("订单查询超时，请稍后重试") from None


def _persist_success_cache(
    report: OrderReport,
    *,
    tenant_id: str,
    user_id: str,
) -> None:
    if not is_cacheable_order_report(report):
        return
    put_cached_order(report, tenant_id=tenant_id, user_id=user_id)


def query_order(
    query: str = "",
    order_no: str | None = None,
    *,
    tenant_id: str = "demo",
    user_id: str = "anonymous",
    last_order_no: str | None = None,
    persist: bool = True,
    require_explicit_order_no: bool = False,
    use_order_pool: bool = True,
) -> OrderReport:
    """查单统一入口：解析单号 → 鉴权查库 → 可选 OSS 归档。

    Graph / Supervisor / HTTP 走这里。``require_explicit_order_no`` 时不从
    ``query`` 里刮单号（ReACT tool 契约：空参就报缺单号，禁止用问句凑）。

    ReACT 工具路径应设 ``persist=False``、``use_order_pool=False``：
    LangGraph 已把同步 tool 卸到线程，避免再进 ``run_order`` 双层池。
    """
    resolved = _resolve_order_number(
        query,
        order_no,
        user_id=user_id,
        last_order_no=last_order_no,
        require_explicit_order_no=require_explicit_order_no,
    )
    args = (
        query,
        order_no,
        tenant_id,
        user_id,
        last_order_no,
        persist,
        require_explicit_order_no,
    )
    try:
        if use_order_pool:
            report = run_order(_query_order_inner, *args)
        else:
            report = _query_order_inner(*args)
    except ToolPoolBusyError as exc:
        return _handle_order_transient_failure(
            resolved or order_no or last_order_no,
            tenant_id=tenant_id,
            user_id=user_id,
            exc=exc,
            busy=True,
        )
    except TimeoutError:
        return _handle_order_transient_failure(
            resolved or order_no or last_order_no,
            tenant_id=tenant_id,
            user_id=user_id,
        )
    _persist_success_cache(report, tenant_id=tenant_id, user_id=user_id)
    return report


def cache_state_updates_for_order(report: OrderReport) -> dict[str, Any]:
    """成功查单后写入会话 structured 槽位。"""
    return session_updates_for_order(report)


def _query_order_inner(
    query: str,
    order_no: str | None,
    tenant_id: str,
    user_id: str,
    last_order_no: str | None,
    persist: bool,
    require_explicit_order_no: bool,
) -> OrderReport:
    if require_explicit_order_no:
        number = (extract_order_no(order_no or "") or (order_no or "").strip()) or None
        if not number:
            inc_tool_call("order_query", "error", tenant_id)
            raise ValueError(MISSING_ORDER_NO)
    else:
        number = resolve_query_order_no(
            query,
            order_no=order_no,
            user_id=user_id,
            last_order_no=last_order_no,
        )
    report = fetch_order_report(number, tenant_id=tenant_id, query=query or None, user_id=user_id)
    if persist:
        report = persist_order_report(report, tenant_id=tenant_id)
    return report


def order_to_tool_observation(
    order_no: str,
    query: str = "",
    *,
    tenant_id: str | None = None,
    user_id: str | None = None,
) -> str:
    """ReACT 工具观察：成功返回与热路径一致的 Markdown；失败返回 JSON error/hint。"""
    tenant = tenant_id or get_current_tenant()
    uid = user_id or get_current_user()
    try:
        report = query_order(
            query,
            order_no,
            tenant_id=tenant,
            user_id=uid,
            require_explicit_order_no=True,
            persist=False,
            use_order_pool=False,
        )
    except ToolEscalationError:
        raise
    except ValueError as exc:
        message = str(exc)
        if MISSING_ORDER_NO in message:
            return _tool_json(
                MISSING_ORDER_NO,
                "请向用户确认完整订单编号，不要编造",
            )
        return _tool_json(message, "请核对订单编号，不要编造新单号")
    except TimeoutError:
        return _tool_json("订单查询超时", "请稍后重试，不要连续重复调用")
    except Exception as exc:
        inc_tool_call("order_query", "error", tenant)
        return _tool_json("订单查询失败", str(exc)[:120])
    if report.current_status == "busy":
        return _tool_json("订单服务繁忙", "请求队列已满，请稍后再试，不要连续重复调用")
    return format_order_answer(report)


@tool(parse_docstring=True)
def tool_order_query(order_no: str, query: str = "") -> str:
    """查询订单/退款/支付进度并生成可视化报告。

    order_no 必填。用户消息、对话历史或「关联订单号」中都没有订单号时：
    不要调用本工具，先向用户询问完整订单编号；严禁编造、猜测或用手机号/日期充数。

    Args:
        order_no: 已确认的订单号，纯数字至少 8 位，例如 20251114001。没有订单号时不要调用。
        query: 用户原话，可选。用于判断是否在问开课/开通时间。
    """
    return order_to_tool_observation(order_no, query)


class OrderCapability:
    """订单能力：Graph / Supervisor / HTTP / ReACT 共用 ``query_order``。"""

    spec = CapabilitySpec(
        id="order_query",
        tool_name="tool_order_query",
        description="按订单号查询支付/开通/退款进度，并生成可视化报告。",
        accepts=("order_no",),
        provides=("product_name", "order_no", "course_code"),
    )

    def bind(self, step: Any, ctx: ExecutionContext, observations: dict[str, Any]) -> dict[str, Any] | None:
        del step, observations
        return {
            "order_no": ctx.order_no,
            "last_order_no": ctx.last_order_no or ctx.order_no,
            "query": ctx.query,
            "persist": ctx.persist,
        }

    async def ainvoke(self, ctx: ExecutionContext, **kwargs: Any) -> CapabilityResult:
        try:
            report = await asyncio.to_thread(
                query_order,
                kwargs.get("query") or ctx.query,
                kwargs.get("order_no") or ctx.order_no,
                tenant_id=ctx.tenant_id,
                user_id=ctx.user_id,
                last_order_no=kwargs.get("last_order_no") or ctx.last_order_no or ctx.order_no,
                persist=bool(kwargs.get("persist", ctx.persist)),
            )
        except ToolEscalationError as exc:
            return CapabilityResult(
                capability_id=self.spec.id,
                ok=False,
                text=exc.decision.message,
                error="escalated",
                escalated=True,
            )
        except (ValueError, TimeoutError) as exc:
            return CapabilityResult(
                capability_id=self.spec.id,
                ok=False,
                text=str(exc),
                error=str(exc),
            )
        observations: dict[str, Any] = {}
        product = usable_product_name(report)
        if product:
            observations["product_name"] = product
        code = usable_course_code(report)
        if code:
            observations["course_code"] = code
        if report.order_no:
            observations["order_no"] = report.order_no
        observations["_cache_state_updates"] = session_updates_for_order(report)
        return CapabilityResult(
            capability_id=self.spec.id,
            ok=report.current_status not in {"busy", "error", "not_found"},
            text=format_order_answer(report),
            data=report,
            observations=observations,
        )

    def as_langchain_tool(self) -> Any:
        return tool_order_query


from app.capabilities.registry import register as _register

_register(OrderCapability())
