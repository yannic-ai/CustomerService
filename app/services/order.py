"""订单领域服务：解析订单号、查库并生成进度可视化报告。"""

from __future__ import annotations

from datetime import datetime

from app.db.models import Order
from app.memory import is_persistent_user
from app.oss.client import get_oss
from app.schemas import OrderProgressStep, OrderReport
from app.security.order_access import (  # noqa: F401  对外重导出
    AccessContext,
    OrderAccessDecision,
    OrderDenyReason,
    authorize as authorize_order_access,
    can_access_order,
    ensure_access as ensure_order_access,
    make_order_scope_filters,
)
from app.tenancy import get_current_user

STATUS_LABELS = {
    "paid": "已支付",
    "active": "已开通",
    "refund_reviewing": "退款审核中",
    "refunded": "已退款",
    "cancelled": "已取消",
}

MERMAID_STATUS = {
    "completed": "✓",
    "in_progress": "●",
    "pending": "○",
    "rejected": "✗",
}


def extract_order_no(text: str) -> str | None:
    """从自然语言中提取订单号，例如 #20251114001。"""
    import re

    match = re.search(r"(?:订单|#|NO\.?)?\s*(\d{8,})", text, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _progress_percent(steps: list[OrderProgressStep]) -> int:
    """按节点完成情况估算进度百分比。"""
    if not steps:
        return 0
    score = 0.0
    unit = 100 / len(steps)
    for step in steps:
        if step.status == "completed":
            score += unit
        elif step.status == "in_progress":
            score += unit * 0.5
    return round(score)


def _ascii_bar(percent: int) -> str:
    """生成 10 格 ASCII 进度条。"""
    filled = round(percent / 10)
    return "█" * filled + "░" * (10 - filled)


def _mermaid(steps: list[OrderProgressStep]) -> str:
    """把进度节点渲染为 Mermaid flowchart。"""
    nodes = []
    edges = []
    for index, step in enumerate(steps):
        mark = MERMAID_STATUS.get(step.status, "?")
        nodes.append(f'    S{index}["{step.name} {mark}"]')
        if index > 0:
            edges.append(f"    S{index - 1} --> S{index}")
    return "flowchart LR\n" + "\n".join(nodes + edges)


def _visualization(report: OrderReport) -> str:
    """拼接 Markdown 报告：进度条 + Mermaid + 明细表。"""
    lines = [
        f"**订单 {report.order_no} · {report.product_name}**",
        f"当前状态：{report.current_status}　进度：{_ascii_bar(report.progress_percent)} {report.progress_percent}%",
        "",
        "```mermaid",
        _mermaid(report.steps),
        "```",
        "",
        "| 节点 | 状态 | 时间 | 说明 |",
        "|---|---|---|---|",
    ]
    status_cn = {
        "completed": "已完成",
        "in_progress": "进行中",
        "pending": "待处理",
        "rejected": "已拒绝",
    }
    for step in report.steps:
        lines.append(
            f"| {step.name} | {status_cn.get(step.status, step.status)} | {step.timestamp or '-'} | {step.note or '-'} |"
        )
    return "\n".join(lines)


OPEN_COURSE_HINTS = ("开课", "上课", "开通")


def _open_course_summary(report: OrderReport) -> str | None:
    """若有「开通课程」已完成节点，返回开课时间摘要。"""
    for step in report.steps:
        if "开通" in step.name and step.status == "completed" and step.timestamp:
            return (
                f"订单 {report.order_no} 关联课程「{report.product_name}」"
                f"已于 {step.timestamp} 开通。"
                + (f"（{step.note}）" if step.note else "")
            )
    return None


def _apply_query_summary(report: OrderReport, query: str | None) -> OrderReport:
    """按用户问句调整摘要：开课/上课/开通时优先突出开通时间。"""
    if not query:
        return report
    if any(token in query for token in OPEN_COURSE_HINTS):
        open_summary = _open_course_summary(report)
        if open_summary:
            report.summary = open_summary
    return report


def normalize_user_id(user_id: str | None) -> str:
    """空值回落为 anonymous。"""
    return (user_id or "").strip() or "anonymous"


def _order_tool_outcome(reason: OrderDenyReason | None) -> str:
    """权限拒绝码 → 查单工具 outcome（未找到 vs 拒权）。"""
    if reason in {OrderDenyReason.ORDER_NOT_FOUND, OrderDenyReason.CROSS_TENANT}:
        return "not_found"
    return "deny"


def _order_scope(order_no: str, tenant_id: str, user_id: str | None):
    """**兼容层**：已改为调用统一入口的 make_order_scope_filters。"""
    return make_order_scope_filters(order_no, tenant_id, user_id)


def resolve_query_order_no(
    query: str,
    order_no: str | None = None,
    user_id: str | None = None,
    last_order_no: str | None = None,
) -> str:
    """解析本轮订单号。匿名必须本轮给出单号或使用本会话已绑定的单号，禁止拿他人槽位回填。"""
    extracted = extract_order_no(query)
    number = (extracted or order_no or "").strip() or None
    if not number:
        raise ValueError("未识别到订单号，请提供类似 #20251114001 的订单编号。")

    uid = normalize_user_id(user_id)
    if is_persistent_user(uid):
        return number

    bound = (last_order_no or "").strip() or None
    mentioned = extracted == number or number in (query or "")
    if mentioned or (bound and number == bound):
        return number
    raise ValueError("请先提供订单编号以绑定后再查询。")


def fetch_order_report(
    order_no: str,
    tenant_id: str = "demo",
    query: str | None = None,
    user_id: str | None = None,
) -> OrderReport:
    """按租户查询订单并组装可视化报告。权限+查库一体，失败抛 ValueError。"""
    from app.observability import inc_tool_call

    uid = normalize_user_id(user_id if user_id is not None else get_current_user())
    decision = authorize_order_access(
        order_no,
        context=AccessContext.DIRECT_QUERY,
        tenant_id=tenant_id,
        user_id=uid,
    )
    if not decision.allowed:
        inc_tool_call("order_query", _order_tool_outcome(decision.reason_code), tenant_id)
        raise ValueError(f"未找到订单 {order_no}")

    order = decision.order
    assert order is not None

    steps = [
        OrderProgressStep(
            name=event.name,
            status=event.status,  # type: ignore[arg-type]
            timestamp=event.occurred_at.strftime("%Y-%m-%d %H:%M") if event.occurred_at else None,
            note=event.note,
        )
        for event in order.events
    ]
    report = OrderReport(
        order_no=order.order_no,
        product_name=order.course.name,
        course_code=order.course.code or "",
        order_type=order.order_type,
        current_status=STATUS_LABELS.get(order.status, order.status),
        progress_percent=_progress_percent(steps),
        amount=order.amount,
        steps=steps,
    )
    report.visualization = _visualization(report)
    if order.status == "refund_reviewing":
        report.summary = (
            f"订单 {order.order_no} 已完成支付与课程开通，退款申请已受理，"
            "目前处于财务审核阶段，通过后将原路退回。"
        )
    else:
        report.summary = f"订单 {order.order_no} 当前状态为{report.current_status}。"
    inc_tool_call("order_query", "ok", tenant_id)
    return _apply_query_summary(report, query)


def persist_order_report(report: OrderReport, tenant_id: str = "demo") -> OrderReport:
    """把报告写入 OSS（或本地目录），并回填 `oss_url`。"""
    filename = f"reports/{report.order_no}-{datetime.now().strftime('%Y%m%d%H%M%S')}.md"
    report.oss_url = get_oss().put_text(filename, report.visualization, tenant_id=tenant_id)
    return report
