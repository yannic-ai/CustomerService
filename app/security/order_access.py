"""订单权限校验 · 单一权威入口。

⚠️ 禁止在本文件外自行拼订单权限的 WHERE 条件或判断表达式。
所有订单数据访问必须通过本模块的对外 API：
  - authorize()              ：完整决策，拿到 OrderAccessDecision（含 reason_code / order 对象）
  - ensure_access()          ：授权通过返回 Order，失败抛 ValueError（推荐 fetch/report 用）
  - can_access_order()       ：布尔式兼容旧代码（新代码请勿用）
  - make_order_scope_filters()：复杂 SQL 需要 WHERE 层过滤时用（与 _decide_access 规则同源）
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db.models import Order
from app.db.session import session_scope
from app.memory import EPHEMERAL_USERS, is_persistent_user
from app.tenancy import get_current_tenant, get_current_user

logger = logging.getLogger("cs.security.order_access")

ORDER_NO_RE = re.compile(r"^\d{8,}$")


class AccessContext(str, Enum):
    """订单访问发生的上下文场景。影响匿名用户的宽松度判定。"""

    DIRECT_QUERY = "direct_query"
    # 用户本轮明确提及订单号，或 Tool / HTTP API 显式传入 order_no

    SLOT_FOLLOWUP = "slot_followup"
    # 来自会话槽位 last_order_no 的跨轮追问（本轮未重新提单号）

    MEMORY_RECALL = "memory_recall"
    # 来自长期用户记忆的跨会话回填（最严格，匿名永久拒绝）

    REACT_BINDING = "react_binding"
    # ReACT Agent 在推理中把 order_no 绑到上下文（参数来自本轮解析）


class OrderDenyReason(str, Enum):
    """拒绝原因码（仅用于审计日志，不得出现在用户响应中）。"""

    ORDER_NOT_FOUND = "order_not_found"
    # 该租户下不存在此订单号

    CROSS_TENANT = "cross_tenant"
    # 订单存在，但属于其他租户（审计需要区分 vs 纯不存在）

    NOT_ORDER_OWNER = "not_order_owner"
    # 登录用户，本租户有订单，但下单人不是该 user

    ANONYMOUS_UNBOUND = "anonymous_unbound"
    # 匿名用户用 SLOT_FOLLOWUP / MEMORY_RECALL 且不符合会话绑定规则

    MALFORMED_ORDER_NO = "malformed_order_no"
    # 订单号格式非法（长度 < 8 或含非数字字符）


@dataclass(frozen=True)
class OrderAccessDecision:
    allowed: bool
    reason_code: OrderDenyReason | None = None
    order_id: int | None = None
    order: Order | None = None  # 允许时返回已加载对象，避免二次查库


# ============================================================
#  内部工具函数
# ============================================================

def _normalize_user_id(user_id: str | None) -> str:
    """空值回落为 anonymous。重复定义以避免循环 import tools.order。"""
    return (user_id or "").strip() or "anonymous"


def _user_visible_deny(order_no: str) -> str:
    """对用户统一文案：不区分不存在 vs 无权，防存在性枚举。"""
    return f"未找到订单 {order_no}"


# ============================================================
#  核心决策函数 · 唯一事实来源
#  规则（与 make_order_scope_filters 保持严格同源）：
#   ┌───────────┬───────────────────────────────────────────────┐
#   │ 匿名用户  │ DIRECT_QUERY  │ tenant + 单号存在             │
#   │ (anon)   │ SLOT_FOLLOWUP  │ 同 DIRECT（会话已绑定约束由    │
#   │           │                │ authorize() 外层处理）       │
#   │           │ REACT_BINDING  │ 同 DIRECT                     │
#   │           │ MEMORY_RECALL  │ × 永久拒绝                   │
#   ├───────────┼───────────────────────────────────────────────┤
#   │ 登录用户  │ 任何 context   │ tenant + 单号 + user_id=下单人│
#   └───────────┴───────────────────────────────────────────────┘
# ============================================================

def _decide_access(
    order_no: str,
    tenant_id: str,
    user_id: str,
    context: AccessContext,
) -> tuple[bool, OrderDenyReason | None, Order | None]:
    # --- 1. 格式校验 ---
    number = (order_no or "").strip()
    if not ORDER_NO_RE.fullmatch(number):
        return False, OrderDenyReason.MALFORMED_ORDER_NO, None

    uid = _normalize_user_id(user_id)
    anon = (uid in EPHEMERAL_USERS) or (uid == "")

    # --- 2. 匿名 + 长期记忆：永久拒绝 ---
    if anon and context == AccessContext.MEMORY_RECALL:
        return False, OrderDenyReason.ANONYMOUS_UNBOUND, None

    # --- 3. 查单（先不带 user_id，以便审计区分 CROSS_TENANT / NOT_OWNER / NOT_FOUND）---
    with session_scope() as session:
        stmt = (
            select(Order)
            .options(selectinload(Order.course), selectinload(Order.events))
            .where(
                Order.order_no == number,
                Order.tenant_id == tenant_id,
            )
        )
        order = session.scalar(stmt)
        if order is None:
            any_tenant = session.scalar(select(Order.id).where(Order.order_no == number))
            reason = OrderDenyReason.CROSS_TENANT if any_tenant else OrderDenyReason.ORDER_NOT_FOUND
            return False, reason, None

        if not anon and order.user_id != uid:
            return False, OrderDenyReason.NOT_ORDER_OWNER, None

        return True, None, order


# ============================================================
#  审计 + metrics（延迟导入，避免与 observability 循环）
# ============================================================

def _record_deny(
    order_no: str,
    tenant_id: str,
    user_id: str,
    context: AccessContext,
    reason: OrderDenyReason,
) -> None:
    try:
        from app.observability import log_event, inc_order_access_deny
    except Exception:  # observability 未初始化时绝不影响权限判断
        return
    try:
        log_event(
            logger,
            "order_access_denied",
            order_no=order_no,
            tenant_id=tenant_id,
            user_id=user_id,
            context=context.value,
            reason=reason.value,
        )
    except Exception:
        pass
    try:
        inc_order_access_deny(tenant_id, reason.value, context.value)
    except Exception:
        pass


# ============================================================
#  对外 API · 四类场景
# ============================================================

def authorize(
    order_no: str,
    *,
    context: AccessContext | str = AccessContext.DIRECT_QUERY,
    tenant_id: str | None = None,
    user_id: str | None = None,
    session_bound_order_no: str | None = None,
) -> OrderAccessDecision:
    """**统一对外权限入口**。返回完整决策对象（含拒绝原因码与订单对象）。

    Args:
        order_no: 要访问的订单号
        context: 调用上下文场景
        tenant_id: 租户 ID；None 时取当前 tenancy 上下文
        user_id: 用户 ID；None 时取当前 tenancy 上下文
        session_bound_order_no: 当 context=SLOT_FOLLOWUP 时必须传，
            校验请求单号是否等于当前会话已绑定单号（匿名用户防串话）
    """
    ctx = AccessContext(context) if isinstance(context, str) else context
    tid = tenant_id or get_current_tenant()
    uid = _normalize_user_id(user_id or get_current_user())

    # --- 匿名 + 槽位追问：必须匹配会话已绑定单号 ---
    anon = (uid in EPHEMERAL_USERS) or (uid == "")
    if anon and ctx == AccessContext.SLOT_FOLLOWUP:
        bound = (session_bound_order_no or "").strip()
        if bound and order_no.strip() != bound:
            _record_deny(order_no, tid, uid, ctx, OrderDenyReason.ANONYMOUS_UNBOUND)
            return OrderAccessDecision(False, OrderDenyReason.ANONYMOUS_UNBOUND)

    allowed, reason, order = _decide_access(order_no, tid, uid, ctx)

    if not allowed:
        _record_deny(order_no, tid, uid, ctx, reason)  # type: ignore[arg-type]
        return OrderAccessDecision(False, reason)

    return OrderAccessDecision(True, order_id=order.id, order=order)


def ensure_access(
    order_no: str,
    *,
    context: AccessContext | str = AccessContext.DIRECT_QUERY,
    tenant_id: str | None = None,
    user_id: str | None = None,
    session_bound_order_no: str | None = None,
) -> Order:
    """authorize() 的「断言式」便利函数：通过则返回已加载 Order，失败抛 ValueError。

    用于 fetch_order_report / HTTP API 这类后续需要直接使用订单对象的场景——
    一条调用替代「authorize + 判 allowed + 再 session.get(Order)」两步，减少重复查库。
    """
    decision = authorize(
        order_no,
        context=context,
        tenant_id=tenant_id,
        user_id=user_id,
        session_bound_order_no=session_bound_order_no,
    )
    if not decision.allowed:
        raise ValueError(_user_visible_deny(order_no))
    assert decision.order is not None
    return decision.order


def can_access_order(
    order_no: str,
    tenant_id: str,
    user_id: str | None,
    *,
    context: AccessContext | str = AccessContext.DIRECT_QUERY,
) -> bool:
    """**兼容 API**：旧代码 `tools.order.can_access_order(no, tenant, user)` 无修改地工作。

    ⚠️ 新代码请优先使用 `authorize()`，可获得拒绝原因码与 Order 对象。
    """
    return authorize(
        order_no,
        context=context,
        tenant_id=tenant_id,
        user_id=user_id,
    ).allowed


def make_order_scope_filters(
    order_no: str,
    tenant_id: str,
    user_id: str | None,
):
    """**WHERE 过滤器生成器**：规则与 `_decide_access` 严格同源。

    ⚠️ 大多数场景请用 `ensure_access()` 直接拿 Order 对象。
    仅当确实需要手写复杂 JOIN / 聚合 SQL 且必须在 WHERE 层过滤时才调用此函数。
    """
    from app.db.models import Order as _Order

    number = (order_no or "").strip()
    filters = [_Order.order_no == number, _Order.tenant_id == tenant_id]
    uid = _normalize_user_id(user_id)
    if is_persistent_user(uid):
        filters.append(_Order.user_id == uid)
    return filters
