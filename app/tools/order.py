"""兼容层：旧路径 ``app.tools.order`` → ``app.services.order``。"""

from app.services.order import (
    AccessContext,
    OrderAccessDecision,
    OrderDenyReason,
    authorize_order_access,
    can_access_order,
    ensure_order_access,
    extract_order_no,
    fetch_order_report,
    make_order_scope_filters,
    normalize_user_id,
    persist_order_report,
    resolve_query_order_no,
)

__all__ = [
    "AccessContext",
    "OrderAccessDecision",
    "OrderDenyReason",
    "authorize_order_access",
    "can_access_order",
    "ensure_order_access",
    "extract_order_no",
    "fetch_order_report",
    "make_order_scope_filters",
    "normalize_user_id",
    "persist_order_report",
    "resolve_query_order_no",
]
