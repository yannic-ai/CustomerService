"""Tool Agent：订单能力的薄适配器（兼容旧 import）。"""

from app.capabilities.order import format_order_answer, query_order
from app.schemas import OrderReport

handle_order_query = query_order

__all__ = ["format_order_answer", "handle_order_query", "query_order", "OrderReport"]
