"""工具失败降级：缓存回退与自动转人工。"""

from __future__ import annotations

import pytest

from app.agents.react import ObservingToolCallLimitMiddleware
from app.capabilities.order import query_order
from app.config import get_settings
from app.executors import ToolPoolBusyError
from app.observability.metrics import render_prometheus, reset_metrics
from app.schemas import OrderProgressStep, OrderReport
from app.tool_cache import (
    _MemoryToolCacheStore,
    get_cached_order,
    put_cached_order,
    reset_tool_cache_store,
    session_tool_cache_scope,
    session_updates_for_order,
)
from app.tool_escalation import ToolEscalationError, escalation_for_transient, tool_failure_tracker_scope


def _sample_report(order_no: str = "20251114001") -> OrderReport:
    return OrderReport(
        order_no=order_no,
        product_name="Python入门课",
        course_code="PY101",
        order_type="course",
        current_status="已支付",
        progress_percent=80,
        amount=199.0,
        steps=[OrderProgressStep(name="支付", status="completed")],
        visualization="进度图",
        summary=f"订单 {order_no} 当前状态为已支付。",
    )


def test_session_cache_hit_on_order_timeout(monkeypatch) -> None:
    reset_tool_cache_store(_MemoryToolCacheStore())
    report = _sample_report()
    session_state = session_updates_for_order(report)
    monkeypatch.setattr(
        "app.services.order.can_access_order",
        lambda order_no, tenant_id, user_id: True,
    )
    monkeypatch.setattr(
        "app.capabilities.order.run_order",
        lambda fn, *args, **kwargs: (_ for _ in ()).throw(TimeoutError("timeout")),
    )

    with session_tool_cache_scope(session_state), tool_failure_tracker_scope():
        result = query_order(
            "查进度",
            report.order_no,
            tenant_id="demo",
            user_id="u1",
            use_order_pool=True,
        )

    assert "分钟前的查询结果" in result.summary
    assert result.order_no == report.order_no


def test_redis_cache_hit_on_busy(monkeypatch) -> None:
    store = _MemoryToolCacheStore()
    reset_tool_cache_store(store)
    report = _sample_report()
    put_cached_order(report, tenant_id="demo", user_id="u1")
    monkeypatch.setattr(
        "app.services.order.can_access_order",
        lambda order_no, tenant_id, user_id: True,
    )
    monkeypatch.setattr(
        "app.capabilities.order.run_order",
        lambda fn, *args, **kwargs: (_ for _ in ()).throw(
            ToolPoolBusyError("order pool busy")
        ),
    )

    with tool_failure_tracker_scope():
        result = query_order(
            "查进度",
            report.order_no,
            tenant_id="demo",
            user_id="u1",
            use_order_pool=True,
        )

    assert "分钟前的查询结果" in result.summary
    cached = get_cached_order(report.order_no, tenant_id="demo", user_id="u1")
    assert cached is not None
    assert cached.source == "redis"


def test_not_found_does_not_use_cache(monkeypatch) -> None:
    reset_tool_cache_store(_MemoryToolCacheStore())
    report = _sample_report()
    session_state = session_updates_for_order(report)

    def _deny(*args, **kwargs):
        raise ValueError("未找到订单 99999999")

    monkeypatch.setattr("app.capabilities.order._query_order_inner", _deny)
    monkeypatch.setattr(
        "app.capabilities.order.run_order",
        lambda fn, *args, **kwargs: fn(*args),
    )

    with session_tool_cache_scope(session_state), tool_failure_tracker_scope():
        with pytest.raises(ValueError, match="未找到订单"):
            query_order(
                "查进度",
                "99999999",
                tenant_id="demo",
                user_id="u1",
                use_order_pool=True,
            )


def test_transient_failures_escalate_without_cache(monkeypatch) -> None:
    reset_tool_cache_store(_MemoryToolCacheStore())
    monkeypatch.setattr(
        "app.capabilities.order.run_order",
        lambda fn, *args, **kwargs: (_ for _ in ()).throw(TimeoutError("timeout")),
    )

    with tool_failure_tracker_scope():
        with pytest.raises(TimeoutError):
            query_order("查", "20251114001", tenant_id="demo", user_id="u1", use_order_pool=True)
        with pytest.raises(ToolEscalationError):
            query_order("查", "20251114001", tenant_id="demo", user_id="u1", use_order_pool=True)


def test_react_limit_middleware_auto_handoff(monkeypatch) -> None:
    from langchain_core.messages import AIMessage

    reset_metrics()
    monkeypatch.setattr(get_settings(), "tool_escalation_on_react_limit", True)
    middleware = ObservingToolCallLimitMiddleware(run_limit=1, exit_behavior="continue")
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "tool_order_query", "args": {"order_no": "1"}, "id": "c1"},
                    {"name": "tool_order_query", "args": {"order_no": "2"}, "id": "c2"},
                ],
            )
        ],
        "run_tool_call_count": {"__all__": 1},
        "thread_tool_call_count": {"__all__": 1},
    }
    result = middleware.after_model(state, runtime=None)
    assert isinstance(result, dict)
    assert result.get("jump_to") == "end"
    assert "转接人工客服" in str(result["messages"][-1].content)
    body = render_prometheus()
    assert "cs_tool_escalation_total" in body


def test_cache_denies_cross_user_session(monkeypatch) -> None:
    reset_tool_cache_store(_MemoryToolCacheStore())
    report = _sample_report()
    session_state = session_updates_for_order(report)
    monkeypatch.setattr(
        "app.services.order.can_access_order",
        lambda order_no, tenant_id, user_id: user_id == "owner",
    )
    monkeypatch.setattr(
        "app.capabilities.order.run_order",
        lambda fn, *args, **kwargs: (_ for _ in ()).throw(TimeoutError("timeout")),
    )

    with session_tool_cache_scope(session_state), tool_failure_tracker_scope():
        with pytest.raises(TimeoutError):
            query_order(
                "查",
                report.order_no,
                tenant_id="demo",
                user_id="other",
                use_order_pool=True,
            )
        with pytest.raises(ToolEscalationError):
            query_order(
                "查",
                report.order_no,
                tenant_id="demo",
                user_id="other",
                use_order_pool=True,
            )


def test_escalation_counter_threshold(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "tool_escalation_transient_threshold", 2)
    with tool_failure_tracker_scope():
        assert escalation_for_transient("order_query", "1") is None
        decision = escalation_for_transient("order_query", "1")
        assert decision is not None
        assert decision.escalate is True
