"""查单 / 课程检索工具成功率与 Agent 跑飞指标。"""

from __future__ import annotations

import time

import pytest

from app.agents.rag import retrieve_course_docs
from app.services.order import fetch_order_report
from app.observability import render_prometheus, reset_metrics
from app.observability.metrics import maybe_inc_runaway_llm_calls


def test_order_tool_ok_metric() -> None:
    reset_metrics()
    report = fetch_order_report("20251114001", tenant_id="demo")
    assert report.order_no == "20251114001"
    body = render_prometheus()
    assert 'cs_tool_calls_total{outcome="ok",tenant="demo",tool="order_query"} 1' in body


def test_order_tool_not_found_metric() -> None:
    reset_metrics()
    with pytest.raises(ValueError, match="未找到订单"):
        fetch_order_report("99999999999", tenant_id="demo")
    body = render_prometheus()
    assert 'cs_tool_calls_total{outcome="not_found",tenant="demo",tool="order_query"} 1' in body


def test_order_tool_deny_metric() -> None:
    reset_metrics()
    with pytest.raises(ValueError, match="未找到订单"):
        fetch_order_report("20251114001", tenant_id="demo", user_id="alice")
    body = render_prometheus()
    assert 'cs_tool_calls_total{outcome="deny",tenant="demo",tool="order_query"} 1' in body


def test_course_retrieve_ok_metric() -> None:
    reset_metrics()
    docs = retrieve_course_docs("Python入门课包含哪些内容？", tenant_id="demo")
    assert docs
    body = render_prometheus()
    assert 'cs_tool_calls_total{outcome="ok",tenant="demo",tool="course_retrieve"} 1' in body


def test_course_retrieve_not_found_metric() -> None:
    reset_metrics()
    docs = retrieve_course_docs("今天天气怎么样", tenant_id="demo")
    assert docs == []
    body = render_prometheus()
    assert (
        'cs_tool_calls_total{outcome="not_found",tenant="demo",tool="course_retrieve"} 1'
        in body
    )


def test_order_tool_timeout_metric(monkeypatch) -> None:
    from app.agents import tool as order_tool
    from app.config import get_settings
    from app.executors import run_order
    from app.tool_cache import _MemoryToolCacheStore, reset_tool_cache_store

    reset_metrics()
    reset_tool_cache_store(_MemoryToolCacheStore())
    settings = get_settings().model_copy(
        update={"order_tool_timeout_seconds": 0.05, "order_tool_workers": 2}
    )
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    monkeypatch.setattr("app.executors.get_settings", lambda: settings)

    def _slow() -> str:
        time.sleep(1.0)
        return "ok"

    with pytest.raises(TimeoutError):
        run_order(_slow)

    # query_order 路径也应记 timeout
    def _slow_handle(*_args, **_kwargs):
        time.sleep(1.0)
        raise AssertionError("should not finish")

    monkeypatch.setattr("app.capabilities.order._query_order_inner", _slow_handle)
    with pytest.raises(TimeoutError, match="超时"):
        order_tool.handle_order_query("查询订单#20251114001", tenant_id="demo")

    body = render_prometheus()
    assert 'outcome="timeout"' in body
    assert 'tool="order_query"' in body


def test_agent_runaway_high_llm_calls(monkeypatch) -> None:
    from app.config import get_settings

    reset_metrics()
    settings = get_settings().model_copy(update={"agent_runaway_llm_calls": 3})
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    maybe_inc_runaway_llm_calls(3)
    assert "cs_agent_runaway_total" not in render_prometheus()
    maybe_inc_runaway_llm_calls(4)
    body = render_prometheus()
    assert 'cs_agent_runaway_total{reason="high_llm_calls"} 1' in body


def test_react_limit_also_counts_runaway() -> None:
    from langchain_core.messages import AIMessage

    from app.agents.react import ObservingToolCallLimitMiddleware

    reset_metrics()
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
        ]
    }
    middleware.after_model(state, runtime=None)
    body = render_prometheus()
    assert "cs_react_limit_hit_total 1" in body
    assert 'cs_agent_runaway_total{reason="react_limit"} 1' in body
