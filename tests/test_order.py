import asyncio

import pytest

from app.agents.intent import recognize_intent
from app.services.order import extract_order_no, fetch_order_report, resolve_query_order_no


def test_extract_order_no() -> None:
    assert extract_order_no("查询订单#20251114001的退款进度") == "20251114001"


def test_fetch_refund_progress() -> None:
    report = fetch_order_report("20251114001", tenant_id="demo")
    assert report.product_name == "Python入门课"
    assert report.course_code == "python-intro"
    assert report.current_status == "退款审核中"
    assert report.progress_percent > 0
    assert "退款审核" in report.visualization


def test_fetch_opencourse_summary() -> None:
    report = fetch_order_report(
        "20251114001",
        tenant_id="demo",
        query="查询订单 20251114001 什么时候开课？",
    )
    assert "2025-11-14 10:22" in report.summary
    assert "开通" in report.summary


def test_owner_can_fetch_order() -> None:
    report = fetch_order_report("20251114001", tenant_id="demo", user_id="u10001")
    assert report.order_no == "20251114001"
    assert report.current_status == "退款审核中"


def test_other_user_cannot_fetch_order() -> None:
    with pytest.raises(ValueError, match="未找到订单"):
        fetch_order_report("20251114001", tenant_id="demo", user_id="alice")


def test_anonymous_must_bind_order_no() -> None:
    with pytest.raises(ValueError, match="请先提供订单编号"):
        resolve_query_order_no("那退款呢", order_no="20251114001", user_id="anonymous")


def test_anonymous_bind_from_query_or_slot() -> None:
    assert (
        resolve_query_order_no(
            "查询订单#20251114001的退款进度",
            user_id="anonymous",
        )
        == "20251114001"
    )
    assert (
        resolve_query_order_no(
            "那退款呢",
            order_no="20251114001",
            user_id="anonymous",
            last_order_no="20251114001",
        )
        == "20251114001"
    )


def test_anonymous_cannot_use_unbound_other_order() -> None:
    with pytest.raises(ValueError, match="请先提供订单编号"):
        resolve_query_order_no(
            "那退款呢",
            order_no="99999999999",
            user_id="anonymous",
            last_order_no="20251114001",
        )


def test_keyword_intent_course_and_order() -> None:
    course = asyncio.run(recognize_intent("Python入门课包含哪些内容？"))
    assert course.intent == "course_consult"
    order = asyncio.run(recognize_intent("查询订单#20251114001的退款进度"))
    assert order.intent == "order"
    assert order.order_no == "20251114001"
