import asyncio

from fastapi.testclient import TestClient

import pytest

from app.agents.rag import consult_course
from app.app import app
from app.concurrency import achat
from app.tenancy import normalize_tenant_id, resolve_request_user
from app.services.order import fetch_order_report


def _chat(message, **kwargs):
    return asyncio.run(achat(message, **kwargs))


def test_normalize_tenant_defaults() -> None:
    assert normalize_tenant_id(None) == "demo"
    assert normalize_tenant_id("") == "demo"
    assert normalize_tenant_id("acme") == "acme"


def test_resolve_request_user_prefers_header() -> None:
    assert resolve_request_user("alice", "u10001") == "alice"
    assert resolve_request_user(None, "u10001") == "u10001"
    assert resolve_request_user("", "  ") == "anonymous"


def test_normalize_tenant_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        normalize_tenant_id("bad tenant!")


def test_order_isolated_by_tenant() -> None:
    report = fetch_order_report("20251114001", tenant_id="demo")
    assert report.product_name == "Python入门课"
    with pytest.raises(ValueError, match="未找到订单"):
        fetch_order_report("20251114001", tenant_id="acme")


def test_chat_order_isolated_by_tenant() -> None:
    demo = _chat("查询订单#20251114001的退款进度", tenant_id="demo")
    assert demo.blocked is False
    assert demo.intent == "order"
    assert "20251114001" in demo.answer

    acme = _chat("查询订单#20251114001的退款进度", tenant_id="acme")
    assert acme.blocked is False
    assert "未找到订单" in acme.answer


def test_course_rag_isolated_by_tenant() -> None:
    demo = asyncio.run(consult_course("Python入门课包含哪些内容？", tenant_id="demo"))
    assert demo.course_name == "Python入门课"
    assert any("python-intro" in s for s in demo.sources) or "Python" in demo.summary

    acme_python = asyncio.run(consult_course("Python入门课包含哪些内容？", tenant_id="acme"))
    # acme 知识库无 Python 大纲，不应返回 demo 的课程文件
    assert all("python-intro" not in s for s in (acme_python.sources or []))
    assert acme_python.course_name != "Python入门课" or "Acme" in acme_python.summary

    acme = asyncio.run(consult_course("Acme 入职培训包含哪些内容？", tenant_id="acme"))
    assert "Acme" in acme.course_name or "入职" in (acme.summary or "")
    assert all("python-intro" not in s for s in (acme.sources or []))


def test_api_order_requires_matching_tenant() -> None:
    with TestClient(app) as client:
        ok = client.get(
            "/api/v1/orders/20251114001",
            headers={"X-Tenant-Id": "demo"},
        )
        owner = client.get(
            "/api/v1/orders/20251114001",
            headers={"X-Tenant-Id": "demo", "X-User-Id": "u10001"},
        )
        forbidden = client.get(
            "/api/v1/orders/20251114001",
            headers={"X-Tenant-Id": "demo", "X-User-Id": "alice"},
        )
        missing = client.get(
            "/api/v1/orders/20251114001",
            headers={"X-Tenant-Id": "acme"},
        )
        defaulted = client.get("/api/v1/orders/20251114001")
    assert ok.status_code == 200
    assert owner.status_code == 200
    assert forbidden.status_code == 404
    assert missing.status_code == 404
    assert defaulted.status_code == 200
