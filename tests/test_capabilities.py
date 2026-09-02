"""能力注册表：HTTP / 图 / Supervisor / ReACT 共用同一入口。"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage

from app.capabilities import get_capability, langchain_tools, list_capabilities, tool_name_for
from app.capabilities.course import retrieve_course_knowledge
from app.capabilities.order import query_order, tool_order_query
from app.schemas import CourseConsultResult
from app.tool_context import tool_context_scope
from app.tenancy import set_current_tenant, set_current_user


def test_builtin_capabilities_registered() -> None:
    ids = {cap.spec.id for cap in list_capabilities()}
    assert ids == {"order_query", "course_retrieve"}
    assert tool_name_for("order_query") == "tool_order_query"
    assert tool_name_for("course_retrieve") == "retrieve_course_knowledge"
    assert get_capability("order_query").spec.provides == ("product_name", "order_no", "course_code")
    assert "product_name" in get_capability("course_retrieve").spec.accepts
    assert "course_code" in get_capability("course_retrieve").spec.accepts


def test_langchain_tools_come_from_registry() -> None:
    names = {getattr(tool, "name", None) for tool in langchain_tools()}
    assert names == {"tool_order_query", "retrieve_course_knowledge"}
    assert tool_order_query is get_capability("order_query").as_langchain_tool()


def test_query_order_and_react_tool_share_lookup() -> None:
    set_current_tenant("demo")
    set_current_user("u10001")
    report = query_order("查询订单#20251114001", tenant_id="demo", user_id="u10001")
    assert report.order_no == "20251114001"
    payload = tool_order_query.invoke({"order_no": "20251114001", "query": "退款进度"})
    assert "20251114001" in payload
    assert "Python" in payload or "退款" in payload or "支付" in payload


def test_react_tool_returns_markdown_on_success() -> None:
    set_current_tenant("demo")
    set_current_user("u10001")
    payload = tool_order_query.invoke({"order_no": "20251114001", "query": "退款进度"})
    assert "20251114001" in payload
    assert "mermaid" in payload or "进度" in payload or "Python" in payload
    assert '"order_no"' not in payload


def test_react_tool_error_still_json() -> None:
    set_current_tenant("acme")
    payload = tool_order_query.invoke({"order_no": "20251114001"})
    assert "未找到订单" in payload
    assert "hint" in payload

def test_new_capability_appears_in_registry_and_react_tools() -> None:
    from langchain.tools import tool

    from app.capabilities import register, unregister
    from app.capabilities.types import CapabilityResult, CapabilitySpec, ExecutionContext

    @tool
    def tool_invoice_query(invoice_no: str) -> str:
        """查询发票。"""
        return invoice_no

    class InvoiceCapability:
        spec = CapabilitySpec(
            id="invoice_query",
            tool_name="tool_invoice_query",
            description="查询发票",
            accepts=("invoice_no",),
            provides=("invoice_no",),
        )

        def bind(self, step, ctx: ExecutionContext, observations):
            del step, observations
            return {"invoice_no": ctx.query}

        async def ainvoke(self, ctx: ExecutionContext, **kwargs):
            del ctx, kwargs
            return CapabilityResult(capability_id="invoice_query", text="ok")

        def as_langchain_tool(self):
            return tool_invoice_query

    register(InvoiceCapability())
    try:
        assert get_capability("invoice_query").spec.tool_name == "tool_invoice_query"
        assert tool_name_for("invoice_query") == "tool_invoice_query"
        names = {getattr(item, "name", None) for item in langchain_tools()}
        assert "tool_invoice_query" in names
    finally:
        unregister("invoice_query")
    assert "invoice_query" not in {cap.spec.id for cap in list_capabilities()}


def test_retrieve_course_knowledge_reads_tool_context(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_retrieve_course(query: str, **kwargs):
        captured["query"] = query
        captured.update(kwargs)
        return CourseConsultResult(
            course_name="Python入门课",
            summary="ok",
            modules=[],
        )

    monkeypatch.setattr("app.capabilities.course.retrieve_course", fake_retrieve_course)

    history = [HumanMessage(content="Python入门课怎么样")]
    with tool_context_scope(
        history=history,
        last_course_query="Python入门课",
        last_course_name="Python入门课",
        session_summary="用户咨询 Python 入门课",
    ):
        payload = asyncio.run(
            retrieve_course_knowledge.ainvoke({"question": "有哪些模块"})
        )

    assert payload == "ok"
    assert captured["query"] == "有哪些模块"
    assert captured["history"] == history
    assert captured["last_course_query"] == "Python入门课"
    assert captured["last_course_name"] == "Python入门课"
    assert captured["session_summary"] == "用户咨询 Python 入门课"


def test_react_tool_skips_oss_persist(monkeypatch) -> None:
    calls = {"n": 0}

    def fake_persist(report, tenant_id: str = "demo"):
        calls["n"] += 1
        return report

    monkeypatch.setattr("app.capabilities.order.persist_order_report", fake_persist)
    set_current_tenant("demo")
    set_current_user("u10001")
    tool_order_query.invoke({"order_no": "20251114001", "query": "退款进度"})
    assert calls["n"] == 0


def test_react_tool_bypasses_order_pool(monkeypatch) -> None:
    def boom(*args, **kwargs):
        del args, kwargs
        raise AssertionError("run_order should not be called from ReACT tool")

    monkeypatch.setattr("app.capabilities.order.run_order", boom)
    set_current_tenant("demo")
    set_current_user("u10001")
    payload = tool_order_query.invoke({"order_no": "20251114001"})
    assert "20251114001" in payload


def test_legacy_tools_order_shim_reexports_service() -> None:
    from app.services.order import extract_order_no as from_service
    from app.tools.order import extract_order_no as from_shim

    assert from_shim is from_service
    assert from_shim("#20251114001") == "20251114001"


def test_hot_path_query_order_still_persists(monkeypatch) -> None:
    calls = {"n": 0}

    def fake_persist(report, tenant_id: str = "demo"):
        calls["n"] += 1
        report.oss_url = "oss://test"
        return report

    monkeypatch.setattr("app.capabilities.order.persist_order_report", fake_persist)
    set_current_tenant("demo")
    set_current_user("u10001")
    report = query_order("查询订单#20251114001", tenant_id="demo", user_id="u10001")
    assert calls["n"] == 1
    assert report.oss_url == "oss://test"
