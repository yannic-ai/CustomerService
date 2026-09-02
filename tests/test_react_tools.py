from types import SimpleNamespace

from langchain.messages import ToolMessage

from app.agents.react import tool_order_query
from app.security.middleware import SecurityAgentMiddleware
from app.tenancy import set_current_tenant


def test_wrap_tool_call_converts_value_error_to_tool_message() -> None:
    middleware = SecurityAgentMiddleware(user_id="u1", tenant_id="demo")
    request = SimpleNamespace(tool_call={"name": "tool_order_query", "id": "call_1"})

    def boom(_request):
        raise ValueError("未找到订单 20251114001")

    result = middleware.wrap_tool_call(request, boom)
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "Traceback" not in result.content
    assert "未找到订单" in result.content
    assert any(event["event"] == "middleware_tool_error" for event in middleware.callback.events)
    assert any(event["event"] == "middleware_tool_call" for event in middleware.callback.events)


def test_wrap_tool_call_masks_pii_in_tool_result() -> None:
    middleware = SecurityAgentMiddleware()
    request = SimpleNamespace(tool_call={"name": "tool_order_query", "id": "call_2"})

    def leak(_request):
        return ToolMessage(
            content="联系 13812345678",
            name="tool_order_query",
            tool_call_id="call_2",
        )

    result = middleware.wrap_tool_call(request, leak)
    assert "[PHONE_MASKED]" in result.content
    assert "13812345678" not in result.content


def test_wrap_tool_call_retries_timeout_once() -> None:
    middleware = SecurityAgentMiddleware()
    request = SimpleNamespace(tool_call={"name": "tool_order_query", "id": "call_3"})
    calls = {"n": 0}

    def flaky(_request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("timed out")
        return ToolMessage(content="ok", name="tool_order_query", tool_call_id="call_3")

    result = middleware.wrap_tool_call(request, flaky)
    assert result.content == "ok"
    assert calls["n"] == 2


def test_tool_order_query_schema_requires_order_no() -> None:
    schema = tool_order_query.tool_call_schema.model_json_schema()
    assert "order_no" in schema.get("required", [])
    assert "question" not in schema.get("properties", {})
    description = f"{schema.get('description') or ''} {tool_order_query.description}"
    assert "不要编造" in description or "严禁编造" in description
    order_no_desc = (schema.get("properties") or {}).get("order_no", {}).get("description") or ""
    assert "订单号" in order_no_desc


def test_tool_order_query_missing_order_no() -> None:
    assert "未识别到订单号" in tool_order_query.invoke(
        {"order_no": "", "query": "帮我查一下进度"}
    )


def test_tool_order_query_does_not_scrape_order_no_from_query() -> None:
    payload = tool_order_query.invoke(
        {"order_no": "", "query": "查询订单#20251114001的退款进度"}
    )
    assert "未识别到订单号" in payload


def test_tool_order_query_missing_order_returns_hint_json() -> None:
    set_current_tenant("acme")
    payload = tool_order_query.invoke({"order_no": "20251114001"})
    assert "未找到订单" in payload or "error" in payload
    assert "hint" in payload

def test_arun_react_offline_fallback_runs_async(monkeypatch) -> None:
    """无 LLM 时 arun_react 退化到 _offline_react，异步包装可正常运行。"""
    import asyncio

    from app.agents import react as react_mod
    from app.agents.react import arun_react

    monkeypatch.setattr(react_mod, "build_react_agent", lambda **kwargs: None)
    answer = asyncio.run(arun_react("Python入门课包含哪些内容", tenant_id="demo"))
    assert answer
    assert "Python" in answer or "模块" in answer


def test_arun_react_falls_back_when_llm_raises(monkeypatch) -> None:
    """Key 已配但 ReACT 调用失败时，同样走离线订单+检索。"""
    import asyncio

    from app.agents import react as react_mod
    from app.agents.react import arun_react

    class Boom:
        async def ainvoke(self, *args, **kwargs):
            del args, kwargs
            raise TimeoutError("upstream down")

    monkeypatch.setattr(react_mod, "build_react_agent", lambda **kwargs: Boom())
    answer = asyncio.run(arun_react("Python入门课包含哪些内容", tenant_id="demo"))
    assert answer
    assert "Python" in answer or "模块" in answer or "暂时无法" in answer
