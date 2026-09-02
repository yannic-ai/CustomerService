from fastapi.testclient import TestClient

from app.app import app
from app.observability import REQUEST_ID_HEADER, reset_metrics
from tests.test_chat_api import parse_sse, sse_done


def test_health_is_liveness() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_checks_db_and_index() -> None:
    with TestClient(app) as client:
        response = client.get("/ready")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["checks"]["db"] == "ok"
    assert payload["checks"]["faiss"] == "ok"
    assert payload["checks"]["redis"] in {"ok", "skipped"}


def test_request_id_echoed_in_header_and_sse() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "转人工"},
            headers={REQUEST_ID_HEADER: "req-eval-001"},
        )
    assert response.status_code == 200
    assert response.headers.get(REQUEST_ID_HEADER) == "req-eval-001"
    events = parse_sse(response)
    meta = [data for event, data in events if event == "meta"]
    assert meta
    assert meta[0]["request_id"] == "req-eval-001"
    assert sse_done(response)["intent"] == "handoff"


def test_metrics_after_chat(monkeypatch) -> None:
    from app.config import get_settings

    real = get_settings()
    offline = real.model_copy(update={"deepseek_api_key": ""})
    monkeypatch.setattr("app.agents.rag.get_settings", lambda: offline)
    monkeypatch.setattr("app.agents.intent.get_settings", lambda: offline)
    monkeypatch.setattr("app.graph.nodes.get_settings", lambda: offline)
    monkeypatch.setattr("app.graph.runner.get_settings", lambda: offline)
    monkeypatch.setattr("app.llm.get_settings", lambda: offline)

    reset_metrics()
    with TestClient(app) as client:
        chat = client.post("/chat", json={"message": "Python入门课包含哪些内容？"})
        assert chat.status_code == 200
        body = client.get("/metrics").text
    assert "cs_http_requests_total" in body
    assert 'path="/chat"' in body
    assert "cs_chat_requests_total" in body
    assert 'intent="course_consult"' in body
    assert "cs_chat_in_flight" in body
    assert 'cs_node_duration_seconds_count{node="security"}' in body
    assert 'cs_node_duration_seconds_count{node="router"}' in body
    assert 'cs_node_duration_seconds_count{node="rag"}' in body
    assert "cs_llm_calls_per_turn_count" in body
    # 离线课程路径不推 token，不应记 TTFT
    assert "cs_ttft_seconds_count" not in body
    assert 'cs_tool_calls_total{outcome="ok",tenant="demo",tool="course_retrieve"}' in body


def test_chitchat_and_handoff_nodes_timed(monkeypatch) -> None:
    """闲聊 / 转人工也走 _timed，应出现在 cs_node_duration_seconds。"""
    from app.config import get_settings

    real = get_settings()
    offline = real.model_copy(update={"deepseek_api_key": ""})
    monkeypatch.setattr("app.agents.intent.get_settings", lambda: offline)
    monkeypatch.setattr("app.graph.nodes.get_settings", lambda: offline)
    monkeypatch.setattr("app.graph.runner.get_settings", lambda: offline)
    monkeypatch.setattr("app.llm.get_settings", lambda: offline)

    reset_metrics()
    with TestClient(app) as client:
        handoff = client.post("/chat", json={"message": "转人工"})
        assert handoff.status_code == 200
        assert sse_done(handoff)["intent"] == "handoff"
        chitchat = client.post("/chat", json={"message": "你好"})
        assert chitchat.status_code == 200
        body = client.get("/metrics").text
    assert 'cs_node_duration_seconds_count{node="handoff"}' in body
    assert 'cs_node_duration_seconds_count{node="chitchat"}' in body
    assert 'cs_node_duration_seconds_count{node="respond"}' in body


def test_react_limit_middleware_increments_on_block() -> None:
    from langchain_core.messages import AIMessage

    from app.agents.react import ObservingToolCallLimitMiddleware
    from app.observability.metrics import render_prometheus

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
        ],
        "run_tool_call_count": {"__all__": 1},
        "thread_tool_call_count": {"__all__": 1},
    }
    result = middleware.after_model(state, runtime=None)
    assert isinstance(result, dict)
    assert result.get("messages")
    assert result.get("jump_to") == "end"
    assert any(isinstance(m, AIMessage) for m in result["messages"])
    body = render_prometheus()
    assert "cs_react_limit_hit_total" in body
    assert "cs_react_limit_hit_total 1" in body


def test_react_limit_middleware_no_increment_without_block() -> None:
    from langchain_core.messages import AIMessage

    from app.agents.react import ObservingToolCallLimitMiddleware
    from app.observability.metrics import render_prometheus

    reset_metrics()
    middleware = ObservingToolCallLimitMiddleware(run_limit=8, exit_behavior="continue")
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "tool_order_query", "args": {"order_no": "1"}, "id": "c1"},
                ],
            )
        ],
        "run_tool_call_count": {},
        "thread_tool_call_count": {},
    }
    result = middleware.after_model(state, runtime=None)
    assert isinstance(result, dict)
    assert "messages" not in result
    body = render_prometheus()
    assert "cs_react_limit_hit_total" not in body or "cs_react_limit_hit_total 0" in body
