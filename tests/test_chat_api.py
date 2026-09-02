import json

from fastapi.testclient import TestClient
from httpx import Response

from app.app import app
from app.graph import GRAPH_MERMAID_PATH, build_graph, graph_mermaid


def parse_sse(response: Response) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in response.text.split("\n\n"):
        event = "message"
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if not data_lines:
            continue
        events.append((event, json.loads("\n".join(data_lines))))
    return events


def sse_done(response: Response) -> dict:
    dones = [data for event, data in parse_sse(response) if event == "done"]
    assert dones, response.text
    return dones[-1]


def test_chat_endpoint() -> None:
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "Python入门课包含哪些内容？"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    events = parse_sse(response)
    names = [event for event, _ in events]
    assert "meta" in names
    assert "status" in names
    assert "done" in names
    payload = sse_done(response)
    assert payload["blocked"] is False
    assert "Python入门课" in payload["answer"]


def test_chat_compat_json() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/chat", json={"message": "转人工"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "handoff"
    assert "人工" in payload["answer"]


def test_greet_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/greet", headers={"X-Tenant-Id": "demo"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["greeting"]
    assert len(payload["options"]) >= 3
    labels = {item["label"] for item in payload["options"]}
    assert "课程咨询" in labels
    assert "转人工" in labels


def test_build_graph_exports_mermaid() -> None:
    compiled = build_graph()
    mermaid = graph_mermaid(compiled)
    for node in ("security", "router", "rag", "tool", "supervisor", "chitchat", "handoff", "respond"):
        assert node in mermaid
    assert "course_tool" not in mermaid
    assert GRAPH_MERMAID_PATH.exists()
    assert "security" in GRAPH_MERMAID_PATH.read_text(encoding="utf-8")
    assert "supervisor" in GRAPH_MERMAID_PATH.read_text(encoding="utf-8")


def test_chat_order_respects_user_header() -> None:
    with TestClient(app) as client:
        owner = client.post(
            "/chat",
            json={"message": "查询订单#20251114001的退款进度"},
            headers={"X-Tenant-Id": "demo", "X-User-Id": "u10001"},
        )
        other = client.post(
            "/chat",
            json={"message": "查询订单#20251114001的退款进度"},
            headers={"X-Tenant-Id": "demo", "X-User-Id": "alice"},
        )
        anon = client.post(
            "/chat",
            json={"message": "查询订单#20251114001的退款进度"},
            headers={"X-Tenant-Id": "demo"},
        )
    assert owner.status_code == 200
    assert "退款" in sse_done(owner)["answer"]
    assert other.status_code == 200
    assert "未找到订单" in sse_done(other)["answer"]
    assert anon.status_code == 200
    assert "退款" in sse_done(anon)["answer"]


def test_chat_user_header_overrides_body() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "查询订单#20251114001的退款进度", "user_id": "u10001"},
            headers={"X-Tenant-Id": "demo", "X-User-Id": "alice"},
        )
    assert response.status_code == 200
    assert "未找到订单" in sse_done(response)["answer"]


def test_graph_page() -> None:
    with TestClient(app) as client:
        html = client.get("/graph")
        source = client.get("/graph.mmd")
    assert html.status_code == 200
    assert "mermaid" in html.text
    assert source.status_code == 200
    payload = source.json()["mermaid"]
    assert "router" in payload
    assert "handoff" in payload
    assert "course_tool" not in payload


def test_chat_session_lock_returns_409() -> None:
    import threading
    import time

    from app.graph import make_thread_id, reset_graph
    from app.session_cache import DictSessionLock

    lock = DictSessionLock(wait_seconds=0.05)
    session_id = "http-lock-session"
    reset_graph(session_lock=lock)
    held = threading.Event()

    def _hold() -> None:
        with lock.hold(make_thread_id("demo", session_id)):
            held.set()
            time.sleep(0.4)

    worker = threading.Thread(target=_hold)
    worker.start()
    try:
        assert held.wait(timeout=1)
        with TestClient(app) as client:
            response = client.post(
                "/chat",
                json={"message": "Python入门课包含哪些内容？", "session_id": session_id},
                headers={"X-Tenant-Id": "demo"},
            )
        assert response.status_code == 409
        assert "同一会话" in response.json()["detail"]
    finally:
        worker.join()
        reset_graph()
