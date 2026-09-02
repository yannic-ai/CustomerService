import asyncio

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

from app.app import app
from app.config import get_settings
from app.observability import reset_metrics
from app.schemas import CourseConsultResult
from tests.test_chat_api import parse_sse, sse_done


def test_chat_course_offline_has_no_token_events(monkeypatch) -> None:
    real = get_settings()
    offline = real.model_copy(update={"deepseek_api_key": ""})
    monkeypatch.setattr("app.agents.rag.get_settings", lambda: offline)
    monkeypatch.setattr("app.agents.intent.get_settings", lambda: offline)
    monkeypatch.setattr("app.graph.nodes.get_settings", lambda: offline)
    monkeypatch.setattr("app.graph.runner.get_settings", lambda: offline)
    monkeypatch.setattr("app.llm.get_settings", lambda: offline)
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "Python入门课包含哪些内容？"})
    assert response.status_code == 200
    events = parse_sse(response)
    names = [event for event, _ in events]
    assert "meta" in names
    assert "status" in names
    assert "done" in names
    assert "token" not in names
    payload = sse_done(response)
    assert "Python入门课" in payload["answer"]
    assert payload.get("usage") is None


def test_chat_token_stream_when_rag_streams(monkeypatch) -> None:
    tokens = ["## Python", "入门课", "\n摘要"]

    async def fake_astream_course_answer(*args, **kwargs):
        del args, kwargs
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
        for piece in tokens:
            writer({"event": "token", "content": piece})
        result = CourseConsultResult(
            course_name="Python入门课",
            summary="摘要",
            sources=["python-intro.md"],
        )
        return "".join(tokens), result

    reset_metrics()
    monkeypatch.setattr("app.graph.nodes.astream_course_answer", fake_astream_course_answer)
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "Python入门课包含哪些内容？"})
        assert response.status_code == 200
        events = parse_sse(response)
        token_events = [data for event, data in events if event == "token"]
        assert [item["content"] for item in token_events] == tokens
        payload = sse_done(response)
        assert payload["answer"] == "".join(tokens)
        body = client.get("/metrics").text
    assert "cs_ttft_seconds_count" in body
    assert 'intent="course_consult"' in body


def test_stream_course_answer_emits_tokens(monkeypatch) -> None:
    from app.agents import rag as rag_mod
    from app.llm_gateway import begin_turn_usage, take_turn_usage, wrap_chat_model
    from app.llm_gateway.stores import MemoryGatewayStore, set_gateway_store
    from app.llm_gateway.types import CallPolicy
    from app.tenancy import set_current_tenant

    class StreamDummy:
        async def astream(self, payload, config=None, **kwargs):
            del payload, config, kwargs
            yield AIMessageChunk(content="模块")
            yield AIMessageChunk(
                content="一",
                usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            )

    store = MemoryGatewayStore()
    set_gateway_store(store)
    set_current_tenant("demo")
    begin_turn_usage()
    real = get_settings()
    monkeypatch.setattr(
        "app.agents.rag.get_settings",
        lambda: real.model_copy(update={"deepseek_api_key": "test-key"}),
    )
    monkeypatch.setattr(
        rag_mod,
        "retrieve_course_docs",
        lambda *a, **k: [
            Document(
                page_content="课程名称：Python入门课\n模块1 环境",
                metadata={"source": "python-intro.md"},
            )
        ],
    )
    monkeypatch.setattr(
        "app.llm.get_chat_model",
        lambda **kwargs: wrap_chat_model(
            StreamDummy(),
            CallPolicy(usage_tag="rag", cache_enabled=False, model="dummy"),
        ),
    )

    emitted: list[str] = []
    monkeypatch.setattr(rag_mod, "_emit_token", lambda content: emitted.append(content))
    answer, structured = asyncio.run(rag_mod.stream_course_answer("Python入门课"))
    assert answer == "模块一"
    assert structured.course_name == "Python入门课"
    assert structured.modules == []
    assert structured.learning_path == []
    assert structured.sources == ["python-intro.md"]
    assert emitted == ["模块", "一"]
    turn = take_turn_usage()
    assert turn.total_tokens == 5
    assert len(turn.steps) == 1
    set_gateway_store(None)
