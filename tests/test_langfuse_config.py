from app.config import get_settings
from app.graph import ChatTurn
from app.observability.langfuse import langfuse_callbacks, trace_metadata


def test_langfuse_disabled_returns_empty_callbacks(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "langfuse_public_key", "")
    monkeypatch.setattr(settings, "langfuse_secret_key", "")
    assert settings.langfuse_enabled is False
    assert langfuse_callbacks() == []


def test_langfuse_enabled_adds_handler(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-test")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk-test")
    monkeypatch.setattr(settings, "langfuse_host", "http://127.0.0.1:3000")

    sentinel = object()

    class FakeHandler:
        pass

    def fake_callbacks():
        return [sentinel]

    monkeypatch.setattr("app.graph.runner.langfuse_callbacks", fake_callbacks)
    turn = ChatTurn("你好", "u1", "demo", "sess-1")
    config = turn.run_config()
    assert sentinel in config["callbacks"]
    meta = config["metadata"]
    assert meta["langfuse_user_id"] == "u1"
    assert meta["langfuse_session_id"] == turn.thread_id
    assert "tenant:demo" in meta["langfuse_tags"]
    assert "request_id" in meta


def test_trace_metadata_without_langfuse(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "langfuse_public_key", "")
    monkeypatch.setattr(settings, "langfuse_secret_key", "")
    meta = trace_metadata(
        user_id="u1",
        tenant_id="demo",
        session_id="s1",
        thread_id="demo:u1:s1",
    )
    assert "langfuse_user_id" not in meta
    assert meta["tenant_id"] == "demo"
