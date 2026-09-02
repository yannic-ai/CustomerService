"""质量闭环：用户赞踩、Prompt 热更新 fallback、自动 score 扩展。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.app import app
from app.observability import render_prometheus, reset_metrics
from app.observability.langfuse import (
    PROMPT_INTENT,
    get_text_prompt,
    score_turn,
    score_user_feedback,
)


def test_get_text_prompt_falls_back_without_langfuse(monkeypatch) -> None:
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "langfuse_public_key", "")
    monkeypatch.setattr(settings, "langfuse_secret_key", "")
    assert get_text_prompt(PROMPT_INTENT, "本地意图提示词") == "本地意图提示词"


def test_get_text_prompt_uses_langfuse_compile(monkeypatch) -> None:
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-test")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk-test")
    monkeypatch.setattr(settings, "langfuse_host", "http://127.0.0.1:3000")

    class FakePrompt:
        def compile(self) -> str:
            return "远程意图提示词"

    class FakeClient:
        def get_prompt(self, name, **kwargs):
            assert name == PROMPT_INTENT
            assert kwargs.get("fallback") == "本地"
            return FakePrompt()

    monkeypatch.setattr("app.observability.langfuse._ensure_env", lambda: True)
    monkeypatch.setattr("langfuse.get_client", lambda: FakeClient())
    assert get_text_prompt(PROMPT_INTENT, "本地") == "远程意图提示词"


def test_score_user_feedback_increments_prometheus(monkeypatch) -> None:
    reset_metrics()
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "langfuse_public_key", "")
    monkeypatch.setattr(settings, "langfuse_secret_key", "")
    result = score_user_feedback(
        trace_id="abc",
        value=1,
        request_id="req-1",
        session_id="sess-1",
    )
    assert result["recorded"] is True
    assert result["langfuse"] is False
    body = render_prometheus()
    assert 'cs_user_feedback_total{value="up"} 1' in body


def test_score_turn_writes_all_auto_scores(monkeypatch) -> None:
    calls: list[tuple[str, float]] = []

    class FakeClient:
        def score_current_trace(self, *, name, value, **_kwargs):
            calls.append((name, value))

    monkeypatch.setattr("app.observability.langfuse._ensure_env", lambda: True)
    monkeypatch.setattr("langfuse.get_client", lambda: FakeClient())
    score_turn(blocked=False, intent="course_consult", rag_empty=True, outcome="ok")
    names = {name: value for name, value in calls}
    assert names["blocked"] == 0.0
    assert names["handoff"] == 0.0
    assert names["rag_empty"] == 1.0
    assert names["turn_ok"] == 1.0


def test_feedback_api_accepts_thumbs(monkeypatch) -> None:
    reset_metrics()
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "langfuse_public_key", "")
    monkeypatch.setattr(settings, "langfuse_secret_key", "")
    with TestClient(app) as client:
        res = client.post(
            "/feedback",
            json={"trace_id": "t1", "value": 0, "request_id": "r1", "session_id": "s1"},
        )
    assert res.status_code == 200
    payload = res.json()
    assert payload["recorded"] is True
    assert 'cs_user_feedback_total{value="down"} 1' in client.get("/metrics").text


def test_feedback_api_rejects_invalid_value() -> None:
    with TestClient(app) as client:
        res = client.post("/feedback", json={"trace_id": "t1", "value": 2})
    assert res.status_code == 422
