"""管理面 ingest API 单测。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.app import app
from app.config import get_settings


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_ingest_api_disabled_without_token(client: TestClient, monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_ingest_token", "")
    response = client.post("/admin/knowledge/ingest")
    assert response.status_code == 503


def test_ingest_api_rejects_bad_token(client: TestClient, monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_ingest_token", "secret-token")
    response = client.post(
        "/admin/knowledge/ingest",
        headers={"X-Admin-Token": "wrong"},
    )
    assert response.status_code == 401


def test_ingest_api_accepts_background_job(client: TestClient, monkeypatch, tmp_path: Path) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_ingest_token", "secret-token")
    calls: list[bool] = []
    monkeypatch.setattr(
        "app.rag.ingest_api.ingest_indexes",
        lambda *, persist, rebuild: calls.append(rebuild) or {},
    )
    response = client.post(
        "/admin/knowledge/ingest",
        headers={"X-Admin-Token": "secret-token"},
        data={"tenant_id": "demo", "rebuild": "false"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "accepted"
    assert payload["tenant_id"] == "demo"
    assert calls == [False]


def test_ingest_api_uploads_markdown(client: TestClient, monkeypatch, tmp_path: Path) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_ingest_token", "secret-token")
    monkeypatch.setattr(settings, "faiss_index_path", str(tmp_path / "faiss"))
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    monkeypatch.setattr("app.rag.ingest_api.KNOWLEDGE_DIR", knowledge)
    monkeypatch.setattr(
        "app.rag.ingest_api.ingest_indexes",
        lambda *, persist, rebuild: {},
    )
    response = client.post(
        "/admin/knowledge/ingest",
        headers={"X-Admin-Token": "secret-token"},
        data={"tenant_id": "demo"},
        files={"file": ("upload.md", b"# demo\n\n## section\n\n- hello\n", "text/markdown")},
    )
    assert response.status_code == 200
    saved = knowledge / "upload.md"
    assert saved.exists()
    assert "hello" in saved.read_text(encoding="utf-8")
