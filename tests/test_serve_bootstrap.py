"""serve / chat 启动门禁：不隐式 ingest，缺索引 fail-fast。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import get_settings
from app.rag.embeddings import get_embeddings
from app.rag.vectorstore import (
    IndexUnavailableError,
    build_faiss_index,
    get_indexes,
    require_index_for_serve,
    reset_indexes,
)


def test_get_indexes_missing_does_not_rebuild(tmp_path: Path) -> None:
    settings = get_settings()
    original = settings.faiss_index_path
    empty = tmp_path / "missing-faiss"
    settings.faiss_index_path = str(empty)
    reset_indexes()
    try:
        with pytest.raises(IndexUnavailableError, match="ingest"):
            get_indexes()
        assert not empty.exists() or not any(empty.rglob("index.faiss"))
    finally:
        settings.faiss_index_path = original
        reset_indexes()
        build_faiss_index(persist=True)


def test_get_indexes_mismatch_does_not_rebuild() -> None:
    settings = get_settings()
    original_dim = settings.embedding_dim
    settings.embedding_dim = 256
    get_embeddings.cache_clear()
    reset_indexes()
    try:
        with pytest.raises(IndexUnavailableError, match="ingest"):
            get_indexes()
        assert (Path(settings.faiss_index_path) / "embedding.json").is_file()
    finally:
        settings.embedding_dim = original_dim
        get_embeddings.cache_clear()
        reset_indexes()
        build_faiss_index(persist=True)


def test_require_index_for_serve_ok() -> None:
    require_index_for_serve()


def test_require_index_for_serve_missing(tmp_path: Path) -> None:
    settings = get_settings()
    original = settings.faiss_index_path
    settings.faiss_index_path = str(tmp_path / "no-index")
    try:
        with pytest.raises(IndexUnavailableError, match="不存在"):
            require_index_for_serve()
    finally:
        settings.faiss_index_path = original


def test_seed_does_not_ingest(monkeypatch, capsys) -> None:
    calls: list[object] = []
    monkeypatch.setattr("app.cli.ingest_indexes", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr("app.cli._ensure_db", lambda: None)
    from app.cli import main

    main(["seed"])
    assert calls == []
    assert "ingest" in capsys.readouterr().out


def test_serve_does_not_ingest(monkeypatch) -> None:
    ingest_calls: list[object] = []
    monkeypatch.setattr("app.cli.ingest_indexes", lambda *a, **k: ingest_calls.append(1))
    monkeypatch.setattr("app.session_cache.require_redis_for_serve", lambda: None)
    monkeypatch.setattr("app.cli.require_index_for_serve", lambda: None)
    started: dict[str, object] = {}

    def fake_run(app, **kwargs):
        started["app"] = app
        started.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    from app.cli import main

    main(["serve"])
    assert ingest_calls == []
    assert started.get("app") == "app.app:app"


def test_serve_exits_when_index_missing(monkeypatch) -> None:
    monkeypatch.setattr("app.session_cache.require_redis_for_serve", lambda: None)
    monkeypatch.setattr(
        "app.cli.require_index_for_serve",
        lambda: (_ for _ in ()).throw(IndexUnavailableError("FAISS 索引不存在")),
    )
    from app.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["serve"])
    assert exc.value.code == 1


def test_chat_exits_when_index_missing(monkeypatch) -> None:
    monkeypatch.setattr("app.cli._ensure_db", lambda: None)
    monkeypatch.setattr(
        "app.cli.require_index_for_serve",
        lambda: (_ for _ in ()).throw(IndexUnavailableError("FAISS 索引不存在")),
    )
    from app.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["chat", "你好"])
    assert exc.value.code == 1
