import json
from pathlib import Path

import pytest

from app.config import get_settings
from app.rag.embeddings import (
    EmbeddingIndexMismatchError,
    current_embedding_spec,
    get_embeddings,
)
from app.rag.vectorstore import (
    assert_embedding_compatible,
    build_faiss_index,
    faiss_index_status,
    ingest_indexes,
    reset_indexes,
)


def test_test_runtime_uses_ngram() -> None:
    spec = current_embedding_spec()
    assert spec["backend"] == "ngram"
    assert spec["dim"] == 384


def test_ingest_writes_embedding_meta() -> None:
    settings = get_settings()
    meta_path = Path(settings.faiss_index_path) / "embedding.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["backend"] == "ngram"
    assert payload["model"] == "ngram"
    assert payload["dim"] == 384


def test_index_rejects_dimension_change() -> None:
    settings = get_settings()
    original_dim = settings.embedding_dim
    settings.embedding_dim = 512
    get_embeddings.cache_clear()
    try:
        with pytest.raises(EmbeddingIndexMismatchError, match="ingest"):
            assert_embedding_compatible(settings.faiss_index_path)
        assert faiss_index_status(settings.faiss_index_path) == "mismatch"
    finally:
        settings.embedding_dim = original_dim
        get_embeddings.cache_clear()


def test_index_rejects_backend_meta_change() -> None:
    settings = get_settings()
    meta_path = Path(settings.faiss_index_path) / "embedding.json"
    original = meta_path.read_text(encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {"backend": "huggingface", "model": "BAAI/bge-small-zh-v1.5", "dim": 512},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    try:
        with pytest.raises(EmbeddingIndexMismatchError, match="huggingface"):
            assert_embedding_compatible(settings.faiss_index_path)
        assert faiss_index_status(settings.faiss_index_path) == "mismatch"
    finally:
        meta_path.write_text(original, encoding="utf-8")


def test_ingest_without_rebuild_flag_rebuilds_on_mismatch() -> None:
    settings = get_settings()
    original_dim = settings.embedding_dim
    settings.embedding_dim = 256
    get_embeddings.cache_clear()
    reset_indexes()
    try:
        ingest_indexes(persist=True)
        spec = current_embedding_spec()
        assert spec["dim"] == 256
        assert faiss_index_status(settings.faiss_index_path) == "ok"
        payload = json.loads(
            (Path(settings.faiss_index_path) / "embedding.json").read_text(encoding="utf-8")
        )
        assert payload["dim"] == 256
    finally:
        settings.embedding_dim = original_dim
        get_embeddings.cache_clear()
        reset_indexes()
        build_faiss_index(persist=True)


def test_rebuild_after_dim_change_updates_meta() -> None:
    settings = get_settings()
    original_dim = settings.embedding_dim
    settings.embedding_dim = 256
    get_embeddings.cache_clear()
    reset_indexes()
    try:
        build_faiss_index(persist=True)
        spec = current_embedding_spec()
        assert spec["dim"] == 256
        assert faiss_index_status(settings.faiss_index_path) == "ok"
    finally:
        settings.embedding_dim = original_dim
        get_embeddings.cache_clear()
        reset_indexes()
        build_faiss_index(persist=True)
