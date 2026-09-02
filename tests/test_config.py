"""Settings 字段范围与交叉校验。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings


def _settings(**overrides) -> Settings:
    """构造 Settings，忽略 .env，避免本机环境干扰单测。"""
    return Settings(_env_file=None, **overrides)


def test_settings_defaults_valid() -> None:
    settings = _settings()
    assert settings.app_port == 8000
    assert settings.embedding_backend == "huggingface"
    assert settings.log_level == "INFO"
    assert settings.log_format == "json"
    assert settings.llm_circuit_failure_threshold == 3
    assert settings.llm_circuit_cooldown_seconds == 30


def test_app_port_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        _settings(app_port=0)


def test_discard_score_cannot_exceed_summarize_score() -> None:
    with pytest.raises(ValidationError, match="context_discard_score"):
        _settings(context_discard_score=0.8, context_summarize_score=0.3)


def test_always_keep_recent_cannot_exceed_max_messages() -> None:
    with pytest.raises(ValidationError, match="context_always_keep_recent"):
        _settings(context_always_keep_recent=100, history_max_messages=50)


def test_embedding_backend_aliases_normalize() -> None:
    assert _settings(embedding_backend="HF").embedding_backend == "huggingface"
    assert _settings(embedding_backend="bge").embedding_backend == "huggingface"
    assert _settings(embedding_backend="hash").embedding_backend == "ngram"


def test_log_format_normalizes_case() -> None:
    assert _settings(log_format="JSON").log_format == "json"
    assert _settings(log_level="debug").log_level == "DEBUG"


def test_app_port_env_out_of_range(monkeypatch) -> None:
    monkeypatch.setenv("APP_PORT", "99999")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_get_settings_returns_singleton() -> None:
    """冒烟：单例可读取且端口合法（不 cache_clear，避免破坏会话级离线隔离）。"""
    settings = get_settings()
    assert settings.app_port >= 1
    assert get_settings() is settings


def test_rag_rerank_top_k_cannot_exceed_fetch_k() -> None:
    with pytest.raises(ValidationError, match="rag_rerank_top_k"):
        _settings(rag_rerank_top_k=30, rag_rerank_fetch_k=10)


def test_rag_fetch_k_min_cannot_exceed_max() -> None:
    with pytest.raises(ValidationError, match="rag_fetch_k_min"):
        _settings(rag_fetch_k_min=40, rag_fetch_k_max=20)


def test_rag_defaults_include_p0_fields() -> None:
    settings = _settings()
    assert settings.rag_rerank_enabled is False
    assert settings.rag_dynamic_k_enabled is True
    assert settings.rag_fetch_k_min == 12
    assert settings.rag_fetch_k_max == 32
    assert settings.rag_rewrite_enabled is True
    assert settings.rag_eval_harvest_enabled is True
    assert settings.rag_eval_harvest_dir == "evals/pending"


def test_settings_groups_by_theme() -> None:
    """容量 / LLM / 记忆字段分属不同 mixin，避免再往单一超大文件堆。"""
    from app.config import CapacitySettings, LLMSettings, MemorySettings, RAGSettings

    settings = _settings()
    assert isinstance(settings, LLMSettings)
    assert isinstance(settings, RAGSettings)
    assert isinstance(settings, CapacitySettings)
    assert isinstance(settings, MemorySettings)
    assert settings.chat_concurrency >= 1
    assert settings.history_token_budget >= 256
