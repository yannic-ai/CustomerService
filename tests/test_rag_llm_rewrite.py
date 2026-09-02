"""LLM/规则多 query 分解单测。"""

from __future__ import annotations

from app.config import get_settings
from app.rag.llm_rewrite import decompose_retrieval_queries


def test_rule_decompose_splits_complex_query() -> None:
    settings = get_settings()
    original = settings.rag_llm_rewrite_enabled
    settings.rag_llm_rewrite_enabled = False
    try:
        queries = decompose_retrieval_queries("Python入门课包含哪些模块和大纲内容")
        assert len(queries) >= 2
        joined = " ".join(queries)
        assert "模块" in joined
        assert "大纲" in joined
    finally:
        settings.rag_llm_rewrite_enabled = original


def test_short_query_not_decomposed() -> None:
    queries = decompose_retrieval_queries("难吗")
    assert queries == ["难吗"]


def test_disabled_llm_rewrite_keeps_single_query(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "rag_llm_rewrite_enabled", False)
    queries = decompose_retrieval_queries("机器学习导论讲什么")
    assert queries == ["机器学习导论讲什么"]
