"""条件式 Query Rewrite 单测。"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from app.agents.rag import _prepare_search_query
from app.config import get_settings
from app.rag.rewrite import maybe_rewrite_query


def test_rewrite_abstract_difficulty() -> None:
    result = maybe_rewrite_query(
        "Python入门课 难吗",
        original_query="难吗",
        last_course_name="Python入门课",
    )
    assert result.rewritten is True
    assert result.reason == "abstract"
    assert "课程难度" in result.query


def test_rewrite_skips_clear_long_query() -> None:
    query = "Python入门课包含哪些内容？"
    result = maybe_rewrite_query(query, original_query=query)
    assert result.rewritten is False
    assert result.query == query


def test_prepare_search_query_followup_with_slot(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "rag_rewrite_enabled", True)
    search_query = _prepare_search_query(
        "这个怎么样",
        None,
        tenant_id="demo",
        last_course_name="Python入门课",
        last_course_query="Python入门课包含哪些内容？",
    )
    assert "Python" in search_query


def test_prepare_search_query_rewrite_disabled(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "rag_rewrite_enabled", False)
    history = [HumanMessage(content="机器学习导论讲什么")]
    search_query = _prepare_search_query(
        "难吗",
        history,
        tenant_id="demo",
        last_course_name="机器学习基础课",
    )
    assert "课程难度" not in search_query
    assert "机器学习" in search_query
