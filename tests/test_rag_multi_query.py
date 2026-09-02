"""多 query 检索融合单测。"""

from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.rag.llm_rewrite import decompose_retrieval_queries
from app.rag.vectorstore import ingest_indexes, reset_indexes, search_tenant_documents


def test_multi_query_search_fuses_subqueries(tmp_path: Path) -> None:
    settings = get_settings()
    original = settings.faiss_index_path
    settings.faiss_index_path = str(tmp_path / "faiss")
    reset_indexes()
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "alpha.md").write_text(
        "# alpha\n\n## 模块A\n\n- 专名MODULE_A_ONLY\n\n## 模块B\n\n- 专名MODULE_B_ONLY\n",
        encoding="utf-8",
    )
    try:
        ingest_indexes(persist=True, knowledge_dir=knowledge)
        subqueries = decompose_retrieval_queries("MODULE_A_ONLY 以及 MODULE_B_ONLY 模块内容")
        assert len(subqueries) >= 2
        result = search_tenant_documents(
            subqueries[0],
            "demo",
            k=4,
            retrieval_queries=subqueries,
        )
        assert len(result.retrieval_queries) >= 2
        text = "\n".join(hit.document.page_content for hit in result.hits)
        assert "MODULE_A_ONLY" in text or "MODULE_B_ONLY" in text
    finally:
        settings.faiss_index_path = original
        reset_indexes()
