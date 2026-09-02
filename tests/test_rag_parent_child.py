from pathlib import Path

from langchain_core.documents import Document

from app.config import get_settings
from app.rag.catalog import enrich_query_with_course_aliases, load_course_aliases
from app.rag.chunking import build_parent_child_documents, is_retrieval_chunk
from app.rag.vectorstore import (
    expand_hit_documents_scored,
    ingest_indexes,
    iter_tenant_documents,
    reset_indexes,
    search_tenant_documents,
)


def test_build_parent_child_marks_roles(tmp_path: Path) -> None:
    path = tmp_path / "python-intro.md"
    path.write_text(
        "# Python\n\n课程名称：Python入门课\n\n## 模块1：语法\n\n- 变量\n- 循环\n",
        encoding="utf-8",
    )
    docs = build_parent_child_documents(path, "demo")
    parents = [doc for doc in docs if doc.metadata.get("chunk_role") == "parent"]
    children = [doc for doc in docs if doc.metadata.get("chunk_role") == "child"]
    summaries = [doc for doc in children if doc.metadata.get("chunk_kind") == "section_summary"]
    assert parents
    assert children
    assert summaries
    assert children[0].metadata.get("parent_id") == parents[0].metadata.get("parent_id")
    assert children[0].metadata.get("module_name")


def test_section_summary_is_searchable(tmp_path: Path) -> None:
    settings = get_settings()
    original = settings.faiss_index_path
    settings.faiss_index_path = str(tmp_path / "faiss")
    reset_indexes()
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "alpha.md").write_text(
        "# alpha\n\n## 课程简介\n\n这是只在简介段出现的专名SECTION_SUMMARY_ONLY。\n\n"
        "## 模块1：细节\n\n- 其他内容 " + ("x" * 500) + "\n",
        encoding="utf-8",
    )
    try:
        ingest_indexes(persist=True, knowledge_dir=knowledge)
        result = search_tenant_documents("SECTION_SUMMARY_ONLY", "demo", k=5)
        assert result.hits
        kinds = {str(hit.document.metadata.get("chunk_kind") or "") for hit in result.hits}
        assert "section_summary" in kinds or any(
            "SECTION_SUMMARY_ONLY" in hit.document.page_content for hit in result.hits
        )
    finally:
        settings.faiss_index_path = original
        reset_indexes()


def test_course_code_fallback_not_full_corpus_dump(tmp_path: Path) -> None:
    settings = get_settings()
    original = settings.faiss_index_path
    settings.faiss_index_path = str(tmp_path / "faiss")
    reset_indexes()
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    body = "\n\n".join(
        f"## 模块{i}：章节{i}\n\n- 专名FALLBACK_PROBE_{i}\n" + ("内容填充 " * 80)
        for i in range(1, 9)
    )
    (knowledge / "wide.md").write_text(
        f"# wide\n\n课程名称：宽课程\n\n{body}",
        encoding="utf-8",
    )
    try:
        ingest_indexes(persist=True, knowledge_dir=knowledge)
        corpus = iter_tenant_documents("demo", course_code="wide")
        child_count = sum(1 for doc in corpus if is_retrieval_chunk(doc))
        result = search_tenant_documents(
            "完全无关的噪声查询ZZZ",
            "demo",
            k=3,
            course_code="wide",
        )
        assert len(result.hits) <= 3
        assert len(result.hits) < child_count
        assert all(is_retrieval_chunk(hit.document) for hit in result.hits)
    finally:
        settings.faiss_index_path = original
        reset_indexes()


def test_parents_are_not_searchable(tmp_path: Path) -> None:
    settings = get_settings()
    original = settings.faiss_index_path
    settings.faiss_index_path = str(tmp_path / "faiss")
    reset_indexes()
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "alpha.md").write_text(
        "# alpha\n\n课程名称：测试课\n\n## 模块1：专名UNIQUE_PARENT_TEST\n\n- 子块内容\n",
        encoding="utf-8",
    )
    try:
        ingest_indexes(persist=True, knowledge_dir=knowledge)
        result = search_tenant_documents("UNIQUE_PARENT_TEST", "demo", k=5)
        for hit in result.hits:
            assert is_retrieval_chunk(hit.document)
            assert hit.document.metadata.get("chunk_role") == "child"
    finally:
        settings.faiss_index_path = original
        reset_indexes()


def test_child_hit_expands_to_parent_context(tmp_path: Path) -> None:
    settings = get_settings()
    original = settings.faiss_index_path
    settings.faiss_index_path = str(tmp_path / "faiss")
    reset_indexes()
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "alpha.md").write_text(
        "# alpha\n\n## 模块1：完整章节\n\n- 第一行\n- 第二行很长内容用于切分 " + ("x" * 500) + "\n",
        encoding="utf-8",
    )
    try:
        ingest_indexes(persist=True, knowledge_dir=knowledge)
        corpus = iter_tenant_documents("demo")
        children = [doc for doc in corpus if doc.metadata.get("chunk_role") == "child"]
        assert children
        expanded = expand_hit_documents_scored([children[0]], corpus, neighbor=0)
        assert expanded
        top = expanded[0].document
        assert top.metadata.get("chunk_role") == "parent"
        assert "完整章节" in top.page_content
    finally:
        settings.faiss_index_path = original
        reset_indexes()


def test_alias_enrichment_adds_course_name() -> None:
    aliases = load_course_aliases("demo")
    assert aliases
    enriched = enrich_query_with_course_aliases("py入门怎么样", tenant_id="demo")
    assert "Python入门课" in enriched
