from app.config import get_settings
from app.rag.retrieval_plan import resolve_retrieval_plan


def test_explicit_course_profile_uses_smaller_fetch_k() -> None:
    settings = get_settings()
    plan = resolve_retrieval_plan(
        "模块有哪些",
        result_k=8,
        course_code="python-intro",
        settings=settings,
    )
    assert plan.profile == "explicit_course"
    assert plan.fetch_k >= settings.rag_fetch_k_min
    assert plan.fetch_k <= settings.rag_fetch_k_max


def test_followup_profile_for_short_query() -> None:
    settings = get_settings()
    plan = resolve_retrieval_plan("难吗", result_k=8, settings=settings)
    assert plan.profile == "followup"


def test_complex_profile_for_comprehensive_query() -> None:
    settings = get_settings()
    plan = resolve_retrieval_plan(
        "Python入门课包含哪些模块和学习路径？",
        result_k=8,
        settings=settings,
    )
    assert plan.profile == "complex"
    assert plan.fetch_k == settings.rag_fetch_k_max


def test_dynamic_k_disabled_uses_legacy_fetch_in_search(monkeypatch) -> None:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    from app.rag.embeddings import NgramEmbeddings
    from app.rag.vectorstore import _fetch_k, get_indexes, search_tenant_documents, set_indexes

    settings = get_settings()
    monkeypatch.setattr(settings, "rag_dynamic_k_enabled", False)
    doc = Document(
        page_content="Python 入门",
        metadata={"source": "python-intro.md", "chunk": 0, "tenant_id": "demo"},
    )
    previous = dict(get_indexes())
    try:
        set_indexes({"demo": FAISS.from_documents([doc], NgramEmbeddings(dim=settings.embedding_dim))})
        result = search_tenant_documents("Python入门课包含哪些内容？", "demo", k=4)
        assert result.fetch_k == _fetch_k(4)
        assert result.query_profile == "complex"
    finally:
        set_indexes(previous)
