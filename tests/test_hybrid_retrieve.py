from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.agents.rag import retrieve_course_docs
from app.config import get_settings
from app.rag.embeddings import NgramEmbeddings
from app.rag.lexical import reciprocal_rank_fusion, tokenize
from app.rag.vectorstore import get_indexes, search_tenant_documents, set_indexes


def _doc(text: str, source: str, tenant_id: str = "demo") -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "chunk": 0,
            "course_file": source.rsplit(".", 1)[0],
            "tenant_id": tenant_id,
        },
    )


def _store(docs: list[Document]) -> FAISS:
    return FAISS.from_documents(docs, NgramEmbeddings(dim=get_settings().embedding_dim))


def test_tokenize_keeps_proper_nouns() -> None:
    tokens = tokenize("NumPy 基础与广播，过拟合和交叉验证")
    assert "numpy" in tokens
    assert "过拟合" in tokens
    assert "交叉验证" in tokens
    assert "广播" in tokens


def test_rrf_prefers_docs_high_in_both_lists() -> None:
    alpha = _doc("alpha", "a.md")
    beta = _doc("beta", "b.md")
    gamma = _doc("gamma", "c.md")
    fused = reciprocal_rank_fusion([[alpha, beta, gamma], [beta, gamma, alpha]], rrf_k=60, limit=3)
    assert fused[0][0].metadata["source"] == "b.md"
    assert fused[0][1] > fused[1][1]


def test_numpy_and_overfit_hit_right_courses() -> None:
    numpy_hits = search_tenant_documents("NumPy 基础与广播", "demo", k=3).hits
    assert numpy_hits
    assert numpy_hits[0].document.metadata.get("source") == "data-analysis.md"
    assert numpy_hits[0].bm25_score > 0

    overfit_hits = search_tenant_documents("过拟合和交叉验证", "demo", k=3).hits
    assert overfit_hits
    assert overfit_hits[0].document.metadata.get("source") == "ml-basics.md"
    assert overfit_hits[0].bm25_score > 0


def test_bm25_lifts_proper_noun_over_semantic_neighbor() -> None:
    previous = dict(get_indexes())
    try:
        set_indexes(
            {
                "demo": _store(
                    [
                        _doc("Python 入门 环境搭建 基础语法 变量 循环", "python-intro.md"),
                        _doc("数据分析 模块1：NumPy 基础 广播 ndarray", "data-analysis.md"),
                    ]
                )
            }
        )
        hits = search_tenant_documents("NumPy 基础与广播", "demo", k=2).hits
        assert hits[0].document.metadata["source"] == "data-analysis.md"
        assert hits[0].bm25_score > hits[1].bm25_score
    finally:
        set_indexes(previous)


def test_weather_still_empty_with_hybrid() -> None:
    docs = retrieve_course_docs("今天天气怎么样", k=8, tenant_id="demo")
    assert docs == []


def test_hybrid_disabled_still_retrieves(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "rag_hybrid_enabled", False)
    hits = search_tenant_documents("Python入门课包含哪些内容？", "demo", k=3).hits
    assert hits
    assert hits[0].bm25_score == 0
    assert any(hit.document.metadata.get("source") == "python-intro.md" for hit in hits)
