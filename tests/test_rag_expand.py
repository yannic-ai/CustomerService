from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.agents.rag import retrieve_course_docs
from app.config import get_settings
from app.rag.embeddings import NgramEmbeddings
from app.rag.vectorstore import expand_hit_documents, get_indexes, set_indexes


def _clause(index: int, *, section: str | None = None, extra: str = "") -> Document:
    heading = section or f"第{index}条"
    return Document(
        page_content=f"{heading} {extra}".strip(),
        metadata={
            "source": "policy.pdf.md",
            "chunk": index,
            "section": heading,
            "course_file": "policy",
            "tenant_id": "demo",
        },
    )


def test_expand_keeps_same_section_and_neighbors_not_whole_file() -> None:
    corpus = [_clause(index) for index in range(20)]
    # 第7条拆成两块：同 section 都应留下
    corpus[7] = Document(
        page_content="第7条 无理由退货 第一款",
        metadata={
            "source": "policy.pdf.md",
            "chunk": 7,
            "section": "第7条",
            "course_file": "policy",
            "tenant_id": "demo",
        },
    )
    corpus.append(
        Document(
            page_content="第7条 无理由退货 第二款 时限说明",
            metadata={
                "source": "policy.pdf.md",
                "chunk": 20,
                "section": "第7条",
                "course_file": "policy",
                "tenant_id": "demo",
            },
        )
    )
    expanded = expand_hit_documents([corpus[7]], corpus, neighbor=1)
    sections = {str(doc.metadata.get("section")) for doc in expanded}
    chunks = {int(doc.metadata["chunk"]) for doc in expanded}
    assert "第7条" in sections
    assert 6 in chunks and 7 in chunks and 8 in chunks
    assert 20 in chunks
    assert 0 not in chunks
    assert 19 not in chunks
    assert len(expanded) == 4


def test_expand_without_section_only_uses_neighbors() -> None:
    corpus = [
        Document(
            page_content=f"段落{index}",
            metadata={"source": "long.md", "chunk": index, "tenant_id": "demo"},
        )
        for index in range(10)
    ]
    expanded = expand_hit_documents([corpus[4]], corpus, neighbor=1)
    chunks = [int(doc.metadata["chunk"]) for doc in expanded]
    assert chunks == [3, 4, 5]


def test_retrieve_does_not_dump_entire_policy_file(monkeypatch) -> None:
    previous = dict(get_indexes())
    clauses = [_clause(index) for index in range(20)]
    clauses[7] = _clause(7, section="第7条", extra="活动期间购买的商品可以无理由退货")
    try:
        set_indexes(
            {
                "demo": FAISS.from_documents(
                    clauses, NgramEmbeddings(dim=get_settings().embedding_dim)
                )
            }
        )
        from app.rag.vectorstore import SearchHit, TenantSearchResult

        monkeypatch.setattr(
            "app.agents.rag.search_tenant_documents",
            lambda query, tenant, k=8, **_kwargs: TenantSearchResult(
                hits=[SearchHit(document=clauses[7], score=1.0, vector_sim=0.9, bm25_score=2.0)],
                fetch_k=k,
                query_profile="default",
                fused_count=1,
                rerank_enabled=False,
            ),
        )
        docs = retrieve_course_docs("无理由退货", k=3, tenant_id="demo")
        chunks = {int(doc.metadata["chunk"]) for doc in docs}
        assert chunks == {6, 7, 8}
    finally:
        set_indexes(previous)
