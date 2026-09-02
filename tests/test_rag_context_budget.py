from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.agents.rag import retrieve_course_docs
from app.config import get_settings
from app.context.window import count_tokens
from app.rag.embeddings import NgramEmbeddings
from app.rag.vectorstore import (
    RagContextItem,
    expand_hit_documents_scored,
    format_rag_context_chunk,
    get_indexes,
    set_indexes,
    trim_rag_context_by_budget,
)


def _doc(
    text: str,
    *,
    source: str = "course.md",
    chunk: int = 0,
    section: str | None = None,
    tenant_id: str = "demo",
) -> Document:
    metadata: dict[str, object] = {
        "source": source,
        "chunk": chunk,
        "course_file": source.rsplit(".", 1)[0],
        "tenant_id": tenant_id,
    }
    if section is not None:
        metadata["section"] = section
    return Document(page_content=text, metadata=metadata)


def _item(
    document: Document,
    *,
    hit_rank: int = 0,
    tier: int = 0,
    distance: int = 0,
) -> RagContextItem:
    return RagContextItem(
        document=document,
        hit_rank=hit_rank,
        tier=tier,
        distance=distance,
    )


def _tokens(document: Document) -> int:
    return count_tokens(format_rag_context_chunk(document))


def test_trim_keeps_hits_before_neighbors() -> None:
    hit = _doc("hit original", chunk=5, section="模块A")
    neighbor = _doc("neighbor far away", chunk=7, section="模块B")
    items = [
        _item(hit, tier=0),
        _item(neighbor, tier=2, distance=2),
    ]
    budget = _tokens(hit) + 1
    docs, stats = trim_rag_context_by_budget(items, budget)
    assert [doc.page_content for doc in docs] == ["hit original"]
    assert stats["context_trimmed"] == 1


def test_trim_keeps_section_before_far_neighbor() -> None:
    hit = _doc("hit", chunk=5, section="模块A")
    section_peer = _doc("same section peer", chunk=6, section="模块A")
    far_neighbor = _doc("far neighbor", chunk=8, section="模块B")
    items = [
        _item(hit, tier=0),
        _item(section_peer, tier=1, distance=1),
        _item(far_neighbor, tier=2, distance=3),
    ]
    budget = _tokens(hit) + _tokens(section_peer) + 1
    docs, stats = trim_rag_context_by_budget(items, budget)
    texts = [doc.page_content for doc in docs]
    assert "hit" in texts
    assert "same section peer" in texts
    assert "far neighbor" not in texts
    assert stats["context_trimmed"] == 1


def test_trim_respects_hit_rank() -> None:
    rank0 = _doc("rank0 hit", chunk=0, section="A")
    rank1 = _doc("rank1 hit", chunk=10, section="B")
    items = [
        _item(rank0, hit_rank=0, tier=0),
        _item(rank1, hit_rank=1, tier=0),
    ]
    budget = _tokens(rank0) + 1
    docs, stats = trim_rag_context_by_budget(items, budget)
    assert [doc.page_content for doc in docs] == ["rank0 hit"]
    assert stats["context_trimmed"] == 1


def test_trim_budget_zero_disables() -> None:
    hit = _doc("hit", chunk=0, section="A")
    neighbor = _doc("neighbor", chunk=2, section="B")
    items = [_item(hit, tier=0), _item(neighbor, tier=2, distance=2)]
    docs, stats = trim_rag_context_by_budget(items, 0)
    assert len(docs) == 2
    assert stats["context_trimmed"] == 0


def test_trim_minimum_one_hit_when_single_chunk_exceeds_budget() -> None:
    huge = _doc("x" * 5000, chunk=0, section="A")
    neighbor = _doc("neighbor", chunk=1, section="B")
    items = [_item(huge, tier=0), _item(neighbor, tier=2, distance=1)]
    docs, stats = trim_rag_context_by_budget(items, 32)
    assert len(docs) == 1
    assert docs[0].page_content.startswith("x")
    assert stats["context_budget_exceeded"] is True


def test_expand_hit_documents_scored_dedupes_by_best_priority() -> None:
    corpus = [
        _doc("section part 1", chunk=7, section="第7条"),
        _doc("section part 2", chunk=20, section="第7条"),
        _doc("neighbor", chunk=8, section="第8条"),
    ]
    hit = corpus[0]
    scored = expand_hit_documents_scored([hit], corpus, neighbor=1)
    keys = {item.document.page_content for item in scored}
    assert "section part 1" in keys
    assert "section part 2" in keys
    assert "neighbor" in keys
    hit_item = next(item for item in scored if item.document is hit)
    assert hit_item.tier == 0


def test_retrieve_course_docs_applies_context_budget(monkeypatch) -> None:
    previous = dict(get_indexes())
    settings = get_settings()
    monkeypatch.setattr(settings, "rag_context_token_budget", 64)
    monkeypatch.setattr(settings, "rag_min_score", 0.0)

    docs = [
        _doc(f"Python 模块 {index} " + ("内容" * 80), chunk=index, section=f"模块{index}")
        for index in range(12)
    ]
    try:
        set_indexes({"demo": FAISS.from_documents(docs, NgramEmbeddings(dim=settings.embedding_dim))})
        result = retrieve_course_docs("Python 模块", k=3, tenant_id="demo")
        assert result
        total = sum(_tokens(doc) for doc in result)
        assert total <= 64 or len(result) == 1
    finally:
        set_indexes(previous)
