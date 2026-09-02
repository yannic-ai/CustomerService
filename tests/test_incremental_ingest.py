from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.agents.rag import retrieve_course_docs
from app.config import get_settings
from app.rag.vectorstore import (
    _iter_store_documents,
    build_faiss_index,
    ingest_indexes,
    reset_indexes,
)


@pytest.fixture
def isolated_knowledge(tmp_path: Path):
    settings = get_settings()
    original_faiss = settings.faiss_index_path
    settings.faiss_index_path = str(tmp_path / "faiss")
    reset_indexes()
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    yield knowledge
    settings.faiss_index_path = original_faiss
    reset_indexes()
    build_faiss_index(persist=True)


def _write_course(knowledge: Path, name: str, body: str) -> None:
    (knowledge / name).write_text(
        f"# {name}\n\n课程名称：{body}\n\n## 模块1：内容\n\n- {body}\n",
        encoding="utf-8",
    )


def _active(store) -> list[Document]:
    return _iter_store_documents(store, include_deleted=False)


def _all_docs(store) -> list[Document]:
    return _iter_store_documents(store, include_deleted=True)


def test_unchanged_ingest_does_not_reembed(isolated_knowledge: Path) -> None:
    _write_course(isolated_knowledge, "alpha.md", "向量课第一版")
    stores = ingest_indexes(persist=True, knowledge_dir=isolated_knowledge)
    ntotal = stores["demo"].index.ntotal
    active = len(_active(stores["demo"]))
    again = ingest_indexes(persist=True, knowledge_dir=isolated_knowledge)
    assert again["demo"].index.ntotal == ntotal
    assert len(_active(again["demo"])) == active


def test_update_writes_new_chunks_then_soft_deletes_old(isolated_knowledge: Path) -> None:
    _write_course(isolated_knowledge, "alpha.md", "旧版过拟合专名")
    first = ingest_indexes(persist=True, knowledge_dir=isolated_knowledge)
    old_total = first["demo"].index.ntotal
    old_active = len(_active(first["demo"]))

    _write_course(isolated_knowledge, "alpha.md", "新版交叉验证专名")
    second = ingest_indexes(persist=True, knowledge_dir=isolated_knowledge)
    store = second["demo"]
    active = _active(store)
    all_docs = _all_docs(store)
    assert store.index.ntotal == old_total + len(active)
    assert len(active) == old_active
    assert any(doc.metadata.get("deleted") for doc in all_docs)
    joined_active = "\n".join(doc.page_content for doc in active)
    assert "交叉验证" in joined_active
    assert "过拟合" not in joined_active

    hits = retrieve_course_docs("交叉验证专名", k=4, tenant_id="demo")
    text = "\n".join(doc.page_content for doc in hits)
    assert "交叉验证" in text
    assert "过拟合" not in text


def test_deleted_file_is_soft_deleted_not_dropped(isolated_knowledge: Path) -> None:
    _write_course(isolated_knowledge, "keep.md", "保留课程 NumPy")
    _write_course(isolated_knowledge, "drop.md", "即将删除的条款")
    first = ingest_indexes(persist=True, knowledge_dir=isolated_knowledge)
    before = first["demo"].index.ntotal
    (isolated_knowledge / "drop.md").unlink()
    second = ingest_indexes(persist=True, knowledge_dir=isolated_knowledge)
    store = second["demo"]
    assert store.index.ntotal == before
    sources = {str(doc.metadata.get("source")) for doc in _active(store)}
    assert "keep.md" in sources
    assert "drop.md" not in sources
    dropped = [
        doc
        for doc in _all_docs(store)
        if doc.metadata.get("source") == "drop.md"
    ]
    assert dropped
    assert all(doc.metadata.get("deleted") for doc in dropped)
    gone = retrieve_course_docs("即将删除的条款", k=4, tenant_id="demo")
    assert gone == []
    kept = retrieve_course_docs("NumPy", k=4, tenant_id="demo")
    assert any("NumPy" in doc.page_content for doc in kept)
