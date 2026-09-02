"""课程主数据与大纲用 course_code 对齐；FAISS generation 可热切换。"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from app.agents.rag import retrieve_course_docs
from app.config import get_settings
from app.db.courses import resolve_course_code
from app.rag.vectorstore import (
    build_faiss_index,
    current_index_revision,
    disk_index_revision,
    get_indexes,
    ingest_indexes,
    iter_tenant_documents,
    replace_index_cache,
    reset_indexes,
    set_indexes,
)
from app.services.order import fetch_order_report


def test_order_report_carries_course_code() -> None:
    report = fetch_order_report("20251114001", tenant_id="demo")
    assert report.course_code == "python-intro"
    assert report.product_name == "Python入门课"


def test_resolve_course_code_prefers_code_over_name() -> None:
    assert resolve_course_code("demo", code="python-intro", name="对不齐的包装名") == "python-intro"
    assert resolve_course_code("demo", name="Python入门课") == "python-intro"
    assert resolve_course_code("demo", name="不存在的课") is None


def test_ingested_chunks_stamp_catalog_name() -> None:
    docs = [
        doc
        for doc in iter_tenant_documents("demo")
        if doc.metadata.get("course_code") == "python-intro"
    ]
    assert docs
    assert docs[0].metadata.get("course_name") == "Python入门课"


def test_course_code_hits_when_display_name_drifts(tmp_path: Path) -> None:
    settings = get_settings()
    original = settings.faiss_index_path
    settings.faiss_index_path = str(tmp_path / "faiss")
    reset_indexes()
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "python-intro.md").write_text(
        "# python-intro\n\n课程名称：大纲里的旧标题\n\n## 模块1：内容\n\n- 切片哈希对齐专名XYZ\n",
        encoding="utf-8",
    )
    try:
        ingest_indexes(persist=True, knowledge_dir=knowledge)
        drifted = "完全对不齐的展示名包装课"
        hits = retrieve_course_docs(drifted, k=4, tenant_id="demo", course_code="python-intro")
        text = "\n".join(doc.page_content for doc in hits)
        assert "切片哈希对齐专名XYZ" in text
        empty = retrieve_course_docs(drifted, k=4, tenant_id="demo")
        assert empty == []
    finally:
        settings.faiss_index_path = original
        reset_indexes()
        build_faiss_index(persist=True)


def test_get_indexes_reloads_when_generation_changes(tmp_path: Path) -> None:
    settings = get_settings()
    original = settings.faiss_index_path
    settings.faiss_index_path = str(tmp_path / "faiss")
    reset_indexes()
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()

    def write(body: str) -> None:
        (knowledge / "alpha.md").write_text(
            f"# alpha\n\n课程名称：{body}\n\n## 模块1：内容\n\n- {body}\n",
            encoding="utf-8",
        )

    try:
        write("第一版专名AAA")
        ingest_indexes(persist=True, knowledge_dir=knowledge)
        reset_indexes()
        worker_b = dict(get_indexes())
        rev1 = current_index_revision()
        assert rev1
        text_v1 = "\n".join(doc.page_content for doc in retrieve_course_docs("AAA", k=4, tenant_id="demo"))
        assert "第一版专名AAA" in text_v1

        reset_indexes()
        write("第二版专名BBB")
        ingest_indexes(persist=True, knowledge_dir=knowledge)
        rev2 = disk_index_revision()
        assert rev2 != rev1

        replace_index_cache(worker_b, revision=rev1, pinned=False)
        hits = retrieve_course_docs("BBB", k=4, tenant_id="demo")
        text = "\n".join(doc.page_content for doc in hits)
        assert "第二版专名BBB" in text
        assert current_index_revision() == rev2
    finally:
        settings.faiss_index_path = original
        reset_indexes()
        build_faiss_index(persist=True)


def test_set_indexes_is_not_overwritten_by_generation(tmp_path: Path) -> None:
    settings = get_settings()
    original = settings.faiss_index_path
    settings.faiss_index_path = str(tmp_path / "faiss")
    reset_indexes()
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "alpha.md").write_text(
        "# alpha\n\n课程名称：磁盘版\n\n## 模块1：内容\n\n- 磁盘版专名\n",
        encoding="utf-8",
    )
    try:
        ingest_indexes(persist=True, knowledge_dir=knowledge)
        from langchain_community.vectorstores import FAISS

        from app.rag.embeddings import NgramEmbeddings

        pinned_doc = Document(
            page_content="课程名称：钉住版\n钉住专名PINNED_ONLY",
            metadata={
                "source": "pinned.md",
                "chunk": 0,
                "course_file": "pinned",
                "course_code": "pinned",
                "tenant_id": "demo",
            },
        )
        pinned = {"demo": FAISS.from_documents([pinned_doc], NgramEmbeddings(dim=settings.embedding_dim))}
        set_indexes(pinned)
        gen_path = Path(settings.faiss_index_path) / "generation.json"
        gen_path.write_text(json.dumps({"id": "forced-other"}, ensure_ascii=False), encoding="utf-8")
        hits = retrieve_course_docs("PINNED_ONLY", k=4, tenant_id="demo")
        text = "\n".join(doc.page_content for doc in hits)
        assert "钉住专名PINNED_ONLY" in text
        assert "磁盘版专名" not in text
    finally:
        settings.faiss_index_path = original
        reset_indexes()
        build_faiss_index(persist=True)
