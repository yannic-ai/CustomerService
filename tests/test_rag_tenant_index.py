import asyncio

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.agents.rag import consult_course, retrieve_course_docs, stream_course_answer
from app.config import get_settings
from app.rag.embeddings import NgramEmbeddings
from app.rag.vectorstore import PLATFORM_TENANT, get_indexes, set_indexes


def _doc(text: str, source: str, tenant_id: str) -> Document:
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


def test_small_tenant_not_crowded_out_by_large_tenant() -> None:
    previous = dict(get_indexes())
    demo_docs = [
        _doc(f"Python 入门 环境搭建 基础语法 {index}", f"python-{index}.md", "demo")
        for index in range(30)
    ]
    acme_docs = [_doc("Python 入门 工单系统 入职培训", "acme-onboarding.md", "acme")]
    try:
        set_indexes({"demo": _store(demo_docs), "acme": _store(acme_docs)})
        docs = retrieve_course_docs("Python 入门", k=8, tenant_id="acme")
        sources = {str(doc.metadata.get("source")) for doc in docs}
        assert "acme-onboarding.md" in sources
        assert not any(str(source).startswith("python-") for source in sources)
    finally:
        set_indexes(previous)


def test_unknown_tenant_does_not_fall_back_to_demo() -> None:
    docs = retrieve_course_docs("Python入门课包含哪些内容？", k=8, tenant_id="no-such-tenant")
    assert docs == []


def test_low_score_hits_are_rejected_as_empty() -> None:
    docs = retrieve_course_docs("今天天气怎么样", k=8, tenant_id="demo")
    assert docs == []
    result = asyncio.run(consult_course("今天天气怎么样", tenant_id="demo"))
    assert result.course_name == "未找到课程"
    assert not result.modules


def test_mismatch_course_on_small_tenant_is_empty() -> None:
    docs = retrieve_course_docs("Python入门课包含哪些内容？", k=8, tenant_id="acme")
    assert docs == []


def test_offline_fallback_prefers_course_name_in_query(monkeypatch) -> None:
    from app.agents import rag as rag_mod

    monkeypatch.setattr(
        rag_mod,
        "retrieve_course_docs",
        lambda *a, **k: [
            Document(
                page_content="课程名称：数据分析实战\n前置要求：完成 Python入门课或同等基础",
                metadata={"source": "data-analysis.md"},
            ),
            Document(
                page_content="课程名称：Python入门课\n适合人群：零基础学员",
                metadata={"source": "python-intro.md"},
            ),
        ],
    )
    result = asyncio.run(consult_course("Python入门课包含哪些内容？", tenant_id="demo"))
    assert result.course_name == "Python入门课"


def test_stream_structured_is_not_regex_outline(monkeypatch) -> None:
    from app.agents import rag as rag_mod

    monkeypatch.setattr(
        rag_mod,
        "retrieve_course_docs",
        lambda *a, **k: [
            Document(
                page_content="课程名称：Python入门课\n模块1：环境搭建\n模块2：流程控制",
                metadata={"source": "python-intro.md"},
            )
        ],
    )
    real = get_settings()
    monkeypatch.setattr(
        rag_mod,
        "get_settings",
        lambda: real.model_copy(update={"deepseek_api_key": "test-key"}),
    )

    class StreamDummy:
        async def astream(self, payload, config=None, **kwargs):
            del payload, config, kwargs
            yield type("Chunk", (), {"content": "只根据检索作答"})()

    monkeypatch.setattr("app.llm.get_chat_model", lambda **kwargs: StreamDummy())
    answer, structured = asyncio.run(stream_course_answer("Python入门课"))
    assert "只根据检索作答" in answer
    assert structured.course_name == "Python入门课"
    assert structured.modules == []
    assert structured.learning_path == []


def test_astream_course_answer_consumes_llm_astream(monkeypatch) -> None:
    """astream_course_answer 应 async for llm.astream 拼接 delta，不进线程池。"""
    import asyncio

    from app.agents import rag as rag_mod
    from app.agents.rag import astream_course_answer

    monkeypatch.setattr(
        rag_mod,
        "retrieve_course_docs",
        lambda *a, **k: [
            Document(
                page_content="课程名称：Python入门课\n模块1：环境搭建",
                metadata={"source": "python-intro.md"},
            )
        ],
    )
    real = get_settings()
    monkeypatch.setattr(
        rag_mod,
        "get_settings",
        lambda: real.model_copy(update={"deepseek_api_key": "test-key"}),
    )

    class AStreamDummy:
        async def astream(self, payload, config=None, **kwargs):
            del payload, config, kwargs
            yield type("Chunk", (), {"content": "只根据检索作答"})

    monkeypatch.setattr("app.llm.get_chat_model", lambda **kwargs: AStreamDummy())
    answer, structured = asyncio.run(astream_course_answer("Python入门课"))
    assert "只根据检索作答" in answer
    assert structured.course_name == "Python入门课"


def test_astream_course_answer_falls_back_when_llm_fails(monkeypatch) -> None:
    """Key 已配但生成失败时，走检索规则结构，而不是把异常抛出图外。"""
    import asyncio

    from app.agents import rag as rag_mod
    from app.agents.rag import astream_course_answer

    monkeypatch.setattr(
        rag_mod,
        "retrieve_course_docs",
        lambda *a, **k: [
            Document(
                page_content="课程名称：Python入门课\n模块1：环境搭建",
                metadata={"source": "python-intro.md"},
            )
        ],
    )
    real = get_settings()
    monkeypatch.setattr(
        rag_mod,
        "get_settings",
        lambda: real.model_copy(update={"deepseek_api_key": "test-key"}),
    )

    class Boom:
        async def astream(self, payload, config=None, **kwargs):
            del payload, config, kwargs
            raise TimeoutError("upstream down")
            yield  # pragma: no cover  # make this an async generator

    monkeypatch.setattr("app.llm.get_chat_model", lambda **kwargs: Boom())
    answer, structured = asyncio.run(astream_course_answer("Python入门课"))
    assert structured.course_name == "Python入门课"
    assert "Python入门课" in answer or "模块" in answer


def test_platform_index_is_visible_to_tenant() -> None:
    previous = dict(get_indexes())
    try:
        set_indexes(
            {
                "acme": _store([_doc("Acme 入职培训工单", "acme-onboarding.md", "acme")]),
                PLATFORM_TENANT: _store(
                    [_doc("活动期间购买的商品可以无理由退货", "return-policy.md", PLATFORM_TENANT)]
                ),
            }
        )
        docs = retrieve_course_docs("无理由退货", k=8, tenant_id="acme")
        sources = {str(doc.metadata.get("source")) for doc in docs}
        assert "return-policy.md" in sources
        demo_docs = retrieve_course_docs("无理由退货", k=8, tenant_id="demo")
        demo_sources = {str(doc.metadata.get("source")) for doc in demo_docs}
        assert "return-policy.md" in demo_sources
        assert "acme-onboarding.md" not in demo_sources
    finally:
        set_indexes(previous)
