"""章节级 evidence 引用单测。"""

from __future__ import annotations

from langchain_core.documents import Document

from app.agents.rag import build_evidence_from_docs, format_course_answer
from app.schemas import CourseConsultResult, EvidenceSection


def test_build_evidence_dedupes_by_section() -> None:
    docs = [
        Document(
            page_content="环境搭建与基础语法内容 " * 5,
            metadata={
                "source": "python-intro.md",
                "section_path": "环境搭建",
                "module_name": "环境搭建与基础语法",
            },
        ),
        Document(
            page_content="重复章节",
            metadata={
                "source": "python-intro.md",
                "section_path": "环境搭建",
                "module_name": "环境搭建与基础语法",
            },
        ),
        Document(
            page_content="面向对象内容",
            metadata={
                "source": "python-intro.md",
                "section_path": "面向对象",
                "module_name": "面向对象编程",
            },
        ),
    ]
    evidence = build_evidence_from_docs(docs)
    assert len(evidence) == 2
    assert evidence[0].module_name == "环境搭建与基础语法"
    assert evidence[0].excerpt.endswith("…") or len(evidence[0].excerpt) <= 120


def test_format_course_answer_includes_evidence_section() -> None:
    result = CourseConsultResult(
        course_name="Python入门课",
        summary="示例摘要",
        modules=[],
        evidence=[
            EvidenceSection(
                source="python-intro.md",
                module_name="环境搭建",
                excerpt="示例摘录",
            )
        ],
    )
    text = format_course_answer(result)
    assert "### 参考依据" in text
    assert "python-intro.md" in text
