"""Parent-Child 分块：子块检索，父块生成。"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

_HEADERS = [("#", "title"), ("##", "section"), ("###", "subsection")]
_CHILD_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
_FALLBACK_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
_SUMMARY_MAX_CHARS = 280


def _section_path(metadata: dict[str, object]) -> str:
    parts = [
        str(metadata.get("title") or "").strip(),
        str(metadata.get("section") or "").strip(),
        str(metadata.get("subsection") or "").strip(),
    ]
    return " / ".join(part for part in parts if part)


def _parent_id(source: str, section: str) -> str:
    slug = re.sub(r"\s+", "_", section.strip()) or "root"
    return f"{source}::{slug}"


def _with_prefix(metadata: dict[str, object], body: str) -> str:
    prefix = _section_path(metadata)
    body = body.strip()
    if prefix and body:
        return f"{prefix}\n{body}"
    return prefix or body


def _first_paragraph(body: str, *, max_chars: int = _SUMMARY_MAX_CHARS) -> str:
    """取正文首段，供 section 摘要 child 召回章节标题与导语。"""
    text = body.strip()
    if not text:
        return ""
    for block in re.split(r"\n\s*\n", text):
        paragraph = block.strip()
        if paragraph:
            if len(paragraph) > max_chars:
                return paragraph[:max_chars].rstrip() + "…"
            return paragraph
    clipped = text[:max_chars].rstrip()
    return clipped + ("…" if len(text) > max_chars else "")


def _build_section_summary(section_path: str, parent_body: str) -> str:
    """章节摘要：标题路径 + 首段，便于跨 section 关键词命中。"""
    intro = _first_paragraph(parent_body)
    if section_path and intro:
        return f"{section_path}\n{intro}"
    return section_path or intro


def _base_metadata(path: Path, tenant_id: str, digest: str) -> dict[str, object]:
    return {
        "source": path.name,
        "course_file": path.stem,
        "course_code": path.stem,
        "tenant_id": tenant_id,
        "content_hash": digest,
        "deleted": False,
        "doc_type": "outline",
    }


def build_parent_child_documents(path: Path, tenant_id: str) -> list[Document]:
    """将 Markdown 切成可检索子块与用于上下文的父块。"""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS)
    sections = splitter.split_text(text)
    if not sections:
        sections = _FALLBACK_SPLITTER.create_documents([text])

    documents: list[Document] = []
    child_index = 0
    for section_index, chunk in enumerate(sections):
        if isinstance(chunk, Document):
            document = chunk
        else:
            document = Document(page_content=str(chunk), metadata={})
        metadata = dict(document.metadata)
        section_name = str(metadata.get("section") or metadata.get("title") or f"part-{section_index}")
        parent_id = _parent_id(path.name, section_name)
        section_path = _section_path(metadata)
        parent_body = _with_prefix(metadata, document.page_content)
        parent = Document(
            page_content=parent_body,
            metadata={
                **_base_metadata(path, tenant_id, digest),
                "chunk": -(section_index + 1),
                "chunk_role": "parent",
                "parent_id": parent_id,
                "section_path": section_path,
                "module_name": section_name,
            },
        )
        documents.append(parent)

        summary_body = _build_section_summary(section_path, parent_body)
        if summary_body:
            documents.append(
                Document(
                    page_content=summary_body,
                    metadata={
                        **_base_metadata(path, tenant_id, digest),
                        "chunk": child_index,
                        "chunk_role": "child",
                        "chunk_kind": "section_summary",
                        "parent_id": parent_id,
                        "section_path": section_path,
                        "module_name": section_name,
                        **metadata,
                    },
                )
            )
            child_index += 1

        child_bodies = _CHILD_SPLITTER.split_text(parent_body)
        if not child_bodies:
            child_bodies = [parent_body]
        for body in child_bodies:
            documents.append(
                Document(
                    page_content=body,
                    metadata={
                        **_base_metadata(path, tenant_id, digest),
                        "chunk": child_index,
                        "chunk_role": "child",
                        "parent_id": parent_id,
                        "section_path": section_path,
                        "module_name": section_name,
                        **metadata,
                    },
                )
            )
            child_index += 1
    return documents


def is_retrieval_chunk(document: Document) -> bool:
    """仅子块参与向量/BM25 检索。"""
    return str(document.metadata.get("chunk_role") or "child") != "parent"
