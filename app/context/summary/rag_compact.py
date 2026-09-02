"""课纲长回复就地压缩。"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, BaseMessage

from app.context.window import message_text

RAG_COMPACT_MIN_CHARS = 400
RAG_COMPACT_SUMMARY_CHARS = 180


def _is_long_rag_reply(text: str) -> bool:
    if len(text) < RAG_COMPACT_MIN_CHARS:
        return False
    has_title = bool(re.search(r"(?m)^##\s+\S", text))
    has_outline = "### 课程模块" in text or "### 学习路径" in text
    return has_title and has_outline


def compact_rag_history(text: str) -> str:
    if not _is_long_rag_reply(text):
        return text
    return _compact_rag_text(text)


def _compact_rag_text(text: str) -> str:
    title = ""
    summary_parts: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not title and stripped.startswith("## "):
            title = stripped[3:].strip()
            continue
        if stripped.startswith("###"):
            break
        if stripped:
            summary_parts.append(stripped)
            if len(" ".join(summary_parts)) >= RAG_COMPACT_SUMMARY_CHARS:
                break
    body = " ".join(summary_parts)[:RAG_COMPACT_SUMMARY_CHARS]
    head = f"## {title}" if title else "## 课程咨询"
    return f"{head}\n{body}".strip() if body else head


def compact_rag_message(message: BaseMessage) -> BaseMessage:
    if not isinstance(message, AIMessage):
        return message
    text = message_text(message)
    stub = compact_rag_history(text)
    if stub == text:
        return message
    return AIMessage(content=stub, id=getattr(message, "id", None))


def apply_rag_compact(
    messages: list[BaseMessage],
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    kept: list[BaseMessage] = []
    updated: list[BaseMessage] = []
    for message in messages:
        new = compact_rag_message(message)
        kept.append(new)
        if new is not message and getattr(new, "id", None):
            updated.append(new)
    return kept, updated
