"""可选 LLM 多 query 分解：复杂问句拆成 2–3 个子检索 query。"""

from __future__ import annotations

import json
import logging
import re

from app.config import get_settings

logger = logging.getLogger("cs.rag.rewrite")

_SPLIT_PATTERN = re.compile(r"[，,、；;]|以及|还有|同时|和|与")
_COMPLEX_HINTS = re.compile(r"(模块|大纲|路径|对比|区别|包含|学习|内容|讲什么|哪些)")


def _rule_decompose(query: str, *, max_queries: int) -> list[str]:
    """规则拆分：长问句且含复杂提示词时按连接词切分。"""
    text = (query or "").strip()
    if not text:
        return []
    if len(text) < 16 or not _COMPLEX_HINTS.search(text):
        return [text]
    parts = [part.strip() for part in _SPLIT_PATTERN.split(text) if part.strip()]
    if len(parts) < 2:
        return [text]
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts[:max_queries]:
        key = part.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(part)
    return deduped or [text]


def _parse_query_list(raw: object) -> list[str]:
    if isinstance(raw, list):
        items = [str(item).strip() for item in raw if str(item).strip()]
        return items
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return []
            return _parse_query_list(parsed)
        if text:
            return [text]
    return []


def _llm_decompose(query: str, *, max_queries: int) -> list[str]:
    from langchain_core.messages import HumanMessage, SystemMessage

    from app.llm import get_chat_model

    model = get_chat_model(temperature=0, usage_tag="rag")
    system = (
        "你是检索 query 分解助手。把课程咨询拆成 2 到 "
        f"{max_queries} 个可独立检索的中文子问题。"
        "只输出 JSON 字符串数组，不要解释。"
    )
    response = model.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=query),
        ]
    )
    content = getattr(response, "content", response)
    if not isinstance(content, str):
        content = str(content or "")
    items = _parse_query_list(content)
    if len(items) < 2:
        return []
    return items[:max_queries]


def decompose_retrieval_queries(query: str) -> list[str]:
    """返回用于检索的 query 列表；单问句时长度为 1。"""
    text = (query or "").strip()
    if not text:
        return []
    settings = get_settings()
    max_queries = settings.rag_llm_rewrite_max_queries
    if settings.rag_llm_rewrite_enabled and settings.llm_enabled:
        try:
            llm_queries = _llm_decompose(text, max_queries=max_queries)
            if len(llm_queries) >= 2:
                return llm_queries
        except Exception as exc:
            logger.warning("LLM query 分解失败，回退规则拆分: %s", exc)
    return _rule_decompose(text, max_queries=max_queries)
