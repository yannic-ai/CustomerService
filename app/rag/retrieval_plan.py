"""检索计划：按 query 复杂度决定 fetch_k，不依赖 LLM。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config.rag import RAGSettings

_FOLLOWUP_HINTS = ("这个", "那个", "难吗", "怎么样", "如何", "可以吗")
_COMPLEX_HINTS = ("哪些", "包含", "区别", "对比", "适合", "路径", "模块", "大纲", "学习", "内容")
_SHORT_QUERY_CHARS = 8


@dataclass(frozen=True)
class RetrievalPlan:
    """一次检索的 k 与 profile。"""

    fetch_k: int
    result_k: int
    profile: str  # explicit_course | followup | complex | default


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def resolve_retrieval_plan(
    query: str,
    *,
    result_k: int,
    course_code: str | None = None,
    settings: RAGSettings,
) -> RetrievalPlan:
    """根据 query 与是否绑定课程编码生成检索计划。"""
    text = (query or "").strip()
    k_min = settings.rag_fetch_k_min
    k_max = settings.rag_fetch_k_max

    if course_code:
        fetch_k = _clamp(max(result_k * 2, k_min), k_min, k_max)
        return RetrievalPlan(fetch_k=fetch_k, result_k=result_k, profile="explicit_course")

    if len(text) < _SHORT_QUERY_CHARS or any(hint in text for hint in _FOLLOWUP_HINTS):
        fetch_k = _clamp(max(result_k * 3, k_min), k_min, k_max)
        return RetrievalPlan(fetch_k=fetch_k, result_k=result_k, profile="followup")

    complex_pattern = re.compile("|".join(re.escape(h) for h in _COMPLEX_HINTS))
    if len(text) >= 16 or complex_pattern.search(text):
        fetch_k = k_max
        return RetrievalPlan(fetch_k=fetch_k, result_k=result_k, profile="complex")

    fetch_k = _clamp(max(result_k * 3, k_min), k_min, k_max)
    return RetrievalPlan(fetch_k=fetch_k, result_k=result_k, profile="default")
