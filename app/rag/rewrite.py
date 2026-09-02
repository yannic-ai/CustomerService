"""条件式 Query Rewrite：规则优先，与 expand_query / 别名补全分层。"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ANAPHORA_HINTS = ("这个", "那个", "它", "这门", "那门", "这门课", "那门课")
_ABSTRACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"难不难|难吗|好不好学|好学吗"), "课程难度 学习要求"),
    (re.compile(r"适合谁|适合什么人|什么人适合|零基础"), "适合人群 前置要求"),
    (re.compile(r"学到什么|能学到什么|学什么|有什么用|收获"), "课程模块 学习目标"),
]
_SHORT_QUERY_CHARS = 12


@dataclass(frozen=True)
class RewriteResult:
    """一次 query 改写结果。"""

    query: str
    rewritten: bool
    reason: str  # followup | abstract | none


def maybe_rewrite_query(
    query: str,
    *,
    original_query: str,
    last_course_name: str | None = None,
    last_course_query: str | None = None,
) -> RewriteResult:
    """仅在指代追问或抽象口语场景追加检索词，清晰长问句原样返回。"""
    text = (query or "").strip()
    if not text:
        return RewriteResult(query=text, rewritten=False, reason="none")

    original = (original_query or "").strip()
    course_hint = (last_course_name or last_course_query or "").strip()

    if len(original) < _SHORT_QUERY_CHARS and any(h in original for h in _ANAPHORA_HINTS):
        if course_hint and course_hint not in text:
            return RewriteResult(
                query=f"{course_hint} {text}".strip(),
                rewritten=True,
                reason="followup",
            )

    for pattern, keywords in _ABSTRACT_PATTERNS:
        if pattern.search(text):
            if keywords in text:
                return RewriteResult(query=text, rewritten=False, reason="none")
            return RewriteResult(
                query=f"{text} {keywords}".strip(),
                rewritten=True,
                reason="abstract",
            )

    return RewriteResult(query=text, rewritten=False, reason="none")
