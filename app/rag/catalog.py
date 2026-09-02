"""课程别名归一化：提升同义课名/口语叫法的检索命中率。"""

from __future__ import annotations

import re

from app.db.courses import load_course_catalog

_EXTRA_ALIASES: dict[str, tuple[str, ...]] = {
    "python-intro": ("Python入门", "python入门", "py入门", "Python零基础"),
    "ml-basics": ("机器学习基础课", "机器学习导论", "ML入门", "机器学习课"),
    "data-analysis": ("数据分析课", "Pandas课", "pandas课程", "NumPy课"),
    "acme-onboarding": ("Acme入职", "入职培训课"),
}


def _normalize_key(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def aliases_for_course_code(code: str) -> tuple[str, ...]:
    return _EXTRA_ALIASES.get(code, ())


def load_course_aliases(tenant_id: str) -> dict[str, str]:
    """别名（归一化键）→ 规范课名。"""
    aliases: dict[str, str] = {}
    catalog = load_course_catalog(tenant_id)
    for (tenant, code), name in catalog.items():
        if tenant != tenant_id or not name:
            continue
        aliases[_normalize_key(name)] = name
        aliases[_normalize_key(code)] = name
        for extra in _EXTRA_ALIASES.get(code, ()):
            aliases[_normalize_key(extra)] = name
    return aliases


def enrich_query_with_course_aliases(query: str, *, tenant_id: str) -> str:
    """若 query 命中别名且未含规范课名，则前置课名辅助检索。"""
    text = (query or "").strip()
    if not text:
        return text
    normalized_query = _normalize_key(text)
    for alias_key, course_name in sorted(
        load_course_aliases(tenant_id).items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if alias_key and alias_key in normalized_query and course_name not in text:
            return f"{course_name} {text}".strip()
    return text
