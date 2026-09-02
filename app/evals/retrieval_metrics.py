"""检索层确定性指标：HitRate@K / Recall@K / MRR。"""

from __future__ import annotations


def _unique_preserve_order(sources: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for source in sources:
        if source and source not in seen:
            seen.add(source)
            ordered.append(source)
    return ordered


def hit_at_k(ranked_sources: list[str], gold_sources: set[str], k: int) -> bool:
    """top-k 是否至少命中一个 gold source。"""
    if not gold_sources:
        return True
    top = _unique_preserve_order(ranked_sources)[:k]
    return any(source in gold_sources for source in top)


def recall_at_k(ranked_sources: list[str], gold_sources: set[str], k: int) -> float:
    """gold sources 在 top-k 中的召回比例。"""
    if not gold_sources:
        return 1.0
    top = set(_unique_preserve_order(ranked_sources)[:k])
    return len(top & gold_sources) / len(gold_sources)


def mrr(ranked_sources: list[str], gold_sources: set[str]) -> float:
    """第一个相关文档排名的倒数。"""
    if not gold_sources:
        return 1.0
    for index, source in enumerate(_unique_preserve_order(ranked_sources), start=1):
        if source in gold_sources:
            return 1.0 / index
    return 0.0
