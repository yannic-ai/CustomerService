"""CrossEncoder 精排：对融合候选做二次排序，失败时降级。"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from functools import lru_cache

from app.rag.vectorstore import SearchHit

logger = logging.getLogger("cs.rag.reranker")

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-rerank")
_warmed_models: set[str] = set()


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def warmup_reranker(model_name: str) -> None:
    """懒加载精排模型；首次 rerank 前调用，避免首请求卡顿。"""
    name = (model_name or "").strip()
    if not name or name in _warmed_models:
        return
    _load_cross_encoder(name)
    _warmed_models.add(name)


def _predict_scores(model_name: str, pairs: list[tuple[str, str]]) -> list[float]:
    warmup_reranker(model_name)
    model = _load_cross_encoder(model_name)
    raw = model.predict(pairs)
    return [float(score) for score in raw]


def rerank_hits(
    query: str,
    hits: list[SearchHit],
    *,
    top_k: int,
    model_name: str,
    timeout_seconds: float,
) -> list[SearchHit]:
    """对候选精排；超时/异常时保留原序并截断 top_k。"""
    if not hits or top_k <= 0:
        return []
    if len(hits) == 1:
        return hits

    pairs = [(query, hit.document.page_content or "") for hit in hits]
    try:
        future = _executor.submit(_predict_scores, model_name, pairs)
        scores = future.result(timeout=timeout_seconds)
    except FuturesTimeoutError:
        logger.warning("Rerank 超时 %.1fs，降级为融合排序", timeout_seconds)
        return hits[:top_k]
    except Exception as exc:
        logger.warning("Rerank 失败，降级为融合排序: %s", exc)
        return hits[:top_k]

    ranked = sorted(
        zip(hits, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        SearchHit(
            document=hit.document,
            score=float(score),
            vector_sim=hit.vector_sim,
            bm25_score=hit.bm25_score,
            low_confidence=hit.low_confidence,
        )
        for hit, score in ranked[:top_k]
    ]
