"""应用层词法检索：jieba 分词 + BM25Plus，供与向量结果做 RRF。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from langchain_core.documents import Document

logger = logging.getLogger("cs.rag")

_DOMAIN_WORDS = (
    "NumPy",
    "numpy",
    "Pandas",
    "pandas",
    "过拟合",
    "交叉验证",
    "机器学习",
    "Python",
    "入门课",
    "数据分析",
    "广播",
)

_jieba_ready = False


def _ensure_jieba() -> None:
    global _jieba_ready
    if _jieba_ready:
        return
    import jieba

    jieba.initialize()
    for word in _DOMAIN_WORDS:
        jieba.add_word(word)
    _jieba_ready = True


def tokenize(text: str) -> list[str]:
    """中文分词；保留 NumPy 这类专名，丢掉标点和单字母。"""
    raw = (text or "").strip()
    if not raw:
        return []
    _ensure_jieba()
    import jieba

    tokens: list[str] = []
    for piece in jieba.lcut(raw):
        token = piece.strip().lower()
        if not token:
            continue
        if re.fullmatch(r"[\W_]+", token, flags=re.UNICODE):
            continue
        if token.isascii() and token.isalnum() and len(token) < 2:
            continue
        if not token.isascii() and len(token) < 2:
            continue
        tokens.append(token)
    return tokens


def document_key(document: Document) -> tuple[str, str]:
    return (str(document.metadata.get("source")), document.page_content)


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    *,
    rrf_k: int = 60,
    limit: int = 8,
) -> list[tuple[Document, float]]:
    """RRF：score = Σ 1/(k + rank)，不调两路权重。"""
    scores: dict[tuple[str, str], float] = {}
    docs: dict[tuple[str, str], Document] = {}
    for ranking in ranked_lists:
        for rank, document in enumerate(ranking, start=1):
            key = document_key(document)
            docs[key] = document
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(docs[key], score) for key, score in ordered[:limit]]


@dataclass
class Bm25Index:
    """单个租户（或平台）语料上的 BM25 索引。"""

    documents: list[Document]
    _engine: object

    @classmethod
    def build(cls, documents: list[Document]) -> Bm25Index | None:
        if not documents:
            return None
        from rank_bm25 import BM25Plus

        tokenized = [tokenize(document.page_content) or [""] for document in documents]
        # Okapi 在小语料上 IDF 常为 0/负；Plus 的 IDF 恒正，专名仍能排上来
        return cls(documents=list(documents), _engine=BM25Plus(tokenized))

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        tokens = tokenize(query)
        if not tokens or k <= 0:
            return []
        scores = self._engine.get_scores(tokens)
        ranked = [
            (self.documents[index], float(score))
            for index, score in enumerate(scores)
            if float(score) > 0
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:k]
