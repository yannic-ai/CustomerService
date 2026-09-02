"""文本向量化：默认 BGE（HuggingFace），可回退 n-gram 哈希。"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np
from langchain_core.embeddings import Embeddings

from app.config import get_settings

_BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索："


class EmbeddingIndexMismatchError(RuntimeError):
    """磁盘索引的 embedding 与当前配置不一致，必须 ingest 重建。"""


class NgramEmbeddings(Embeddings):
    """零依赖中文友好向量：字符 bigram 哈希，便于测试与离线回退。"""

    def __init__(self, dim: int = 384) -> None:
        """指定向量维度，需与 FAISS 索引一致。"""
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        """将文本转为 L2 归一化的 bigram 哈希向量。"""
        vec = np.zeros(self.dim, dtype=np.float32)
        normalized = re.sub(r"\s+", "", (text or "").lower())
        if not normalized:
            return vec.tolist()
        grams = [normalized[i : i + 2] for i in range(len(normalized) - 1)] or [normalized]
        for gram in grams:
            digest = hashlib.md5(gram.encode("utf-8")).hexdigest()
            vec[int(digest, 16) % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        if norm:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文档，供 FAISS 建库。"""
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """向量化检索查询。"""
        return self._embed(text)


class PrefixedQueryEmbeddings(Embeddings):
    """检索时给 query 加指令前缀；文档侧不加。"""

    def __init__(self, inner: Embeddings, query_prefix: str) -> None:
        self._inner = inner
        self._query_prefix = query_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(f"{self._query_prefix}{text}")


def _huggingface_embeddings(model_name: str) -> Embeddings:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "默认 embedding 为 BGE，请确认已安装项目依赖；"
            "或设置 EMBEDDING_BACKEND=ngram 回退哈希向量。"
        ) from exc

    inner = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
    if "bge" in model_name.lower():
        return PrefixedQueryEmbeddings(inner, _BGE_QUERY_PREFIX)
    return inner


def _normalize_backend(raw: str | None) -> str:
    backend = (raw or "huggingface").strip().lower()
    if backend in {"hf", "bge"}:
        return "huggingface"
    if backend in {"hash"}:
        return "ngram"
    return backend


@lru_cache
def get_embeddings() -> Embeddings:
    """按配置返回嵌入模型。huggingface 为默认；ngram 无需额外依赖。"""
    settings = get_settings()
    backend = _normalize_backend(settings.embedding_backend)
    if backend == "ngram":
        return NgramEmbeddings(dim=settings.embedding_dim)
    if backend == "huggingface":
        return _huggingface_embeddings(settings.embedding_model)
    raise RuntimeError(
        f"未知 EMBEDDING_BACKEND={settings.embedding_backend!r}，可选 huggingface / ngram"
    )


def current_embedding_spec() -> dict[str, str | int]:
    """当前进程实际使用的 backend / 模型 / 向量维度。"""
    settings = get_settings()
    backend = _normalize_backend(settings.embedding_backend)
    embeddings = get_embeddings()
    dim = len(embeddings.embed_query("维度探测"))
    model = "ngram" if backend == "ngram" else settings.embedding_model
    return {"backend": backend, "model": model, "dim": dim}


def format_embedding_spec(spec: dict[str, str | int]) -> str:
    return f"{spec.get('backend')}/{spec.get('model')} dim={spec.get('dim')}"
