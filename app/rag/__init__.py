"""课程知识检索：嵌入模型、FAISS 与应用层 BM25。"""

from app.rag.embeddings import get_embeddings
from app.rag.vectorstore import (
    IndexUnavailableError,
    build_faiss_index,
    get_indexes,
    ingest_indexes,
    load_faiss_index,
    load_faiss_indexes,
    require_index_for_serve,
)

__all__ = [
    "get_embeddings",
    "IndexUnavailableError",
    "build_faiss_index",
    "get_indexes",
    "ingest_indexes",
    "load_faiss_index",
    "load_faiss_indexes",
    "require_index_for_serve",
]
