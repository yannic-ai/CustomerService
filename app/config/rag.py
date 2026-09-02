"""向量检索与切片策略。"""

from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from app.config.paths import DATA_DIR

EmbeddingBackend = Literal["huggingface", "ngram"]


class RAGSettings(BaseModel):
    """FAISS / embedding / 混合检索。新增检索参数只改本文件。"""

    faiss_index_path: str = str(DATA_DIR / "faiss_index")
    embedding_backend: EmbeddingBackend = "huggingface"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    # BGE-small-zh-v1.5 为 512 维；改模型或改此值后必须 ingest 重建
    embedding_dim: int = Field(default=512, ge=32, le=4096)
    # 向量相似度低于此值且 BM25 未命中则空召回；0 关闭门控
    rag_min_score: float = Field(default=0.2, ge=0, le=1)
    # 向量 + 应用层 BM25 + RRF；课名/模块名/NumPy 等专名走词法
    rag_hybrid_enabled: bool = True
    rag_rrf_k: int = Field(default=60, ge=1)
    # 命中后补同 section 与前后各 N 个切片；不要整文件倒入
    rag_neighbor_chunks: int = Field(default=1, ge=0, le=20)
    # 扩展后检索上下文 token 预算；0 关闭裁剪（整片丢弃，不截断片内）
    rag_context_token_budget: int = Field(default=4096, ge=0)
    # CrossEncoder 精排；默认关闭，失败降级为融合排序
    rag_rerank_enabled: bool = False
    rag_rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rag_rerank_fetch_k: int = Field(default=24, ge=1)
    rag_rerank_top_k: int = Field(default=6, ge=1)
    rag_rerank_timeout_seconds: float = Field(default=3.0, gt=0)
    # 按 query 复杂度动态调整 fetch_k
    rag_dynamic_k_enabled: bool = True
    rag_fetch_k_min: int = Field(default=12, ge=1)
    rag_fetch_k_max: int = Field(default=32, ge=1)
    # 条件式 query rewrite；关闭时 maybe_rewrite_query 透传
    rag_rewrite_enabled: bool = True
    # 规则 rewrite 之后可选 LLM/规则多 query 分解；默认关，不影响现网
    rag_llm_rewrite_enabled: bool = False
    rag_llm_rewrite_max_queries: int = Field(default=3, ge=2, le=5)
    # 失败样本回流 evals/pending
    rag_eval_harvest_enabled: bool = True
    rag_eval_harvest_dir: str = "evals/pending"
    rag_eval_snapshot_ttl_seconds: int = Field(default=3600, ge=60)

    @field_validator("embedding_backend", mode="before")
    @classmethod
    def _normalize_embedding_backend(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        backend = value.strip().lower()
        if backend in {"hf", "bge"}:
            return "huggingface"
        if backend == "hash":
            return "ngram"
        return backend

    @model_validator(mode="after")
    def _check_rag_constraints(self) -> Self:
        if self.rag_rerank_top_k > self.rag_rerank_fetch_k:
            raise ValueError("rag_rerank_top_k 不能大于 rag_rerank_fetch_k")
        if self.rag_fetch_k_min > self.rag_fetch_k_max:
            raise ValueError("rag_fetch_k_min 不能大于 rag_fetch_k_max")
        return self
