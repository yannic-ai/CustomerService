"""课程知识切片与 FAISS 索引的构建 / 加载。"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import KNOWLEDGE_DIR, get_settings
from app.rag.chunking import build_parent_child_documents, is_retrieval_chunk
from app.rag.embeddings import (
    EmbeddingIndexMismatchError,
    current_embedding_spec,
    format_embedding_spec,
    get_embeddings,
)
from app.rag.lexical import Bm25Index, document_key, reciprocal_rank_fusion


class IndexUnavailableError(RuntimeError):
    """磁盘索引缺失或不兼容；查询与服务启动禁止隐式重建。"""

logger = logging.getLogger("cs.rag")

# 平台内置知识库：所有租户可检索。对应书中 tenant_id IS NULL。
PLATFORM_TENANT = "platform"
_EMBEDDING_META = "embedding.json"
_GENERATION_FILE = "generation.json"

_indexes: dict[str, FAISS] | None = None
_indexes_revision: str | None = None
_indexes_pinned: bool = False
_lexical: dict[str, Bm25Index | None] = {}
_lexical_lock = threading.Lock()
_index_lock = threading.RLock()


@dataclass(frozen=True)
class SearchHit:
    """一次检索命中：score 为 RRF（或关闭混合时的向量相似度）。"""

    document: Document
    score: float
    vector_sim: float = 0.0
    bm25_score: float = 0.0
    low_confidence: bool = False

    def passes_min_score(self, min_score: float) -> bool:
        """向量过线，或 BM25 命中专名，都算有效召回。"""
        if min_score <= 0:
            return True
        return self.vector_sim >= min_score or self.bm25_score > 0


@dataclass(frozen=True)
class TenantSearchResult:
    """search 层结果与诊断元数据（不含邻片扩展）。"""

    hits: list[SearchHit]
    fetch_k: int
    query_profile: str
    fused_count: int
    rerank_enabled: bool
    rerank_top_k: int = 0
    low_confidence_fallback: bool = False
    retrieval_queries: tuple[str, ...] = ()


def _chunk_markdown(path: Path, tenant_id: str) -> list[Document]:
    """将单个 Markdown 切为 parent-child 文档（子块检索、父块生成）。"""
    return build_parent_child_documents(path, tenant_id)


def load_knowledge_documents(knowledge_dir: Path | None = None) -> list[Document]:
    """读取 Markdown 大纲。

    - `data/knowledge/*.md` → tenant_id=demo（默认租户）
    - `data/knowledge/<tenant>/*.md` → 对应租户
    - `data/knowledge/platform/*.md` → 平台内置，所有租户可检索
    """
    root = knowledge_dir or KNOWLEDGE_DIR
    settings = get_settings()
    docs: list[Document] = []

    for path in sorted(root.glob("*.md")):
        docs.extend(_chunk_markdown(path, settings.default_tenant))

    for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
        tenant_id = subdir.name
        for path in sorted(subdir.glob("*.md")):
            docs.extend(_chunk_markdown(path, tenant_id))

    _stamp_catalog(docs)
    return docs


def document_course_code(document: Document) -> str:
    """切片对应的课程编码：优先 ``course_code``，兼容旧索引的 ``course_file`` / ``source``。"""
    code = str(document.metadata.get("course_code") or document.metadata.get("course_file") or "")
    if code:
        return code
    source = str(document.metadata.get("source") or "")
    if source.endswith(".md"):
        return Path(source).removesuffix(".md")
    return ""


def _matches_course(document: Document, course_code: str | None) -> bool:
    if not course_code:
        return True
    return document_course_code(document) == course_code


def _stamp_catalog(documents: list[Document]) -> None:
    """用订单主数据的课名盖到切片上；文件名（code）仍是对齐键。"""
    if not documents:
        return
    from app.db.courses import load_course_catalog
    from app.rag.catalog import aliases_for_course_code

    catalog = load_course_catalog()
    if not catalog:
        return
    for document in documents:
        code = document_course_code(document)
        if not code:
            continue
        name = catalog.get((_tenant_of(document), code))
        if name:
            document.metadata["course_code"] = code
            document.metadata["course_name"] = name
            extras = aliases_for_course_code(code)
            if extras:
                document.metadata["course_aliases"] = ",".join(extras)


def _tenant_of(document: Document) -> str:
    return str(document.metadata.get("tenant_id") or get_settings().default_tenant)


def _group_documents(documents: list[Document]) -> dict[str, list[Document]]:
    grouped: dict[str, list[Document]] = {}
    for document in documents:
        grouped.setdefault(_tenant_of(document), []).append(document)
    return grouped


def _stores_from_groups(grouped: dict[str, list[Document]]) -> dict[str, FAISS]:
    embeddings = get_embeddings()
    return {
        tenant_id: FAISS.from_documents(docs, embeddings)
        for tenant_id, docs in grouped.items()
        if docs
    }


def _clear_legacy_root_index(root: Path) -> None:
    for name in ("index.faiss", "index.pkl"):
        path = root / name
        if path.is_file():
            path.unlink()


def _tenant_index_dirs(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    found: dict[str, Path] = {}
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "index.faiss").exists():
            found[child.name] = child
    return found


def _is_deleted(document: Document) -> bool:
    return bool(document.metadata.get("deleted"))


def _iter_store_documents(store: FAISS, *, include_deleted: bool = False) -> list[Document]:
    documents: list[Document] = []
    search_fn = getattr(getattr(store, "docstore", None), "search", None)
    index_map = getattr(store, "index_to_docstore_id", {}) or {}
    if not callable(search_fn):
        return documents
    for doc_id in index_map.values():
        document = search_fn(doc_id)
        if not isinstance(document, Document):
            continue
        if not include_deleted and _is_deleted(document):
            continue
        documents.append(document)
    return documents


def _partition_store(store: FAISS) -> dict[str, FAISS]:
    """把旧的整库索引按 metadata.tenant_id 拆成租户子索引。"""
    grouped = _group_documents(_iter_store_documents(store))
    if not grouped:
        return {}
    return _stores_from_groups(grouped)


def _embedding_meta_path(root: Path) -> Path:
    return root / _EMBEDDING_META


def _peek_index_dim(root: Path) -> int | None:
    import faiss

    tenant_dirs = _tenant_index_dirs(root)
    if tenant_dirs:
        faiss_file = next(iter(tenant_dirs.values())) / "index.faiss"
    else:
        faiss_file = root / "index.faiss"
    if not faiss_file.is_file():
        return None
    return int(faiss.read_index(str(faiss_file)).d)


def _load_embedding_meta(root: Path) -> dict[str, str | int] | None:
    path = _embedding_meta_path(root)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _write_embedding_meta(root: Path, spec: dict[str, str | int]) -> None:
    _embedding_meta_path(root).write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def assert_embedding_compatible(root: Path | None = None) -> None:
    """索引 embedding 必须与当前配置一致；换模型或维度只能 ingest 重建。"""
    index_root = Path(root or get_settings().faiss_index_path)
    if not _tenant_index_dirs(index_root) and not (index_root / "index.faiss").exists():
        return
    current = current_embedding_spec()
    stored = _load_embedding_meta(index_root)
    if stored is None:
        dim = _peek_index_dim(index_root)
        if dim is None:
            return
        if int(dim) != int(current["dim"]) or current["backend"] != "ngram":
            raise EmbeddingIndexMismatchError(
                f"现有 FAISS 索引未记录 embedding 元数据（维度 {dim}），"
                f"当前配置为 {format_embedding_spec(current)}。"
                "换模型或换维度必须重建索引：python main.py ingest"
            )
        logger.warning("索引缺少 embedding.json，已按维度 %s 兼容加载；建议 ingest 补齐元数据", dim)
        return
    stored_norm = {
        "backend": str(stored.get("backend") or ""),
        "model": str(stored.get("model") or ""),
        "dim": int(stored.get("dim") or 0),
    }
    current_norm = {
        "backend": str(current["backend"]),
        "model": str(current["model"]),
        "dim": int(current["dim"]),
    }
    if stored_norm != current_norm:
        raise EmbeddingIndexMismatchError(
            f"FAISS 索引由 {format_embedding_spec(stored_norm)} 构建，"
            f"当前配置为 {format_embedding_spec(current_norm)}。"
            "换模型或换维度必须重建索引：python main.py ingest"
        )


def _persist_indexes(root: Path, stores: dict[str, FAISS]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _clear_legacy_root_index(root)
    stale = set(_tenant_index_dirs(root)) - set(stores)
    for tenant_id in stale:
        shutil.rmtree(root / tenant_id)
    for tenant_id, store in stores.items():
        path = root / tenant_id
        path.mkdir(parents=True, exist_ok=True)
        store.save_local(str(path))
    spec = current_embedding_spec()
    _write_embedding_meta(root, spec)
    generation = _write_generation(root)
    logger.info(
        "FAISS indexes saved to %s (%s chunks, tenants=%s, embedding=%s, generation=%s)",
        root,
        sum(store.index.ntotal for store in stores.values()),
        sorted(stores),
        format_embedding_spec(spec),
        generation,
    )


def _generation_path(root: Path) -> Path:
    return root / _GENERATION_FILE


def _write_generation(root: Path) -> str:
    ident = str(time.time_ns())
    _generation_path(root).write_text(
        json.dumps({"id": ident}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ident


def disk_index_revision(root: Path | None = None) -> str:
    """磁盘上的索引世代。ingest 落盘后变化，其它 worker 据此热加载。"""
    index_root = Path(root or get_settings().faiss_index_path)
    path = _generation_path(index_root)
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            ident = str(payload.get("id") or "").strip()
            if ident:
                return ident
        except json.JSONDecodeError:
            pass
        return f"mtime:{path.stat().st_mtime_ns}"
    meta = _embedding_meta_path(index_root)
    if meta.is_file():
        return f"mtime:{meta.stat().st_mtime_ns}"
    return "0"


def _load_store(path: Path) -> FAISS:
    return FAISS.load_local(
        str(path),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def _read_indexes() -> dict[str, FAISS] | None:
    root = Path(get_settings().faiss_index_path)
    tenant_dirs = _tenant_index_dirs(root)
    legacy = root / "index.faiss"
    if not tenant_dirs and not legacy.exists():
        return None
    assert_embedding_compatible(root)
    if tenant_dirs:
        return {tenant_id: _load_store(path) for tenant_id, path in tenant_dirs.items()}
    logger.warning("检测到整库 FAISS 索引，已按 tenant_id 拆开；请执行 ingest 持久化为租户子目录")
    return _partition_store(_load_store(root))


def faiss_index_status(index_path: str | Path | None = None) -> str:
    """ok / missing / mismatch。mismatch 表示换了模型或维度，必须 ingest。"""
    root = Path(index_path or get_settings().faiss_index_path)
    if not (root / "index.faiss").exists() and not _tenant_index_dirs(root):
        return "missing"
    try:
        assert_embedding_compatible(root)
    except EmbeddingIndexMismatchError:
        return "mismatch"
    return "ok"


def faiss_index_ready(index_path: str | Path | None = None) -> bool:
    """就绪检查：存在与当前 embedding 兼容的索引。"""
    return faiss_index_status(index_path) == "ok"


def _clear_lexical_cache() -> None:
    with _lexical_lock:
        _lexical.clear()


def _install_indexes(
    stores: dict[str, FAISS] | None,
    *,
    pinned: bool = False,
    revision: str | None = None,
) -> None:
    global _indexes, _indexes_pinned, _indexes_revision
    _indexes = stores
    _indexes_pinned = bool(pinned and stores is not None)
    _indexes_revision = revision
    _clear_lexical_cache()


def reset_indexes() -> None:
    """测试用：丢掉内存中的租户索引。"""
    with _index_lock:
        _install_indexes(None, pinned=False, revision=None)


def set_indexes(indexes: dict[str, FAISS] | None) -> None:
    """测试用：注入租户索引并钉住，避免被磁盘 generation 覆盖。"""
    with _index_lock:
        _install_indexes(indexes, pinned=indexes is not None, revision=None)


def replace_index_cache(
    indexes: dict[str, FAISS] | None,
    *,
    revision: str | None = None,
    pinned: bool = False,
) -> None:
    """安装一份内存索引。``pinned=False`` 时下次 ``get_indexes`` 会对照磁盘 generation。"""
    with _index_lock:
        _install_indexes(indexes, pinned=pinned, revision=revision)


def current_index_revision() -> str | None:
    """当前进程已加载的 generation；测试注入未钉 revision 时为 None。"""
    return _indexes_revision


def _read_indexes_or_rebuild() -> dict[str, FAISS] | None:
    """兼容则加载；缺索引或 embedding 不匹配则返回 None，由调用方全量重建。"""
    try:
        return _read_indexes()
    except EmbeddingIndexMismatchError as exc:
        logger.warning("%s；改为全量重建", exc)
        return None


def get_indexes() -> dict[str, FAISS]:
    """懒加载全部租户 FAISS 索引；磁盘 generation 变化时热切换。

    缺失或不兼容时抛 ``IndexUnavailableError``，不在查询路径上全量重建。
    修复入口是显式 ``python main.py ingest``（``ingest_indexes`` 仍可重建）。
    """
    with _index_lock:
        if _indexes_pinned and _indexes is not None:
            return _indexes
        root = Path(get_settings().faiss_index_path)
        revision = disk_index_revision(root)
        if _indexes is not None and _indexes_revision == revision:
            return _indexes
        try:
            loaded = _read_indexes()
        except EmbeddingIndexMismatchError as exc:
            raise IndexUnavailableError(str(exc)) from exc
        if loaded is not None:
            _install_indexes(loaded, pinned=False, revision=revision)
            return loaded
        raise IndexUnavailableError(
            f"FAISS 索引不存在或无法加载：{root}。"
            "请先执行 python main.py ingest"
        )


def require_index_for_serve() -> None:
    """HTTP 启动门禁：索引必须存在且与当前 embedding 兼容，禁止隐式重建。"""
    status = faiss_index_status()
    if status == "ok":
        return
    if status == "missing":
        raise IndexUnavailableError(
            "FAISS 索引不存在。请先执行 python main.py ingest"
        )
    raise IndexUnavailableError(
        "FAISS 索引与当前 embedding 配置不兼容。"
        "请执行 python main.py ingest --rebuild"
    )


def search_scopes(tenant_id: str, *, course_code: str | None = None) -> list[str]:
    """当前租户索引 + 平台内置索引（若存在）。绑定具体课时不搜平台库。"""
    indexes = get_indexes()
    if course_code:
        return [tenant_id] if tenant_id in indexes else []
    scopes = [tenant_id]
    if PLATFORM_TENANT in indexes and tenant_id != PLATFORM_TENANT:
        scopes.append(PLATFORM_TENANT)
    return [scope for scope in scopes if scope in indexes]


def iter_tenant_documents(tenant_id: str, *, course_code: str | None = None) -> list[Document]:
    """当前租户与平台索引里的在用切片，供命中后按节 / 相邻 chunk 补齐。"""
    with _index_lock:
        indexes = get_indexes()
        documents: list[Document] = []
        for scope in search_scopes(tenant_id, course_code=course_code):
            documents.extend(_iter_store_documents(indexes[scope]))
        if course_code:
            documents = [doc for doc in documents if _matches_course(doc, course_code)]
        return documents


def _chunk_index(document: Document) -> int | None:
    raw = document.metadata.get("chunk")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _section_key(document: Document) -> tuple[str, str] | None:
    """同文件 + 同 ## section。空 section 不按节扩张，避免扫进整份无标题文档。"""
    source = str(document.metadata.get("source") or "")
    section = str(document.metadata.get("section") or "").strip()
    if not source or not section:
        return None
    return (source, section)


def format_rag_context_chunk(document: Document) -> str:
    """与 RAG prompt 注入格式一致，供 token 估算。"""
    return f"来源:{document.metadata.get('source')}\n{document.page_content}"


@dataclass(frozen=True)
class RagContextItem:
    """扩展后的检索上下文项，带裁剪优先级。"""

    document: Document
    hit_rank: int  # 来自第几个 hit，0 = 最佳
    tier: int  # 0=hit / 1=section / 2=neighbor
    distance: int  # 同节 index 差或邻片 |offset|

    def sort_key(self) -> tuple[int, int, int]:
        return (self.hit_rank, self.tier, self.distance)


def _better_item(current: RagContextItem | None, candidate: RagContextItem) -> bool:
    """candidate 是否比 current 优先级更高（应替换）。"""
    if current is None:
        return True
    return candidate.sort_key() < current.sort_key()


def _parent_lookup(corpus: list[Document]) -> dict[str, Document]:
    parents: dict[str, Document] = {}
    for document in corpus:
        if str(document.metadata.get("chunk_role")) != "parent":
            continue
        parent_id = str(document.metadata.get("parent_id") or "")
        if parent_id:
            parents[parent_id] = document
    return parents


def _context_document(hit: Document, parents: dict[str, Document]) -> Document:
    """子块命中时优先用父块作为生成上下文。"""
    if str(hit.metadata.get("chunk_role")) != "child":
        return hit
    parent_id = str(hit.metadata.get("parent_id") or "")
    if parent_id and parent_id in parents:
        return parents[parent_id]
    return hit


def expand_hit_documents_scored(
    hits: list[Document],
    corpus: list[Document],
    *,
    neighbor: int = 1,
) -> list[RagContextItem]:
    """命中后优先换父块，再补同 section 与相邻子块。"""
    if not hits:
        return []
    window = max(0, int(neighbor))
    parents = _parent_lookup(corpus)
    by_source_chunk: dict[tuple[str, int], Document] = {}
    by_section: dict[tuple[str, str], list[Document]] = {}
    for document in corpus:
        if not is_retrieval_chunk(document):
            continue
        source = str(document.metadata.get("source") or "")
        index = _chunk_index(document)
        if source and index is not None:
            by_source_chunk[(source, index)] = document
        key = _section_key(document)
        if key is not None:
            by_section.setdefault(key, []).append(document)

    selected: dict[tuple[str, str], RagContextItem] = {}

    def consider(item: RagContextItem) -> None:
        key = document_key(item.document)
        existing = selected.get(key)
        if _better_item(existing, item):
            selected[key] = item

    for hit_rank, hit in enumerate(hits):
        context_doc = _context_document(hit, parents)
        consider(RagContextItem(document=context_doc, hit_rank=hit_rank, tier=0, distance=0))
        source = str(hit.metadata.get("source") or "")
        section_key = _section_key(hit)
        hit_index = _chunk_index(hit)
        if section_key is not None and hit_index is not None:
            for document in by_section.get(section_key, []):
                if document is hit:
                    continue
                resolved = _context_document(document, parents)
                doc_index = _chunk_index(document)
                distance = abs(doc_index - hit_index) if doc_index is not None else 0
                consider(
                    RagContextItem(
                        document=resolved,
                        hit_rank=hit_rank,
                        tier=1,
                        distance=distance,
                    )
                )
        if source and hit_index is not None and hit_index >= 0:
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                neighbor_doc = by_source_chunk.get((source, hit_index + offset))
                if neighbor_doc is not None:
                    consider(
                        RagContextItem(
                            document=_context_document(neighbor_doc, parents),
                            hit_rank=hit_rank,
                            tier=2,
                            distance=abs(offset),
                        )
                    )

    return sorted(selected.values(), key=lambda item: item.sort_key())


def trim_rag_context_by_budget(
    items: list[RagContextItem],
    budget: int,
) -> tuple[list[Document], dict[str, int | bool]]:
    """按优先级整片裁剪至 token 预算；至少保留 top-1 hit 原片。

    返回 (documents, stats)，stats 含 context_tokens / context_trimmed / context_budget_exceeded。
    """
    from app.context.window import count_tokens

    if not items:
        return [], {"context_tokens": 0, "context_trimmed": 0, "context_budget_exceeded": False}

    ordered = sorted(items, key=lambda item: item.sort_key())
    if budget <= 0:
        docs = _sort_rag_documents(item.document for item in ordered)
        tokens = sum(count_tokens(format_rag_context_chunk(doc)) for doc in docs)
        return docs, {
            "context_tokens": tokens,
            "context_trimmed": 0,
            "context_budget_exceeded": False,
        }

    kept: list[RagContextItem] = []
    total = 0
    for item in ordered:
        cost = count_tokens(format_rag_context_chunk(item.document))
        if kept and total + cost > budget:
            continue
        kept.append(item)
        total += cost

    top_hit = next((item for item in ordered if item.tier == 0 and item.hit_rank == 0), None)
    budget_exceeded = False
    if top_hit is not None and top_hit not in kept:
        kept = [top_hit]
        total = count_tokens(format_rag_context_chunk(top_hit.document))
        budget_exceeded = total > budget
        if budget_exceeded:
            logger.warning(
                "RAG top-1 命中片 token=%s 超过预算 %s，仍保留整片",
                total,
                budget,
            )
    elif top_hit is not None and top_hit in kept and total > budget:
        budget_exceeded = True

    trimmed = len(ordered) - len(kept)
    docs = _sort_rag_documents(item.document for item in kept)
    return docs, {
        "context_tokens": total,
        "context_trimmed": trimmed,
        "context_budget_exceeded": budget_exceeded,
    }


def _sort_rag_documents(documents: object) -> list[Document]:
    return sorted(
        documents,
        key=lambda doc: (
            str(doc.metadata.get("source") or ""),
            _chunk_index(doc) if _chunk_index(doc) is not None else 10**9,
        ),
    )


def expand_hit_documents(
    hits: list[Document],
    corpus: list[Document],
    *,
    neighbor: int = 1,
) -> list[Document]:
    """命中后补同 section 与相邻切片，不把整个源文件倒进上下文。"""
    return _sort_rag_documents(
        item.document
        for item in expand_hit_documents_scored(hits, corpus, neighbor=neighbor)
    )


def l2_to_similarity(distance: float) -> float:
    """把 FAISS L2 距离映射成 [0, 1] 相似度；距离越小越相似。"""
    return max(0.0, 1.0 - float(distance) / 2.0)


def _fetch_k(k: int) -> int:
    return max(k * 3, 12)


def _vector_inner_k(fetch_k: int, *, course_code: str | None, total: int) -> int:
    multiplier = 8 if course_code else 4
    return min(total, max(fetch_k * multiplier, 32 if course_code else 16))


def _search_vector(
    query: str,
    tenant_id: str,
    fetch_k: int,
    *,
    course_code: str | None = None,
) -> list[tuple[Document, float]]:
    indexes = get_indexes()
    scored: list[tuple[Document, float]] = []
    for scope in search_scopes(tenant_id, course_code=course_code):
        store = indexes[scope]
        if store.index.ntotal <= 0:
            continue
        inner_k = _vector_inner_k(fetch_k, course_code=course_code, total=store.index.ntotal)
        scored.extend(store.similarity_search_with_score(query, k=inner_k))
    ranked: list[tuple[Document, float]] = []
    seen: set[tuple[str, str]] = set()
    for document, distance in sorted(scored, key=lambda item: item[1]):
        if _is_deleted(document) or not _matches_course(document, course_code):
            continue
        if not is_retrieval_chunk(document):
            continue
        key = document_key(document)
        if key in seen:
            continue
        seen.add(key)
        ranked.append((document, l2_to_similarity(distance)))
        if len(ranked) >= fetch_k:
            break
    return ranked


def _bm25_index(scope: str) -> Bm25Index | None:
    with _lexical_lock:
        if scope in _lexical:
            return _lexical[scope]
    store = get_indexes().get(scope)
    built = Bm25Index.build(
        [doc for doc in _iter_store_documents(store) if is_retrieval_chunk(doc)]
    ) if store is not None else None
    with _lexical_lock:
        return _lexical.setdefault(scope, built)


def _search_bm25(
    query: str,
    tenant_id: str,
    fetch_k: int,
    *,
    course_code: str | None = None,
) -> list[tuple[Document, float]]:
    best: dict[tuple[str, str], tuple[Document, float]] = {}
    lexical_fetch = max(fetch_k * 4, 16) if course_code else fetch_k
    for scope in search_scopes(tenant_id, course_code=course_code):
        index = _bm25_index(scope)
        if index is None:
            continue
        for document, score in index.search(query, lexical_fetch):
            if not _matches_course(document, course_code):
                continue
            key = document_key(document)
            previous = best.get(key)
            if previous is None or score > previous[1]:
                best[key] = (document, score)
    return sorted(best.values(), key=lambda item: item[1], reverse=True)[:fetch_k]


def _fuse_hits(
    vector_ranked: list[tuple[Document, float]],
    bm25_ranked: list[tuple[Document, float]],
    *,
    rrf_k: int,
    limit: int,
) -> list[SearchHit]:
    vector_map = {document_key(document): sim for document, sim in vector_ranked}
    bm25_map = {document_key(document): score for document, score in bm25_ranked}
    fused = reciprocal_rank_fusion(
        [
            [document for document, _ in vector_ranked],
            [document for document, _ in bm25_ranked],
        ],
        rrf_k=rrf_k,
        limit=limit,
    )
    return [
        SearchHit(
            document=document,
            score=rrf,
            vector_sim=vector_map.get(document_key(document), 0.0),
            bm25_score=bm25_map.get(document_key(document), 0.0),
        )
        for document, rrf in fused
    ]


def _hits_from_ranked_lists(
    ranked_lists: list[list[Document]],
    vector_maps: list[dict[tuple[str, str], float]],
    bm25_maps: list[dict[tuple[str, str], float]],
    *,
    rrf_k: int,
    limit: int,
) -> list[SearchHit]:
    fused = reciprocal_rank_fusion(ranked_lists, rrf_k=rrf_k, limit=limit)
    vector_map: dict[tuple[str, str], float] = {}
    bm25_map: dict[tuple[str, str], float] = {}
    for item in vector_maps:
        for key, value in item.items():
            vector_map[key] = max(vector_map.get(key, 0.0), value)
    for item in bm25_maps:
        for key, value in item.items():
            bm25_map[key] = max(bm25_map.get(key, 0.0), value)
    return [
        SearchHit(
            document=document,
            score=rrf,
            vector_sim=vector_map.get(document_key(document), 0.0),
            bm25_score=bm25_map.get(document_key(document), 0.0),
        )
        for document, rrf in fused
    ]


def _single_query_hits(
    query: str,
    tenant_id: str,
    fetch_k: int,
    fuse_limit: int,
    *,
    course_code: str | None,
    settings,
) -> tuple[list[SearchHit], list[tuple[Document, float]], list[tuple[Document, float]]]:
    vector_ranked = _search_vector(query, tenant_id, fetch_k, course_code=course_code)
    bm25_ranked = (
        _search_bm25(query, tenant_id, fetch_k, course_code=course_code)
        if settings.rag_hybrid_enabled
        else []
    )
    if not settings.rag_hybrid_enabled:
        hits = [
            SearchHit(document=document, score=sim, vector_sim=sim, bm25_score=0.0)
            for document, sim in vector_ranked[:fuse_limit]
        ]
    else:
        hits = _fuse_hits(
            vector_ranked,
            bm25_ranked,
            rrf_k=settings.rag_rrf_k,
            limit=fuse_limit,
        )
    return hits, vector_ranked, bm25_ranked


def _multi_query_hits(
    queries: list[str],
    tenant_id: str,
    fetch_k: int,
    fuse_limit: int,
    *,
    course_code: str | None,
    settings,
) -> tuple[list[SearchHit], list[tuple[Document, float]], list[tuple[Document, float]]]:
    ranked_lists: list[list[Document]] = []
    vector_maps: list[dict[tuple[str, str], float]] = []
    bm25_maps: list[dict[tuple[str, str], float]] = []
    last_vector: list[tuple[Document, float]] = []
    last_bm25: list[tuple[Document, float]] = []
    for sub_query in queries:
        vector_ranked = _search_vector(sub_query, tenant_id, fetch_k, course_code=course_code)
        bm25_ranked = (
            _search_bm25(sub_query, tenant_id, fetch_k, course_code=course_code)
            if settings.rag_hybrid_enabled
            else []
        )
        last_vector = vector_ranked
        last_bm25 = bm25_ranked
        vector_maps.append({document_key(doc): sim for doc, sim in vector_ranked})
        bm25_maps.append({document_key(doc): score for doc, score in bm25_ranked})
        if settings.rag_hybrid_enabled:
            per_query = _fuse_hits(
                vector_ranked,
                bm25_ranked,
                rrf_k=settings.rag_rrf_k,
                limit=fuse_limit,
            )
            ranked_lists.append([hit.document for hit in per_query])
        else:
            ranked_lists.append([document for document, _ in vector_ranked[:fuse_limit]])
    hits = _hits_from_ranked_lists(
        ranked_lists,
        vector_maps,
        bm25_maps,
        rrf_k=settings.rag_rrf_k,
        limit=fuse_limit,
    )
    return hits, last_vector, last_bm25


def search_tenant_documents(
    query: str,
    tenant_id: str,
    k: int = 8,
    *,
    course_code: str | None = None,
    retrieval_queries: list[str] | None = None,
) -> TenantSearchResult:
    """只在租户（及平台）子索引里检索。默认向量 + BM25 + RRF + 可选 rerank。

    ``course_code`` 按切片 metadata 过滤；名字对不齐时仍可按编码命中该课大纲。
    """
    from app.rag.retrieval_plan import resolve_retrieval_plan

    with _index_lock:
        settings = get_settings()
        plan = resolve_retrieval_plan(
            query,
            result_k=k,
            course_code=course_code,
            settings=settings,
        )
        fetch_k = plan.fetch_k if settings.rag_dynamic_k_enabled else _fetch_k(k)
        fuse_limit = (
            max(fetch_k, settings.rag_rerank_fetch_k)
            if settings.rag_rerank_enabled
            else fetch_k
        )
        active_queries = [part.strip() for part in (retrieval_queries or [query]) if part.strip()] or [query]
        if len(active_queries) > 1:
            hits, vector_ranked, bm25_ranked = _multi_query_hits(
                active_queries,
                tenant_id,
                fetch_k,
                fuse_limit,
                course_code=course_code,
                settings=settings,
            )
        else:
            hits, vector_ranked, bm25_ranked = _single_query_hits(
                query,
                tenant_id,
                fetch_k,
                fuse_limit,
                course_code=course_code,
                settings=settings,
            )

        fused_count = len(hits)
        rerank_enabled = settings.rag_rerank_enabled and bool(hits)
        rerank_top_k = 0
        if rerank_enabled:
            from app.rag.reranker import rerank_hits

            rerank_top_k = min(settings.rag_rerank_top_k, k)
            hits = rerank_hits(
                query,
                hits,
                top_k=rerank_top_k,
                model_name=settings.rag_rerank_model,
                timeout_seconds=settings.rag_rerank_timeout_seconds,
            )
        else:
            hits = hits[:k]

        low_confidence_fallback = False
        if course_code and not hits:
            relaxed = _fuse_hits(
                vector_ranked,
                bm25_ranked if settings.rag_hybrid_enabled else [],
                rrf_k=settings.rag_rrf_k,
                limit=k,
            ) if settings.rag_hybrid_enabled else [
                SearchHit(document=document, score=sim, vector_sim=sim, bm25_score=0.0)
                for document, sim in vector_ranked[:k]
            ]
            if relaxed:
                hits = [
                    SearchHit(
                        document=item.document,
                        score=item.score,
                        vector_sim=item.vector_sim,
                        bm25_score=item.bm25_score,
                        low_confidence=True,
                    )
                    for item in relaxed
                ]
                low_confidence_fallback = True

        return TenantSearchResult(
            hits=hits[:k],
            fetch_k=fetch_k,
            query_profile=plan.profile,
            fused_count=fused_count,
            rerank_enabled=rerank_enabled,
            rerank_top_k=rerank_top_k,
            low_confidence_fallback=low_confidence_fallback,
            retrieval_queries=tuple(active_queries),
        )


def _live_source_hashes(store: FAISS) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for document in _iter_store_documents(store, include_deleted=False):
        source = str(document.metadata.get("source") or "")
        if source:
            hashes[source] = str(document.metadata.get("content_hash") or "")
    return hashes


def _soft_delete_stale(
    store: FAISS,
    sources: set[str],
    keep_hash: dict[str, str],
) -> int:
    """把旧版本标删，刚写入且 hash 匹配的新切片留下。"""
    retired = 0
    for document in _iter_store_documents(store, include_deleted=True):
        source = str(document.metadata.get("source") or "")
        if source not in sources or _is_deleted(document):
            continue
        digest = str(document.metadata.get("content_hash") or "")
        if keep_hash.get(source) and digest == keep_hash[source]:
            continue
        document.metadata["deleted"] = True
        retired += 1
    return retired


def ingest_indexes(
    persist: bool = True,
    *,
    rebuild: bool = False,
    knowledge_dir: Path | None = None,
) -> dict[str, FAISS]:
    """增量 ingest：先写新 chunk，再软删旧切片；未改动文件不重嵌。"""
    with _index_lock:
        if rebuild:
            return build_faiss_index(persist=persist, knowledge_dir=knowledge_dir)
        existing = _indexes if _indexes is not None else _read_indexes_or_rebuild()
        if not existing:
            return build_faiss_index(persist=persist, knowledge_dir=knowledge_dir)

        fresh = load_knowledge_documents(knowledge_dir)
        root = knowledge_dir or KNOWLEDGE_DIR
        if not fresh:
            raise RuntimeError(f"知识库为空：{root}")
        grouped = _group_documents(fresh)
        working = dict(existing)
        added = 0
        retired = 0
        unchanged_files = 0

        for tenant_id in sorted(set(working) | set(grouped)):
            wanted = grouped.get(tenant_id, [])
            wanted_by_source: dict[str, list[Document]] = {}
            wanted_hash: dict[str, str] = {}
            for document in wanted:
                source = str(document.metadata.get("source") or "")
                wanted_by_source.setdefault(source, []).append(document)
                wanted_hash[source] = str(document.metadata.get("content_hash") or "")

            store = working.get(tenant_id)
            live_hash = _live_source_hashes(store) if store is not None else {}
            new_docs: list[Document] = []
            retire_sources: set[str] = set()
            for source, digest in wanted_hash.items():
                if digest and live_hash.get(source) == digest:
                    unchanged_files += 1
                    continue
                new_docs.extend(wanted_by_source[source])
                if source in live_hash:
                    retire_sources.add(source)
            for source in live_hash:
                if source not in wanted_hash:
                    retire_sources.add(source)

            if not new_docs and not retire_sources:
                continue
            if store is None:
                working[tenant_id] = FAISS.from_documents(new_docs, get_embeddings())
                added += len(new_docs)
                continue
            # 先写新切片，再标删旧切片，避免空窗；锁内完成，避免新旧同时可检索
            if new_docs:
                store.add_documents(new_docs)
                added += len(new_docs)
            if retire_sources:
                retired += _soft_delete_stale(store, retire_sources, wanted_hash)

        if persist and (added or retired):
            _persist_indexes(Path(get_settings().faiss_index_path), working)
            revision = disk_index_revision()
            _install_indexes(working, pinned=False, revision=revision)
        elif persist:
            _install_indexes(working, pinned=False, revision=disk_index_revision())
        else:
            _install_indexes(working, pinned=True, revision=f"memory:{time.time_ns()}")
        logger.info(
            "incremental ingest added=%s retired=%s unchanged_files=%s tenants=%s",
            added,
            retired,
            unchanged_files,
            sorted(working),
        )
        return working


def build_faiss_index(
    persist: bool = True,
    knowledge_dir: Path | None = None,
) -> dict[str, FAISS]:
    """全量重建租户子索引。embedding 变更或 --rebuild 时用。"""
    with _index_lock:
        root = knowledge_dir or KNOWLEDGE_DIR
        documents = load_knowledge_documents(root)
        if not documents:
            raise RuntimeError(f"知识库为空：{root}")
        stores = _stores_from_groups(_group_documents(documents))
        if persist:
            _persist_indexes(Path(get_settings().faiss_index_path), stores)
            _install_indexes(stores, pinned=False, revision=disk_index_revision())
        else:
            _install_indexes(stores, pinned=True, revision=f"memory:{time.time_ns()}")
        return stores


def load_faiss_indexes() -> dict[str, FAISS]:
    """加载已有租户索引；不存在则自动构建。"""
    return get_indexes()


def load_faiss_index() -> dict[str, FAISS]:
    """兼容旧调用：加载全部租户索引。"""
    return load_faiss_indexes()
