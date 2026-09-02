"""线上失败样本回流：写入 evals/pending 待人工审核。"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from app.config import ROOT_DIR, get_settings
from app.evalsets import EVAL_DIR

PendingKind = Literal["rag", "faithfulness"]

_DEFAULT_PENDING_DIR = EVAL_DIR / "pending"


@dataclass
class TurnSnapshot:
    """一轮对话结束时的 RAG 上下文，供赞踩关联。"""

    trace_id: str
    tenant_id: str
    original_query: str
    search_query: str = ""
    intent: str = "chitchat"
    rag_empty: bool = False
    sources: list[str] = field(default_factory=list)
    course_name: str = ""
    rewrite_reason: str = ""
    answer_preview: str = ""
    session_id: str = ""
    request_id: str = ""


_snapshot_cache: OrderedDict[str, tuple[float, TurnSnapshot]] = OrderedDict()
_seen_pending: set[str] = set()


def _pending_dir() -> Path:
    settings = get_settings()
    raw = (settings.rag_eval_harvest_dir or "").strip()
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT_DIR / path
        return path
    return _DEFAULT_PENDING_DIR


def _harvest_enabled() -> bool:
    return bool(get_settings().rag_eval_harvest_enabled)


def _case_id(tenant: str, query: str, reason: str) -> str:
    digest = hashlib.sha256(f"{tenant}|{query}|{reason}".encode()).hexdigest()[:12]
    return f"pending-{digest}"


def _dedupe_key(tenant: str, query: str, reason: str) -> str:
    return hashlib.sha256(f"{tenant}|{query}|{reason}".encode()).hexdigest()


def _prune_snapshot_cache() -> None:
    settings = get_settings()
    ttl = max(60, int(settings.rag_eval_snapshot_ttl_seconds))
    now = time.time()
    expired = [key for key, (ts, _) in _snapshot_cache.items() if now - ts > ttl]
    for key in expired:
        _snapshot_cache.pop(key, None)
    max_size = 1000
    while len(_snapshot_cache) > max_size:
        _snapshot_cache.popitem(last=False)


def cache_turn_snapshot(snapshot: TurnSnapshot) -> None:
    """缓存本轮快照，供 feedback 关联。"""
    tid = (snapshot.trace_id or "").strip()
    if not tid:
        return
    _snapshot_cache[tid] = (time.time(), snapshot)
    _snapshot_cache.move_to_end(tid)
    _prune_snapshot_cache()


def get_turn_snapshot(trace_id: str) -> TurnSnapshot | None:
    tid = (trace_id or "").strip()
    if not tid:
        return None
    entry = _snapshot_cache.get(tid)
    if entry is None:
        return None
    ts, snapshot = entry
    ttl = max(60, int(get_settings().rag_eval_snapshot_ttl_seconds))
    if time.time() - ts > ttl:
        _snapshot_cache.pop(tid, None)
        return None
    return snapshot


def append_pending(case: dict[str, Any], *, kind: PendingKind = "rag") -> bool:
    """去重后 append 到 pending JSONL；返回是否写入。"""
    if not _harvest_enabled():
        return False
    tenant = str(case.get("tenant_id") or "demo")
    query = str(case.get("query") or "")
    reason = str(case.get("source") or case.get("id") or "unknown")
    key = _dedupe_key(tenant, query, reason)
    if key in _seen_pending:
        return False

    pending_dir = _pending_dir()
    pending_dir.mkdir(parents=True, exist_ok=True)
    filename = "rag.jsonl" if kind == "rag" else "faithfulness.jsonl"
    path = pending_dir / filename

    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            existing_key = _dedupe_key(
                str(row.get("tenant_id") or "demo"),
                str(row.get("query") or ""),
                str(row.get("source") or row.get("id") or ""),
            )
            if existing_key == key:
                _seen_pending.add(key)
                return False

    if "id" not in case:
        case["id"] = _case_id(tenant, query, reason)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    _seen_pending.add(key)
    return True


def record_retrieval_miss(
    *,
    tenant_id: str,
    query: str,
    search_query: str = "",
    reason: str = "rag_empty",
    trace_id: str = "",
    rewrite_reason: str = "",
) -> bool:
    """空召回或低分拒答写入 pending。"""
    case: dict[str, Any] = {
        "source": reason,
        "query": query,
        "tenant_id": tenant_id,
        "expect_empty": True,
        "meta": {
            "trace_id": trace_id,
            "search_query": search_query,
            "rewrite_reason": rewrite_reason,
        },
    }
    return append_pending(case, kind="rag")


def record_feedback_down(trace_id: str, *, comment: str | None = None) -> bool:
    """用户踩时从 snapshot 生成 faithfulness 候选。"""
    snapshot = get_turn_snapshot(trace_id)
    if snapshot is None:
        return False
    case: dict[str, Any] = {
        "source": "user_feedback_down",
        "query": snapshot.original_query,
        "tenant_id": snapshot.tenant_id,
        "expect_course_contains": snapshot.course_name or None,
        "meta": {
            "trace_id": trace_id,
            "search_query": snapshot.search_query,
            "rewrite_reason": snapshot.rewrite_reason,
            "sources": snapshot.sources,
            "comment": comment or "",
            "rag_empty": snapshot.rag_empty,
        },
    }
    case = {k: v for k, v in case.items() if v is not None}
    return append_pending(case, kind="faithfulness")


def export_eval_failure(
    *,
    test_name: str,
    case_id: str,
    row: dict[str, Any],
    message: str,
) -> bool:
    """离线评测失败导出到 pending。"""
    tenant = str(row.get("tenant_id") or "demo")
    case: dict[str, Any] = {
        "source": f"eval_fail:{test_name}",
        "query": row.get("query", ""),
        "tenant_id": tenant,
        "meta": {"case_id": case_id, "message": message, "original": row},
    }
    if row.get("expect_empty"):
        case["expect_empty"] = True
    if row.get("expect_sources"):
        case["expect_sources"] = row["expect_sources"]
    if row.get("expect_not_found"):
        case["expect_not_found"] = True
    kind: PendingKind = "faithfulness" if test_name == "faithfulness" else "rag"
    return append_pending(case, kind=kind)


def list_pending_files() -> list[Path]:
    pending_dir = _pending_dir()
    if not pending_dir.is_dir():
        return []
    return sorted(pending_dir.glob("*.jsonl"))


def merge_pending(*, apply: bool = False) -> list[str]:
    """预览或合并 pending 到正式 evals 集。"""
    lines_out: list[str] = []
    for pending_path in list_pending_files():
        target_name = pending_path.name
        target = EVAL_DIR / target_name
        rows: list[str] = []
        for line in pending_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            rows.append(text)
        if not rows:
            continue
        preview = f"{target_name}: +{len(rows)} rows -> {target}"
        lines_out.append(preview)
        if apply:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(row + "\n")
            pending_path.write_text("", encoding="utf-8")
    return lines_out


def reset_harvest_state_for_tests() -> None:
    """测试隔离：清空内存缓存与去重集。"""
    _snapshot_cache.clear()
    _seen_pending.clear()


def maybe_harvest_eval_failure(
    *,
    test_name: str,
    case_id: str,
    row: dict[str, Any],
    message: str,
) -> None:
    """EVAL_HARVEST=1 时将评测失败写入 pending。"""
    import os

    if os.environ.get("EVAL_HARVEST") != "1":
        return
    export_eval_failure(test_name=test_name, case_id=case_id, row=row, message=message)
