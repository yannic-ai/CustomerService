"""多进程快照：写入共享目录，刮 /metrics 时合并存活 worker。"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from app.observability.metrics.names import FLUSH_INTERVAL_SECONDS, TOOL_POOL_LABELS
from app.observability.metrics.store import (
    Hist,
    _COUNTERS,
    _HISTOGRAMS,
    label_key,
)

logger = logging.getLogger("cs.metrics")

_MULTIPROC_DIR: Path | None = None


def configure_metrics_multiprocess(
    directory: str | None = None,
    *,
    workers: int = 1,
) -> Path | None:
    """启用跨 worker 快照目录。``workers>1`` 或目录/环境变量任一成立即开启。

    须在 ``uvicorn.run(..., workers=N)`` 之前调用，以便子进程继承
    ``PROMETHEUS_MULTIPROC_DIR``。
    """
    global _MULTIPROC_DIR
    explicit = (directory or os.environ.get("PROMETHEUS_MULTIPROC_DIR") or "").strip()
    if not explicit and int(workers or 1) <= 1:
        return _MULTIPROC_DIR
    if explicit:
        path = Path(explicit)
    else:
        from app.config import DATA_DIR

        path = DATA_DIR / "prometheus_multiproc"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = str(path)
    _MULTIPROC_DIR = path
    _cleanup_dead_pid_files(path)
    logger.info("Prometheus 多进程目录 %s workers=%s", path, workers)
    return path


def configure_metrics_multiprocess_from_settings(settings: Any | None = None) -> Path | None:
    """按 Settings / 环境变量启用多进程指标。"""
    if settings is None:
        from app.config import get_settings

        settings = get_settings()
    directory = (getattr(settings, "prometheus_multiproc_dir", "") or "").strip() or None
    workers = int(getattr(settings, "uvicorn_workers", 1) or 1)
    return configure_metrics_multiprocess(directory, workers=workers)


def metrics_multiproc_enabled() -> bool:
    """当前进程是否会把快照写到共享目录。"""
    return _MULTIPROC_DIR is not None


def disable_metrics_multiprocess() -> None:
    """测试用：关掉聚合并删掉本 pid 文件。"""
    global _MULTIPROC_DIR
    mark_metrics_process_dead()
    _MULTIPROC_DIR = None
    os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)


def mark_metrics_process_dead() -> None:
    """进程退出前删掉本 pid 快照，避免把已死 worker 的计数永远加进去。"""
    remove_pid_file(os.getpid())


def flush_metrics_if_multiprocess() -> None:
    """对话槽 / 工具池水位变化时强制刷盘，让其他 worker 刮到的 gauge 不过期。"""
    flush_snapshot(force=True)


def snapshot_path(pid: int, directory: Path | None = None) -> Path:
    root = directory or _MULTIPROC_DIR
    assert root is not None
    return root / f"cs-{pid}.json"


_snapshot_path = snapshot_path


def remove_pid_file(pid: int) -> None:
    if _MULTIPROC_DIR is None:
        return
    path = snapshot_path(pid)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return


_remove_pid_file = remove_pid_file


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


_pid_alive = pid_alive


def _current_pid_alive(pid: int) -> bool:
    """允许测试 patch ``app.observability.metrics._pid_alive``。"""
    from app.observability import metrics as metrics_pkg

    fn = getattr(metrics_pkg, "_pid_alive", pid_alive)
    return fn(pid)


def _cleanup_dead_pid_files(directory: Path) -> None:
    for path in directory.glob("cs-*.json"):
        pid = pid_from_name(path.name)
        if pid is None or not _current_pid_alive(pid):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue


def pid_from_name(name: str) -> int | None:
    if not name.startswith("cs-") or not name.endswith(".json"):
        return None
    raw = name[3:-5]
    if not raw.isdigit():
        return None
    return int(raw)


_pid_from_name = pid_from_name


def local_in_flight() -> int:
    try:
        from app.concurrency import in_flight_count

        return int(in_flight_count())
    except Exception:
        return 0


_local_in_flight = local_in_flight


def empty_tool_pool_gauges() -> dict[str, int]:
    return {pool: 0 for pool in TOOL_POOL_LABELS}


_empty_tool_pool_gauges = empty_tool_pool_gauges


def local_tool_pool_in_flight() -> dict[str, int]:
    try:
        from app.executors import tool_pool_in_flight_counts

        counts = tool_pool_in_flight_counts()
    except Exception:
        counts = {}
    return {pool: int(counts.get(pool, 0) or 0) for pool in TOOL_POOL_LABELS}


_local_tool_pool_in_flight = local_tool_pool_in_flight


def merge_tool_pool_gauges(dest: dict[str, int], raw: object) -> None:
    if not isinstance(raw, dict):
        return
    for pool in TOOL_POOL_LABELS:
        dest[pool] = dest.get(pool, 0) + int(raw.get(pool) or 0)


_merge_tool_pool_gauges = merge_tool_pool_gauges


def hist_to_json(hist: Hist) -> dict[str, Any]:
    bounds: list[float | None] = []
    for bound in hist.bounds:
        bounds.append(None if bound == float("inf") else float(bound))
    return {
        "bounds": bounds,
        "buckets": [int(x) for x in hist.buckets],
        "count": int(hist.count),
        "sum": float(hist.sum),
    }


def hist_from_json(payload: dict[str, Any]) -> Hist | None:
    raw_bounds = payload.get("bounds") or []
    bounds = tuple(float("inf") if item is None else float(item) for item in raw_bounds)
    if not bounds:
        return None
    hist = Hist(bounds)
    buckets = payload.get("buckets") or []
    if len(buckets) != len(bounds):
        return None
    hist.buckets = [int(x) for x in buckets]
    hist.count = int(payload.get("count") or 0)
    hist.sum = float(payload.get("sum") or 0.0)
    return hist


def _snapshot_payload_unlocked() -> dict[str, Any]:
    counters: dict[str, list[dict[str, Any]]] = {}
    for name, series in _COUNTERS.items():
        counters[name] = [{"labels": dict(labels), "value": value} for labels, value in series.items()]
    histograms: dict[str, list[dict[str, Any]]] = {}
    for name, series in _HISTOGRAMS.items():
        histograms[name] = [
            {"labels": dict(labels), **hist_to_json(hist)} for labels, hist in series.items()
        ]
    return {
        "pid": os.getpid(),
        "counters": counters,
        "histograms": histograms,
        "gauges": {
            "cs_chat_in_flight": local_in_flight(),
            "cs_tool_pool_in_flight": local_tool_pool_in_flight(),
        },
    }


def flush_snapshot(*, force: bool = False) -> None:
    from app.observability.metrics import store as metrics_store

    if _MULTIPROC_DIR is None:
        return
    now = time.monotonic()
    with metrics_store._LOCK:
        if not force and not metrics_store._DIRTY:
            return
        if not force and (now - metrics_store._LAST_FLUSH) < FLUSH_INTERVAL_SECONDS:
            return
        payload = _snapshot_payload_unlocked()
        metrics_store._DIRTY = False
        metrics_store._LAST_FLUSH = now
    path = snapshot_path(os.getpid())
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        logger.debug("metrics 快照写入失败: %s", exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


_flush_snapshot = flush_snapshot


def _load_snapshot(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def add_counter(
    dest: dict[str, dict[tuple[tuple[str, str], ...], float]],
    name: str,
    labels: dict[str, str],
    value: float,
) -> None:
    series = dest.setdefault(name, {})
    key = label_key(labels)
    series[key] = series.get(key, 0.0) + float(value)


def add_histogram(
    dest: dict[str, dict[tuple[tuple[str, str], ...], Hist]],
    name: str,
    labels: dict[str, str],
    hist: Hist,
) -> None:
    series = dest.setdefault(name, {})
    key = label_key(labels)
    existing = series.get(key)
    if existing is None:
        merged = Hist(hist.bounds)
        merged.buckets = list(hist.buckets)
        merged.count = hist.count
        merged.sum = hist.sum
        series[key] = merged
        return
    if existing.bounds != hist.bounds:
        return
    existing.count += hist.count
    existing.sum += hist.sum
    for index, amount in enumerate(hist.buckets):
        existing.buckets[index] += amount


def merge_snapshots(
    files: list[Path],
) -> tuple[
    dict[str, dict[tuple[tuple[str, str], ...], float]],
    dict[str, dict[tuple[tuple[str, str], ...], Hist]],
    int,
    dict[str, int],
]:
    counters: dict[str, dict[tuple[tuple[str, str], ...], float]] = {}
    histograms: dict[str, dict[tuple[tuple[str, str], ...], Hist]] = {}
    in_flight = 0
    tool_pool = empty_tool_pool_gauges()
    for path in files:
        payload = _load_snapshot(path)
        if not payload:
            continue
        for name, rows in (payload.get("counters") or {}).items():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                add_counter(counters, name, dict(row.get("labels") or {}), float(row.get("value") or 0))
        for name, rows in (payload.get("histograms") or {}).items():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                hist = hist_from_json(row)
                if hist is None:
                    continue
                add_histogram(histograms, name, dict(row.get("labels") or {}), hist)
        gauges = payload.get("gauges") or {}
        if isinstance(gauges, dict):
            in_flight += int(gauges.get("cs_chat_in_flight") or 0)
            merge_tool_pool_gauges(tool_pool, gauges.get("cs_tool_pool_in_flight"))
    return counters, histograms, in_flight, tool_pool


def collect_multiprocess() -> tuple[
    dict[str, dict[tuple[tuple[str, str], ...], float]],
    dict[str, dict[tuple[tuple[str, str], ...], Hist]],
    int,
    dict[str, int],
] | None:
    directory = _MULTIPROC_DIR
    if directory is None:
        return None
    flush_snapshot(force=True)
    files: list[Path] = []
    for path in directory.glob("cs-*.json"):
        pid = pid_from_name(path.name)
        if pid is None or not _current_pid_alive(pid):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        files.append(path)
    if not files:
        return None
    return merge_snapshots(files)


_collect_multiprocess = collect_multiprocess
