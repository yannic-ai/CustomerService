"""进程内计数器 / 直方图。``_inc`` / ``_observe`` 是唯一写入入口。"""

from __future__ import annotations

import os
import threading

from app.observability.metrics.names import DURATION_BUCKETS

_LOCK = threading.Lock()
_COUNTERS: dict[str, dict[tuple[tuple[str, str], ...], float]] = {}
_HISTOGRAMS: dict[str, dict[tuple[tuple[str, str], ...], Hist]] = {}
_DIRTY = False
_LAST_FLUSH = 0.0


class Hist:
    __slots__ = ("bounds", "buckets", "count", "sum")

    def __init__(self, bounds: tuple[float, ...] = DURATION_BUCKETS) -> None:
        self.bounds = bounds
        self.buckets = [0] * len(bounds)
        self.count = 0
        self.sum = 0.0


# 兼容旧测试与内部命名
_Hist = Hist


def label_key(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


_label_key = label_key


def inc(name: str, amount: float = 1, **labels: str) -> None:
    global _DIRTY
    key = label_key(labels)
    with _LOCK:
        series = _COUNTERS.setdefault(name, {})
        series[key] = series.get(key, 0.0) + amount
        _DIRTY = True
    from app.observability.metrics.multiproc import flush_snapshot

    flush_snapshot()


def observe(
    name: str,
    value: float,
    *,
    buckets: tuple[float, ...] = DURATION_BUCKETS,
    **labels: str,
) -> None:
    global _DIRTY
    key = label_key(labels)
    with _LOCK:
        series = _HISTOGRAMS.setdefault(name, {})
        hist = series.get(key)
        if hist is None:
            hist = Hist(buckets)
            series[key] = hist
        hist.count += 1
        hist.sum += value
        for index, bound in enumerate(hist.bounds):
            if value <= bound:
                hist.buckets[index] += 1
        _DIRTY = True
    from app.observability.metrics.multiproc import flush_snapshot

    flush_snapshot()


_inc = inc
_observe = observe


def reset_metrics() -> None:
    """测试用：清空本进程计数，并删掉本 pid 的共享快照。"""
    global _DIRTY
    with _LOCK:
        _COUNTERS.clear()
        _HISTOGRAMS.clear()
        _DIRTY = False
    from app.observability.metrics.multiproc import remove_pid_file

    remove_pid_file(os.getpid())
