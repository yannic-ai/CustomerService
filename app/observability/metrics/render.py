"""Prometheus 文本导出。"""

from __future__ import annotations

from collections.abc import Iterable

from app.observability.metrics.names import METRIC_HELP, TOOL_POOL_LABELS
from app.observability.metrics.store import Hist, _COUNTERS, _HISTOGRAMS, _LOCK, label_key


def _esc(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{key}="{_esc(value)}"' for key, value in labels)
    return "{" + inner + "}"


def _format_counter_value(value: float) -> str:
    """整数计数保持整型；成本等浮点计数保留小数。"""
    if abs(value - round(value)) < 1e-9:
        return f"{round(value):.0f}"
    return f"{value:.6f}"


def _counter_lines(name: str, series: dict[tuple[tuple[str, str], ...], float]) -> Iterable[str]:
    yield f"# HELP {name} {METRIC_HELP.get(name, name)}"
    yield f"# TYPE {name} counter"
    for labels, value in sorted(series.items()):
        yield f"{name}{_format_labels(labels)} {_format_counter_value(value)}"


def _hist_lines(name: str, series: dict[tuple[tuple[str, str], ...], Hist]) -> Iterable[str]:
    yield f"# HELP {name} {METRIC_HELP.get(name, name)}"
    yield f"# TYPE {name} histogram"
    for labels, hist in sorted(series.items()):
        label_map = dict(labels)
        for bound, count in zip(hist.bounds, hist.buckets):
            le = "+Inf" if bound == float("inf") else f"{bound:g}"
            bucket_labels = label_key({**label_map, "le": le})
            yield f"{name}_bucket{_format_labels(bucket_labels)} {count}"
        yield f"{name}_sum{_format_labels(labels)} {hist.sum:.6f}"
        yield f"{name}_count{_format_labels(labels)} {hist.count}"


def _tool_pool_gauge_lines(counts: dict[str, int]) -> Iterable[str]:
    yield f"# HELP cs_tool_pool_in_flight {METRIC_HELP['cs_tool_pool_in_flight']}"
    yield "# TYPE cs_tool_pool_in_flight gauge"
    for pool in sorted(TOOL_POOL_LABELS):
        yield f'cs_tool_pool_in_flight{{pool="{pool}"}} {int(counts.get(pool, 0))}'


def render_prometheus() -> str:
    """导出 Prometheus 文本；多 worker 时合并共享目录里所有存活进程。"""
    from app.observability.metrics.multiproc import (
        collect_multiprocess,
        local_in_flight,
        local_tool_pool_in_flight,
    )

    merged = collect_multiprocess()
    if merged is not None:
        counters, histograms, in_flight, tool_pool = merged
    else:
        with _LOCK:
            counters = {k: dict(v) for k, v in _COUNTERS.items()}
            histograms = {k: dict(v) for k, v in _HISTOGRAMS.items()}
        in_flight = local_in_flight()
        tool_pool = local_tool_pool_in_flight()
    try:
        from app.chat_slots import get_chat_slot_store

        store = get_chat_slot_store()
        if not store.degraded:
            in_flight = int(store.in_flight())
    except Exception:
        pass
    lines: list[str] = []
    for name in sorted(counters):
        lines.extend(_counter_lines(name, counters[name]))
    for name in sorted(histograms):
        lines.extend(_hist_lines(name, histograms[name]))
    lines.append(f"# HELP cs_chat_in_flight {METRIC_HELP['cs_chat_in_flight']}")
    lines.append("# TYPE cs_chat_in_flight gauge")
    lines.append(f"cs_chat_in_flight {in_flight}")
    lines.extend(_tool_pool_gauge_lines(tool_pool))
    lines.append("")
    return "\n".join(lines)
