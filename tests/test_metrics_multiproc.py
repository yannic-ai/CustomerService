"""多 worker 时 /metrics 合并共享目录里所有存活进程的快照。"""

from __future__ import annotations

import json
import os
from pathlib import Path

from app.observability.metrics import (
    configure_metrics_multiprocess,
    disable_metrics_multiprocess,
    inc_http,
    metrics_multiproc_enabled,
    observe_http,
    render_prometheus,
    reset_metrics,
)


def _write_worker_snapshot(
    directory: Path,
    pid: int,
    *,
    http_count: float = 0,
    in_flight: int = 0,
    duration_sum: float = 0.0,
    duration_count: int = 0,
    tool_pool: dict[str, int] | None = None,
) -> Path:
    path = directory / f"cs-{pid}.json"
    counters = []
    if http_count:
        counters.append(
            {
                "labels": {"method": "GET", "path": "/chat", "status": "200"},
                "value": http_count,
            }
        )
    histograms = []
    if duration_count:
        histograms.append(
            {
                "labels": {"method": "GET", "path": "/chat", "status": "200"},
                "bounds": [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, None],
                "buckets": [0, 0, duration_count, duration_count, duration_count, duration_count, duration_count, duration_count, duration_count, duration_count],
                "count": duration_count,
                "sum": duration_sum,
            }
        )
    gauges: dict[str, object] = {"cs_chat_in_flight": in_flight}
    if tool_pool is not None:
        gauges["cs_tool_pool_in_flight"] = tool_pool
    path.write_text(
        json.dumps(
            {
                "pid": pid,
                "counters": {"cs_http_requests_total": counters} if counters else {},
                "histograms": {"cs_http_duration_seconds": histograms} if histograms else {},
                "gauges": gauges,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_single_process_does_not_write_snapshot(tmp_path: Path) -> None:
    reset_metrics()
    disable_metrics_multiprocess()
    inc_http("GET", "/chat", 200)
    assert list(tmp_path.glob("cs-*.json")) == []
    assert metrics_multiproc_enabled() is False
    body = render_prometheus()
    assert 'cs_http_requests_total{method="GET",path="/chat",status="200"} 1' in body


def test_multiproc_merges_other_live_worker(tmp_path: Path, monkeypatch) -> None:
    reset_metrics()
    configure_metrics_multiprocess(str(tmp_path), workers=2)
    other_pid = 424242
    monkeypatch.setattr(
        "app.observability.metrics._pid_alive",
        lambda pid: pid in {os.getpid(), other_pid},
    )
    try:
        inc_http("GET", "/chat", 200)
        _write_worker_snapshot(tmp_path, other_pid, http_count=4, in_flight=3)
        body = render_prometheus()
        assert 'cs_http_requests_total{method="GET",path="/chat",status="200"} 5' in body
        assert "cs_chat_in_flight 3" in body
        assert (tmp_path / f"cs-{os.getpid()}.json").exists()
    finally:
        disable_metrics_multiprocess()
        reset_metrics()


def test_multiproc_drops_dead_worker_file(tmp_path: Path, monkeypatch) -> None:
    reset_metrics()
    configure_metrics_multiprocess(str(tmp_path), workers=2)
    dead_pid = 424243
    monkeypatch.setattr(
        "app.observability.metrics._pid_alive",
        lambda pid: pid == os.getpid(),
    )
    try:
        inc_http("GET", "/chat", 200)
        dead = _write_worker_snapshot(tmp_path, dead_pid, http_count=9, in_flight=5)
        body = render_prometheus()
        assert not dead.exists()
        assert 'cs_http_requests_total{method="GET",path="/chat",status="200"} 1' in body
        assert "cs_chat_in_flight 0" in body
    finally:
        disable_metrics_multiprocess()
        reset_metrics()


def test_multiproc_merges_histograms(tmp_path: Path, monkeypatch) -> None:
    reset_metrics()
    configure_metrics_multiprocess(str(tmp_path), workers=2)
    other_pid = 424244
    monkeypatch.setattr(
        "app.observability.metrics._pid_alive",
        lambda pid: pid in {os.getpid(), other_pid},
    )
    try:
        observe_http("GET", "/chat", 200, 0.2)
        _write_worker_snapshot(
            tmp_path,
            other_pid,
            duration_count=2,
            duration_sum=0.4,
        )
        body = render_prometheus()
        assert 'cs_http_duration_seconds_count{method="GET",path="/chat",status="200"} 3' in body
        assert 'cs_http_duration_seconds_sum{method="GET",path="/chat",status="200"} 0.600000' in body
    finally:
        disable_metrics_multiprocess()
        reset_metrics()


def test_multiproc_merges_tool_pool_in_flight(tmp_path: Path, monkeypatch) -> None:
    reset_metrics()
    configure_metrics_multiprocess(str(tmp_path), workers=2)
    other_pid = 424245
    monkeypatch.setattr(
        "app.observability.metrics._pid_alive",
        lambda pid: pid in {os.getpid(), other_pid},
    )
    try:
        _write_worker_snapshot(
            tmp_path,
            other_pid,
            tool_pool={"order": 2, "course": 1},
        )
        body = render_prometheus()
        assert 'cs_tool_pool_in_flight{pool="order"} 2' in body
        assert 'cs_tool_pool_in_flight{pool="course"} 1' in body
    finally:
        disable_metrics_multiprocess()
        reset_metrics()


def test_workers_gt_one_without_dir_uses_data_folder(monkeypatch, tmp_path: Path) -> None:
    reset_metrics()
    monkeypatch.setattr("app.config.DATA_DIR", tmp_path)
    try:
        path = configure_metrics_multiprocess(None, workers=2)
        assert path == tmp_path / "prometheus_multiproc"
        assert path is not None and path.is_dir()
        assert os.environ.get("PROMETHEUS_MULTIPROC_DIR") == str(path)
    finally:
        disable_metrics_multiprocess()
        os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
        reset_metrics()
