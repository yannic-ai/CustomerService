"""工具线程池背压单测：无界队列 → 有界信号量，池满 busy、信号量不泄漏、本地执行兼容。"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.executors import (
    ToolPoolBusyError,
    _DEFAULT_QUEUE_WAIT_SECONDS,
    _get_order_pool,
    _pool_capacity,
    _submit,
    run_order,
    shutdown_tool_pools,
)
from app.observability import reset_metrics


@pytest.fixture(autouse=True)
def _reset_pools_and_metrics() -> None:
    shutdown_tool_pools()
    reset_metrics()
    yield
    shutdown_tool_pools()
    reset_metrics()


# ---------------------------------------------------------------------------
# 1. 容量计算
# ---------------------------------------------------------------------------
def test_pool_capacity_default_factor_2() -> None:
    # workers=4, factor=2 → cap=8
    assert _pool_capacity(4) == 8


def test_pool_capacity_min_1() -> None:
    # workers=0 在 get_pool 内部会返回 None；_pool_capacity 兜底：max(1, max(1, 0))*factor = 1*2 = 2
    # workers=1 时 cap = max(1, 1) * 2 = 2
    assert _pool_capacity(1) == 2
    # workers=0 时也保证至少 1
    assert _pool_capacity(0) >= 1


# ---------------------------------------------------------------------------
# 2. pool=None / workers=0 → 本地直接执行（兼容原行为）
# ---------------------------------------------------------------------------
def test_submit_without_pool_runs_locally() -> None:
    result = _submit(None, None, "order", lambda x: x * 2, 21)
    assert result == 42


def test_run_order_without_workers_runs_locally(monkeypatch) -> None:
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "order_tool_workers", 0)
    # 强制重建池
    shutdown_tool_pools()
    assert run_order(lambda: 7) == 7


# ---------------------------------------------------------------------------
# 3. 背压触发：当信号量被占满时，新 submit → queue_wait 超时 → ToolPoolBusyError
# ---------------------------------------------------------------------------
def test_backpressure_triggers_tool_pool_busy() -> None:
    """占空信号量后，submit 极短 queue_wait → 必触发 ToolPoolBusyError。

    直接手动 acquire 占满 semaphore（不启动真实任务，更稳定、无死锁）。
    """
    pool, sem = _get_order_pool()
    assert pool is not None and sem is not None
    cap = _pool_capacity(4)  # 默认 4 workers → cap=8

    acquired_tokens: list = []
    try:
        # 1) 手动把信号量所有槽位占空
        for _ in range(cap):
            ok = sem.acquire(timeout=1)
            assert ok, "failed to pre-fill semaphore slots"
            acquired_tokens.append(True)
        assert sem._value == 0  # type: ignore[attr-defined]

        # 2) 此时 submit 拿不到槽位 → 0.001s 后 busy
        with pytest.raises(ToolPoolBusyError) as excinfo:
            _submit(
                pool,
                sem,
                "order",
                lambda: None,
                queue_wait_timeout=0.001,
            )
        assert "order pool busy" in str(excinfo.value)
    finally:
        # 还回去
        for _ in acquired_tokens:
            sem.release()


# ---------------------------------------------------------------------------
# 4. 信号量泄漏测试：正常 / 超时 / 异常 三条路径结束后 sem.value 应回到初始 cap
# ---------------------------------------------------------------------------
def test_semaphore_not_leaked_on_normal() -> None:
    pool, sem = _get_order_pool()
    assert pool is not None and sem is not None
    cap = _pool_capacity(4)
    assert sem._value == cap  # type: ignore[attr-defined]

    def _fn() -> int:
        return 123

    for _ in range(5):
        assert _submit(pool, sem, "order", _fn) == 123
    # 任务完成后 value 必须回到 cap（允许由于 done_callback 异步性 +1 临时偏差，但最终稳定）
    time.sleep(0.05)
    assert sem._value == cap  # type: ignore[attr-defined]


def test_semaphore_not_leaked_on_timeout() -> None:
    pool, sem = _get_order_pool()
    assert pool is not None and sem is not None
    cap = _pool_capacity(4)

    def _slow() -> int:
        time.sleep(2.0)
        return 1

    with pytest.raises(TimeoutError):
        _submit(pool, sem, "order", _slow, timeout=0.05)

    # 超时后 future.cancel() 被调用；但任务可能已经开始跑，需等它真正结束
    time.sleep(2.2)
    assert sem._value == cap  # type: ignore[attr-defined]


def test_semaphore_not_leaked_on_exception() -> None:
    pool, sem = _get_order_pool()
    assert pool is not None and sem is not None
    cap = _pool_capacity(4)

    def _boom() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError):
        _submit(pool, sem, "order", _boom)

    time.sleep(0.05)
    assert sem._value == cap  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 5. 重入保护：在池内 worker 上再次 run_order 不应死锁（应直接本地执行不 acquire）
# ---------------------------------------------------------------------------
def test_reentrant_call_runs_inline_no_deadlock() -> None:
    pool, sem = _get_order_pool()
    assert pool is not None and sem is not None
    cap = _pool_capacity(4)

    def _inner() -> int:
        # 池内再次 run_order → TLS.pool_key 匹配 → 直接本地执行
        return run_order(lambda: 99)

    result = _submit(pool, sem, "order", _inner)
    assert result == 99
    time.sleep(0.05)
    assert sem._value == cap  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 6. 观测指标：被拒绝时 rejected 计数器 +1；排队时间直方图记录
# ---------------------------------------------------------------------------
def test_busy_records_rejected_metric() -> None:
    from app.observability.metrics import _COUNTERS

    pool, sem = _get_order_pool()
    assert pool is not None and sem is not None
    # 直接把 sem 占空
    while sem._value > 0:  # type: ignore[attr-defined]
        sem.acquire(blocking=False)

    try:
        with pytest.raises(ToolPoolBusyError):
            _submit(pool, sem, "order", lambda: 1, queue_wait_timeout=0.001)
    finally:
        # 还回去
        while sem._value < _pool_capacity(4):  # type: ignore[attr-defined]
            sem.release()

    # 检查 cs_tool_pool_rejected_total 计数器
    key = (("pool", "order"),)
    rejected = _COUNTERS.get("cs_tool_pool_rejected_total", {}).get(key, 0)
    assert rejected >= 1


def test_submit_records_queue_wait_histogram() -> None:
    from app.observability.metrics import _HISTOGRAMS

    pool, sem = _get_order_pool()
    assert pool is not None and sem is not None
    _submit(pool, sem, "order", lambda: None)
    time.sleep(0.05)

    key = (("pool", "order"),)
    hist = _HISTOGRAMS.get("cs_tool_pool_queue_seconds", {}).get(key)
    assert hist is not None
    assert hist.count >= 1
    assert hist.sum >= 0.0


# ---------------------------------------------------------------------------
# 7. queue_wait 兜底默认值：未配置时不永久阻塞
# ---------------------------------------------------------------------------
def test_default_queue_wait_finite() -> None:
    assert isinstance(_DEFAULT_QUEUE_WAIT_SECONDS, float)
    # 默认值不应永久（不是 inf，不是很大）
    assert 0 < _DEFAULT_QUEUE_WAIT_SECONDS <= 30


# ---------------------------------------------------------------------------
# 8. Prometheus 导出工具池实时水位
# ---------------------------------------------------------------------------
def test_prometheus_always_exports_tool_pool_in_flight_zeros() -> None:
    from app.executors import tool_pool_in_flight_counts
    from app.observability.metrics import render_prometheus

    body = render_prometheus()
    assert 'cs_tool_pool_in_flight{pool="course"} 0' in body
    assert 'cs_tool_pool_in_flight{pool="order"} 0' in body
    assert tool_pool_in_flight_counts() == {"order": 0, "course": 0}


def test_prometheus_tool_pool_in_flight_tracks_running_task() -> None:
    from app.executors import tool_pool_in_flight_counts
    from app.observability.metrics import render_prometheus

    started = threading.Event()
    release = threading.Event()

    def _hold() -> str:
        started.set()
        assert release.wait(timeout=5)
        return "ok"

    worker = threading.Thread(target=lambda: run_order(_hold), daemon=True)
    worker.start()
    assert started.wait(timeout=5)
    try:
        counts = tool_pool_in_flight_counts()
        assert counts["order"] >= 1
        body = render_prometheus()
        assert f'cs_tool_pool_in_flight{{pool="order"}} {counts["order"]}' in body
        assert 'cs_tool_pool_in_flight{pool="course"} 0' in body
    finally:
        release.set()
        worker.join(timeout=5)
        assert not worker.is_alive()
    assert tool_pool_in_flight_counts()["order"] == 0
    assert 'cs_tool_pool_in_flight{pool="order"} 0' in render_prometheus()
