"""订单工具的独立线程池（带背压：有界队列 + 超时等待 + 拒绝计数）。"""

from __future__ import annotations

import atexit
import contextvars
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import TypeVar

from app.config import get_settings

logger = logging.getLogger("cs.pools")
T = TypeVar("T")

_TLS = threading.local()
_ORDER_POOL: ThreadPoolExecutor | None = None
_ORDER_SEMAPHORE: threading.Semaphore | None = None
_ORDER_CAPACITY = 0
_ATEXIT_REGISTERED = False
# queue_wait 未显式配置时的兜底上限（避免永久阻塞调用方）
_DEFAULT_QUEUE_WAIT_SECONDS = 5.0


def tool_pool_in_flight_counts() -> dict[str, int]:
    """各池当前占用 + 排队槽数；池未创建时为 0。

    ``course`` 键恒为 0（课程检索已 async 化，不再使用独立线程池），
    保留仅为 Prometheus 指标向后兼容。
    """
    return {
        "order": _occupied(_ORDER_SEMAPHORE, _ORDER_CAPACITY),
        "course": 0,
    }


def _occupied(semaphore: threading.Semaphore | None, capacity: int) -> int:
    if semaphore is None or capacity <= 0:
        return 0
    try:
        available = int(getattr(semaphore, "_value", 0))
    except Exception:
        return 0
    return max(0, int(capacity) - available)


def _flush_pool_in_flight() -> None:
    try:
        from app.observability.metrics import flush_metrics_if_multiprocess

        flush_metrics_if_multiprocess()
    except Exception:
        return


class ToolPoolBusyError(Exception):
    """工具线程池队列已满或 wait 超时；调用方应降级 / 返回 503。"""


def _register_atexit() -> None:
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(shutdown_tool_pools)
        _ATEXIT_REGISTERED = True


def _pool_capacity(workers: int) -> int:
    """信号量容量（在执行 + 排队）= workers × queue_factor，至少 1。"""
    try:
        factor = max(1, int(get_settings().tool_pool_queue_factor))
    except Exception:
        factor = 2
    return max(1, max(1, int(workers)) * factor)


def _get_order_pool() -> tuple[ThreadPoolExecutor | None, threading.Semaphore | None]:
    global _ORDER_POOL, _ORDER_SEMAPHORE, _ORDER_CAPACITY
    workers = get_settings().order_tool_workers
    if workers <= 0:
        return None, None
    if _ORDER_POOL is None:
        cap = _pool_capacity(workers)
        _ORDER_POOL = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cs-order")
        _ORDER_SEMAPHORE = threading.Semaphore(cap)
        _ORDER_CAPACITY = cap
        _register_atexit()
        logger.info("订单工具线程池 workers=%s queue_cap=%s", workers, cap)
    return _ORDER_POOL, _ORDER_SEMAPHORE


def _submit(
    pool: ThreadPoolExecutor | None,
    semaphore: threading.Semaphore | None,
    pool_key: str,
    fn: Callable[..., T],
    *args,
    timeout: float | None = None,
    queue_wait_timeout: float | None = None,
    **kwargs,
) -> T:
    """提交到指定池；已在该池 worker 上或 workers=0 时直接执行。

    背压流程：
      1. 已在同池内重入 / 无池 → 直接本地同步执行（无 acquire，避免死锁）
      2. acquire(semaphore, timeout=queue_wait_timeout)
         - 超时 → 记 ToolPoolBusyError + metrics.rejected
         - 成功 → 记录排队耗时直方图 queue_seconds
      3. pool.submit(_inner)；future 完成时（done_callback）释放信号量
         - 保证 cancel / timeout / 正常结束 / 异常 4 条路径最终都会 release
      4. 等待 future.result(timeout=timeout)；超时 → future.cancel() 并抛 TimeoutError
    """
    # 重入保护 + 无池本地执行 放在最前面，不占用信号量
    if pool is None or getattr(_TLS, "pool_key", None) == pool_key:
        return fn(*args, **kwargs)

    ctx = contextvars.copy_context()
    started = time.monotonic()
    acquired = False
    try:
        if semaphore is not None:
            wait = (
                float(queue_wait_timeout)
                if queue_wait_timeout is not None and queue_wait_timeout > 0
                else _DEFAULT_QUEUE_WAIT_SECONDS
            )
            acquired = semaphore.acquire(timeout=wait)
            if not acquired:
                from app.observability import inc_tool_pool_rejected

                inc_tool_pool_rejected(pool_key)
                raise ToolPoolBusyError(
                    f"{pool_key} pool busy; queue_wait>{wait:g}s"
                )
            _flush_pool_in_flight()

        waited = time.monotonic() - started
        from app.observability import observe_tool_pool_queue

        observe_tool_pool_queue(pool_key, waited)

        def _inner() -> T:
            _TLS.pool_key = pool_key
            try:
                return ctx.run(fn, *args, **kwargs)
            finally:
                _TLS.pool_key = None

        future: Future[T] = pool.submit(_inner)

        # 释放权交给 done_callback（即使 cancel 返回 False 的已启动任务，结束后也会触发）
        if semaphore is not None:
            captured_sem: threading.Semaphore = semaphore
            captured_pool = pool_key

            def _release(
                _f: Future[T],
                _s: threading.Semaphore = captured_sem,
                _pk: str = captured_pool,
            ) -> None:
                try:
                    _s.release()
                except Exception:
                    logger.exception(
                        "tool pool semaphore release failed pool=%s", _pk
                    )
                _flush_pool_in_flight()

            future.add_done_callback(_release)
            acquired = False  # callback 负责释放

        if timeout is not None and timeout > 0:
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError:
                future.cancel()
                raise TimeoutError(
                    f"{pool_key} tool timed out after {timeout:g}s"
                ) from None
        return future.result()
    finally:
        # acquire 成功但 submit/future 赋值前路径抛出 → 手动释放
        if acquired and semaphore is not None:
            try:
                semaphore.release()
            except Exception:
                logger.exception(
                    "tool pool semaphore finally-release failed pool=%s", pool_key
                )
            _flush_pool_in_flight()


def run_order(fn: Callable[..., T], *args, **kwargs) -> T:
    """在订单线程池执行 fn；超时抛 TimeoutError；池满抛 ToolPoolBusyError。"""
    pool, sem = _get_order_pool()
    settings = get_settings()
    timeout = float(settings.order_tool_timeout_seconds or 0) or None
    return _submit(
        pool,
        sem,
        "order",
        fn,
        *args,
        timeout=timeout,
        queue_wait_timeout=settings.tool_pool_queue_wait_seconds,
        **kwargs,
    )


def shutdown_tool_pools() -> None:
    """关闭工具线程池，清理全局引用。"""
    global _ORDER_POOL, _ORDER_SEMAPHORE, _ORDER_CAPACITY
    if _ORDER_POOL is not None:
        _ORDER_POOL.shutdown(wait=True)
    _ORDER_POOL = None
    _ORDER_SEMAPHORE = None
    _ORDER_CAPACITY = 0
