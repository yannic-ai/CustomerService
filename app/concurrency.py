"""请求级并发：集群对话槽 + 图执行线程池。"""

from __future__ import annotations

import asyncio
import atexit
import logging
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.chat_slots import ChatSlotBusy, get_chat_slot_store, reset_chat_slot_store
from app.config import get_settings
from app.executors import shutdown_tool_pools
from app.graph import chat_astream
from app.schemas import ChatResponse
from app.tenancy import normalize_tenant_id

logger = logging.getLogger("cs.concurrency")


class ChatBusyError(Exception):
    """同时对话数已达上限（集群或该租户）。"""

    def __init__(self, reason: str = "cluster") -> None:
        self.reason = reason if reason in {"cluster", "tenant"} else "cluster"
        super().__init__("对话繁忙")


_GRAPH_POOL: ThreadPoolExecutor | None = None
_ATEXIT_REGISTERED = False


def _register_atexit() -> None:
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(shutdown_concurrency)
        _ATEXIT_REGISTERED = True


def in_flight_count() -> int:
    """当前占用的对话并发槽（有 Redis 时为集群合计），供 /metrics 刮取。"""
    return get_chat_slot_store().in_flight()


def get_graph_executor() -> ThreadPoolExecutor | None:
    """图执行线程池；GRAPH_WORKERS<=0 时在当前事件循环线程直接跑（测试用）。"""
    global _GRAPH_POOL
    workers = get_settings().graph_workers
    if workers <= 0:
        return None
    if _GRAPH_POOL is None:
        _GRAPH_POOL = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cs-graph")
        _register_atexit()
        logger.info("图执行线程池 workers=%s", workers)
    return _GRAPH_POOL


def _slot_limits() -> tuple[int, int, int]:
    settings = get_settings()
    global_limit = max(1, int(getattr(settings, "chat_concurrency", 8) or 8))
    raw_tenant = getattr(settings, "chat_tenant_concurrency", 4)
    try:
        tenant_limit = int(raw_tenant)
    except (TypeError, ValueError):
        tenant_limit = 4
    if tenant_limit <= 0:
        tenant_limit = global_limit
    tenant_limit = max(1, min(tenant_limit, global_limit))
    ttl = max(1, int(getattr(settings, "chat_slot_ttl_seconds", 180) or 180))
    return global_limit, tenant_limit, ttl


def _resolve_tenant(tenant_id: str | None) -> str:
    try:
        return normalize_tenant_id(tenant_id)
    except ValueError:
        return get_settings().default_tenant


def _occupy(tenant_id: str) -> str:
    global_limit, tenant_limit, ttl = _slot_limits()
    try:
        return get_chat_slot_store().acquire(
            tenant_id,
            global_limit=global_limit,
            tenant_limit=tenant_limit,
            ttl_seconds=ttl,
        )
    except ChatSlotBusy as exc:
        raise ChatBusyError(exc.reason) from exc


def _vacate(tenant_id: str, token: str) -> None:
    if not token:
        return
    get_chat_slot_store().release(tenant_id, token)


async def _acquire_chat_slot(tenant_id: str) -> str:
    store = get_chat_slot_store()
    if store.degraded:
        token = _occupy(tenant_id)
    else:
        token = await asyncio.to_thread(_occupy, tenant_id)
    from app.observability.metrics import flush_metrics_if_multiprocess

    flush_metrics_if_multiprocess()
    return token


async def _release_chat_slot(tenant_id: str, token: str) -> None:
    store = get_chat_slot_store()
    if store.degraded:
        _vacate(tenant_id, token)
    else:
        await asyncio.to_thread(_vacate, tenant_id, token)
    from app.observability.metrics import flush_metrics_if_multiprocess

    flush_metrics_if_multiprocess()


async def achat(
    message: str,
    user_id: str = "anonymous",
    tenant_id: str | None = None,
    session_id: str | None = None,
) -> ChatResponse:
    """异步对话入口：限流后原生 await chat_astream 收集 done 事件。

    图节点已 async 化，不再经图线程池转发同步 chat()。
    """
    tenant = _resolve_tenant(tenant_id)
    token = await _acquire_chat_slot(tenant)
    try:
        last: ChatResponse | None = None
        async for event, data in chat_astream(
            message,
            user_id=user_id,
            tenant_id=tenant_id,
            session_id=session_id,
        ):
            if event == "done":
                last = ChatResponse.model_validate(data)
        if last is None:
            raise RuntimeError("图执行未产生结果")
        return last
    finally:
        await _release_chat_slot(tenant, token)


async def achat_stream(
    message: str,
    user_id: str = "anonymous",
    tenant_id: str | None = None,
    session_id: str | None = None,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """SSE 入口：占并发槽后原生 await chat_astream，不再经图线程池转发。

    节点 async 化后双层线程池（外层 _GRAPH_POOL + 内层 asyncio.run）已移除，
    会话锁仍由 chat_astream 内部按 thread_id 持有。
    """
    tenant = _resolve_tenant(tenant_id)
    token = await _acquire_chat_slot(tenant)
    try:
        async for event in chat_astream(
            message,
            user_id=user_id,
            tenant_id=tenant_id,
            session_id=session_id,
        ):
            yield event
    finally:
        await _release_chat_slot(tenant, token)


def shutdown_concurrency() -> None:
    """关闭图线程池与工具线程池，并丢掉进程内对话槽。"""
    global _GRAPH_POOL
    if _GRAPH_POOL is not None:
        _GRAPH_POOL.shutdown(wait=True)
        _GRAPH_POOL = None
    reset_chat_slot_store()
    from app.observability.metrics import flush_metrics_if_multiprocess

    flush_metrics_if_multiprocess()
    shutdown_tool_pools()
