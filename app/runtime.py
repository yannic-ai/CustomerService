"""进程级运行时装配：图、checkpointer、会话锁、长期记忆、Gateway store。

生产路径在首次 ``get_app_context()`` 时按 Redis 是否可用构造一套对象，再编译图。
测试用 ``install_app_context`` / ``in_memory_app_context`` 一次替换整套依赖，
不必分别 ``reset_graph`` + ``reset_user_store`` + ``set_gateway_store``。

图节点仍通过 LangGraph runtime ``get_store()`` 取长期记忆；Gateway / 会话锁
在调用时从本上下文读取。不使用 ContextVar 作为主存储：``GRAPH_WORKERS>0``
时执行在线程池里，ContextVar 不会传到 worker 线程。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from app.llm_gateway.stores import GatewayStore, MemoryGatewayStore, build_gateway_store
from app.memory import build_user_store
from app.session_cache import (
    AsyncSessionLock,
    DictSessionLock,
    SessionLock,
    build_checkpointer,
    get_redis_client,
    set_session_lock,
)

logger = logging.getLogger("cs.runtime")

_LOCK = threading.Lock()
_PROCESS: AppContext | None = None


@dataclass
class AppContext:
    """一次装配出的基础设施句柄；graph 在首次访问时编译。"""

    redis: Any | None
    checkpointer: Any
    session_lock: SessionLock | AsyncSessionLock
    user_store: BaseStore
    gateway_store: GatewayStore | None
    visualize: bool = True
    _graph: Any = field(default=None, repr=False, compare=False)

    @property
    def graph(self):
        if self._graph is None:
            from app.graph.builder import build_graph

            self._graph = build_graph(
                checkpointer=self.checkpointer,
                store=self.user_store,
                visualize=self.visualize,
            )
        return self._graph

    def invalidate_graph(self) -> None:
        """checkpointer / user_store 替换后丢掉已编译图。"""
        self._graph = None

    def ensure_gateway_store(self) -> GatewayStore:
        if self.gateway_store is None:
            self.gateway_store = build_gateway_store(client=self.redis)
        return self.gateway_store


def in_memory_app_context(
    *,
    checkpointer: Any | None = None,
    session_lock: SessionLock | AsyncSessionLock | None = None,
    user_store: BaseStore | None = None,
    gateway_store: GatewayStore | None = None,
    redis: Any | None = None,
) -> AppContext:
    """纯内存装配，供测试一次注入整套替身。"""
    lock = session_lock or DictSessionLock()
    return AppContext(
        redis=redis,
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
        session_lock=lock,
        user_store=user_store if user_store is not None else InMemoryStore(),
        gateway_store=gateway_store if gateway_store is not None else MemoryGatewayStore(),
        visualize=False,
    )


def build_app_context(
    *,
    redis: Any | None = None,
    use_redis: bool = True,
    checkpointer: Any | None = None,
    session_lock: SessionLock | AsyncSessionLock | None = None,
    user_store: BaseStore | None = None,
    gateway_store: GatewayStore | None = None,
    visualize: bool = True,
) -> AppContext:
    """按当前配置构造一套运行时对象；不写入进程单例。"""
    client = redis
    if use_redis and client is None:
        client = get_redis_client(required=False)

    if checkpointer is None:
        checkpointer = build_checkpointer(session_lock=session_lock)
        lock = session_lock
        if lock is None:
            from app.session_cache import get_session_lock as _raw_lock

            lock = _raw_lock()
    else:
        lock = session_lock or DictSessionLock()
        set_session_lock(lock)

    if user_store is None:
        user_store = build_user_store(client=client)
    if gateway_store is None:
        gateway_store = build_gateway_store(client=client)

    return AppContext(
        redis=client,
        checkpointer=checkpointer,
        session_lock=lock,
        user_store=user_store,
        gateway_store=gateway_store,
        visualize=visualize,
    )


def peek_app_context() -> AppContext | None:
    """已安装的进程上下文；尚未装配时为 None。"""
    return _PROCESS


def get_app_context() -> AppContext:
    """返回进程上下文；没有则按 Redis 配置惰性装配。"""
    global _PROCESS
    if _PROCESS is not None:
        return _PROCESS
    with _LOCK:
        if _PROCESS is None:
            _PROCESS = build_app_context()
            set_session_lock(_PROCESS.session_lock)
            logger.info(
                "已装配 AppContext redis=%s checkpointer=%s user_store=%s gateway=%s",
                type(_PROCESS.redis).__name__ if _PROCESS.redis is not None else "none",
                type(_PROCESS.checkpointer).__name__,
                type(_PROCESS.user_store).__name__,
                type(_PROCESS.gateway_store).__name__
                if _PROCESS.gateway_store is not None
                else "lazy",
            )
        return _PROCESS


def install_app_context(ctx: AppContext) -> AppContext | None:
    """替换进程上下文，返回替换前的实例（便于测试还原）。"""
    global _PROCESS
    with _LOCK:
        previous = _PROCESS
        _PROCESS = ctx
        set_session_lock(ctx.session_lock)
        return previous


@contextmanager
def using_app_context(ctx: AppContext) -> Iterator[AppContext]:
    """临时安装上下文，退出时还原。"""
    previous = install_app_context(ctx)
    try:
        yield ctx
    finally:
        if previous is None:
            clear_app_context()
        else:
            install_app_context(previous)


def clear_app_context() -> None:
    """丢掉进程上下文，下次 ``get_app_context()`` 重新装配。"""
    global _PROCESS
    with _LOCK:
        _PROCESS = None


def replace_app_context(
    *,
    checkpointer: Any | None = None,
    session_lock: SessionLock | AsyncSessionLock | None = None,
    user_store: BaseStore | None = None,
    gateway_store: GatewayStore | None | object = ...,
    redis: Any | None | object = ...,
    invalidate_graph: bool = False,
) -> AppContext:
    """原地替换当前上下文的部分依赖。

    ``gateway_store=None`` 表示下次按 redis 惰性重建；省略该参数则保持原值。
    """
    ctx = get_app_context()
    if checkpointer is not None:
        ctx.checkpointer = checkpointer
        invalidate_graph = True
    if session_lock is not None:
        ctx.session_lock = session_lock
        set_session_lock(session_lock)
    if user_store is not None:
        ctx.user_store = user_store
        invalidate_graph = True
    if gateway_store is not ...:
        ctx.gateway_store = gateway_store  # type: ignore[assignment]
    if redis is not ...:
        ctx.redis = redis  # type: ignore[assignment]
    if invalidate_graph:
        ctx.invalidate_graph()
    return ctx
