"""短期会话 checkpoint：每次读写都走共享缓存，未配置 Redis 时仅进程内。"""

from __future__ import annotations

import asyncio
import logging
import pickle
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Protocol

import ormsgpack
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata

logger = logging.getLogger("cs.session_cache")

# Redis Cluster Hash Tag 约定：
#   短期会话的所有相关 key（checkpoint + 锁）必须共享 hash tag {thread_id}，
#   确保在 Cluster 模式下路由到同一节点，以便未来 MULTI/pipeline/事务类操作。
#   thread_id 格式为 "tenant:user:session"（见 graph.make_thread_id）。
#   Key 结构：
#     cs:session:{tenant:user:session}          ← checkpoint blob
#     cs:session:lock:{tenant:user:session}     ← 互斥锁
KEY_PREFIX = "cs:session:"
LOCK_PREFIX = "cs:session:lock:"
# 热缓存信封：CSK1 + msgpack；无此前缀则尝试 pickle（TTL 内旧会话）
ENVELOPE_MAGIC = b"CSK1"
_UNLOCK_LUA = """
if redis.call("get", KEYS[1]) == ARGV[1] then
  return redis.call("del", KEYS[1])
else
  return 0
end
"""

_SESSION_LOCK: "SessionLock | AsyncSessionLock | None" = None
_REDIS: Any = None
_ASYNC_REDIS: Any = None
_REDIS_GUARD = threading.Lock()
_ASYNC_REDIS_GUARD = threading.Lock()


def _typed_to_list(pair: Any) -> list[Any]:
    typ, data = pair
    return [typ, data]


def _list_to_typed(pair: Any) -> tuple[str, bytes]:
    typ, data = pair
    if isinstance(data, memoryview):
        data = bytes(data)
    elif isinstance(data, bytearray):
        data = bytes(data)
    elif isinstance(data, str):
        data = data.encode("utf-8")
    return str(typ), bytes(data)


def encode_checkpoint_blob(payload: dict[str, Any]) -> bytes:
    """把 InMemorySaver 三本字典编成 CSK1 + msgpack（元组 key 改 list）。"""
    storage_out: dict[str, dict[str, list[Any]]] = {}
    for ns, checkpoints in (payload.get("storage") or {}).items():
        encoded: dict[str, list[Any]] = {}
        for checkpoint_id, saved in checkpoints.items():
            dumped, meta, parent = saved
            encoded[str(checkpoint_id)] = [
                _typed_to_list(dumped),
                _typed_to_list(meta),
                parent,
            ]
        storage_out[str(ns)] = encoded

    writes_out: list[list[Any]] = []
    for outer_key, inner in (payload.get("writes") or {}).items():
        thread_id, checkpoint_ns, checkpoint_id = outer_key
        items: list[list[Any]] = []
        for inner_key, value in inner.items():
            task_id, channel, typed, task_path = value
            items.append(
                [
                    list(inner_key),
                    [task_id, channel, _typed_to_list(typed), task_path],
                ]
            )
        writes_out.append([[thread_id, checkpoint_ns, checkpoint_id], items])

    blobs_out: list[list[Any]] = []
    for blob_key, typed in (payload.get("blobs") or {}).items():
        thread_id, checkpoint_ns, channel, version = blob_key
        blobs_out.append(
            [[thread_id, checkpoint_ns, channel, version], _typed_to_list(typed)]
        )

    packed = ormsgpack.packb(
        {"v": 1, "storage": storage_out, "writes": writes_out, "blobs": blobs_out}
    )
    return ENVELOPE_MAGIC + packed


def decode_checkpoint_blob(raw: bytes) -> dict[str, Any] | None:
    """解码热缓存；CSK1 走 msgpack，否则尝试 pickle。失败返回 None。"""
    if not raw:
        return None
    if raw.startswith(ENVELOPE_MAGIC):
        try:
            loaded = ormsgpack.unpackb(raw[len(ENVELOPE_MAGIC) :])
        except Exception:
            return None
        if not isinstance(loaded, dict):
            return None
        return _payload_from_msgpack(loaded)
    try:
        loaded = pickle.loads(raw)
    except Exception:
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


def _payload_from_msgpack(loaded: dict[str, Any]) -> dict[str, Any] | None:
    try:
        storage: dict[str, dict[str, Any]] = {}
        for ns, checkpoints in (loaded.get("storage") or {}).items():
            decoded: dict[str, Any] = {}
            for checkpoint_id, saved in checkpoints.items():
                dumped, meta, parent = saved
                decoded[str(checkpoint_id)] = (
                    _list_to_typed(dumped),
                    _list_to_typed(meta),
                    parent,
                )
            storage[str(ns)] = decoded

        writes: dict[tuple[Any, ...], dict[tuple[Any, ...], Any]] = {}
        for outer_key, items in loaded.get("writes") or []:
            thread_id, checkpoint_ns, checkpoint_id = outer_key
            inner: dict[tuple[Any, ...], Any] = {}
            for inner_key, value in items:
                task_id, channel, typed, task_path = value
                inner[tuple(inner_key)] = (
                    task_id,
                    channel,
                    _list_to_typed(typed),
                    task_path,
                )
            writes[(thread_id, checkpoint_ns, checkpoint_id)] = inner

        blobs: dict[tuple[Any, ...], tuple[str, bytes]] = {}
        for blob_key, typed in loaded.get("blobs") or []:
            thread_id, checkpoint_ns, channel, version = blob_key
            blobs[(thread_id, checkpoint_ns, channel, version)] = _list_to_typed(typed)
    except Exception:
        return None
    return {"storage": storage, "writes": writes, "blobs": blobs}


class SessionCache(Protocol):
    """字节缓存：get / set / delete / touch。"""

    def get(self, key: str) -> bytes | None: ...

    def set(self, key: str, value: bytes, ttl: int) -> None: ...

    def delete(self, key: str) -> None: ...

    def touch(self, key: str, ttl: int) -> None: ...


class DictSessionCache:
    """进程内字典，测试用；模拟 Redis TTL 但不真正过期。"""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self._data.get(key)

    def set(self, key: str, value: bytes, ttl: int) -> None:
        self._data[key] = value

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def touch(self, key: str, ttl: int) -> None:
        pass  # 进程内字典不真正过期，无需刷新


class RedisSessionCache:
    """redis-py 客户端封装。"""

    def __init__(self, client: Any) -> None:
        self._client = client

    def get(self, key: str) -> bytes | None:
        value = self._client.get(key)
        if value is None:
            return None
        if isinstance(value, str):
            return value.encode("utf-8")
        return bytes(value)

    def set(self, key: str, value: bytes, ttl: int) -> None:
        if ttl > 0:
            self._client.set(key, value, ex=ttl)
        else:
            self._client.set(key, value)

    def delete(self, key: str) -> None:
        self._client.delete(key)

    def touch(self, key: str, ttl: int) -> None:
        """刷新 key 的 TTL（滑动过期）；key 不存在时 no-op。"""
        if ttl > 0:
            try:
                self._client.expire(key, ttl)
            except Exception:
                logger.exception("刷新会话 TTL 失败 key=%s", key)


class AsyncSessionCache(Protocol):
    """异步字节缓存：aget / aset / adelete / atouch。"""

    async def aget(self, key: str) -> bytes | None: ...

    async def aset(self, key: str, value: bytes, ttl: int) -> None: ...

    async def adelete(self, key: str) -> None: ...

    async def atouch(self, key: str, ttl: int) -> None: ...


class AsyncDictSessionCache:
    """异步包装 DictSessionCache：底层是进程内字典，不阻塞事件循环。"""

    def __init__(self, inner: DictSessionCache | None = None) -> None:
        self._inner = inner or DictSessionCache()

    async def aget(self, key: str) -> bytes | None:
        return self._inner.get(key)

    async def aset(self, key: str, value: bytes, ttl: int) -> None:
        self._inner.set(key, value, ttl)

    async def adelete(self, key: str) -> None:
        self._inner.delete(key)

    async def atouch(self, key: str, ttl: int) -> None:
        self._inner.touch(key, ttl)


class AsyncRedisSessionCache:
    """redis.asyncio 客户端封装，原生非阻塞。"""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def aget(self, key: str) -> bytes | None:
        value = await self._client.get(key)
        if value is None:
            return None
        if isinstance(value, str):
            return value.encode("utf-8")
        return bytes(value)

    async def aset(self, key: str, value: bytes, ttl: int) -> None:
        if ttl > 0:
            await self._client.set(key, value, ex=ttl)
        else:
            await self._client.set(key, value)

    async def adelete(self, key: str) -> None:
        await self._client.delete(key)

    async def atouch(self, key: str, ttl: int) -> None:
        """刷新 key 的 TTL（滑动过期）；key 不存在时 no-op。"""
        if ttl > 0:
            try:
                await self._client.expire(key, ttl)
            except Exception:
                logger.exception("刷新会话 TTL 失败 key=%s", key)


class RedisUnavailableError(RuntimeError):
    """Redis 未配置或连不上；HTTP 服务禁止回退进程内存。"""


class SessionBusyError(Exception):
    """同一 thread_id 已有对话在执行，未在等待窗口内拿到锁。"""

    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id
        super().__init__(f"会话繁忙 thread_id={thread_id}")


class SessionLock(Protocol):
    """按 thread_id 互斥，覆盖整轮对话。"""

    def acquire(self, thread_id: str) -> str: ...

    def release(self, thread_id: str, token: str) -> None: ...

    @contextmanager
    def hold(self, thread_id: str) -> Iterator[None]: ...


def _lock_settings() -> tuple[int, float, float]:
    from app.config import get_settings

    settings = get_settings()
    ttl = max(1, int(settings.session_lock_ttl_seconds))
    wait = max(0.0, float(settings.session_lock_wait_seconds))
    interval = max(0.01, float(settings.session_lock_retry_interval_seconds))
    return ttl, wait, interval


class DictSessionLock:
    """进程内按 thread_id 互斥，测试与未配 Redis 时使用。"""

    def __init__(
        self,
        *,
        wait_seconds: float | None = None,
        ttl_seconds: int | None = None,
        retry_interval: float | None = None,
    ) -> None:
        default_ttl, default_wait, default_interval = _lock_settings()
        self._wait = default_wait if wait_seconds is None else max(0.0, wait_seconds)
        self._ttl = default_ttl if ttl_seconds is None else max(1, ttl_seconds)
        self._interval = default_interval if retry_interval is None else max(0.01, retry_interval)
        self._meta = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def _lock_for(self, thread_id: str) -> threading.Lock:
        with self._meta:
            lock = self._locks.get(thread_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[thread_id] = lock
            return lock

    def acquire(self, thread_id: str) -> str:
        lock = self._lock_for(thread_id)
        if lock.acquire(timeout=self._wait):
            return "local"
        raise SessionBusyError(thread_id)

    def release(self, thread_id: str, token: str) -> None:
        del token
        self._lock_for(thread_id).release()

    @contextmanager
    def hold(self, thread_id: str) -> Iterator[None]:
        token = self.acquire(thread_id)
        try:
            yield
        finally:
            self.release(thread_id, token)

    async def aacquire(self, thread_id: str) -> str:
        lock = self._lock_for(thread_id)
        deadline = time.monotonic() + self._wait
        while True:
            if lock.acquire(blocking=False):
                return "local"
            if time.monotonic() >= deadline:
                raise SessionBusyError(thread_id)
            await asyncio.sleep(self._interval)

    async def arelease(self, thread_id: str, token: str) -> None:
        self.release(thread_id, token)

    @asynccontextmanager
    async def ahold(self, thread_id: str) -> AsyncIterator[None]:
        token = await self.aacquire(thread_id)
        try:
            yield
        finally:
            await self.arelease(thread_id, token)


class RedisSessionLock:
    """Redis SET NX + token 释放，跨 worker 互斥同一 thread_id。"""

    def __init__(
        self,
        client: Any,
        *,
        wait_seconds: float | None = None,
        ttl_seconds: int | None = None,
        retry_interval: float | None = None,
    ) -> None:
        default_ttl, default_wait, default_interval = _lock_settings()
        self._client = client
        self._wait = default_wait if wait_seconds is None else max(0.0, wait_seconds)
        self._ttl_ms = max(1000, int((default_ttl if ttl_seconds is None else ttl_seconds) * 1000))
        self._interval = default_interval if retry_interval is None else max(0.01, retry_interval)

    def _key(self, thread_id: str) -> str:
        # 与 CachedInMemorySaver._cache_key 共用同一 hash tag {thread_id}
        # → Cluster 下 lock + checkpoint 落到同一 slot
        return f"{LOCK_PREFIX}{{{thread_id}}}"

    def acquire(self, thread_id: str) -> str:
        token = uuid.uuid4().hex
        key = self._key(thread_id)
        deadline = time.monotonic() + self._wait
        while True:
            ok = self._client.set(key, token, nx=True, px=self._ttl_ms)
            if ok:
                return token
            if time.monotonic() >= deadline:
                raise SessionBusyError(thread_id)
            time.sleep(self._interval)

    def release(self, thread_id: str, token: str) -> None:
        try:
            self._client.eval(_UNLOCK_LUA, 1, self._key(thread_id), token)
        except Exception:
            logger.exception("释放会话锁失败 thread_id=%s", thread_id)

    @contextmanager
    def hold(self, thread_id: str) -> Iterator[None]:
        token = self.acquire(thread_id)
        try:
            yield
        finally:
            self.release(thread_id, token)


class AsyncSessionLock(Protocol):
    """异步按 thread_id 互斥，覆盖整轮对话；支持 ``async with lock.ahold(...)``。"""

    degraded: bool

    async def aacquire(self, thread_id: str) -> str: ...

    async def arelease(self, thread_id: str, token: str) -> None: ...

    def ahold(self, thread_id: str) -> "AsyncIterator[None]": ...


class AsyncDictSessionLock:
    """进程内异步锁：threading.Lock + asyncio.sleep 轮询，跨事件循环可用。

    事件循环单一时无并发；``asyncio.run`` 每次新建循环也能用，因为底层是
    threading.Lock 而非 asyncio.Lock（后者绑定到首次使用的循环）。
    """

    degraded = True

    def __init__(
        self,
        *,
        wait_seconds: float | None = None,
        ttl_seconds: int | None = None,
        retry_interval: float | None = None,
    ) -> None:
        import asyncio

        default_ttl, default_wait, default_interval = _lock_settings()
        self._wait = default_wait if wait_seconds is None else max(0.0, wait_seconds)
        self._ttl = default_ttl if ttl_seconds is None else max(1, ttl_seconds)
        self._interval = default_interval if retry_interval is None else max(0.01, retry_interval)
        self._meta = threading.Lock()
        self._held: set[str] = set()
        self._asyncio = asyncio

    async def aacquire(self, thread_id: str) -> str:
        deadline = time.monotonic() + self._wait
        while True:
            with self._meta:
                if thread_id not in self._held:
                    self._held.add(thread_id)
                    return "local"
            if time.monotonic() >= deadline:
                raise SessionBusyError(thread_id)
            await self._asyncio.sleep(self._interval)

    async def arelease(self, thread_id: str, token: str) -> None:
        del token
        with self._meta:
            self._held.discard(thread_id)

    @asynccontextmanager
    async def ahold(self, thread_id: str) -> AsyncIterator[None]:
        token = await self.aacquire(thread_id)
        try:
            yield
        finally:
            await self.arelease(thread_id, token)

    # 兼容旧同步调用方（部分测试仍用 ``with lock.hold(...)``）
    def acquire(self, thread_id: str) -> str:
        token = "local"
        deadline = time.monotonic() + self._wait
        while True:
            with self._meta:
                if thread_id not in self._held:
                    self._held.add(thread_id)
                    return token
            if time.monotonic() >= deadline:
                raise SessionBusyError(thread_id)
            time.sleep(self._interval)

    def release(self, thread_id: str, token: str) -> None:
        del token
        with self._meta:
            self._held.discard(thread_id)

    @contextmanager
    def hold(self, thread_id: str) -> Iterator[None]:
        token = self.acquire(thread_id)
        try:
            yield
        finally:
            self.release(thread_id, token)


class AsyncRedisSessionLock:
    """redis.asyncio SET NX + token 释放，跨 worker 互斥同一 thread_id。"""

    degraded = False

    def __init__(
        self,
        client: Any,
        *,
        wait_seconds: float | None = None,
        ttl_seconds: int | None = None,
        retry_interval: float | None = None,
    ) -> None:
        import asyncio

        default_ttl, default_wait, default_interval = _lock_settings()
        self._client = client
        self._wait = default_wait if wait_seconds is None else max(0.0, wait_seconds)
        self._ttl_ms = max(1000, int((default_ttl if ttl_seconds is None else ttl_seconds) * 1000))
        self._interval = default_interval if retry_interval is None else max(0.01, retry_interval)
        self._asyncio = asyncio

    def _key(self, thread_id: str) -> str:
        return f"{LOCK_PREFIX}{{{thread_id}}}"

    async def aacquire(self, thread_id: str) -> str:
        token = uuid.uuid4().hex
        key = self._key(thread_id)
        deadline = time.monotonic() + self._wait
        while True:
            ok = await self._client.set(key, token, nx=True, px=self._ttl_ms)
            if ok:
                return token
            if time.monotonic() >= deadline:
                raise SessionBusyError(thread_id)
            await self._asyncio.sleep(self._interval)

    async def arelease(self, thread_id: str, token: str) -> None:
        try:
            await self._client.eval(_UNLOCK_LUA, 1, self._key(thread_id), token)
        except Exception:
            logger.exception("释放会话锁失败 thread_id=%s", thread_id)

    @asynccontextmanager
    async def ahold(self, thread_id: str) -> AsyncIterator[None]:
        token = await self.aacquire(thread_id)
        try:
            yield
        finally:
            await self.arelease(thread_id, token)


AnySessionLock = SessionLock | AsyncSessionLock


@asynccontextmanager
async def ahold_session(lock: Any, thread_id: str) -> AsyncIterator[None]:
    """异步持有会话锁：优先 ``ahold``（asyncio.sleep / redis.asyncio），否则把同步 acquire 卸到线程。"""
    ahold = getattr(lock, "ahold", None)
    if ahold is not None:
        async with ahold(thread_id):
            yield
    else:
        token = await asyncio.to_thread(lock.acquire, thread_id)
        try:
            yield
        finally:
            await asyncio.to_thread(lock.release, thread_id, token)


def get_session_lock() -> "SessionLock | AsyncSessionLock":
    """当前会话锁；优先进程 AppContext，否则模块回退。"""
    from app.runtime import peek_app_context

    ctx = peek_app_context()
    if ctx is not None:
        return ctx.session_lock
    global _SESSION_LOCK
    if _SESSION_LOCK is None:
        _SESSION_LOCK = DictSessionLock()
    return _SESSION_LOCK


def set_session_lock(lock: SessionLock | AsyncSessionLock | None) -> None:
    """测试用：替换或清空会话锁；已装配 AppContext 时同步写回。"""
    global _SESSION_LOCK
    _SESSION_LOCK = lock
    from app.runtime import peek_app_context

    ctx = peek_app_context()
    if ctx is not None:
        ctx.session_lock = lock or DictSessionLock()


class CachedInMemorySaver(InMemorySaver):
    """以共享缓存为唯一事实源的 checkpointer。

    每次 get / put 都从缓存整线程替换本地副本，写完立刻回写。
    persist 只保留每个 namespace 的最新 checkpoint，外壳为 CSK1+msgpack。
    本机内存只是单次调用的工作区，多 worker 不会读到彼此覆盖前的旧槽位。

    未使用官方 RedisSaver：它依赖 RedisJSON / RediSearch（Redis 8 或 Redis Stack），
    与当前普通 REDIS_URL 不兼容。

    支持同步与异步两套缓存：
      - 同步 ``cache``：用于 sync ``get_tuple`` / ``put`` / ``put_writes``（测试 / 兼容）
      - 异步 ``async_cache``：用于 ``aget_tuple`` / ``aput`` / ``aput_writes``
        （生产异步路径，Redis 走 redis.asyncio 原生非阻塞）
    未提供 ``async_cache`` 时异步方法回退同步 ``cache``：进程内字典不阻塞，
    Redis 同步客户端会堵住事件循环，生产路径必须传入 ``AsyncRedisSessionCache``。
    """

    def __init__(
        self,
        cache: SessionCache,
        *,
        ttl_seconds: int = 604800,
        async_cache: AsyncSessionCache | None = None,
    ) -> None:
        super().__init__()
        self._cache = cache
        self._async_cache = async_cache
        self._ttl = max(0, int(ttl_seconds))
        self._lock = threading.RLock()

    def _cache_key(self, thread_id: str) -> str:
        # Redis Cluster hash tag：{thread_id} 确保同一会话的 cache + lock 路由到同一节点
        return f"{KEY_PREFIX}{{{thread_id}}}"

    def _evict_thread(self, thread_id: str) -> None:
        self.storage.pop(thread_id, None)
        for key in [k for k in self.writes if k[0] == thread_id]:
            self.writes.pop(key, None)
        for key in [k for k in self.blobs if k[0] == thread_id]:
            self.blobs.pop(key, None)

    def _load(self, thread_id: str) -> None:
        cache_key = self._cache_key(thread_id)
        raw = self._cache.get(cache_key)
        payload: dict[str, Any] | None = None
        if raw:
            payload = decode_checkpoint_blob(raw)
            if payload is None:
                logger.warning("会话缓存无法解码，已丢弃 thread_id=%s", thread_id)
                self._cache.delete(cache_key)
            else:
                # 滑动过期：命中时刷新 TTL，活跃会话续期、不活跃自动清理
                self._cache.touch(cache_key, self._ttl)
        self._evict_thread(thread_id)
        if not payload:
            return
        for ns, checkpoints in (payload.get("storage") or {}).items():
            self.storage[thread_id][ns].update(checkpoints)
        for key, writes in (payload.get("writes") or {}).items():
            self.writes[key].update(writes)
        self.blobs.update(payload.get("blobs") or {})

    async def _aload(self, thread_id: str) -> None:
        """异步版 _load：优先用 async_cache，否则回退同步 cache。"""
        cache_key = self._cache_key(thread_id)
        payload: dict[str, Any] | None = None
        if self._async_cache is not None:
            raw = await self._async_cache.aget(cache_key)
            if raw:
                payload = decode_checkpoint_blob(raw)
                if payload is None:
                    logger.warning("会话缓存无法解码，已丢弃 thread_id=%s", thread_id)
                    await self._async_cache.adelete(cache_key)
                else:
                    await self._async_cache.atouch(cache_key, self._ttl)
        else:
            raw = self._cache.get(cache_key)
            if raw:
                payload = decode_checkpoint_blob(raw)
                if payload is None:
                    logger.warning("会话缓存无法解码，已丢弃 thread_id=%s", thread_id)
                    self._cache.delete(cache_key)
                else:
                    self._cache.touch(cache_key, self._ttl)
        self._evict_thread(thread_id)
        if not payload:
            return
        for ns, checkpoints in (payload.get("storage") or {}).items():
            self.storage[thread_id][ns].update(checkpoints)
        for key, writes in (payload.get("writes") or {}).items():
            self.writes[key].update(writes)
        self.blobs.update(payload.get("blobs") or {})

    def _prune_thread(self, thread_id: str) -> None:
        """每个 namespace 只留最新 checkpoint，并丢掉未引用的 writes / blobs。"""
        ns_map = self.storage.get(thread_id)
        if not ns_map:
            return
        keep_cps: set[tuple[str, str]] = set()
        keep_blobs: set[tuple[Any, ...]] = set()
        for ns, checkpoints in list(ns_map.items()):
            if not checkpoints:
                ns_map.pop(ns, None)
                continue
            latest_id = max(checkpoints.keys())
            for checkpoint_id in list(checkpoints):
                if checkpoint_id != latest_id:
                    checkpoints.pop(checkpoint_id, None)
            keep_cps.add((ns, latest_id))
            saved = checkpoints.get(latest_id)
            if not saved:
                continue
            try:
                ckpt = self.serde.loads_typed(saved[0])
            except Exception:
                logger.exception(
                    "裁剪 checkpoint 时无法解码 channel_versions thread_id=%s ns=%s",
                    thread_id,
                    ns,
                )
                continue
            versions = ckpt.get("channel_versions") or {}
            for channel, version in versions.items():
                keep_blobs.add((thread_id, ns, channel, version))

        for key in [k for k in self.writes if k[0] == thread_id]:
            _, ns, checkpoint_id = key
            if (ns, checkpoint_id) not in keep_cps:
                self.writes.pop(key, None)

        for key in [k for k in self.blobs if k[0] == thread_id]:
            if key not in keep_blobs:
                self.blobs.pop(key, None)

    def _persist(self, thread_id: str) -> None:
        self._prune_thread(thread_id)
        ns_map = self.storage.get(thread_id)
        if not ns_map:
            self._cache.delete(self._cache_key(thread_id))
            return
        payload = {
            "storage": {ns: dict(cps) for ns, cps in ns_map.items()},
            "writes": {k: dict(v) for k, v in self.writes.items() if k[0] == thread_id},
            "blobs": {k: v for k, v in self.blobs.items() if k[0] == thread_id},
        }
        self._cache.set(
            self._cache_key(thread_id),
            encode_checkpoint_blob(payload),
            self._ttl,
        )

    async def _apersist(self, thread_id: str) -> None:
        """异步版 _persist：优先用 async_cache，否则回退同步 cache。"""
        self._prune_thread(thread_id)
        ns_map = self.storage.get(thread_id)
        cache_key = self._cache_key(thread_id)
        if not ns_map:
            if self._async_cache is not None:
                await self._async_cache.adelete(cache_key)
            else:
                self._cache.delete(cache_key)
            return
        payload = {
            "storage": {ns: dict(cps) for ns, cps in ns_map.items()},
            "writes": {k: dict(v) for k, v in self.writes.items() if k[0] == thread_id},
            "blobs": {k: v for k, v in self.blobs.items() if k[0] == thread_id},
        }
        blob = encode_checkpoint_blob(payload)
        if self._async_cache is not None:
            await self._async_cache.aset(cache_key, blob, self._ttl)
        else:
            self._cache.set(cache_key, blob, self._ttl)

    def get_tuple(self, config: RunnableConfig):
        thread_id = str(config["configurable"]["thread_id"])
        with self._lock:
            self._load(thread_id)
            return super().get_tuple(config)

    async def aget_tuple(self, config: RunnableConfig):
        """原生异步 get_tuple：await async_cache（redis.asyncio）而非走同步线程。"""
        thread_id = str(config["configurable"]["thread_id"])
        await self._aload(thread_id)
        return InMemorySaver.get_tuple(self, config)

    def get_delta_channel_history(self, *, config: RunnableConfig, channels):
        thread_id = str(config["configurable"]["thread_id"])
        with self._lock:
            self._load(thread_id)
            return super().get_delta_channel_history(config=config, channels=channels)

    async def aget_delta_channel_history(self, *, config: RunnableConfig, channels):
        thread_id = str(config["configurable"]["thread_id"])
        await self._aload(thread_id)
        return InMemorySaver.get_delta_channel_history(self, config=config, channels=channels)

    def list(self, config: RunnableConfig | None, **kwargs):
        if not config:
            return super().list(config, **kwargs)
        thread_id = str(config["configurable"]["thread_id"])
        with self._lock:
            self._load(thread_id)
            items = list(super().list(config, **kwargs))
        return iter(items)

    async def alist(self, config: RunnableConfig | None, **kwargs):
        if not config:
            async for item in super().alist(config, **kwargs):
                yield item
            return
        thread_id = str(config["configurable"]["thread_id"])
        await self._aload(thread_id)
        for item in self.list(config, **kwargs):
            yield item

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = str(config["configurable"]["thread_id"])
        with self._lock:
            self._load(thread_id)
            result = super().put(config, checkpoint, metadata, new_versions)
            self._persist(thread_id)
            self._evict_thread(thread_id)
            return result

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """原生异步 put：await async_cache 写回。"""
        thread_id = str(config["configurable"]["thread_id"])
        await self._aload(thread_id)
        result = InMemorySaver.put(self, config, checkpoint, metadata, new_versions)
        await self._apersist(thread_id)
        self._evict_thread(thread_id)
        return result

    def put_writes(self, config: RunnableConfig, writes, task_id: str, task_path: str = "") -> None:
        thread_id = str(config["configurable"]["thread_id"])
        with self._lock:
            self._load(thread_id)
            super().put_writes(config, writes, task_id, task_path)
            self._persist(thread_id)
            self._evict_thread(thread_id)

    async def aput_writes(
        self, config: RunnableConfig, writes, task_id: str, task_path: str = ""
    ) -> None:
        """原生异步 put_writes：await async_cache 写回。"""
        thread_id = str(config["configurable"]["thread_id"])
        await self._aload(thread_id)
        InMemorySaver.put_writes(self, config, writes, task_id, task_path)
        await self._apersist(thread_id)
        self._evict_thread(thread_id)

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            super().delete_thread(thread_id)
            self._cache.delete(self._cache_key(thread_id))

    async def adelete_thread(self, thread_id: str) -> None:
        super().delete_thread(thread_id)
        if self._async_cache is not None:
            await self._async_cache.adelete(self._cache_key(thread_id))
        else:
            self._cache.delete(self._cache_key(thread_id))


def connect_redis(url: str) -> Any:
    """创建同步 Redis 客户端；decode_responses=False 以保存 checkpoint 字节。"""
    import redis

    return redis.Redis.from_url(url, decode_responses=False)


def connect_async_redis(url: str) -> Any:
    """创建异步 Redis 客户端（redis.asyncio），用于原生非阻塞路径。"""
    import redis.asyncio

    return redis.asyncio.Redis.from_url(url, decode_responses=False)


def reset_redis_client() -> None:
    """测试用：丢掉同步与异步 Redis 单例。"""
    global _REDIS, _ASYNC_REDIS
    _REDIS = None
    _ASYNC_REDIS = None


def get_redis_client(*, required: bool = False) -> Any | None:
    """返回已 ping 通的同步 Redis 客户端。

    未配置 ``REDIS_URL`` 时返回 None；已配置但 ping 失败则抛 ``RedisUnavailableError``，禁止回退内存。
    ``required=True`` 表示调用方明确要求必须有 Redis（例如多 worker）。
    """
    global _REDIS
    from app.config import get_settings

    url = (get_settings().redis_url or "").strip()
    if not url:
        if required:
            raise RedisUnavailableError(
                "当前运行模式需要 REDIS_URL。"
                "请先执行 docker compose up -d redis，并在 .env 中设置 "
                "REDIS_URL=redis://127.0.0.1:6379/0"
            )
        return None
    with _REDIS_GUARD:
        if _REDIS is not None:
            return _REDIS
        client = connect_redis(url)
        try:
            client.ping()
        except Exception as exc:
            raise RedisUnavailableError(
                f"Redis 不可用（{url}），禁止回退进程内存。"
            ) from exc
        _REDIS = client
        logger.info("已连接 Redis（同步） %s", url)
        return _REDIS


def get_async_redis_client(*, required: bool = False) -> Any | None:
    """返回异步 Redis 客户端（redis.asyncio）。

    依赖同步 ``get_redis_client`` 完成 ping 校验；通过后创建异步客户端
    （不再重复 ping，避免在启动期引入事件循环）。
    """
    global _ASYNC_REDIS
    from app.config import get_settings

    url = (get_settings().redis_url or "").strip()
    if not url:
        if required:
            raise RedisUnavailableError(
                "当前运行模式需要 REDIS_URL。"
                "请先执行 docker compose up -d redis，并在 .env 中设置 "
                "REDIS_URL=redis://127.0.0.1:6379/0"
            )
        return None
    # 先用同步客户端验证可达（已有 ping 缓存则直接返回）
    get_redis_client(required=required)
    with _ASYNC_REDIS_GUARD:
        if _ASYNC_REDIS is not None:
            return _ASYNC_REDIS
        client = connect_async_redis(url)
        _ASYNC_REDIS = client
        logger.info("已连接 Redis（异步 redis.asyncio） %s", url)
        return _ASYNC_REDIS


def require_redis_for_serve() -> None:
    """HTTP 启动门禁：按模式选择后端，禁止「配了 Redis 却回退内存」。

    - 未配 REDIS_URL 且单 worker：允许进程内存，打告警
    - 未配 REDIS_URL 且 UVICORN_WORKERS>1：拒绝启动（多进程会分裂会话）
    - 已配 REDIS_URL：必须 ping 通，连不上拒绝启动
    """
    from app.config import get_settings

    settings = get_settings()
    url = (settings.redis_url or "").strip()
    workers = max(1, settings.uvicorn_workers)
    if url:
        get_redis_client(required=False)
        return
    if workers > 1:
        raise RedisUnavailableError(
            f"UVICORN_WORKERS={workers} 必须配置 REDIS_URL，否则会话、记忆、配额和并发槽会按进程分裂。"
            "请执行 docker compose up -d redis，并设置 REDIS_URL=redis://127.0.0.1:6379/0"
        )
    logger.warning(
        "未配置 REDIS_URL：以单进程内存运行，重启或多副本无法续聊。"
        "需要共享会话时请启动 Redis 并设置 REDIS_URL。"
    )


def ensure_configured_redis() -> None:
    """若已配置 REDIS_URL 则必须连得上（uvicorn worker 启动时再验一次）。"""
    get_redis_client(required=False)


def build_checkpointer(
    *,
    cache: SessionCache | None = None,
    session_lock: SessionLock | AsyncSessionLock | None = None,
    async_cache: AsyncSessionCache | None = None,
):
    """有 Redis URL 或显式 cache 时每次读写共享缓存，否则纯内存。

    已配置 Redis 时 ping 失败直接抛错，禁止回退 InMemorySaver。
    同时安装按 thread_id 的会话锁：Redis 用 ``AsyncRedisSessionLock``（SET NX +
    asyncio.sleep），未配 Redis 时用进程内 ``DictSessionLock``。
    生产 Redis 路径把 ``AsyncRedisSessionCache`` 传给 checkpointer，避免
    ``astream`` 的 ``aput`` / ``aget_tuple`` 回退同步 redis-py。
    """
    from langgraph.checkpoint.memory import InMemorySaver

    from app.config import get_settings

    settings = get_settings()
    ttl = settings.session_ttl_seconds
    if cache is not None:
        set_session_lock(session_lock or DictSessionLock())
        return CachedInMemorySaver(cache, ttl_seconds=ttl, async_cache=async_cache)
    client = get_redis_client(required=False)
    if client is None:
        set_session_lock(session_lock or DictSessionLock())
        return InMemorySaver()
    async_client = get_async_redis_client(required=False)
    if async_cache is None and async_client is not None:
        async_cache = AsyncRedisSessionCache(async_client)
    if session_lock is None:
        session_lock = (
            AsyncRedisSessionLock(async_client)
            if async_client is not None
            else RedisSessionLock(client)
        )
    logger.info("短期记忆使用 Redis checkpointer（每次读写） ttl=%ss", ttl)
    set_session_lock(session_lock)
    return CachedInMemorySaver(
        RedisSessionCache(client),
        ttl_seconds=ttl,
        async_cache=async_cache,
    )
