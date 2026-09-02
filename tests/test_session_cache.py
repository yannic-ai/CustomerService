import asyncio
import threading
import time

import pytest

from app.concurrency import achat
from app.graph import make_thread_id, reset_graph
from app.session_cache import (
    ENVELOPE_MAGIC,
    CachedInMemorySaver,
    DictSessionCache,
    DictSessionLock,
    RedisSessionLock,
    SessionBusyError,
    decode_checkpoint_blob,
    encode_checkpoint_blob,
)


def _chat(message, **kwargs):
    return asyncio.run(achat(message, **kwargs))


def test_session_survives_new_checkpointer() -> None:
    cache = DictSessionCache()
    reset_graph(checkpointer=CachedInMemorySaver(cache, ttl_seconds=3600))
    try:
        session_id = "persist-order-session"
        first = _chat(
            "查询订单#20251114001的退款进度",
            session_id=session_id,
            tenant_id="demo",
        )
        assert first.intent == "order"
        assert "20251114001" in first.answer

        reset_graph(checkpointer=CachedInMemorySaver(cache, ttl_seconds=3600))
        second = _chat("那退款呢", session_id=session_id, tenant_id="demo")
        assert second.blocked is False
        assert second.intent == "order"
        assert "20251114001" in second.answer
    finally:
        reset_graph()


def test_live_saver_reloads_peer_instance_updates() -> None:
    """同一 saver 进程必须每次从共享缓存读，不能 hydrate-once 后一直用本机内存。"""
    cache = DictSessionCache()
    saver_a = CachedInMemorySaver(cache, ttl_seconds=3600)
    saver_b = CachedInMemorySaver(cache, ttl_seconds=3600)
    session_id = "multi-worker-session"
    try:
        reset_graph(checkpointer=saver_a)
        first = _chat(
            "查询订单#20251114001的退款进度",
            session_id=session_id,
            tenant_id="demo",
        )
        assert first.intent == "order"

        reset_graph(checkpointer=saver_b)
        peer = _chat(
            "Python入门课包含哪些内容？",
            session_id=session_id,
            tenant_id="demo",
        )
        assert peer.intent == "course_consult"

        reset_graph(checkpointer=saver_a)
        follow = _chat("这个怎么样", session_id=session_id, tenant_id="demo")
        assert follow.blocked is False
        assert follow.intent == "course_consult"
        assert "Python" in follow.answer or "模块" in follow.answer
    finally:
        reset_graph()


def test_dict_lock_blocks_same_thread_id() -> None:
    lock = DictSessionLock(wait_seconds=0.15)
    held = threading.Event()

    def _hold() -> None:
        with lock.hold("demo:anonymous:sess"):
            held.set()
            time.sleep(0.4)

    worker = threading.Thread(target=_hold)
    worker.start()
    assert held.wait(timeout=1)
    with pytest.raises(SessionBusyError) as exc:
        lock.acquire("demo:anonymous:sess")
    assert exc.value.thread_id == "demo:anonymous:sess"
    worker.join()


def test_dict_lock_allows_different_thread_ids() -> None:
    lock = DictSessionLock(wait_seconds=1)
    started = threading.Barrier(2)
    done: list[str] = []

    def _hold(thread_id: str) -> None:
        with lock.hold(thread_id):
            started.wait(timeout=1)
            time.sleep(0.1)
            done.append(thread_id)

    workers = [
        threading.Thread(target=_hold, args=("t-a",)),
        threading.Thread(target=_hold, args=("t-b",)),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2)
    assert set(done) == {"t-a", "t-b"}


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def set(self, name: str, value: str, nx: bool = False, px: int | None = None) -> bool:
        del px
        if nx and name in self.store:
            return False
        self.store[name] = value
        return True

    def eval(self, script: str, numkeys: int, *args: str) -> int:
        del script, numkeys
        key, token = args
        if self.store.get(key) == token:
            del self.store[key]
            return 1
        return 0


def test_dict_lock_ahold_blocks_same_thread_id() -> None:
    lock = DictSessionLock(wait_seconds=0.15)
    held = threading.Event()

    def _hold() -> None:
        with lock.hold("demo:anonymous:sess"):
            held.set()
            time.sleep(0.4)

    async def _contend() -> None:
        with pytest.raises(SessionBusyError) as exc:
            await lock.aacquire("demo:anonymous:sess")
        assert exc.value.thread_id == "demo:anonymous:sess"

    worker = threading.Thread(target=_hold)
    worker.start()
    try:
        assert held.wait(timeout=1)
        asyncio.run(_contend())
    finally:
        worker.join()


def test_redis_lock_token_release_is_safe() -> None:
    client = _FakeRedis()
    lock_a = RedisSessionLock(client, wait_seconds=0.05, ttl_seconds=5, retry_interval=0.01)
    lock_b = RedisSessionLock(client, wait_seconds=0.05, ttl_seconds=5, retry_interval=0.01)
    token = lock_a.acquire("sess-1")
    with pytest.raises(SessionBusyError):
        lock_b.acquire("sess-1")
    lock_b.release("sess-1", "stolen-token")
    assert "cs:session:lock:{sess-1}" in client.store
    lock_a.release("sess-1", token)
    assert "cs:session:lock:{sess-1}" not in client.store
    lock_b.acquire("sess-1")


class _FakeAsyncRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def set(self, name: str, value: str, nx: bool = False, px: int | None = None) -> bool:
        del px
        if nx and name in self.store:
            return False
        self.store[name] = value
        return True

    async def eval(self, script: str, numkeys: int, *args: str) -> int:
        del script, numkeys
        key, token = args
        if self.store.get(key) == token:
            del self.store[key]
            return 1
        return 0


def test_async_redis_lock_token_release_is_safe() -> None:
    from app.session_cache import AsyncRedisSessionLock

    client = _FakeAsyncRedis()
    lock_a = AsyncRedisSessionLock(client, wait_seconds=0.05, ttl_seconds=5, retry_interval=0.01)
    lock_b = AsyncRedisSessionLock(client, wait_seconds=0.05, ttl_seconds=5, retry_interval=0.01)

    async def _run() -> None:
        token = await lock_a.aacquire("sess-1")
        with pytest.raises(SessionBusyError):
            await lock_b.aacquire("sess-1")
        await lock_b.arelease("sess-1", "stolen-token")
        assert "cs:session:lock:{sess-1}" in client.store
        await lock_a.arelease("sess-1", token)
        assert "cs:session:lock:{sess-1}" not in client.store
        await lock_b.aacquire("sess-1")

    asyncio.run(_run())


def test_ahold_session_falls_back_to_thread_for_sync_lock() -> None:
    from app.session_cache import ahold_session

    client = _FakeRedis()
    lock = RedisSessionLock(client, wait_seconds=0.05, ttl_seconds=5, retry_interval=0.01)

    async def _run() -> None:
        async with ahold_session(lock, "sess-ahold"):
            assert "cs:session:lock:{sess-ahold}" in client.store
        assert "cs:session:lock:{sess-ahold}" not in client.store

    asyncio.run(_run())


def test_build_checkpointer_wires_async_redis(monkeypatch) -> None:
    from app.session_cache import (
        AsyncRedisSessionCache,
        AsyncRedisSessionLock,
        RedisSessionCache,
        build_checkpointer,
        get_session_lock,
        reset_redis_client,
        set_session_lock,
    )

    fake_sync = object()
    fake_async = object()
    monkeypatch.setattr("app.session_cache.get_redis_client", lambda required=False: fake_sync)
    monkeypatch.setattr("app.session_cache.get_async_redis_client", lambda required=False: fake_async)
    try:
        saver = build_checkpointer()
        assert isinstance(saver._cache, RedisSessionCache)
        assert isinstance(saver._async_cache, AsyncRedisSessionCache)
        assert saver._async_cache._client is fake_async
        lock = get_session_lock()
        assert isinstance(lock, AsyncRedisSessionLock)
        assert lock._client is fake_async
    finally:
        reset_redis_client()
        set_session_lock(None)
        reset_graph()


def test_aget_tuple_uses_async_cache() -> None:
    class _TrackingAsyncCache:
        def __init__(self, inner: DictSessionCache) -> None:
            self._inner = inner
            self.aget_calls = 0

        async def aget(self, key: str) -> bytes | None:
            self.aget_calls += 1
            return self._inner.get(key)

        async def aset(self, key: str, value: bytes, ttl: int) -> None:
            self._inner.set(key, value, ttl)

        async def adelete(self, key: str) -> None:
            self._inner.delete(key)

        async def atouch(self, key: str, ttl: int) -> None:
            self._inner.touch(key, ttl)

    cache = DictSessionCache()
    async_cache = _TrackingAsyncCache(cache)
    saver = CachedInMemorySaver(cache, ttl_seconds=1800, async_cache=async_cache)
    thread_id = "async-cache-session"
    cache.set(
        saver._cache_key(thread_id),
        encode_checkpoint_blob({"storage": {}, "writes": {}, "blobs": {}}),
        ttl=1800,
    )
    asyncio.run(saver.aget_tuple({"configurable": {"thread_id": thread_id}}))
    assert async_cache.aget_calls == 1


def test_chat_same_session_raises_busy() -> None:
    lock = DictSessionLock(wait_seconds=0.05)
    reset_graph(session_lock=lock)
    session_id = "lock-conflict-session"
    thread_id = make_thread_id("demo", session_id)
    held = threading.Event()

    def _hold() -> None:
        with lock.hold(thread_id):
            held.set()
            time.sleep(0.3)

    worker = threading.Thread(target=_hold)
    worker.start()
    try:
        assert held.wait(timeout=1)
        with pytest.raises(SessionBusyError):
            _chat("Python入门课包含哪些内容？", session_id=session_id, tenant_id="demo")
    finally:
        worker.join()
        reset_graph()


def test_serve_allows_memory_when_single_worker() -> None:
    from app.session_cache import require_redis_for_serve, reset_redis_client

    reset_redis_client()
    require_redis_for_serve()


def test_serve_requires_redis_when_multi_worker(monkeypatch) -> None:
    from app.config import get_settings
    from app.session_cache import RedisUnavailableError, require_redis_for_serve, reset_redis_client

    settings = get_settings()
    monkeypatch.setattr(settings, "uvicorn_workers", 2)
    reset_redis_client()
    with pytest.raises(RedisUnavailableError, match="UVICORN_WORKERS"):
        require_redis_for_serve()


def test_build_checkpointer_does_not_fallback_when_redis_down(monkeypatch) -> None:
    from app.config import get_settings
    from app.session_cache import (
        RedisUnavailableError,
        build_checkpointer,
        reset_redis_client,
    )

    settings = get_settings()

    class _Dead:
        def ping(self) -> None:
            raise ConnectionError("down")

    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:1/0")
    monkeypatch.setattr("app.session_cache.connect_redis", lambda _url: _Dead())
    reset_redis_client()
    try:
        with pytest.raises(RedisUnavailableError, match="禁止回退"):
            build_checkpointer()
    finally:
        reset_redis_client()


# ---------------- Hash tag + 滑动过期测试 ---------------- #


def test_cache_key_has_hash_tag() -> None:
    """cache key 带 {thread_id} hash tag，确保 Redis Cluster 同节点路由。"""
    saver = CachedInMemorySaver(DictSessionCache(), ttl_seconds=3600)
    key = saver._cache_key("demo:sess-1")
    assert key.startswith("cs:session:{")
    assert key.endswith("}")
    assert "demo:sess-1" in key
    # hash tag 是花括号内的内容
    assert "{demo:sess-1}" in key


def test_lock_key_has_same_hash_tag_as_cache() -> None:
    """lock key 和 cache key 的 hash tag 相同，Cluster 下同节点。"""
    import re

    from app.session_cache import KEY_PREFIX, LOCK_PREFIX

    thread_id = "demo:sess-1"
    cache_key = f"{KEY_PREFIX}{{{thread_id}}}"
    lock_key = f"{LOCK_PREFIX}{{{thread_id}}}"

    # 提取 hash tag（花括号内内容）
    cache_tag = re.search(r"\{(.+)\}", cache_key)
    lock_tag = re.search(r"\{(.+)\}", lock_key)
    assert cache_tag is not None
    assert lock_tag is not None
    assert cache_tag.group(1) == lock_tag.group(1) == thread_id


def test_dict_cache_touch_is_noop() -> None:
    """DictSessionCache.touch 不报错且不影响数据。"""
    cache = DictSessionCache()
    cache.set("k1", b"v1", ttl=60)
    cache.touch("k1", 120)
    assert cache.get("k1") == b"v1"
    cache.touch("nonexistent", 60)  # 不存在的 key 不报错


class _TrackingCache(DictSessionCache):
    """记录 touch 调用的缓存包装。"""

    def __init__(self) -> None:
        super().__init__()
        self.touch_calls: list[tuple[str, int]] = []

    def touch(self, key: str, ttl: int) -> None:
        self.touch_calls.append((key, ttl))


def test_sliding_ttl_refreshed_on_read() -> None:
    """读取命中的会话时刷新 TTL（滑动过期）。"""
    cache = _TrackingCache()
    saver = CachedInMemorySaver(cache, ttl_seconds=1800)
    thread_id = "ttl-session"

    cache_key = saver._cache_key(thread_id)
    cache.set(
        cache_key,
        encode_checkpoint_blob({"storage": {}, "writes": {}, "blobs": {}}),
        ttl=1800,
    )
    assert len(cache.touch_calls) == 0  # 写入不触发 touch

    config = {"configurable": {"thread_id": thread_id}}
    saver.get_tuple(config)
    assert len(cache.touch_calls) == 1
    assert cache.touch_calls[0][1] == 1800


def test_sliding_ttl_not_refreshed_on_miss() -> None:
    """缓存未命中时不触发 touch。"""
    cache = _TrackingCache()
    saver = CachedInMemorySaver(cache, ttl_seconds=1800)
    config = {"configurable": {"thread_id": "empty-session"}}
    saver.get_tuple(config)  # 缓存未命中
    assert len(cache.touch_calls) == 0


class _FakeRedisCache:
    """支持 expire 的 fake Redis，用于测试 RedisSessionCache.touch。"""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.expire_calls: list[tuple[str, int]] = []

    def get(self, key: str) -> bytes | None:
        return self.store.get(key)

    def set(self, name: str, value: bytes, ex: int | None = None) -> None:
        self.store[name] = value

    def delete(self, key: str) -> None:
        self.store.pop(key, None)

    def expire(self, key: str, ttl: int) -> bool:
        self.expire_calls.append((key, ttl))
        return key in self.store


def test_redis_cache_touch_calls_expire() -> None:
    """RedisSessionCache.touch 调用 redis.expire 刷新 TTL。"""
    from app.session_cache import RedisSessionCache

    client = _FakeRedisCache()
    cache = RedisSessionCache(client)
    cache.set("k1", b"v1", ttl=60)
    cache.touch("k1", 120)
    assert len(client.expire_calls) == 1
    assert client.expire_calls[0] == ("k1", 120)


def test_redis_cache_touch_handles_error() -> None:
    """RedisSessionCache.touch 异常时不传播。"""
    from app.session_cache import RedisSessionCache

    class _BrokenRedis:
        def expire(self, key: str, ttl: int) -> None:
            raise RuntimeError("连接断开")

    cache = RedisSessionCache(_BrokenRedis())
    cache.touch("k1", 60)  # 不抛异常


def _thread_payload(cache: DictSessionCache, saver: CachedInMemorySaver, session_id: str) -> dict:
    thread_id = make_thread_id("demo", session_id)
    raw = cache.get(saver._cache_key(thread_id))
    assert raw is not None
    payload = decode_checkpoint_blob(raw)
    assert payload is not None
    return payload


def test_persist_uses_csk1_envelope() -> None:
    """新写入的会话是 CSK1+msgpack，不是 pickle。"""
    cache = DictSessionCache()
    saver = CachedInMemorySaver(cache, ttl_seconds=3600)
    session_id = "csk1-session"
    try:
        reset_graph(checkpointer=saver)
        _chat("查询订单#20251114001的退款进度", session_id=session_id, tenant_id="demo")
        thread_id = make_thread_id("demo", session_id)
        raw = cache.get(saver._cache_key(thread_id))
        assert raw is not None
        assert raw.startswith(ENVELOPE_MAGIC)
        payload = decode_checkpoint_blob(raw)
        assert payload is not None
        assert payload["storage"]
    finally:
        reset_graph()


def test_persist_keeps_only_latest_checkpoint_per_ns() -> None:
    """多轮对话后每个 namespace 只留 1 个 checkpoint_id。"""
    cache = DictSessionCache()
    saver = CachedInMemorySaver(cache, ttl_seconds=3600)
    session_id = "prune-session"
    try:
        reset_graph(checkpointer=saver)
        _chat("查询订单#20251114001的退款进度", session_id=session_id, tenant_id="demo")
        first = _thread_payload(cache, saver, session_id)
        first_counts = [len(cps) for cps in first["storage"].values()]
        _chat("那退款呢", session_id=session_id, tenant_id="demo")
        second = _thread_payload(cache, saver, session_id)
        second_counts = [len(cps) for cps in second["storage"].values()]
        assert first_counts
        assert second_counts
        assert all(count == 1 for count in first_counts)
        assert all(count == 1 for count in second_counts)
    finally:
        reset_graph()


def test_legacy_pickle_blob_still_loads() -> None:
    """TTL 内旧 pickle 会话仍能读出并续聊。"""
    import pickle

    cache = DictSessionCache()
    saver = CachedInMemorySaver(cache, ttl_seconds=3600)
    session_id = "legacy-pickle-session"
    try:
        reset_graph(checkpointer=saver)
        first = _chat(
            "查询订单#20251114001的退款进度",
            session_id=session_id,
            tenant_id="demo",
        )
        assert first.intent == "order"
        thread_id = make_thread_id("demo", session_id)
        cache_key = saver._cache_key(thread_id)
        payload = decode_checkpoint_blob(cache.get(cache_key) or b"")
        assert payload is not None
        cache.set(cache_key, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), ttl=3600)
        reset_graph(checkpointer=CachedInMemorySaver(cache, ttl_seconds=3600))
        second = _chat("那退款呢", session_id=session_id, tenant_id="demo")
        assert second.intent == "order"
        assert "20251114001" in second.answer
    finally:
        reset_graph()


def test_corrupt_blob_is_deleted_and_get_tuple_returns_none() -> None:
    """损坏字节当 miss：不抛错，并删掉坏 key。"""
    cache = DictSessionCache()
    saver = CachedInMemorySaver(cache, ttl_seconds=1800)
    thread_id = "corrupt-session"
    cache_key = saver._cache_key(thread_id)
    cache.set(cache_key, b"CSK1not-valid-msgpack", ttl=1800)
    result = saver.get_tuple({"configurable": {"thread_id": thread_id}})
    assert result is None
    assert cache.get(cache_key) is None


def test_evicts_local_copy_after_persist() -> None:
    """回写成功后驱逐本机副本，不以进程内存为事实源。"""
    cache = DictSessionCache()
    saver = CachedInMemorySaver(cache, ttl_seconds=3600)
    session_id = "evict-session"
    try:
        reset_graph(checkpointer=saver)
        _chat("查询订单#20251114001的退款进度", session_id=session_id, tenant_id="demo")
        thread_id = make_thread_id("demo", session_id)
        assert thread_id not in saver.storage
        assert cache.get(saver._cache_key(thread_id))
    finally:
        reset_graph()
