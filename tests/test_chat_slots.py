"""集群对话并发槽：全局上限、租户上限、Redis 共享与过期回收。"""

from __future__ import annotations

import pytest

from app.chat_slots import (
    ChatSlotBusy,
    MemoryChatSlotStore,
    RedisChatSlotStore,
    GLOBAL_KEY,
    tenant_slot_key,
)


class _FakeRedis:
    """覆盖容量闸门用到的 ZSET EVAL。"""

    def __init__(self) -> None:
        self.zsets: dict[str, dict[str, float]] = {}

    def _z(self, key: str) -> dict[str, float]:
        return self.zsets.setdefault(str(key), {})

    @staticmethod
    def _purge(held: dict[str, float], now: float) -> None:
        for member, score in list(held.items()):
            if score <= now:
                del held[member]

    def eval(self, script: str, numkeys: int, *args: object) -> int:
        keys = [str(item) for item in args[:numkeys]]
        argv = list(args[numkeys:])
        if "ZADD" in script:
            now = float(argv[0])
            global_limit = int(float(argv[1]))
            tenant_limit = int(float(argv[2]))
            expire_at = float(argv[3])
            token = str(argv[4])
            global_held = self._z(keys[0])
            tenant_held = self._z(keys[1])
            self._purge(global_held, now)
            self._purge(tenant_held, now)
            if len(global_held) >= global_limit:
                return -1
            if len(tenant_held) >= tenant_limit:
                return -2
            global_held[token] = expire_at
            tenant_held[token] = expire_at
            return 1
        if "ZCARD" in script:
            now = float(argv[0])
            held = self._z(keys[0])
            self._purge(held, now)
            return len(held)
        token = str(argv[0])
        self._z(keys[0]).pop(token, None)
        self._z(keys[1]).pop(token, None)
        return 1


def test_memory_global_limit() -> None:
    store = MemoryChatSlotStore()
    first = store.acquire("demo", global_limit=1, tenant_limit=1, ttl_seconds=60)
    with pytest.raises(ChatSlotBusy) as exc:
        store.acquire("acme", global_limit=1, tenant_limit=1, ttl_seconds=60)
    assert exc.value.reason == "cluster"
    store.release("demo", first)
    store.acquire("acme", global_limit=1, tenant_limit=1, ttl_seconds=60)


def test_memory_tenant_limit_does_not_block_other_tenant() -> None:
    store = MemoryChatSlotStore()
    store.acquire("demo", global_limit=2, tenant_limit=1, ttl_seconds=60)
    with pytest.raises(ChatSlotBusy) as exc:
        store.acquire("demo", global_limit=2, tenant_limit=1, ttl_seconds=60)
    assert exc.value.reason == "tenant"
    store.acquire("acme", global_limit=2, tenant_limit=1, ttl_seconds=60)
    assert store.in_flight() == 2


def test_memory_expired_lease_is_reclaimed() -> None:
    store = MemoryChatSlotStore()
    store.acquire("demo", global_limit=1, tenant_limit=1, ttl_seconds=10, now=100.0)
    assert store.in_flight(now=100.0) == 1
    store.acquire("demo", global_limit=1, tenant_limit=1, ttl_seconds=10, now=111.0)
    assert store.in_flight(now=111.0) == 1


def test_redis_slots_are_shared_across_workers() -> None:
    client = _FakeRedis()
    worker_a = RedisChatSlotStore(client)
    worker_b = RedisChatSlotStore(client)
    token = worker_a.acquire("demo", global_limit=1, tenant_limit=1, ttl_seconds=60, now=1.0)
    with pytest.raises(ChatSlotBusy) as exc:
        worker_b.acquire("acme", global_limit=1, tenant_limit=1, ttl_seconds=60, now=1.0)
    assert exc.value.reason == "cluster"
    assert worker_b.in_flight(now=1.0) == 1
    worker_a.release("demo", token)
    worker_b.acquire("acme", global_limit=1, tenant_limit=1, ttl_seconds=60, now=1.0)


def test_redis_tenant_cap_is_shared() -> None:
    client = _FakeRedis()
    worker_a = RedisChatSlotStore(client)
    worker_b = RedisChatSlotStore(client)
    worker_a.acquire("demo", global_limit=4, tenant_limit=1, ttl_seconds=60, now=1.0)
    with pytest.raises(ChatSlotBusy) as exc:
        worker_b.acquire("demo", global_limit=4, tenant_limit=1, ttl_seconds=60, now=1.0)
    assert exc.value.reason == "tenant"
    worker_b.acquire("acme", global_limit=4, tenant_limit=1, ttl_seconds=60, now=1.0)
    assert tenant_slot_key("demo").startswith("cs:chat:tenant:{cs:chat}:")
    assert GLOBAL_KEY.endswith("{cs:chat}")
