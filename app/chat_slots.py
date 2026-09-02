"""集群级对话并发槽：全局上限 + 租户上限，有 Redis 时跨 worker 共享。"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Literal, Protocol

logger = logging.getLogger("cs.chat_slots")

BusyReason = Literal["cluster", "tenant"]

# Redis Cluster：全局槽与租户槽必须同 slot 才能 EVAL。
# 全部打上 {cs:chat}，容量闸门 key 很小，热点可接受。
GLOBAL_KEY = "cs:chat:inflight:{cs:chat}"
TENANT_KEY_PREFIX = "cs:chat:tenant:{cs:chat}:"

_ACQUIRE_LUA = """
local now = tonumber(ARGV[1])
local global_limit = tonumber(ARGV[2])
local tenant_limit = tonumber(ARGV[3])
local expire_at = tonumber(ARGV[4])
local token = ARGV[5]
redis.call("ZREMRANGEBYSCORE", KEYS[1], "-inf", now)
redis.call("ZREMRANGEBYSCORE", KEYS[2], "-inf", now)
if redis.call("ZCARD", KEYS[1]) >= global_limit then
  return -1
end
if redis.call("ZCARD", KEYS[2]) >= tenant_limit then
  return -2
end
redis.call("ZADD", KEYS[1], expire_at, token)
redis.call("ZADD", KEYS[2], expire_at, token)
return 1
"""

_RELEASE_LUA = """
redis.call("ZREM", KEYS[1], ARGV[1])
redis.call("ZREM", KEYS[2], ARGV[1])
return 1
"""

_COUNT_LUA = """
local now = tonumber(ARGV[1])
redis.call("ZREMRANGEBYSCORE", KEYS[1], "-inf", now)
return redis.call("ZCARD", KEYS[1])
"""

_STORE: "ChatSlotStore | None" = None
_FALLBACK_WARNED = False


class ChatSlotBusy(Exception):
    """并发槽已满；``reason`` 为 cluster（全集群）或 tenant（该租户）。"""

    def __init__(self, reason: BusyReason) -> None:
        self.reason: BusyReason = reason
        super().__init__(reason)


class ChatSlotStore(Protocol):
    """占用 / 释放对话槽。"""

    degraded: bool

    def acquire(
        self,
        tenant_id: str,
        *,
        global_limit: int,
        tenant_limit: int,
        ttl_seconds: int,
        now: float | None = None,
    ) -> str: ...

    def release(self, tenant_id: str, token: str) -> None: ...

    def in_flight(self, *, now: float | None = None) -> int: ...


def tenant_slot_key(tenant_id: str) -> str:
    return f"{TENANT_KEY_PREFIX}{tenant_id}"


def _new_token() -> str:
    return uuid.uuid4().hex


def _clock(now: float | None) -> float:
    return time.time() if now is None else float(now)


class MemoryChatSlotStore:
    """进程内槽位；无 REDIS_URL 时的降级实现。"""

    degraded = True

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._global: dict[str, float] = {}
        self._tenants: dict[str, dict[str, float]] = {}

    def _purge(self, now: float) -> None:
        stale = [token for token, exp in self._global.items() if exp <= now]
        for token in stale:
            self._global.pop(token, None)
            for held in self._tenants.values():
                held.pop(token, None)

    def acquire(
        self,
        tenant_id: str,
        *,
        global_limit: int,
        tenant_limit: int,
        ttl_seconds: int,
        now: float | None = None,
    ) -> str:
        now = _clock(now)
        expire_at = now + max(1, int(ttl_seconds))
        with self._lock:
            self._purge(now)
            if len(self._global) >= global_limit:
                raise ChatSlotBusy("cluster")
            held = self._tenants.setdefault(tenant_id, {})
            if len(held) >= tenant_limit:
                raise ChatSlotBusy("tenant")
            token = _new_token()
            self._global[token] = expire_at
            held[token] = expire_at
            return token

    def release(self, tenant_id: str, token: str) -> None:
        with self._lock:
            self._global.pop(token, None)
            held = self._tenants.get(tenant_id)
            if held is not None:
                held.pop(token, None)

    def in_flight(self, *, now: float | None = None) -> int:
        now = _clock(now)
        with self._lock:
            self._purge(now)
            return len(self._global)


class RedisChatSlotStore:
    """Redis ZSET 租约：过期成员自动剔除，worker 崩溃不会永久占槽。"""

    degraded = False

    def __init__(self, client: Any) -> None:
        self._client = client

    def acquire(
        self,
        tenant_id: str,
        *,
        global_limit: int,
        tenant_limit: int,
        ttl_seconds: int,
        now: float | None = None,
    ) -> str:
        now = _clock(now)
        token = _new_token()
        expire_at = now + max(1, int(ttl_seconds))
        result = self._client.eval(
            _ACQUIRE_LUA,
            2,
            GLOBAL_KEY,
            tenant_slot_key(tenant_id),
            now,
            int(global_limit),
            int(tenant_limit),
            expire_at,
            token,
        )
        code = int(result)
        if code == -1:
            raise ChatSlotBusy("cluster")
        if code == -2:
            raise ChatSlotBusy("tenant")
        return token

    def release(self, tenant_id: str, token: str) -> None:
        try:
            self._client.eval(
                _RELEASE_LUA,
                2,
                GLOBAL_KEY,
                tenant_slot_key(tenant_id),
                token,
            )
        except Exception:
            logger.exception("释放对话并发槽失败 tenant=%s", tenant_id)

    def in_flight(self, *, now: float | None = None) -> int:
        now = _clock(now)
        result = self._client.eval(_COUNT_LUA, 1, GLOBAL_KEY, now)
        return int(result or 0)


def set_chat_slot_store(store: ChatSlotStore | None) -> None:
    """测试用：替换或清空槽位存储单例。"""
    global _STORE
    _STORE = store


def reset_chat_slot_store() -> None:
    """测试用：丢掉槽位单例。"""
    set_chat_slot_store(None)


def get_chat_slot_store() -> ChatSlotStore:
    """有 Redis 则用集群槽；未配置 REDIS_URL 时用进程内。已配置但连不上不降级。"""
    global _STORE, _FALLBACK_WARNED
    if _STORE is not None:
        return _STORE
    from app.session_cache import get_redis_client

    client = get_redis_client(required=False)
    if client is not None:
        _STORE = RedisChatSlotStore(client)
        logger.info("对话并发槽使用 Redis（集群全局 + 租户上限）")
        return _STORE
    if not _FALLBACK_WARNED:
        logger.warning(
            "未配置 REDIS_URL：对话并发槽仅为单进程近似，多 worker 会放大 CHAT_CONCURRENCY"
        )
        _FALLBACK_WARNED = True
    _STORE = MemoryChatSlotStore()
    return _STORE
