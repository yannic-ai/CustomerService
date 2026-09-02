"""LLM Gateway 状态存储：日 token 计数与短响应缓存。"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date
from typing import Any, Protocol

from app.llm_gateway.types import QuotaSnapshot, TokenUsage

logger = logging.getLogger("cs.llm_gateway")

# Redis Cluster Hash Tag 约定：
#   - 配额：按 {tenant:day} 打 tag，同一租户同一天的所有 HINCRBY/HGETALL/EXPIRE
#     路由到同一节点（虽然单 hash key 天然同 slot，但保持 tag 便于未来与其他
#     同租户+日期 key 联合事务）
#     Key 结构：cs:llm:quota:{tenant:YYYY-MM-DD}
#   - 短缓存：仅 GET/SET 单 key，无跨 key 事务需求；且 key 已带 tenant 前缀，
#     故不额外强制 tag，避免热点过于集中
#     Key 结构：cs:llm:cache:<tenant>:<fingerprint>
QUOTA_PREFIX = "cs:llm:quota:"
CACHE_PREFIX = "cs:llm:cache:"
QUOTA_TTL_SECONDS = 60 * 60 * 48

_FALLBACK_WARNED = False


def utc_day() -> str:
    """UTC 日期键。"""
    return date.today().isoformat()


class GatewayStore(Protocol):
    """配额计数 + 字节缓存。"""

    degraded: bool

    def get_quota(self, tenant_id: str, day: str | None = None) -> QuotaSnapshot: ...

    def add_usage(
        self,
        tenant_id: str,
        usage: TokenUsage,
        *,
        day: str | None = None,
    ) -> QuotaSnapshot: ...

    def add_cache_hit(self, tenant_id: str, day: str | None = None) -> None: ...

    def get_cache(self, key: str) -> bytes | None: ...

    def set_cache(self, key: str, value: bytes, ttl: int) -> None: ...


class MemoryGatewayStore:
    """进程内配额与缓存；无 Redis 时的降级实现。"""

    degraded = True

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._quota: dict[tuple[str, str], dict[str, int]] = {}
        self._cache: dict[str, tuple[bytes, float]] = {}

    def _row(self, tenant_id: str, day: str) -> dict[str, int]:
        key = (tenant_id, day)
        row = self._quota.get(key)
        if row is None:
            row = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "requests": 0,
                "cache_hits": 0,
            }
            self._quota[key] = row
        return row

    def get_quota(self, tenant_id: str, day: str | None = None) -> QuotaSnapshot:
        day = day or utc_day()
        with self._lock:
            row = self._row(tenant_id, day)
            return QuotaSnapshot(tenant_id=tenant_id, day=day, **row)

    def add_usage(
        self,
        tenant_id: str,
        usage: TokenUsage,
        *,
        day: str | None = None,
    ) -> QuotaSnapshot:
        day = day or utc_day()
        usage = usage.normalized()
        with self._lock:
            row = self._row(tenant_id, day)
            row["prompt_tokens"] += usage.prompt_tokens
            row["completion_tokens"] += usage.completion_tokens
            row["total_tokens"] += usage.total_tokens
            row["requests"] += 1
            return QuotaSnapshot(tenant_id=tenant_id, day=day, **row)

    def add_cache_hit(self, tenant_id: str, day: str | None = None) -> None:
        day = day or utc_day()
        with self._lock:
            self._row(tenant_id, day)["cache_hits"] += 1

    def get_cache(self, key: str) -> bytes | None:
        with self._lock:
            item = self._cache.get(key)
            if item is None:
                return None
            value, expires = item
            if expires and expires < time.time():
                self._cache.pop(key, None)
                return None
            return value

    def set_cache(self, key: str, value: bytes, ttl: int) -> None:
        expires = time.time() + ttl if ttl > 0 else 0.0
        with self._lock:
            self._cache[key] = (value, expires)


class RedisGatewayStore:
    """Redis HASH 日计数 + SET 缓存。"""

    degraded = False

    def __init__(self, client: Any) -> None:
        self._client = client

    def _quota_key(self, tenant_id: str, day: str) -> str:
        # Hash tag {tenant:day} → Cluster 下同一租户同一天配额路由到同一 slot
        return f"{QUOTA_PREFIX}{{{tenant_id}:{day}}}"

    def _cache_key(self, key: str) -> str:
        # 短缓存是单 key GET/SET，无跨 key 事务需求，不强制加 hash tag
        return f"{CACHE_PREFIX}{key}"

    def get_quota(self, tenant_id: str, day: str | None = None) -> QuotaSnapshot:
        day = day or utc_day()
        raw = self._client.hgetall(self._quota_key(tenant_id, day)) or {}
        decoded: dict[str, int] = {}
        for k, v in raw.items():
            name = k.decode("utf-8") if isinstance(k, bytes) else str(k)
            decoded[name] = int(v)
        return QuotaSnapshot(
            tenant_id=tenant_id,
            day=day,
            prompt_tokens=decoded.get("prompt_tokens", 0),
            completion_tokens=decoded.get("completion_tokens", 0),
            total_tokens=decoded.get("total_tokens", 0),
            requests=decoded.get("requests", 0),
            cache_hits=decoded.get("cache_hits", 0),
        )

    def add_usage(
        self,
        tenant_id: str,
        usage: TokenUsage,
        *,
        day: str | None = None,
    ) -> QuotaSnapshot:
        day = day or utc_day()
        usage = usage.normalized()
        key = self._quota_key(tenant_id, day)
        pipe = self._client.pipeline()
        pipe.hincrby(key, "prompt_tokens", usage.prompt_tokens)
        pipe.hincrby(key, "completion_tokens", usage.completion_tokens)
        pipe.hincrby(key, "total_tokens", usage.total_tokens)
        pipe.hincrby(key, "requests", 1)
        pipe.expire(key, QUOTA_TTL_SECONDS)
        pipe.hgetall(key)
        results = pipe.execute()
        decoded = results[-1] or {}
        values: dict[str, int] = {}
        for k, v in decoded.items():
            name = k.decode("utf-8") if isinstance(k, bytes) else str(k)
            values[name] = int(v)
        return QuotaSnapshot(
            tenant_id=tenant_id,
            day=day,
            prompt_tokens=values.get("prompt_tokens", 0),
            completion_tokens=values.get("completion_tokens", 0),
            total_tokens=values.get("total_tokens", 0),
            requests=values.get("requests", 0),
            cache_hits=values.get("cache_hits", 0),
        )

    def add_cache_hit(self, tenant_id: str, day: str | None = None) -> None:
        day = day or utc_day()
        key = self._quota_key(tenant_id, day)
        pipe = self._client.pipeline()
        pipe.hincrby(key, "cache_hits", 1)
        pipe.expire(key, QUOTA_TTL_SECONDS)
        pipe.execute()

    def get_cache(self, key: str) -> bytes | None:
        value = self._client.get(self._cache_key(key))
        if value is None:
            return None
        if isinstance(value, str):
            return value.encode("utf-8")
        return bytes(value)

    def set_cache(self, key: str, value: bytes, ttl: int) -> None:
        full = self._cache_key(key)
        if ttl > 0:
            self._client.set(full, value, ex=ttl)
        else:
            self._client.set(full, value)


def build_gateway_store(*, client: Any | None = None) -> GatewayStore:
    """显式 client 或已连接 Redis 时用 Redis；否则进程内近似。"""
    global _FALLBACK_WARNED
    from app.session_cache import get_redis_client

    redis_client = client if client is not None else get_redis_client(required=False)
    if redis_client is not None:
        logger.info("LLM Gateway 使用 Redis 存储配额与短缓存")
        return RedisGatewayStore(redis_client)
    if not _FALLBACK_WARNED:
        logger.warning(
            "未配置 REDIS_URL：LLM 日限额与短响应缓存仅为单进程近似，多 worker 不共享"
        )
        _FALLBACK_WARNED = True
    return MemoryGatewayStore()


def set_gateway_store(store: GatewayStore | None) -> None:
    """测试用：替换或清空 Gateway 存储。空值表示下次按 Redis 惰性重建。"""
    from app.runtime import replace_app_context

    replace_app_context(gateway_store=store)


def get_gateway_store() -> GatewayStore:
    """有 Redis 则用 Redis；未配置 REDIS_URL 时用进程内。已配置但连不上不降级。"""
    from app.runtime import get_app_context

    return get_app_context().ensure_gateway_store()
