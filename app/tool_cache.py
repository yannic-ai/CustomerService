"""工具结果缓存：会话 structured + Redis TTL 双层回退。"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Literal

from app.config import get_settings
from app.observability import inc_tool_cache_hit
from app.schemas import CourseConsultResult, OrderReport

logger = logging.getLogger("cs.tool_cache")

TOOL_ORDER = "order_query"
TOOL_COURSE = "course_retrieve"
CACHE_PREFIX = "cs:tool:result:"

ToolCacheSource = Literal["session", "redis"]


@dataclass(frozen=True)
class CachedToolResult:
    """缓存命中结果，含来源与时间戳。"""

    payload: OrderReport | CourseConsultResult
    cached_at: float
    source: ToolCacheSource


@dataclass(frozen=True)
class SessionToolCacheState:
    """当前会话内上一轮成功的 structured 快照。"""

    last_order_structured: dict[str, Any] | None = None
    last_course_structured: dict[str, Any] | None = None
    last_course_structured_key: str | None = None


_EMPTY_SESSION = SessionToolCacheState()
_session_ctx: ContextVar[SessionToolCacheState] = ContextVar(
    "session_tool_cache", default=_EMPTY_SESSION
)


def get_session_tool_cache() -> SessionToolCacheState:
    return _session_ctx.get()


@contextmanager
def session_tool_cache_scope(state: dict[str, Any] | None) -> Iterator[SessionToolCacheState]:
    """从图状态注入会话缓存快照，供工具层读取。"""
    if not state:
        yield _EMPTY_SESSION
        return
    ctx = SessionToolCacheState(
        last_order_structured=_as_dict(state.get("last_order_structured")),
        last_course_structured=_as_dict(state.get("last_course_structured")),
        last_course_structured_key=_strip(state.get("last_course_structured_key")),
    )
    token = _session_ctx.set(ctx)
    try:
        yield ctx
    finally:
        _session_ctx.reset(token)


def normalize_course_cache_key(query: str, *, course_code: str | None = None) -> str:
    """课程缓存键：优先 course_code，否则归一化 query。"""
    code = _strip(course_code)
    if code:
        return f"code:{code}"
    text = re.sub(r"\s+", " ", (query or "").strip().lower())
    return f"q:{text[:120]}" if text else "q:unknown"


def is_cacheable_order_report(report: OrderReport) -> bool:
    return report.current_status not in {"busy", "error", "not_found"}


def is_cacheable_course_result(result: CourseConsultResult) -> bool:
    return (result.course_name or "").strip() not in {"", "未找到课程"}


def stale_prefix(age_seconds: float) -> str:
    minutes = max(1, int(age_seconds / 60))
    return (
        f"以下为约 {minutes} 分钟前的查询结果，当前服务繁忙/超时，"
        "信息可能已有变化。\n\n"
    )


def annotate_stale_order(report: OrderReport, *, age_seconds: float) -> OrderReport:
    prefix = stale_prefix(age_seconds)
    return report.model_copy(
        update={
            "summary": f"{prefix}{report.summary or ''}".strip(),
        }
    )


def annotate_stale_course(result: CourseConsultResult, *, age_seconds: float) -> CourseConsultResult:
    prefix = stale_prefix(age_seconds)
    return result.model_copy(
        update={
            "summary": f"{prefix}{result.summary or ''}".strip(),
        }
    )


def session_updates_for_order(report: OrderReport) -> dict[str, Any]:
    if not is_cacheable_order_report(report):
        return {}
    payload = report.model_dump(mode="json")
    payload["_cached_at"] = time.time()
    return {"last_order_structured": payload}


def session_updates_for_course(result: CourseConsultResult, resource_key: str) -> dict[str, Any]:
    if not is_cacheable_course_result(result):
        return {}
    payload = result.model_dump(mode="json")
    payload["_cached_at"] = time.time()
    return {
        "last_course_structured": payload,
        "last_course_structured_key": resource_key,
    }


def get_cached_order(
    order_no: str,
    *,
    tenant_id: str,
    user_id: str,
) -> CachedToolResult | None:
    """会话 structured 优先，再读 Redis。"""
    settings = get_settings()
    if not settings.tool_cache_enabled:
        return None
    number = (order_no or "").strip()
    if not number:
        return None
    from app.services.order import can_access_order

    if not can_access_order(number, tenant_id, user_id):
        return None

    session = get_session_tool_cache()
    if session.last_order_structured:
        cached_no = str(session.last_order_structured.get("order_no") or "").strip()
        if cached_no == number:
            try:
                report = OrderReport.model_validate(session.last_order_structured)
                cached_at = float(session.last_order_structured.get("_cached_at") or time.time())
                inc_tool_cache_hit(TOOL_ORDER, "session")
                return CachedToolResult(
                    payload=report,
                    cached_at=cached_at,
                    source="session",
                )
            except Exception:
                logger.debug("会话订单缓存解析失败 order_no=%s", number, exc_info=True)

    raw = _store().get(_redis_key(tenant_id, user_id, TOOL_ORDER, number))
    if not raw:
        return None
    try:
        blob = json.loads(raw)
        report = OrderReport.model_validate(blob["payload"])
        cached_at = float(blob.get("cached_at") or time.time())
        inc_tool_cache_hit(TOOL_ORDER, "redis")
        return CachedToolResult(
            payload=report,
            cached_at=cached_at,
            source="redis",
        )
    except Exception:
        logger.debug("Redis 订单缓存解析失败 order_no=%s", number, exc_info=True)
        return None


def put_cached_order(
    report: OrderReport,
    *,
    tenant_id: str,
    user_id: str,
) -> None:
    settings = get_settings()
    if not settings.tool_cache_enabled or not is_cacheable_order_report(report):
        return
    number = (report.order_no or "").strip()
    if not number:
        return
    ttl = int(settings.tool_cache_order_ttl_seconds)
    if ttl <= 0:
        return
    payload = {
        "payload": report.model_dump(mode="json"),
        "cached_at": time.time(),
    }
    _store().set(_redis_key(tenant_id, user_id, TOOL_ORDER, number), json.dumps(payload), ttl)


def get_cached_course(
    resource_key: str,
    *,
    tenant_id: str,
    user_id: str,
) -> CachedToolResult | None:
    settings = get_settings()
    if not settings.tool_cache_enabled:
        return None
    key = (resource_key or "").strip()
    if not key:
        return None

    session = get_session_tool_cache()
    if session.last_course_structured and session.last_course_structured_key == key:
        try:
            result = CourseConsultResult.model_validate(session.last_course_structured)
            cached_at = float(session.last_course_structured.get("_cached_at") or time.time())
            inc_tool_cache_hit(TOOL_COURSE, "session")
            return CachedToolResult(
                payload=result,
                cached_at=cached_at,
                source="session",
            )
        except Exception:
            logger.debug("会话课程缓存解析失败 key=%s", key, exc_info=True)

    raw = _store().get(_redis_key(tenant_id, user_id, TOOL_COURSE, key))
    if not raw:
        return None
    try:
        blob = json.loads(raw)
        result = CourseConsultResult.model_validate(blob["payload"])
        cached_at = float(blob.get("cached_at") or time.time())
        inc_tool_cache_hit(TOOL_COURSE, "redis")
        return CachedToolResult(
            payload=result,
            cached_at=cached_at,
            source="redis",
        )
    except Exception:
        logger.debug("Redis 课程缓存解析失败 key=%s", key, exc_info=True)
        return None


def put_cached_course(
    result: CourseConsultResult,
    resource_key: str,
    *,
    tenant_id: str,
    user_id: str,
) -> None:
    settings = get_settings()
    if not settings.tool_cache_enabled or not is_cacheable_course_result(result):
        return
    key = (resource_key or "").strip()
    if not key:
        return
    ttl = int(settings.tool_cache_course_ttl_seconds)
    if ttl <= 0:
        return
    payload = {
        "payload": result.model_dump(mode="json"),
        "cached_at": time.time(),
    }
    _store().set(_redis_key(tenant_id, user_id, TOOL_COURSE, key), json.dumps(payload), ttl)


def _redis_key(tenant_id: str, user_id: str, tool: str, resource_key: str) -> str:
    safe = resource_key.replace(":", "_")
    return f"{CACHE_PREFIX}{tenant_id}:{user_id}:{tool}:{safe}"


def _as_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    return None


def _strip(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


class _ToolCacheStore:
    def get(self, key: str) -> str | None:
        raise NotImplementedError

    def set(self, key: str, value: str, ttl: int) -> None:
        raise NotImplementedError


class _MemoryToolCacheStore(_ToolCacheStore):
    def __init__(self) -> None:
        self._data: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._data.get(key)
            if not row:
                return None
            value, expires = row
            if expires and time.time() > expires:
                self._data.pop(key, None)
                return None
            return value

    def set(self, key: str, value: str, ttl: int) -> None:
        expires = time.time() + ttl if ttl > 0 else 0.0
        with self._lock:
            self._data[key] = (value, expires)


class _RedisToolCacheStore(_ToolCacheStore):
    def __init__(self, client: Any) -> None:
        self._client = client

    def get(self, key: str) -> str | None:
        raw = self._client.get(key)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)

    def set(self, key: str, value: str, ttl: int) -> None:
        if ttl > 0:
            self._client.setex(key, ttl, value)
        else:
            self._client.set(key, value)


_MEMORY_STORE = _MemoryToolCacheStore()
_STORE: _ToolCacheStore | None = None
_STORE_GUARD = threading.Lock()


def _store() -> _ToolCacheStore:
    global _STORE
    if _STORE is not None:
        return _STORE
    with _STORE_GUARD:
        if _STORE is not None:
            return _STORE
        from app.session_cache import get_redis_client

        client = get_redis_client(required=False)
        if client is not None:
            _STORE = _RedisToolCacheStore(client)
        else:
            _STORE = _MEMORY_STORE
        return _STORE


def reset_tool_cache_store(store: _ToolCacheStore | None = None) -> None:
    """测试用：替换或重置工具缓存后端。"""
    global _STORE
    with _STORE_GUARD:
        _STORE = store
