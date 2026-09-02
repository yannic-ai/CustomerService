"""用户长期记忆：基于 LangGraph Store，按租户 + 用户保存结构化事实。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    PutOp,
    SearchOp,
)
from langgraph.store.memory import InMemoryStore

from app.context import MISSING_COURSE_NAMES, SessionSlots

logger = logging.getLogger("cs.memory")

PROFILE_KEY = "profile"
EPHEMERAL_USERS = frozenset({"", "anonymous", "anon"})
MAX_RECENT = 5

# Redis Cluster Hash Tag 约定：
#   同一用户的长期记忆 key 必须共享 hash tag {tenant:user}，
#   保证 Cluster 下同一用户的所有记忆字段路由到同一节点，便于未来做事务或批量操作。
#   Key 结构：
#     cs:memory:{tenant:user}tenants/<tenant>/users/<user>:profile   ← 用户档案
MEMORY_KEY_PREFIX = "cs:memory:"


class RedisUserStore(BaseStore):
    """把用户档案存 Redis JSON，多 worker 共享；TTL 用秒级 EXPIRE。"""

    supports_ttl = True

    def __init__(self, client: Any, *, ttl_minutes: int = 0) -> None:
        self._client = client
        self._ttl_minutes = max(0, int(ttl_minutes))

    @staticmethod
    def _hash_tag_for_namespace(namespace: tuple[str, ...]) -> str:
        """从命名空间提取 hash tag。

        - 标准用户命名空间 ``("tenants", tenant_id, "users", user_id, ...)``
          → tag = ``{tenant_id:user_id}``
        - 其他格式回退为最后两段拼接，极端情况退化为 ``{namespace[0]}``，
          保证始终有合法 tag 以避免 Cluster 跨 slot 事务失败。
        """
        if len(namespace) >= 4 and namespace[0] == "tenants" and namespace[2] == "users":
            tenant = (namespace[1] or "").strip() or "unknown"
            user = (namespace[3] or "").strip() or "unknown"
            return f"{{{tenant}:{user}}}"
        if len(namespace) >= 2:
            return f"{{{namespace[-2]}:{namespace[-1]}}}"
        if namespace:
            return f"{{{namespace[0]}}}"
        return "{default}"

    def _key(self, namespace: tuple[str, ...], key: str) -> str:
        parts = "/".join(part.replace("/", "_") for part in namespace)
        tag = self._hash_tag_for_namespace(namespace)
        # hash tag 紧跟前缀，避免前缀参与 CRC16 计算影响 slot 分布
        return f"{MEMORY_KEY_PREFIX}{tag}{parts}:{key}"

    def _ttl_seconds(self, op_ttl: float | None) -> int:
        minutes = self._ttl_minutes if op_ttl is None else op_ttl
        if not minutes or minutes <= 0:
            return 0
        return max(1, int(float(minutes) * 60))

    def _get(self, op: GetOp) -> Item | None:
        redis_key = self._key(op.namespace, op.key)
        raw = self._client.get(redis_key)
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, dict) or "value" not in payload:
            raise TypeError(f"长期记忆格式无效 key={redis_key}")
        ttl_seconds = self._ttl_seconds(None)
        if op.refresh_ttl and ttl_seconds > 0:
            self._client.expire(redis_key, ttl_seconds)
        created = payload.get("created_at") or datetime.now(UTC).isoformat()
        updated = payload.get("updated_at") or created
        return Item(
            value=payload["value"],
            key=op.key,
            namespace=op.namespace,
            created_at=created,
            updated_at=updated,
        )

    def _put(self, op: PutOp) -> None:
        redis_key = self._key(op.namespace, op.key)
        if op.value is None:
            self._client.delete(redis_key)
            return
        now = datetime.now(UTC).isoformat()
        existing = self._client.get(redis_key)
        created = now
        if existing:
            try:
                old = json.loads(existing if isinstance(existing, str) else existing.decode("utf-8"))
                created = old.get("created_at") or now
            except Exception:
                created = now
        payload = json.dumps(
            {"value": op.value, "created_at": created, "updated_at": now},
            ensure_ascii=False,
        )
        ttl_seconds = self._ttl_seconds(op.ttl)
        if ttl_seconds > 0:
            self._client.set(redis_key, payload, ex=ttl_seconds)
        else:
            self._client.set(redis_key, payload)

    def batch(self, ops):
        results: list[Any] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._get(op))
            elif isinstance(op, PutOp):
                self._put(op)
                results.append(None)
            elif isinstance(op, (SearchOp, ListNamespacesOp)):
                results.append([])
            else:
                results.append(None)
        return results

    async def abatch(self, ops):
        return self.batch(ops)


def get_user_store() -> BaseStore:
    """长期记忆 Store：来自进程 AppContext。"""
    from app.runtime import get_app_context

    return get_app_context().user_store


def build_user_store(*, client: Any | None = None) -> BaseStore:
    """显式 client 或已连接 Redis 时用 RedisUserStore。"""
    from app.config import get_settings
    from app.session_cache import get_redis_client

    redis_client = client if client is not None else get_redis_client(required=False)
    if redis_client is None:
        return InMemoryStore()
    ttl = get_settings().user_memory_ttl_minutes
    logger.info("长期记忆使用 Redis Store ttl=%smin", ttl)
    return RedisUserStore(redis_client, ttl_minutes=ttl)


def reset_user_store(*, store: BaseStore | None = None) -> BaseStore:
    """测试用：换成空 Store，并丢掉已编译图（compile 时绑的是旧 Store）。"""
    from app.runtime import replace_app_context

    next_store = store if store is not None else InMemoryStore()
    replace_app_context(user_store=next_store, invalidate_graph=True)
    return next_store


def memory_namespace(tenant_id: str, user_id: str) -> tuple[str, ...]:
    """长期记忆命名空间，租户与用户双重隔离。"""
    return ("tenants", tenant_id, "users", user_id)


def is_persistent_user(user_id: str | None) -> bool:
    """匿名用户不写长期记忆，避免所有未登录请求共享一份档案。"""
    return bool((user_id or "").strip()) and (user_id or "").strip() not in EPHEMERAL_USERS


@dataclass
class UserMemory:
    """跨会话保留的结构化事实，不存原始对话。"""

    last_intent: str | None = None
    last_order_no: str | None = None
    last_course_query: str | None = None
    last_course_name: str | None = None
    handoff_pending: bool = False
    recent_order_nos: list[str] = field(default_factory=list)
    recent_course_names: list[str] = field(default_factory=list)
    updated_at: str | None = None

    def to_value(self) -> dict[str, Any]:
        """转为 Store 可序列化字典。"""
        return {
            "last_intent": self.last_intent,
            "last_order_no": self.last_order_no,
            "last_course_query": self.last_course_query,
            "last_course_name": self.last_course_name,
            "handoff_pending": self.handoff_pending,
            "recent_order_nos": self.recent_order_nos[:MAX_RECENT],
            "recent_course_names": self.recent_course_names[:MAX_RECENT],
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_value(cls, value: dict[str, Any] | None) -> UserMemory:
        """从 Store 记录恢复。"""
        data = value or {}
        return cls(
            last_intent=data.get("last_intent") or None,
            last_order_no=data.get("last_order_no") or None,
            last_course_query=data.get("last_course_query") or None,
            last_course_name=data.get("last_course_name") or None,
            handoff_pending=bool(data.get("handoff_pending")),
            recent_order_nos=list(data.get("recent_order_nos") or []),
            recent_course_names=list(data.get("recent_course_names") or []),
            updated_at=data.get("updated_at"),
        )


def _active_store() -> BaseStore:
    """图节点内优先用 runtime store，否则回退进程单例。"""
    try:
        from langgraph.config import get_store

        return get_store()
    except Exception:
        return get_user_store()


def load_user_memory(tenant_id: str, user_id: str) -> UserMemory:
    """读取用户档案；Store miss 时从 MySQL 回源并写回热缓存。"""
    if not is_persistent_user(user_id):
        return UserMemory()
    store = _active_store()
    ns = memory_namespace(tenant_id, user_id)
    item = store.get(ns, PROFILE_KEY)
    if item is not None:
        return UserMemory.from_value(item.value)

    from app.db.archive import load_user_profile

    archived = load_user_profile(tenant_id, user_id)
    if not archived:
        return UserMemory()
    memory = UserMemory.from_value(archived)
    try:
        kwargs: dict[str, Any] = {}
        from app.config import get_settings

        ttl_minutes = get_settings().user_memory_ttl_minutes
        if getattr(store, "supports_ttl", False) and ttl_minutes > 0:
            kwargs["ttl"] = float(ttl_minutes)
        store.put(ns, PROFILE_KEY, memory.to_value(), **kwargs)
    except Exception:
        logger.exception(
            "用户档案回填 Store 失败 tenant=%s user=%s", tenant_id, user_id
        )
    return memory


def overlay_user_memory(slots: SessionSlots, memory: UserMemory) -> SessionSlots:
    """会话槽位优先，空缺时用长期记忆回填。"""
    return SessionSlots(
        last_intent=slots.last_intent or memory.last_intent,
        last_order_no=slots.last_order_no or memory.last_order_no,
        last_course_query=slots.last_course_query or memory.last_course_query,
        last_course_name=slots.last_course_name or memory.last_course_name,
        handoff_pending=slots.handoff_pending or memory.handoff_pending,
    )


def _prepend_unique(items: list[str], value: str | None) -> list[str]:
    if not value:
        return items[:MAX_RECENT]
    rest = [item for item in items if item != value]
    return [value, *rest][:MAX_RECENT]


def save_user_memory(
    tenant_id: str,
    user_id: str,
    *,
    intent: str | None = None,
    order_no: str | None = None,
    course_query: str | None = None,
    course_name: str | None = None,
    handoff_pending: bool | None = None,
) -> UserMemory | None:
    """合并写入用户档案；匿名用户跳过。"""
    if not is_persistent_user(user_id):
        return None
    from app.config import get_settings

    settings = get_settings()
    if not settings.user_memory_enabled:
        return None

    current = load_user_memory(tenant_id, user_id)
    name = (course_name or "").strip()
    if name in MISSING_COURSE_NAMES:
        name = ""

    if intent:
        current.last_intent = intent
    if order_no:
        current.last_order_no = order_no
        current.recent_order_nos = _prepend_unique(current.recent_order_nos, order_no)
    if course_query:
        current.last_course_query = course_query
    if name:
        current.last_course_name = name
        current.recent_course_names = _prepend_unique(current.recent_course_names, name)
    if handoff_pending is not None:
        current.handoff_pending = handoff_pending
    current.updated_at = datetime.now(UTC).isoformat()

    kwargs: dict[str, Any] = {}
    store = _active_store()
    ttl_minutes = settings.user_memory_ttl_minutes
    if getattr(store, "supports_ttl", False) and ttl_minutes > 0:
        kwargs["ttl"] = float(ttl_minutes)

    value = current.to_value()
    store.put(memory_namespace(tenant_id, user_id), PROFILE_KEY, value, **kwargs)

    from app.db.archive import upsert_user_profile

    upsert_user_profile(tenant_id, user_id, value)
    return current


def persist_from_state(state: dict[str, Any]) -> UserMemory | None:
    """从图状态抽取本轮事实并写入长期记忆。"""
    from app.graph.state import IdentityView, MemoryView, RouteView

    ident = IdentityView.from_state(state)
    route = RouteView.from_state(state)
    memory = MemoryView.from_state(state)
    order_no = route.order_no or memory.last_order_no
    if order_no and is_persistent_user(ident.user_id):
        from app.services.order import can_access_order

        if not can_access_order(order_no, ident.tenant_id, ident.user_id):
            order_no = None
    return save_user_memory(
        ident.tenant_id,
        ident.user_id,
        intent=route.intent or None,
        order_no=order_no,
        course_query=route.course_query or memory.last_course_query,
        course_name=memory.last_course_name,
        handoff_pending=memory.handoff_pending if route.intent == "handoff" else None,
    )
