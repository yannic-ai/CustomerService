import asyncio

from app.concurrency import achat
from app.memory import (
    load_user_memory,
    memory_namespace,
    save_user_memory,
    is_persistent_user,
)


def _chat(message, **kwargs):
    return asyncio.run(achat(message, **kwargs))


def test_anonymous_is_not_persistent() -> None:
    assert is_persistent_user("anonymous") is False
    assert is_persistent_user("alice") is True


def test_save_and_load_user_memory() -> None:
    saved = save_user_memory(
        "demo",
        "mem-alice",
        intent="order",
        order_no="20251114001",
        course_name="Python入门课",
        course_query="Python入门课包含哪些内容？",
    )
    assert saved is not None
    loaded = load_user_memory("demo", "mem-alice")
    assert loaded.last_order_no == "20251114001"
    assert loaded.last_course_name == "Python入门课"
    assert loaded.last_intent == "order"
    assert "20251114001" in loaded.recent_order_nos


def test_anonymous_save_is_noop() -> None:
    assert save_user_memory("demo", "anonymous", order_no="20251114001") is None
    loaded = load_user_memory("demo", "anonymous")
    assert loaded.last_order_no is None


def test_memory_isolated_by_tenant_and_user() -> None:
    save_user_memory("demo", "mem-bob", order_no="20251114001")
    assert load_user_memory("acme", "mem-bob").last_order_no is None
    assert load_user_memory("demo", "mem-carol").last_order_no is None
    assert memory_namespace("demo", "mem-bob") != memory_namespace("acme", "mem-bob")


def test_long_term_memory_survives_new_session() -> None:
    user_id = "u10001"
    first = _chat(
        "查询订单#20251114001的退款进度",
        user_id=user_id,
        session_id="sess-1",
        tenant_id="demo",
    )
    assert first.blocked is False
    assert first.intent == "order"

    # 新 session，无对话历史，只靠 Store 里的订单号
    second = _chat(
        "那退款呢",
        user_id=user_id,
        session_id="sess-2",
        tenant_id="demo",
    )
    assert second.blocked is False
    assert second.intent == "order"
    assert "20251114001" in second.answer
    assert "退款" in second.answer


def test_long_term_memory_not_shared_across_users() -> None:
    _chat(
        "查询订单#20251114001的退款进度",
        user_id="u10001",
        session_id="s-a",
        tenant_id="demo",
    )
    other = _chat(
        "那退款呢",
        user_id="mem-user-b",
        session_id="s-b",
        tenant_id="demo",
    )
    assert other.intent == "ambiguous"
    assert "订单编号" in other.answer or "订单号" in other.answer


def test_unowned_order_is_not_backfilled_from_memory() -> None:
    save_user_memory("demo", "alice-idor", order_no="20251114001")
    follow = _chat(
        "那退款呢",
        user_id="alice-idor",
        session_id="idor-sess",
        tenant_id="demo",
    )
    assert follow.intent == "ambiguous"
    assert "订单编号" in follow.answer or "订单号" in follow.answer
    assert "退款审核" not in follow.answer


class _FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self.data.get(key)

    def set(self, key: str, value: str | bytes, ex: int | None = None, nx: bool = False, px: int | None = None) -> bool:
        del ex, nx, px
        if isinstance(value, str):
            value = value.encode("utf-8")
        self.data[key] = value
        return True

    def delete(self, key: str) -> None:
        self.data.pop(key, None)

    def expire(self, key: str, seconds: int) -> bool:
        del seconds
        return key in self.data


def test_redis_store_survives_new_instance() -> None:
    from app.memory import RedisUserStore

    client = _FakeRedis()
    ns = ("tenants", "demo", "users", "redis-alice")
    first = RedisUserStore(client, ttl_minutes=30)
    first.put(ns, "profile", {"last_order_no": "20251114001"})
    second = RedisUserStore(client, ttl_minutes=30)
    item = second.get(ns, "profile")
    assert item is not None
    assert item.value["last_order_no"] == "20251114001"
