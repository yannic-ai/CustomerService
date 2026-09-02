"""Redis / checkpointer 清空后，从 MySQL 归档回填续聊。"""

import asyncio

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app.concurrency import achat
from app.db.archive import load_session_archive, load_user_profile
from app.graph import reset_graph
from app.memory import load_user_memory, reset_user_store


def _chat(message, **kwargs):
    return asyncio.run(achat(message, **kwargs))


def test_session_survives_empty_checkpointer_via_db() -> None:
    """热 checkpoint 清空后，同 session 仍能从 session_archives 续聊。"""
    session_id = "archive-order-session"
    user_id = "u10001"
    try:
        first = _chat(
            "查询订单#20251114001的退款进度",
            user_id=user_id,
            session_id=session_id,
            tenant_id="demo",
        )
        assert first.intent == "order"
        assert "20251114001" in first.answer

        archived = load_session_archive("demo", user_id, session_id)
        assert archived is not None
        assert archived.get("last_order_no") == "20251114001"
        assert any(
            m.get("role") == "human" and "20251114001" in (m.get("content") or "")
            for m in archived.get("messages") or []
        )

        reset_graph(checkpointer=InMemorySaver())
        second = _chat(
            "那退款呢",
            user_id=user_id,
            session_id=session_id,
            tenant_id="demo",
        )
        assert second.blocked is False
        assert second.intent == "order"
        assert "20251114001" in second.answer
    finally:
        reset_graph()


def test_user_profile_survives_empty_store_via_db() -> None:
    """Store 清空后，跨 session 仍能从 user_profiles 回填订单号。"""
    user_id = "u10001"
    try:
        first = _chat(
            "查询订单#20251114001的退款进度",
            user_id=user_id,
            session_id="archive-profile-s1",
            tenant_id="demo",
        )
        assert first.intent == "order"

        profile = load_user_profile("demo", user_id)
        assert profile is not None
        assert profile.get("last_order_no") == "20251114001"

        reset_user_store(store=InMemoryStore())
        reset_graph(checkpointer=InMemorySaver())

        loaded = load_user_memory("demo", user_id)
        assert loaded.last_order_no == "20251114001"

        second = _chat(
            "那退款呢",
            user_id=user_id,
            session_id="archive-profile-s2",
            tenant_id="demo",
        )
        assert second.blocked is False
        assert second.intent == "order"
        assert "20251114001" in second.answer
    finally:
        reset_user_store()
        reset_graph()


def test_session_archive_isolated_by_tenant_and_user() -> None:
    """demo 归档不能 hydrate 给 acme 或其他用户。"""
    session_id = "archive-shared-key"
    try:
        _chat(
            "查询订单#20251114001的退款进度",
            user_id="u10001",
            session_id=session_id,
            tenant_id="demo",
        )
        assert load_session_archive("demo", "u10001", session_id) is not None
        assert load_session_archive("acme", "u10001", session_id) is None
        assert load_session_archive("demo", "other-user", session_id) is None

        reset_graph(checkpointer=InMemorySaver())
        other = _chat(
            "那退款呢",
            user_id="other-user",
            session_id=session_id,
            tenant_id="demo",
        )
        assert other.intent == "ambiguous"
        assert "订单编号" in other.answer or "订单号" in other.answer
    finally:
        reset_graph()


def test_anonymous_session_archive_without_user_profile() -> None:
    """匿名只归档同 session，不写 user_profiles。"""
    session_id = "archive-anon-session"
    try:
        first = _chat(
            "查询订单#20251114001的退款进度",
            user_id="anonymous",
            session_id=session_id,
            tenant_id="demo",
        )
        assert first.intent == "order"
        assert load_session_archive("demo", "anonymous", session_id) is not None
        assert load_user_profile("demo", "anonymous") is None

        reset_graph(checkpointer=InMemorySaver())
        second = _chat(
            "那退款呢",
            user_id="anonymous",
            session_id=session_id,
            tenant_id="demo",
        )
        assert second.blocked is False
        assert second.intent == "order"
        assert "20251114001" in second.answer
    finally:
        reset_graph()
