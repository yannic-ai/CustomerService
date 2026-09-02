"""三层记忆一致性：checkpoint（热）+ user memory（Store）+ archive（冷库）。

覆盖恢复、并发与异常：热层丢失后冷层能回填；热层在时不重复合并；
写库失败不拖垮对话；同会话互斥、跨会话不串档。

查单相关场景使用种子用户 ``u10001``（订单 #20251114001 的下单人），
否则鉴权失败后不会写入长期记忆 / 槽位。
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app.concurrency import achat
from app.context import SessionSlots
from app.db.archive import (
    decode_messages,
    load_session_archive,
    load_user_profile,
)
from app.graph import reset_graph
from app.graph.hydrate import ahydrate_inputs_from_archive, hydrate_inputs_from_archive
from app.graph.state import make_thread_id
from app.memory import (
    UserMemory,
    load_user_memory,
    overlay_user_memory,
    reset_user_store,
    save_user_memory,
)
from app.session_cache import DictSessionLock, SessionBusyError


def _chat(message, **kwargs):
    return asyncio.run(achat(message, **kwargs))

OWNER = "u10001"
ORDER_QUERY = "查询订单#20251114001的退款进度"
FOLLOWUP = "那退款呢"


# ---------------------------------------------------------------------------
# 恢复：热层 miss → 冷层回填；热层 hit → 不合并归档
# ---------------------------------------------------------------------------


def test_dual_cold_miss_same_session_via_archive() -> None:
    """checkpoint + Store 都清空后，同 session 仍从 session_archives 续聊。"""
    session_id = "cons-dual-session"
    try:
        first = _chat(ORDER_QUERY, user_id=OWNER, session_id=session_id, tenant_id="demo")
        assert first.intent == "order"
        assert "20251114001" in first.answer
        assert load_session_archive("demo", OWNER, session_id) is not None
        assert load_user_profile("demo", OWNER) is not None

        reset_user_store(store=InMemoryStore())
        reset_graph(checkpointer=InMemorySaver())

        # Store 已空：先验证 profile 回源，再清掉逼同 session 只靠 archive
        assert load_user_memory("demo", OWNER).last_order_no == "20251114001"
        reset_user_store(store=InMemoryStore())

        second = _chat(FOLLOWUP, user_id=OWNER, session_id=session_id, tenant_id="demo")
        assert second.blocked is False
        assert second.intent == "order"
        assert "20251114001" in second.answer
    finally:
        reset_user_store()
        reset_graph()


def test_dual_cold_miss_new_session_via_user_profile() -> None:
    """checkpoint + Store 清空后，换 session 只能靠 user_profiles 回填。"""
    try:
        _chat(ORDER_QUERY, user_id=OWNER, session_id="cons-profile-s1", tenant_id="demo")
        assert load_user_profile("demo", OWNER)["last_order_no"] == "20251114001"

        reset_user_store(store=InMemoryStore())
        reset_graph(checkpointer=InMemorySaver())

        third = _chat(FOLLOWUP, user_id=OWNER, session_id="cons-profile-s2", tenant_id="demo")
        assert third.blocked is False
        assert third.intent == "order"
        assert "20251114001" in third.answer
    finally:
        reset_user_store()
        reset_graph()


def test_hydrate_skips_archive_when_checkpoint_hot(monkeypatch) -> None:
    """已有热 checkpoint 时不合并归档，避免 messages 重复。"""
    calls = {"load": 0}

    def boom_load(*_a, **_k):
        calls["load"] += 1
        raise AssertionError("热 checkpoint 存在时不应读 archive")

    monkeypatch.setattr("app.graph.hydrate.checkpoint_exists", lambda *_a, **_k: True)
    monkeypatch.setattr("app.graph.hydrate.load_session_archive", boom_load)

    inputs = {"query": "hi", "user_id": "u", "tenant_id": "demo", "session_id": "s"}
    hydrate_inputs_from_archive(
        inputs,
        tenant_id="demo",
        user_id="u",
        session_id="s",
        thread_id="demo:u:s",
    )
    assert "messages" not in inputs
    assert calls["load"] == 0


def test_hydrate_merges_archive_on_checkpoint_miss(monkeypatch) -> None:
    """checkpoint miss 时把归档槽位 / 消息并入 inputs。"""
    monkeypatch.setattr("app.graph.hydrate.checkpoint_exists", lambda *_a, **_k: False)
    monkeypatch.setattr(
        "app.graph.hydrate.load_session_archive",
        lambda *_a, **_k: {
            "messages": [
                {"role": "human", "content": "查单"},
                {"role": "ai", "content": "已查询"},
            ],
            "session_summary": "用户关心退款",
            "last_intent": "order",
            "last_order_no": "20251114001",
            "last_course_query": None,
            "last_course_name": None,
            "handoff_pending": False,
        },
    )
    inputs: dict = {"query": FOLLOWUP}
    hydrate_inputs_from_archive(
        inputs,
        tenant_id="demo",
        user_id=OWNER,
        session_id="s",
        thread_id="t",
    )
    assert inputs["last_order_no"] == "20251114001"
    assert inputs["session_summary"] == "用户关心退款"
    assert len(inputs["messages"]) == 2
    assert isinstance(inputs["messages"][0], HumanMessage)


def test_ahydrate_skips_archive_when_checkpoint_hot(monkeypatch) -> None:
    calls = {"load": 0}

    def boom_load(*_a, **_k):
        calls["load"] += 1
        raise AssertionError("热 checkpoint 存在时不应读 archive")

    async def hot(*_a, **_k):
        return True

    monkeypatch.setattr("app.graph.hydrate.acheckpoint_exists", hot)
    monkeypatch.setattr("app.graph.hydrate.load_session_archive", boom_load)
    inputs = {"query": "hi", "user_id": "u", "tenant_id": "demo", "session_id": "s"}
    asyncio.run(
        ahydrate_inputs_from_archive(
            inputs,
            tenant_id="demo",
            user_id="u",
            session_id="s",
            thread_id="demo:u:s",
        )
    )
    assert "messages" not in inputs
    assert calls["load"] == 0


def test_ahydrate_merges_archive_on_checkpoint_miss(monkeypatch) -> None:
    async def miss(*_a, **_k):
        return False

    monkeypatch.setattr("app.graph.hydrate.acheckpoint_exists", miss)
    monkeypatch.setattr(
        "app.graph.hydrate.load_session_archive",
        lambda *_a, **_k: {
            "messages": [
                {"role": "human", "content": "查单"},
                {"role": "ai", "content": "已查询"},
            ],
            "session_summary": "用户关心退款",
            "last_intent": "order",
            "last_order_no": "20251114001",
            "last_course_query": None,
            "last_course_name": None,
            "handoff_pending": False,
        },
    )
    inputs: dict = {"query": FOLLOWUP}
    asyncio.run(
        ahydrate_inputs_from_archive(
            inputs,
            tenant_id="demo",
            user_id=OWNER,
            session_id="s",
            thread_id="t",
        )
    )
    assert inputs["last_order_no"] == "20251114001"
    assert inputs["session_summary"] == "用户关心退款"
    assert len(inputs["messages"]) == 2
    assert isinstance(inputs["messages"][0], HumanMessage)


def test_acheckpoint_exists_error_treated_as_miss(monkeypatch) -> None:
    from app.graph import hydrate as hydrate_mod
    from app.runtime import in_memory_app_context, using_app_context

    class BoomChecker:
        async def aget_tuple(self, *_a, **_k):
            raise RuntimeError("redis down")

    monkeypatch.setattr(
        "app.graph.hydrate.load_session_archive",
        lambda *_a, **_k: {
            "messages": [],
            "last_order_no": "20251114001",
            "last_intent": "order",
            "session_summary": "",
            "handoff_pending": False,
        },
    )
    with using_app_context(in_memory_app_context(checkpointer=BoomChecker())):
        inputs: dict = {"query": "x"}
        assert asyncio.run(hydrate_mod.acheckpoint_exists("any")) is False
        asyncio.run(
            ahydrate_inputs_from_archive(
                inputs,
                tenant_id="demo",
                user_id="u",
                session_id="s",
                thread_id="t",
            )
        )
        assert inputs["last_order_no"] == "20251114001"


def test_session_slots_win_over_user_memory() -> None:
    """会话槽位优先于长期记忆，避免旧档案覆盖本会话已绑定单号。"""
    slots = SessionSlots(last_order_no="20251114002", last_intent="order")
    memory = UserMemory(last_order_no="20251114001", last_intent="course_consult")
    merged = overlay_user_memory(slots, memory)
    assert merged.last_order_no == "20251114002"
    assert merged.last_intent == "order"


def test_archive_roundtrip_matches_checkpoint_slots() -> None:
    """respond 后 archive 中的槽位与同 session 续聊所用事实一致。"""
    session_id = "cons-roundtrip-s"
    try:
        _chat(ORDER_QUERY, user_id=OWNER, session_id=session_id, tenant_id="demo")
        archived = load_session_archive("demo", OWNER, session_id)
        profile = load_user_profile("demo", OWNER)
        assert archived is not None and profile is not None
        assert archived["last_order_no"] == profile["last_order_no"] == "20251114001"

        reset_graph(checkpointer=InMemorySaver())
        follow = _chat(FOLLOWUP, user_id=OWNER, session_id=session_id, tenant_id="demo")
        assert "20251114001" in follow.answer

        after = load_session_archive("demo", OWNER, session_id)
        assert after is not None
        assert after["last_order_no"] == "20251114001"
        assert len(after.get("messages") or []) >= len(archived.get("messages") or [])
    finally:
        reset_graph()


# ---------------------------------------------------------------------------
# 并发：同会话互斥；跨会话不串档；档案可并发写
# ---------------------------------------------------------------------------


def test_same_session_busy_keeps_archive_intact() -> None:
    """同 session 被锁占用时抛 SessionBusy；已有归档不被写坏。"""
    session_id = "cons-lock-session"
    lock = DictSessionLock(wait_seconds=0.05)
    try:
        first = _chat(ORDER_QUERY, user_id=OWNER, session_id=session_id, tenant_id="demo")
        assert first.intent == "order"
        before = load_session_archive("demo", OWNER, session_id)
        assert before is not None
        assert before.get("last_order_no") == "20251114001"

        reset_graph(session_lock=lock)
        held = threading.Event()
        thread_id = make_thread_id("demo", session_id, OWNER)

        def _hold() -> None:
            with lock.hold(thread_id):
                held.set()
                time.sleep(0.4)

        worker = threading.Thread(target=_hold)
        worker.start()
        try:
            assert held.wait(timeout=1)
            raised = False
            try:
                _chat(FOLLOWUP, user_id=OWNER, session_id=session_id, tenant_id="demo")
            except SessionBusyError:
                raised = True
            assert raised is True
        finally:
            worker.join()

        after = load_session_archive("demo", OWNER, session_id)
        assert after is not None
        assert after.get("last_order_no") == before.get("last_order_no")
        assert isinstance(after.get("messages"), list)
        assert after["messages"]
    finally:
        reset_graph()


def test_concurrent_different_sessions_same_user_no_cross_talk() -> None:
    """同一登录用户两个 session 并发：都能完成，互不覆盖对方归档。"""
    results: dict[str, str] = {}

    def _run(session_id: str) -> None:
        resp = _chat(ORDER_QUERY, user_id=OWNER, session_id=session_id, tenant_id="demo")
        results[session_id] = resp.intent

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = [
                pool.submit(_run, "cons-multi-a"),
                pool.submit(_run, "cons-multi-b"),
            ]
            for fut in as_completed(futs):
                fut.result()

        assert results["cons-multi-a"] == "order"
        assert results["cons-multi-b"] == "order"
        a = load_session_archive("demo", OWNER, "cons-multi-a")
        b = load_session_archive("demo", OWNER, "cons-multi-b")
        assert a is not None and b is not None
        assert a.get("last_order_no") == b.get("last_order_no") == "20251114001"
        profile = load_user_profile("demo", OWNER)
        assert profile is not None
        assert profile.get("last_order_no") == "20251114001"
    finally:
        reset_graph()


def test_concurrent_save_user_memory_keeps_last_order() -> None:
    """并发写 Store/档案：不应抛错，last_order_no 仍为写入过的单号之一。"""
    user_id = "cons-mem-race"
    orders = ["20251114001", "20251114002"]

    def _save(order_no: str) -> None:
        save_user_memory("demo", user_id, intent="order", order_no=order_no)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(_save, orders))

    loaded = load_user_memory("demo", user_id)
    assert loaded.last_order_no in orders
    profile = load_user_profile("demo", user_id)
    assert profile is not None
    assert profile.get("last_order_no") in orders


# ---------------------------------------------------------------------------
# 异常：写/读失败降级，不拖垮主路径
# ---------------------------------------------------------------------------


def test_archive_write_failure_does_not_break__chat(monkeypatch) -> None:
    """session_archives 写入失败只打日志，对话仍返回。"""

    class BoomSession:
        def __enter__(self):
            raise RuntimeError("db down")

        def __exit__(self, *args):
            return False

    # 只打归档模块的 session_scope，订单/种子库不受影响
    monkeypatch.setattr("app.db.archive.session_scope", lambda: BoomSession())
    resp = _chat(
        ORDER_QUERY,
        user_id=OWNER,
        session_id="cons-arch-write-fail-s",
        tenant_id="demo",
    )
    assert resp.blocked is False
    assert resp.intent == "order"
    assert "20251114001" in resp.answer


def test_archive_read_failure_falls_back_to_fresh_session(monkeypatch) -> None:
    """checkpoint miss 且读归档失败时，按新会话继续，不抛给调用方。"""
    monkeypatch.setattr("app.graph.hydrate.checkpoint_exists", lambda *_a, **_k: False)

    def boom_load(*_a, **_k):
        raise RuntimeError("archive unavailable")

    monkeypatch.setattr("app.graph.hydrate.load_session_archive", boom_load)

    inputs = {"query": FOLLOWUP}
    hydrate_inputs_from_archive(
        inputs,
        tenant_id="demo",
        user_id="u",
        session_id="s",
        thread_id="t",
    )
    assert inputs == {"query": FOLLOWUP}
    assert "messages" not in inputs
    assert "last_order_no" not in inputs


def test_load_session_archive_swallows_db_errors(monkeypatch) -> None:
    """archive.load_session_archive 自己吞异常并返回 None。"""

    class BoomSession:
        def __enter__(self):
            raise RuntimeError("db")

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("app.db.archive.session_scope", lambda: BoomSession())
    assert load_session_archive("demo", "u", "s") is None


def test_checkpoint_exists_error_treated_as_miss(monkeypatch) -> None:
    """读 checkpoint 抛错视为 miss，仍尝试从 archive 回填。"""
    from app.graph import hydrate as hydrate_mod
    from app.runtime import in_memory_app_context, using_app_context

    class BoomChecker:
        def get_tuple(self, *_a, **_k):
            raise RuntimeError("redis down")

    monkeypatch.setattr(
        "app.graph.hydrate.load_session_archive",
        lambda *_a, **_k: {
            "messages": [],
            "last_order_no": "20251114001",
            "last_intent": "order",
            "session_summary": "",
            "handoff_pending": False,
        },
    )
    with using_app_context(in_memory_app_context(checkpointer=BoomChecker())):
        assert hydrate_mod.checkpoint_exists("any") is False
        inputs: dict = {"query": "x"}
        hydrate_inputs_from_archive(
            inputs,
            tenant_id="demo",
            user_id="u",
            session_id="s",
            thread_id="t",
        )
        assert inputs["last_order_no"] == "20251114001"


def test_user_profile_read_failure_yields_empty_memory(monkeypatch) -> None:
    """Store miss + 读 user_profiles 失败 → 空档案，跨 session 追问需澄清。"""
    user_id = "cons-profile-read-fail"
    reset_user_store(store=InMemoryStore())
    monkeypatch.setattr("app.db.archive.load_user_profile", lambda *_a, **_k: None)
    try:
        mem = load_user_memory("demo", user_id)
        assert mem.last_order_no is None

        reset_graph(checkpointer=InMemorySaver())
        resp = _chat(FOLLOWUP, user_id=user_id, session_id="cons-fresh-s", tenant_id="demo")
        assert resp.intent == "ambiguous"
        assert "订单" in (resp.answer or "")
    finally:
        reset_user_store()
        reset_graph()


def test_user_profile_write_failure_keeps_hot_store(monkeypatch) -> None:
    """写 user_profiles 失败时，热 Store 仍保留事实。"""

    class BoomSession:
        def __enter__(self):
            raise RuntimeError("profile db down")

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("app.db.archive.session_scope", lambda: BoomSession())
    user_id = "cons-profile-write-fail"
    saved = save_user_memory("demo", user_id, intent="order", order_no="20251114001")
    assert saved is not None
    assert load_user_memory("demo", user_id).last_order_no == "20251114001"
    # 冷层未写入
    assert load_user_profile("demo", user_id) is None


def test_decode_messages_skips_corrupt_items() -> None:
    """归档消息损坏时跳过坏项，不拖垮 hydrate。"""
    messages = decode_messages(
        [
            {"role": "human", "content": "正常问句"},
            "not-a-dict",
            {"role": "tool", "content": "忽略"},
            {"role": "ai", "content": "正常回答"},
            {"role": "human"},  # content 缺失 → 空串仍可解码
        ]
    )
    assert len(messages) == 3
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "正常问句"


def test_corrupt_archive_payload_does_not_break_hydrate(monkeypatch) -> None:
    """归档 state 缺字段时仍可 hydrate。"""
    monkeypatch.setattr("app.graph.hydrate.checkpoint_exists", lambda *_a, **_k: False)
    monkeypatch.setattr(
        "app.graph.hydrate.load_session_archive",
        lambda *_a, **_k: {"messages": [{"role": "human", "content": "旧问"}]},
    )
    inputs: dict = {"query": ORDER_QUERY, "user_id": OWNER, "tenant_id": "demo"}
    hydrate_inputs_from_archive(
        inputs,
        tenant_id="demo",
        user_id=OWNER,
        session_id="cons-corrupt",
        thread_id=make_thread_id("demo", "cons-corrupt", OWNER),
    )
    assert len(inputs["messages"]) == 1
    assert inputs.get("last_order_no") is None
    assert inputs.get("handoff_pending") is False
