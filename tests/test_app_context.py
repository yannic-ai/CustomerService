"""AppContext：一次装配图 / checkpointer / Store / Gateway，测试不必分别 reset。"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app.graph import get_checkpointer, get_graph, reset_graph
from app.llm_gateway.stores import MemoryGatewayStore, get_gateway_store, set_gateway_store
from app.memory import get_user_store, reset_user_store
from app.runtime import (
    get_app_context,
    in_memory_app_context,
    using_app_context,
)
from app.session_cache import DictSessionLock, get_session_lock


def test_in_memory_context_wires_all_handles() -> None:
    checkpointer = InMemorySaver()
    user_store = InMemoryStore()
    gateway = MemoryGatewayStore()
    lock = DictSessionLock()
    ctx = in_memory_app_context(
        checkpointer=checkpointer,
        user_store=user_store,
        gateway_store=gateway,
        session_lock=lock,
    )
    with using_app_context(ctx):
        assert get_app_context() is ctx
        assert get_checkpointer() is checkpointer
        assert get_user_store() is user_store
        assert get_gateway_store() is gateway
        assert get_session_lock() is lock
        assert get_graph() is ctx.graph


def test_install_swaps_runtime_without_module_monkeypatch() -> None:
    """替换整套依赖只 install 一次，不必 reset_graph + reset_user_store + set_gateway_store。"""
    first = in_memory_app_context()
    second_store = InMemoryStore()
    second = in_memory_app_context(user_store=second_store)
    with using_app_context(first):
        assert get_user_store() is first.user_store
        with using_app_context(second):
            assert get_user_store() is second_store
            assert get_graph() is second.graph
        assert get_user_store() is first.user_store


def test_reset_wrappers_still_mutate_process_context() -> None:
    ctx = in_memory_app_context()
    with using_app_context(ctx):
        saver = InMemorySaver()
        reset_graph(checkpointer=saver)
        assert get_checkpointer() is saver
        assert ctx._graph is None

        store = InMemoryStore()
        reset_user_store(store=store)
        assert get_user_store() is store
        assert ctx._graph is None

        gateway = MemoryGatewayStore()
        set_gateway_store(gateway)
        assert get_gateway_store() is gateway


def test_chat_writes_to_installed_user_store() -> None:
    """对话走已安装上下文的 Store，不必再 reset_user_store 后 reset_graph。"""
    import asyncio

    from app.concurrency import achat
    from app.memory import PROFILE_KEY, memory_namespace

    store = InMemoryStore()
    with using_app_context(in_memory_app_context(user_store=store)):
        result = asyncio.run(
            achat(
                "查询订单#20251114001的退款进度",
                user_id="u10001",
                session_id="appctx-order",
                tenant_id="demo",
            )
        )
        assert result.intent == "order"
        item = store.get(memory_namespace("demo", "u10001"), PROFILE_KEY)
        assert item is not None
        assert item.value.get("last_order_no") == "20251114001"
