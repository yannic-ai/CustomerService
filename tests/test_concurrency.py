import asyncio
from types import SimpleNamespace

import pytest

from app.concurrency import ChatBusyError, achat, achat_stream, shutdown_concurrency
from app.schemas import ChatResponse


def teardown_function() -> None:
    shutdown_concurrency()


def test_achat_busy_returns_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.concurrency.get_settings",
        lambda: SimpleNamespace(chat_concurrency=1, chat_tenant_concurrency=1, graph_workers=2),
    )

    async def slow_astream(*_args, **_kwargs):
        await asyncio.sleep(0.4)
        yield ("done", ChatResponse(answer="ok", intent="chitchat").model_dump())

    monkeypatch.setattr("app.concurrency.chat_astream", slow_astream)

    async def _run() -> None:
        first = asyncio.create_task(achat("one"))
        await asyncio.sleep(0.05)
        raised = False
        try:
            await achat("two")
        except ChatBusyError:
            raised = True
        await first
        assert raised

    asyncio.run(_run())


def test_achat_stream_busy_returns_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.concurrency.get_settings",
        lambda: SimpleNamespace(chat_concurrency=1, chat_tenant_concurrency=1, graph_workers=0),
    )

    async def slow_stream(*_args, **_kwargs):
        yield ("meta", {"session_id": "s1"})
        await asyncio.sleep(0.4)
        yield ("done", {"answer": "ok", "intent": "chitchat", "blocked": False})

    monkeypatch.setattr("app.concurrency.chat_astream", slow_stream)

    async def _run() -> None:
        agen = achat_stream("one")
        first_event = await agen.__anext__()
        assert first_event[0] == "meta"
        raised = False
        try:
            await anext(achat_stream("two"))
        except ChatBusyError:
            raised = True
        async for _ in agen:
            pass
        assert raised

    asyncio.run(_run())


def test_noisy_tenant_does_not_fill_cluster(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.concurrency.get_settings",
        lambda: SimpleNamespace(
            chat_concurrency=2,
            chat_tenant_concurrency=1,
            chat_slot_ttl_seconds=60,
            graph_workers=0,
            default_tenant="demo",
        ),
    )

    async def slow_astream(*_args, **_kwargs):
        await asyncio.sleep(0.4)
        yield ("done", ChatResponse(answer="ok", intent="chitchat").model_dump())

    monkeypatch.setattr("app.concurrency.chat_astream", slow_astream)

    async def _run() -> None:
        first = asyncio.create_task(achat("one", tenant_id="demo"))
        await asyncio.sleep(0.05)
        with pytest.raises(ChatBusyError) as busy:
            await achat("two", tenant_id="demo")
        assert busy.value.reason == "tenant"
        other = await achat("three", tenant_id="acme")
        assert other.answer == "ok"
        await first

    asyncio.run(_run())
