import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from app.llm_gateway import (
    CallPolicy,
    LlmCircuitOpenError,
    TenantQuotaExceededError,
    begin_turn_usage,
    snapshot_llm_breaker,
    take_turn_usage,
    wrap_chat_model,
)
from app.llm_gateway.stores import MemoryGatewayStore, set_gateway_store
from app.tenancy import set_current_tenant


class DummyModel:
    def __init__(self, fail_times: int = 0) -> None:
        self.calls = 0
        self.fail_times = fail_times

    def invoke(self, payload, config=None, **kwargs):
        del payload, config, kwargs
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TimeoutError("timed out")
        return AIMessage(
            content="hello",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )

    def stream(self, payload, config=None, **kwargs):
        del payload, config, kwargs
        self.calls += 1
        yield AIMessageChunk(content="he")
        yield AIMessageChunk(content="llo")
        yield AIMessageChunk(
            content="",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )

    async def astream(self, payload, config=None, **kwargs):
        for chunk in self.stream(payload, config=config, **kwargs):
            yield chunk

    async def ainvoke(self, payload, config=None, **kwargs):
        return self.invoke(payload, config=config, **kwargs)

    def with_structured_output(self, schema, **kwargs):
        del schema, kwargs
        return self

    def bind_tools(self, *args, **kwargs):
        del args, kwargs
        return self


@pytest.fixture
def memory_store():
    store = MemoryGatewayStore()
    set_gateway_store(store)
    yield store
    set_gateway_store(None)


@pytest.fixture
def gateway_settings(monkeypatch):
    settings = SimpleNamespace(
        llm_daily_token_limit=100,
        llm_cache_enabled=True,
        llm_cache_ttl_seconds=60,
        llm_cache_max_chars=4000,
        llm_max_retries=2,
        llm_timeout_seconds=60,
        llm_circuit_failure_threshold=5,
        llm_circuit_cooldown_seconds=30.0,
        llm_prompt_price_per_mtok_yuan=1.0,
        llm_completion_price_per_mtok_yuan=2.0,
        redis_url="",
    )
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    from app.llm_gateway import reset_llm_breaker

    reset_llm_breaker()
    yield settings
    reset_llm_breaker()


def _model(inner: DummyModel, *, cache: bool = True) -> object:
    return wrap_chat_model(
        inner,
        CallPolicy(
            usage_tag="intent",
            cache_enabled=cache,
            enforce_quota=True,
            temperature=0,
            model="dummy",
        ),
    )


def test_accounts_tokens_and_cache_hit(memory_store, gateway_settings) -> None:
    set_current_tenant("acme")
    inner = DummyModel()
    llm = _model(inner)
    first = llm.invoke([{"role": "user", "content": "hi"}])
    second = llm.invoke([{"role": "user", "content": "hi"}])
    assert first.content == second.content == "hello"
    assert inner.calls == 1
    snap = memory_store.get_quota("acme")
    assert snap.total_tokens == 15
    assert snap.requests == 1
    assert snap.cache_hits == 1
    assert memory_store.degraded is True


def test_quota_blocks_next_call(memory_store, gateway_settings) -> None:
    gateway_settings.llm_daily_token_limit = 15
    set_current_tenant("acme")
    inner = DummyModel()
    llm = _model(inner, cache=False)
    llm.invoke("one")
    with pytest.raises(TenantQuotaExceededError):
        llm.invoke("two")
    assert inner.calls == 1
    assert memory_store.get_quota("acme").total_tokens == 15


def test_react_policy_skips_cache(memory_store, gateway_settings) -> None:
    set_current_tenant("demo")
    inner = DummyModel()
    llm = wrap_chat_model(
        inner,
        CallPolicy(usage_tag="react", cache_enabled=False, model="dummy"),
    )
    llm.invoke("q")
    llm.invoke("q")
    assert inner.calls == 2
    assert memory_store.get_quota("demo").total_tokens == 30
    assert memory_store.get_quota("demo").cache_hits == 0


def test_retries_transient_timeout(memory_store, gateway_settings) -> None:
    set_current_tenant("demo")
    inner = DummyModel(fail_times=1)
    llm = _model(inner, cache=False)
    result = llm.invoke("retry me")
    assert result.content == "hello"
    assert inner.calls == 2


def test_bind_tools_disables_cache(memory_store, gateway_settings) -> None:
    set_current_tenant("demo")
    inner = DummyModel()
    llm = _model(inner, cache=True).bind_tools([])
    llm.invoke("a")
    llm.invoke("a")
    assert inner.calls == 2


def test_stream_folds_last_nonzero_usage(memory_store, gateway_settings) -> None:
    set_current_tenant("demo")
    begin_turn_usage()
    inner = DummyModel()
    llm = _model(inner, cache=False)
    chunks = list(llm.stream("hi"))
    assert "".join(str(c.content) for c in chunks) == "hello"
    snap = memory_store.get_quota("demo")
    assert snap.total_tokens == 15
    assert snap.requests == 1
    turn = take_turn_usage()
    assert turn.total_tokens == 15
    assert len(turn.steps) == 1


def test_stream_accumulates_quota_across_steps(memory_store, gateway_settings) -> None:
    set_current_tenant("demo")
    begin_turn_usage()
    inner = DummyModel()
    llm = _model(inner, cache=False)
    list(llm.stream("one"))
    list(llm.stream("two"))
    assert memory_store.get_quota("demo").total_tokens == 30
    turn = take_turn_usage()
    assert turn.total_tokens == 30
    assert len(turn.steps) == 2


def test_stream_quota_blocks_second_step(memory_store, gateway_settings) -> None:
    gateway_settings.llm_daily_token_limit = 15
    set_current_tenant("demo")
    inner = DummyModel()
    llm = _model(inner, cache=False)
    list(llm.stream("one"))
    with pytest.raises(TenantQuotaExceededError):
        list(llm.stream("two"))
    assert memory_store.get_quota("demo").total_tokens == 15


def test_metrics_split_prompt_completion_and_cost(memory_store, gateway_settings) -> None:
    from app.observability import reset_metrics
    from app.observability.metrics import render_prometheus

    reset_metrics()
    set_current_tenant("acme")
    llm = _model(DummyModel(), cache=False)
    llm.invoke("cost-me")
    body = render_prometheus()
    assert 'cs_llm_tokens_total{direction="prompt",tag="intent",tenant="acme"} 10' in body
    assert 'cs_llm_tokens_total{direction="completion",tag="intent",tenant="acme"} 5' in body
    # 10 * 1.0 / 1e6 + 5 * 2.0 / 1e6 = 0.000020
    assert 'cs_llm_cost_yuan_total{tag="intent",tenant="acme"} 0.000020' in body


def test_cache_hit_does_not_add_tokens_or_cost(memory_store, gateway_settings) -> None:
    from app.observability import reset_metrics
    from app.observability.metrics import render_prometheus

    reset_metrics()
    set_current_tenant("acme")
    llm = _model(DummyModel(), cache=True)
    llm.invoke("same")
    llm.invoke("same")
    body = render_prometheus()
    assert 'direction="prompt"' in body
    assert 'cs_llm_tokens_total{direction="prompt",tag="intent",tenant="acme"} 10' in body
    assert 'cs_llm_cost_yuan_total{tag="intent",tenant="acme"} 0.000020' in body
    assert 'cache_hit="true"' in body


def test_ainvoke_accounts_tokens_and_cache_hit(memory_store, gateway_settings) -> None:
    set_current_tenant("acme")
    inner = DummyModel()
    llm = _model(inner)
    first = asyncio.run(llm.ainvoke([{"role": "user", "content": "hi"}]))
    second = asyncio.run(llm.ainvoke([{"role": "user", "content": "hi"}]))
    assert first.content == second.content == "hello"
    assert inner.calls == 1
    snap = memory_store.get_quota("acme")
    assert snap.total_tokens == 15
    assert snap.requests == 1
    assert snap.cache_hits == 1
    assert memory_store.degraded is True


def test_ainvoke_retries_transient_timeout(memory_store, gateway_settings) -> None:
    set_current_tenant("demo")
    inner = DummyModel(fail_times=1)
    llm = _model(inner, cache=False)
    result = asyncio.run(llm.ainvoke("retry me"))
    assert result.content == "hello"
    assert inner.calls == 2


def test_cache_write_uses_llk1_msgpack(memory_store, gateway_settings) -> None:
    from app.llm_gateway.cache_codec import ENVELOPE_MAGIC, decode_llm_cache

    set_current_tenant("acme")
    llm = _model(DummyModel())
    result = llm.invoke("hi")
    blobs = [value for value, _expires in memory_store._cache.values()]
    assert blobs
    assert all(blob.startswith(ENVELOPE_MAGIC) for blob in blobs)
    assert all(not blob.startswith(b"\x80") for blob in blobs)
    loaded = decode_llm_cache(blobs[0])
    assert isinstance(loaded, AIMessage)
    assert loaded.content == result.content == "hello"


def test_legacy_pickle_cache_is_ignored(memory_store, gateway_settings) -> None:
    import pickle

    set_current_tenant("acme")
    inner = DummyModel()
    llm = _model(inner)
    first = llm.invoke("same")
    for key, (_value, expires) in list(memory_store._cache.items()):
        memory_store._cache[key] = (
            pickle.dumps(first, protocol=pickle.HIGHEST_PROTOCOL),
            expires,
        )
    second = llm.invoke("same")
    assert inner.calls == 2
    assert second.content == "hello"
    from app.llm_gateway.cache_codec import ENVELOPE_MAGIC

    assert all(
        value.startswith(ENVELOPE_MAGIC) for value, _expires in memory_store._cache.values()
    )


def test_corrupt_cache_blob_is_miss(memory_store, gateway_settings) -> None:
    set_current_tenant("acme")
    inner = DummyModel()
    llm = _model(inner)
    llm.invoke("same")
    for key, (_value, expires) in list(memory_store._cache.items()):
        memory_store._cache[key] = (b"LLK1not-valid-msgpack", expires)
    llm.invoke("same")
    assert inner.calls == 2


def test_structured_output_cache_roundtrip(memory_store, gateway_settings) -> None:
    from app.schemas import RouteDecision

    class StructuredDummy:
        def __init__(self) -> None:
            self.calls = 0

        def with_structured_output(self, schema, **kwargs):
            del schema, kwargs
            return self

        def invoke(self, payload, config=None, **kwargs):
            del payload, config, kwargs
            self.calls += 1
            return RouteDecision(intent="chitchat", target="chitchat", reason="llm")

        async def ainvoke(self, payload, config=None, **kwargs):
            return self.invoke(payload, config=config, **kwargs)

    set_current_tenant("acme")
    inner = StructuredDummy()
    llm = wrap_chat_model(
        inner,
        CallPolicy(usage_tag="intent", cache_enabled=True, model="dummy"),
    ).with_structured_output(RouteDecision)
    first = llm.invoke("hi")
    second = llm.invoke("hi")
    assert isinstance(first, RouteDecision)
    assert isinstance(second, RouteDecision)
    assert first.intent == second.intent == "chitchat"
    assert inner.calls == 1
    assert memory_store.get_quota("acme").cache_hits == 1


def test_encode_decode_aimessage_and_pydantic() -> None:
    from app.llm_gateway.cache_codec import decode_llm_cache, encode_llm_cache
    from app.schemas import RouteDecision

    message = AIMessage(
        content="hello",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    blob = encode_llm_cache(message)
    assert blob is not None
    loaded = decode_llm_cache(blob)
    assert isinstance(loaded, AIMessage)
    assert loaded.content == "hello"
    assert loaded.usage_metadata["input_tokens"] == 10

    decision = RouteDecision(intent="order", target="tool", order_no="20251114001")
    blob = encode_llm_cache(decision)
    assert blob is not None
    assert decode_llm_cache(blob) is None
    loaded = decode_llm_cache(blob, schema_type=RouteDecision)
    assert isinstance(loaded, RouteDecision)
    assert loaded.order_no == "20251114001"
    assert decode_llm_cache(b"\x80\x04pickle", schema_type=RouteDecision) is None


def test_circuit_opens_after_consecutive_failures_and_fail_fast(
    memory_store, gateway_settings
) -> None:
    gateway_settings.llm_circuit_failure_threshold = 2
    gateway_settings.llm_max_retries = 0
    set_current_tenant("demo")
    inner = DummyModel(fail_times=99)
    llm = _model(inner, cache=False)
    with pytest.raises(TimeoutError):
        llm.invoke("one")
    with pytest.raises(TimeoutError):
        llm.invoke("two")
    snap = snapshot_llm_breaker()
    assert snap.state == "open"
    assert snap.failures == 2
    calls_after_open = inner.calls
    with pytest.raises(LlmCircuitOpenError):
        llm.invoke("three")
    assert inner.calls == calls_after_open


def test_circuit_success_resets_consecutive_failures(memory_store, gateway_settings) -> None:
    gateway_settings.llm_circuit_failure_threshold = 3
    gateway_settings.llm_max_retries = 0
    set_current_tenant("demo")
    inner = DummyModel(fail_times=1)
    llm = _model(inner, cache=False)
    with pytest.raises(TimeoutError):
        llm.invoke("fail")
    assert snapshot_llm_breaker().failures == 1
    inner.fail_times = 0
    inner.calls = 0
    llm.invoke("ok")
    snap = snapshot_llm_breaker()
    assert snap.state == "closed"
    assert snap.failures == 0


def test_circuit_half_open_allows_one_probe(memory_store, gateway_settings) -> None:
    gateway_settings.llm_circuit_failure_threshold = 1
    gateway_settings.llm_circuit_cooldown_seconds = 0.0
    gateway_settings.llm_max_retries = 0
    set_current_tenant("demo")
    inner = DummyModel(fail_times=1)
    llm = _model(inner, cache=False)
    with pytest.raises(TimeoutError):
        llm.invoke("open")
    assert snapshot_llm_breaker().state == "half_open"
    inner.fail_times = 0
    result = llm.invoke("probe")
    assert result.content == "hello"
    assert snapshot_llm_breaker().state == "closed"


def test_circuit_open_still_serves_cache(memory_store, gateway_settings) -> None:
    gateway_settings.llm_circuit_failure_threshold = 1
    gateway_settings.llm_max_retries = 0
    set_current_tenant("acme")
    inner = DummyModel()
    llm = _model(inner, cache=True)
    first = llm.invoke("same")
    inner.fail_times = 99
    inner.calls = 0
    with pytest.raises(TimeoutError):
        llm.invoke("other")
    assert snapshot_llm_breaker().state == "open"
    cached = llm.invoke("same")
    assert cached.content == first.content == "hello"
    assert inner.calls == 1


def test_quota_does_not_trip_circuit(memory_store, gateway_settings) -> None:
    gateway_settings.llm_daily_token_limit = 15
    gateway_settings.llm_circuit_failure_threshold = 1
    set_current_tenant("acme")
    inner = DummyModel()
    llm = _model(inner, cache=False)
    llm.invoke("one")
    with pytest.raises(TenantQuotaExceededError):
        llm.invoke("two")
    assert snapshot_llm_breaker().state == "closed"
    assert snapshot_llm_breaker().failures == 0

