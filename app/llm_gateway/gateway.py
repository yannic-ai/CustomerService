"""统一 LLM Gateway：超时、重试、熔断、记账、租户日限额、短响应缓存。"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import replace
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from app.llm_gateway.cache_codec import decode_llm_cache, encode_llm_cache
from app.llm_gateway.circuit import guard_llm_circuit, record_llm_failure, record_llm_success
from app.llm_gateway.exceptions import LlmCircuitOpenError, TenantQuotaExceededError
from app.llm_gateway.stores import get_gateway_store
from app.llm_gateway.types import CallPolicy, TokenUsage, TurnUsage, TurnUsageStep
from app.observability import inc_llm, inc_llm_error, log_event
from app.tenancy import get_current_tenant

logger = logging.getLogger("cs.llm_gateway")

_TRANSIENT_TYPES = (TimeoutError, ConnectionError, OSError)
_TURN_USAGE: contextvars.ContextVar[list[TurnUsageStep] | None] = contextvars.ContextVar(
    "cs_llm_turn_usage",
    default=None,
)


def begin_turn_usage() -> None:
    """清空并开启本轮对话的 usage 累加。"""
    _TURN_USAGE.set([])


def take_turn_usage() -> TurnUsage:
    """取出本轮累计 usage，并清空 contextvar。"""
    steps = list(_TURN_USAGE.get() or [])
    _TURN_USAGE.set(None)
    prompt = sum(step.prompt_tokens for step in steps)
    completion = sum(step.completion_tokens for step in steps)
    total = sum(step.total_tokens for step in steps)
    return TurnUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        steps=steps,
    )


def _record_turn_step(policy: CallPolicy, usage: TokenUsage, *, cached: bool = False) -> None:
    steps = _TURN_USAGE.get()
    if steps is None:
        return
    steps.append(
        TurnUsageStep(
            tag=policy.usage_tag,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cached=cached,
        )
    )


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, _TRANSIENT_TYPES):
        return True
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return "timeout" in name or "timeout" in message or "connection" in name


def _usage_from_message(message: Any) -> TokenUsage:
    if message is None:
        return TokenUsage()
    # AIMessage / AIMessageChunk 都带 usage_metadata
    meta = getattr(message, "usage_metadata", None) or {}
    if meta:
        prompt = int(meta.get("input_tokens") or meta.get("prompt_tokens") or 0)
        completion = int(meta.get("output_tokens") or meta.get("completion_tokens") or 0)
        total = int(meta.get("total_tokens") or (prompt + completion))
        return TokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)
    resp = getattr(message, "response_metadata", None) or {}
    token_usage = resp.get("token_usage") or resp.get("usage") or {}
    if token_usage:
        prompt = int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0)
        completion = int(token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0)
        total = int(token_usage.get("total_tokens") or (prompt + completion))
        return TokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)
    return TokenUsage()


def _estimate_usage(payload: Any, result: Any) -> TokenUsage:
    from app.context import count_tokens

    prompt = count_tokens(_payload_text(payload))
    completion = count_tokens(_result_text(result))
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


def _payload_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        parts: list[str] = []
        for item in payload:
            if isinstance(item, BaseMessage):
                parts.append(str(item.content))
            elif isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(payload)


def _result_text(result: Any) -> str:
    if isinstance(result, AIMessage):
        return str(result.content)
    if hasattr(result, "model_dump_json"):
        return result.model_dump_json()
    return str(result)


def _result_chars(result: Any) -> int:
    return len(_result_text(result))


def _cache_fingerprint(payload: Any, policy: CallPolicy, kwargs: dict[str, Any]) -> str:
    blob = {
        "payload": _payload_text(payload),
        "tag": policy.usage_tag,
        "schema": policy.schema_name,
        "model": policy.model,
        "temperature": policy.temperature,
        "stop": kwargs.get("stop"),
    }
    raw = json.dumps(blob, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_llm_cache(store: Any, cache_key: str, schema_type: Any) -> Any | None:
    cached = store.get_cache(cache_key)
    if not cached:
        return None
    result = decode_llm_cache(cached, schema_type=schema_type)
    if result is None:
        logger.warning("LLM 短缓存反序列化失败")
    return result


def _write_llm_cache(store: Any, cache_key: str, result: Any, ttl: int) -> None:
    try:
        blob = encode_llm_cache(result)
        if blob is None:
            logger.warning("LLM 短缓存跳过写入：结果无法按 schema 序列化")
            return
        store.set_cache(cache_key, blob, ttl)
    except Exception:
        logger.exception("写入 LLM 短缓存失败")


def _emit_llm_call(
    *,
    tenant: str,
    policy: CallPolicy,
    usage: TokenUsage,
    daily: int,
    elapsed_ms: int,
    cache: bool,
) -> None:
    prompt = 0 if cache else usage.prompt_tokens
    completion = 0 if cache else usage.completion_tokens
    total = 0 if cache else usage.total_tokens
    log_event(
        logger,
        "llm_call",
        tag=policy.usage_tag,
        model=policy.model,
        prompt_tokens=prompt,
        completion_tokens=completion,
        tokens=total,
        daily=daily,
        elapsed_ms=elapsed_ms,
        cache=cache,
    )
    inc_llm(
        tenant,
        policy.usage_tag,
        cache_hit=cache,
        prompt_tokens=prompt,
        completion_tokens=completion,
        elapsed_seconds=elapsed_ms / 1000.0,
    )


class GatewayChatModel(Runnable[Any, Any]):
    """包装底层 Chat 模型或 bind/structured 后的 Runnable。"""

    def __init__(self, inner: Any, policy: CallPolicy) -> None:
        super().__init__()
        self._inner = inner
        self._policy = policy

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def InputType(self) -> Any:  # noqa: N802
        return getattr(self._inner, "InputType", Any)

    @property
    def OutputType(self) -> Any:  # noqa: N802
        return getattr(self._inner, "OutputType", Any)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> GatewayChatModel:
        """结构化输出后仍走 Gateway。"""
        name = getattr(schema, "__name__", None) or getattr(schema, "__qualname__", str(schema))
        inner = self._inner.with_structured_output(schema, **kwargs)
        return GatewayChatModel(
            inner,
            replace(self._policy, schema_name=str(name), schema_type=schema),
        )

    def bind_tools(self, *args: Any, **kwargs: Any) -> GatewayChatModel:
        """绑定工具后关闭短缓存。"""
        inner = self._inner.bind_tools(*args, **kwargs)
        return GatewayChatModel(inner, replace(self._policy, cache_enabled=False))

    def bind(self, **kwargs: Any) -> GatewayChatModel:
        inner = self._inner.bind(**kwargs)
        temperature = kwargs.get("temperature", self._policy.temperature)
        return GatewayChatModel(inner, replace(self._policy, temperature=float(temperature)))

    def with_config(
        self,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> GatewayChatModel:
        inner = self._inner.with_config(config, **kwargs)
        return GatewayChatModel(inner, self._policy)

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        return self._gated("invoke", input, config, kwargs)

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        return await self._agated("ainvoke", input, config, kwargs)

    def stream(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Iterator[Any]:
        self._assert_quota()
        tenant = get_current_tenant()
        started = time.monotonic()
        last: Any = None
        best_usage = TokenUsage()
        try:
            guard_llm_circuit()
            for chunk in self._inner.stream(input, config=config, **kwargs):
                last = chunk
                chunk_usage = _usage_from_message(chunk)
                if chunk_usage.total_tokens:
                    best_usage = chunk_usage
                yield chunk
        except TenantQuotaExceededError:
            raise
        except LlmCircuitOpenError:
            inc_llm_error(self._policy.usage_tag, "LlmCircuitOpenError")
            raise
        except Exception as exc:
            record_llm_failure()
            inc_llm_error(self._policy.usage_tag, type(exc).__name__)
            raise
        else:
            record_llm_success()
        elapsed_ms = int((time.monotonic() - started) * 1000)
        usage = best_usage if best_usage.total_tokens else _usage_from_message(last)
        if not usage.total_tokens:
            usage = _estimate_usage(input, last)
        usage = usage.normalized()
        snapshot = self._commit(usage, tenant=tenant)
        _emit_llm_call(
            tenant=tenant,
            policy=self._policy,
            usage=usage,
            daily=snapshot.total_tokens,
            elapsed_ms=elapsed_ms,
            cache=False,
        )

    async def astream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        self._assert_quota()
        tenant = get_current_tenant()
        started = time.monotonic()
        last: Any = None
        best_usage = TokenUsage()
        try:
            guard_llm_circuit()
            async for chunk in self._inner.astream(input, config=config, **kwargs):
                last = chunk
                chunk_usage = _usage_from_message(chunk)
                if chunk_usage.total_tokens:
                    best_usage = chunk_usage
                yield chunk
        except TenantQuotaExceededError:
            raise
        except LlmCircuitOpenError:
            inc_llm_error(self._policy.usage_tag, "LlmCircuitOpenError")
            raise
        except Exception as exc:
            record_llm_failure()
            inc_llm_error(self._policy.usage_tag, type(exc).__name__)
            raise
        else:
            record_llm_success()
        elapsed_ms = int((time.monotonic() - started) * 1000)
        usage = best_usage if best_usage.total_tokens else _usage_from_message(last)
        if not usage.total_tokens:
            usage = _estimate_usage(input, last)
        usage = usage.normalized()
        snapshot = self._commit(usage, tenant=tenant)
        _emit_llm_call(
            tenant=tenant,
            policy=self._policy,
            usage=usage,
            daily=snapshot.total_tokens,
            elapsed_ms=elapsed_ms,
            cache=False,
        )

    def _gated(self, method: str, payload: Any, config: RunnableConfig | None, kwargs: dict[str, Any]) -> Any:
        from app.config import get_settings

        settings = get_settings()
        tenant = get_current_tenant()
        store = get_gateway_store()
        policy = self._policy
        self._assert_quota(tenant=tenant, store=store)

        cache_on = bool(policy.cache_enabled and settings.llm_cache_enabled)
        cache_key = ""
        if cache_on:
            cache_key = f"{tenant}:{_cache_fingerprint(payload, policy, kwargs)}"
            cached = _read_llm_cache(store, cache_key, policy.schema_type)
            if cached is not None:
                store.add_cache_hit(tenant)
                _record_turn_step(policy, TokenUsage(), cached=True)
                _emit_llm_call(
                    tenant=tenant,
                    policy=policy,
                    usage=TokenUsage(),
                    daily=0,
                    elapsed_ms=0,
                    cache=True,
                )
                return cached

        started = time.monotonic()
        result = self._call_with_circuit(
            lambda: self._retry_call(method, payload, config, kwargs, settings.llm_max_retries)
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)

        usage = _usage_from_message(result)
        if not usage.total_tokens:
            usage = _estimate_usage(payload, result)
        usage = usage.normalized()
        snapshot = self._commit(usage, tenant=tenant, store=store)
        _emit_llm_call(
            tenant=tenant,
            policy=policy,
            usage=usage,
            daily=snapshot.total_tokens,
            elapsed_ms=elapsed_ms,
            cache=False,
        )

        if cache_on and _result_chars(result) <= settings.llm_cache_max_chars:
            _write_llm_cache(store, cache_key, result, settings.llm_cache_ttl_seconds)
        return result

    async def _agated(
        self,
        method: str,
        payload: Any,
        config: RunnableConfig | None,
        kwargs: dict[str, Any],
    ) -> Any:
        # 真异步闸门：LLM 调用走 await（method="ainvoke"），不再退回同步 _gated/线程池。
        # Redis store 调用仍为同步（ms 级，与 astream 一致），后续可改 asyncio.to_thread。
        from app.config import get_settings

        settings = get_settings()
        tenant = get_current_tenant()
        store = get_gateway_store()
        policy = self._policy
        self._assert_quota(tenant=tenant, store=store)

        cache_on = bool(policy.cache_enabled and settings.llm_cache_enabled)
        cache_key = ""
        if cache_on:
            cache_key = f"{tenant}:{_cache_fingerprint(payload, policy, kwargs)}"
            cached = _read_llm_cache(store, cache_key, policy.schema_type)
            if cached is not None:
                store.add_cache_hit(tenant)
                _record_turn_step(policy, TokenUsage(), cached=True)
                _emit_llm_call(
                    tenant=tenant,
                    policy=policy,
                    usage=TokenUsage(),
                    daily=0,
                    elapsed_ms=0,
                    cache=True,
                )
                return cached

        started = time.monotonic()
        result = await self._acall_with_circuit(
            lambda: self._aretry_call(
                method, payload, config, kwargs, settings.llm_max_retries
            )
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)

        usage = _usage_from_message(result)
        if not usage.total_tokens:
            usage = _estimate_usage(payload, result)
        usage = usage.normalized()
        snapshot = self._commit(usage, tenant=tenant, store=store)
        _emit_llm_call(
            tenant=tenant,
            policy=policy,
            usage=usage,
            daily=snapshot.total_tokens,
            elapsed_ms=elapsed_ms,
            cache=False,
        )

        if cache_on and _result_chars(result) <= settings.llm_cache_max_chars:
            _write_llm_cache(store, cache_key, result, settings.llm_cache_ttl_seconds)
        return result

    def _call_with_circuit(self, fn: Callable[[], Any]) -> Any:
        """熔断在重试环外侧：打开则 fail-fast；整次调用成败才计数。"""
        guard_llm_circuit()
        try:
            result = fn()
        except TenantQuotaExceededError:
            raise
        except LlmCircuitOpenError:
            inc_llm_error(self._policy.usage_tag, "LlmCircuitOpenError")
            raise
        except Exception:
            record_llm_failure()
            raise
        record_llm_success()
        return result

    async def _acall_with_circuit(self, factory: Callable[[], Any]) -> Any:
        guard_llm_circuit()
        try:
            result = await factory()
        except TenantQuotaExceededError:
            raise
        except LlmCircuitOpenError:
            inc_llm_error(self._policy.usage_tag, "LlmCircuitOpenError")
            raise
        except Exception:
            record_llm_failure()
            raise
        record_llm_success()
        return result

    async def _aretry_call(
        self,
        method: str,
        payload: Any,
        config: RunnableConfig | None,
        kwargs: dict[str, Any],
        max_retries: int,
    ) -> Any:
        attempts = max(1, int(max_retries) + 1)
        last_error: BaseException | None = None
        fn: Callable[..., Any] = getattr(self._inner, method)
        for attempt in range(attempts):
            try:
                return await fn(payload, config=config, **kwargs)
            except TenantQuotaExceededError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt + 1 < attempts and _is_transient(exc):
                    logger.warning(
                        "LLM 瞬态错误(异步)，重试 %s/%s: %s",
                        attempt + 1,
                        attempts,
                        exc,
                    )
                    continue
                inc_llm_error(self._policy.usage_tag, type(exc).__name__)
                raise
        assert last_error is not None
        raise last_error

    def _retry_call(
        self,
        method: str,
        payload: Any,
        config: RunnableConfig | None,
        kwargs: dict[str, Any],
        max_retries: int,
    ) -> Any:
        attempts = max(1, int(max_retries) + 1)
        last_error: BaseException | None = None
        fn: Callable[..., Any] = getattr(self._inner, method)
        for attempt in range(attempts):
            try:
                return fn(payload, config=config, **kwargs)
            except TenantQuotaExceededError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt + 1 < attempts and _is_transient(exc):
                    logger.warning(
                        "LLM 瞬态错误，重试 %s/%s: %s",
                        attempt + 1,
                        attempts,
                        exc,
                    )
                    continue
                inc_llm_error(self._policy.usage_tag, type(exc).__name__)
                raise
        assert last_error is not None
        raise last_error

    def _assert_quota(self, *, tenant: str | None = None, store: Any = None) -> None:
        from app.config import get_settings

        settings = get_settings()
        if not self._policy.enforce_quota:
            return
        limit = int(settings.llm_daily_token_limit)
        if limit <= 0:
            return
        tenant = tenant or get_current_tenant()
        store = store or get_gateway_store()
        snapshot = store.get_quota(tenant)
        if snapshot.total_tokens >= limit:
            raise TenantQuotaExceededError(tenant, snapshot.total_tokens, limit)

    def _commit(
        self,
        usage: TokenUsage,
        *,
        tenant: str | None = None,
        store: Any = None,
    ) -> Any:
        tenant = tenant or get_current_tenant()
        store = store or get_gateway_store()
        snapshot = store.add_usage(tenant, usage)
        _record_turn_step(self._policy, usage, cached=False)
        return snapshot


def wrap_chat_model(inner: Any, policy: CallPolicy) -> GatewayChatModel:
    """给底层模型套上 Gateway。"""
    return GatewayChatModel(inner, policy)
