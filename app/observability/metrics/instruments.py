"""业务埋点：新增指标时改本文件，并在 ``names.py`` 登记 HELP。"""

from __future__ import annotations

from app.observability.metrics.names import (
    CHAT_BUSY_REASONS,
    ESCALATION_REASONS,
    LLM_CALLS_BUCKETS,
    QUEUE_BUCKETS,
    RUNAWAY_REASONS,
    TOOL_CACHE_SOURCES,
    TOOL_ORDER_OUTCOMES,
    TOOL_POOL_LABELS,
)
from app.observability.metrics.store import inc, observe


def inc_http(method: str, path: str, status: int) -> None:
    inc("cs_http_requests_total", method=method, path=path, status=str(status))


def observe_http(method: str, path: str, status: int, elapsed_seconds: float) -> None:
    observe(
        "cs_http_duration_seconds",
        elapsed_seconds,
        method=method,
        path=path,
        status=str(status),
    )


def inc_chat(tenant: str, intent: str, blocked: bool, outcome: str) -> None:
    inc(
        "cs_chat_requests_total",
        tenant=tenant,
        intent=intent or "unknown",
        blocked="true" if blocked else "false",
        outcome=outcome,
    )


def observe_chat(intent: str, elapsed_seconds: float) -> None:
    observe("cs_chat_duration_seconds", elapsed_seconds, intent=intent or "unknown")


def observe_node(node: str, elapsed_seconds: float) -> None:
    observe("cs_node_duration_seconds", elapsed_seconds, node=node or "unknown")


def observe_ttft(intent: str, elapsed_seconds: float) -> None:
    observe("cs_ttft_seconds", elapsed_seconds, intent=intent or "unknown")


def observe_llm_calls(intent: str, calls: int) -> None:
    observe(
        "cs_llm_calls_per_turn",
        float(max(0, calls)),
        buckets=LLM_CALLS_BUCKETS,
        intent=intent or "unknown",
    )


def inc_react_limit_hit() -> None:
    inc("cs_react_limit_hit_total")


def inc_tool_cache_hit(tool: str, source: str) -> None:
    safe_source = source if source in TOOL_CACHE_SOURCES else "redis"
    inc("cs_tool_cache_hit_total", tool=tool or "unknown", source=safe_source)


def inc_tool_escalation(reason: str) -> None:
    safe = reason if reason in ESCALATION_REASONS else "transient_exhausted"
    inc("cs_tool_escalation_total", reason=safe)


def inc_user_feedback(value: str) -> None:
    """用户赞踩。value=up|down。"""
    safe = value if value in {"up", "down"} else "down"
    inc("cs_user_feedback_total", value=safe)


def inc_tool_call(tool: str, outcome: str, tenant: str = "demo") -> None:
    """记一次工具调用。查单 / 课程检索 outcome：ok / not_found / deny / timeout / busy / error。"""
    safe_outcome = outcome if outcome in TOOL_ORDER_OUTCOMES else "error"
    inc(
        "cs_tool_calls_total",
        tool=tool or "unknown",
        outcome=safe_outcome,
        tenant=tenant or "demo",
    )


def inc_agent_runaway(reason: str) -> None:
    """Agent 跑飞：单轮 LLM 超阈值或 ReACT 撞工具上限。"""
    safe = reason if reason in RUNAWAY_REASONS else "high_llm_calls"
    inc("cs_agent_runaway_total", reason=safe)


def maybe_inc_runaway_llm_calls(calls: int) -> None:
    """单轮 LLM 次数超过配置阈值则记跑飞（供告警）。"""
    from app.config import get_settings

    threshold = int(get_settings().agent_runaway_llm_calls)
    if threshold > 0 and int(calls) > threshold:
        inc_agent_runaway("high_llm_calls")
        try:
            from app.observability.logging import log_event
            import logging

            log_event(
                logging.getLogger("cs.runaway"),
                "agent_runaway",
                reason="high_llm_calls",
                llm_calls=int(calls),
                threshold=threshold,
            )
        except Exception:
            pass


def inc_chat_busy(reason: str = "cluster") -> None:
    safe = reason if reason in CHAT_BUSY_REASONS else "cluster"
    inc("cs_chat_busy_total", reason=safe)


def inc_session_busy() -> None:
    inc("cs_session_busy_total")


def inc_tool_pool_rejected(pool: str) -> None:
    """工具线程池背压拒绝计数。"""
    safe_pool = pool if pool in TOOL_POOL_LABELS else "unknown"
    inc("cs_tool_pool_rejected_total", pool=safe_pool)


def observe_tool_pool_queue(pool: str, duration: float) -> None:
    """工具线程池排队等待耗时直方图。"""
    safe_pool = pool if pool in TOOL_POOL_LABELS else "unknown"
    observe(
        "cs_tool_pool_queue_seconds",
        max(0.0, float(duration)),
        buckets=QUEUE_BUCKETS,
        pool=safe_pool,
    )


def estimate_llm_cost_yuan(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    prompt_price_per_mtok: float | None = None,
    completion_price_per_mtok: float | None = None,
) -> float:
    """按百万 token 单价估算成本（元）；单价缺省读 Settings。"""
    if prompt_price_per_mtok is None or completion_price_per_mtok is None:
        from app.config import get_settings

        settings = get_settings()
        if prompt_price_per_mtok is None:
            prompt_price_per_mtok = float(settings.llm_prompt_price_per_mtok_yuan)
        if completion_price_per_mtok is None:
            completion_price_per_mtok = float(settings.llm_completion_price_per_mtok_yuan)
    prompt = max(0, int(prompt_tokens)) * float(prompt_price_per_mtok) / 1_000_000.0
    completion = max(0, int(completion_tokens)) * float(completion_price_per_mtok) / 1_000_000.0
    return prompt + completion


def inc_llm(
    tenant: str,
    tag: str,
    *,
    cache_hit: bool,
    tokens: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    elapsed_seconds: float = 0,
) -> None:
    """记 LLM 请求、分列 token 与估算成本。

    ``tokens`` 仅作兼容回退：未传 prompt/completion 时整笔记到 prompt。
    cache hit 不累加 token / 成本。
    """
    hit = "true" if cache_hit else "false"
    inc("cs_llm_requests_total", tenant=tenant, tag=tag, cache_hit=hit)
    prompt = max(0, int(prompt_tokens))
    completion = max(0, int(completion_tokens))
    if not cache_hit and prompt == 0 and completion == 0 and tokens:
        prompt = max(0, int(tokens))
    if not cache_hit:
        if prompt:
            inc(
                "cs_llm_tokens_total",
                float(prompt),
                tenant=tenant,
                tag=tag,
                direction="prompt",
            )
        if completion:
            inc(
                "cs_llm_tokens_total",
                float(completion),
                tenant=tenant,
                tag=tag,
                direction="completion",
            )
        cost = estimate_llm_cost_yuan(prompt, completion)
        if cost > 0:
            inc("cs_llm_cost_yuan_total", cost, tenant=tenant, tag=tag)
    observe("cs_llm_duration_seconds", elapsed_seconds, tag=tag, cache_hit=hit)


def inc_llm_error(tag: str, error_class: str) -> None:
    inc("cs_llm_errors_total", tag=tag, error_class=error_class)


def inc_llm_circuit(event: str) -> None:
    """熔断打开或打开期内拒绝。event=open|reject。"""
    safe = event if event in {"open", "reject"} else "reject"
    inc("cs_llm_circuit_total", event=safe)


def inc_rag_empty(tenant: str, reason: str = "empty") -> None:
    """空召回或低分拒答。reason=empty|low_score。"""
    inc("cs_rag_empty_total", tenant=tenant, reason=reason)


def inc_rag_route_upgrade(tenant: str) -> None:
    """RAG 空召回后因订单信号改走 ReACT。"""
    inc("cs_rag_route_upgrade_total", tenant=tenant)


def inc_order_access_deny(tenant: str, reason: str, context: str) -> None:
    inc(
        "cs_order_access_deny_total",
        tenant=tenant,
        reason=reason,
        context=context,
    )
