"""LLM 连续失败熔断：挡在单次 _retry_call 之外，避免上游持续故障时逐请求重试放大延迟。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from app.llm_gateway.exceptions import LlmCircuitOpenError

_LOCK = threading.Lock()
_STATE: _BreakerState | None = None


@dataclass
class CircuitSnapshot:
    """熔断器当前快照，供测试与排障。"""

    state: str
    failures: int
    cooldown_remaining: float


@dataclass
class _BreakerState:
    failures: int = 0
    opened_at: float | None = None
    probe_in_flight: bool = False


def _settings() -> tuple[int, float]:
    from app.config import get_settings

    cfg = get_settings()
    return int(cfg.llm_circuit_failure_threshold), float(cfg.llm_circuit_cooldown_seconds)


def _snapshot_locked(state: _BreakerState, *, cooldown: float, now: float) -> CircuitSnapshot:
    if state.opened_at is None:
        return CircuitSnapshot(state="closed", failures=state.failures, cooldown_remaining=0.0)
    remaining = max(0.0, cooldown - (now - state.opened_at))
    label = "half_open" if remaining <= 0 else "open"
    return CircuitSnapshot(state=label, failures=state.failures, cooldown_remaining=remaining)


def snapshot_llm_breaker() -> CircuitSnapshot:
    """读取当前熔断状态（无副作用）。"""
    _, cooldown = _settings()
    with _LOCK:
        state = _STATE or _BreakerState()
        return _snapshot_locked(state, cooldown=cooldown, now=time.monotonic())


def reset_llm_breaker() -> None:
    """测试用：清空连续失败计数与打开状态。"""
    global _STATE
    with _LOCK:
        _STATE = _BreakerState()


def guard_llm_circuit() -> None:
    """调用上游前检查；打开期内直接拒绝，冷却结束只放行一个探测请求。"""
    global _STATE
    threshold, cooldown = _settings()
    if threshold <= 0:
        return
    with _LOCK:
        if _STATE is None:
            _STATE = _BreakerState()
        state = _STATE
        if state.opened_at is None:
            return
        now = time.monotonic()
        if now - state.opened_at < cooldown:
            failures = state.failures
        elif state.probe_in_flight:
            failures = state.failures
        else:
            state.probe_in_flight = True
            return
    _reject(failures, cooldown)


def record_llm_success() -> None:
    """整次调用（含重试）成功则闭合。"""
    global _STATE
    with _LOCK:
        _STATE = _BreakerState()


def record_llm_failure() -> None:
    """整次调用（含重试）失败记一次；达到阈值则打开。"""
    global _STATE
    threshold, cooldown = _settings()
    just_opened = False
    with _LOCK:
        if _STATE is None:
            _STATE = _BreakerState()
        state = _STATE
        state.probe_in_flight = False
        if threshold <= 0:
            return
        was_open = state.opened_at is not None
        state.failures += 1
        if state.failures >= threshold:
            state.opened_at = time.monotonic()
            just_opened = not was_open
    if just_opened:
        from app.observability import inc_llm_circuit

        inc_llm_circuit("open")


def _reject(failures: int, cooldown: float) -> None:
    from app.observability import inc_llm_circuit

    inc_llm_circuit("reject")
    raise LlmCircuitOpenError(failures, cooldown)
