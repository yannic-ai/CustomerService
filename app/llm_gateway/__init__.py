"""LLM Gateway：超时、重试、熔断、token 记账、租户日限额、短响应缓存。"""

from app.llm_gateway.circuit import reset_llm_breaker, snapshot_llm_breaker
from app.llm_gateway.exceptions import (
    LLM_UNAVAILABLE_ANSWER,
    LlmCircuitOpenError,
    TenantQuotaExceededError,
    is_llm_degradable,
)
from app.llm_gateway.gateway import (
    GatewayChatModel,
    begin_turn_usage,
    take_turn_usage,
    wrap_chat_model,
)
from app.llm_gateway.stores import get_gateway_store, set_gateway_store
from app.llm_gateway.types import CallPolicy, TokenUsage, TurnUsage

__all__ = [
    "CallPolicy",
    "GatewayChatModel",
    "LLM_UNAVAILABLE_ANSWER",
    "LlmCircuitOpenError",
    "TenantQuotaExceededError",
    "TokenUsage",
    "TurnUsage",
    "begin_turn_usage",
    "get_gateway_store",
    "is_llm_degradable",
    "reset_llm_breaker",
    "set_gateway_store",
    "snapshot_llm_breaker",
    "take_turn_usage",
    "wrap_chat_model",
]
