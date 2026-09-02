"""LLM Gateway 调用策略与记账结果。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CallPolicy:
    """单次模型调用的 Gateway 策略。"""

    usage_tag: str = "default"
    cache_enabled: bool = False
    enforce_quota: bool = True
    schema_name: str = ""
    schema_type: Any = field(default=None, hash=False, compare=False, repr=False)
    temperature: float = 0.1
    model: str = ""


@dataclass
class TokenUsage:
    """一次调用的 token 用量。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached: bool = False

    def normalized(self) -> TokenUsage:
        """补全 total；若仅有 total 则保留。"""
        total = self.total_tokens or (self.prompt_tokens + self.completion_tokens)
        return TokenUsage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=total,
            cached=self.cached,
        )


@dataclass
class QuotaSnapshot:
    """租户当日累计。"""

    tenant_id: str
    day: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0
    cache_hits: int = 0
    extra: dict[str, int] = field(default_factory=dict)


@dataclass
class TurnUsageStep:
    """本轮对话中一次 LLM 调用的记账。"""

    tag: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached: bool = False


@dataclass
class TurnUsage:
    """本轮对话累计的 LLM usage（可写入 SSE done）。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    steps: list[TurnUsageStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "steps": len(self.steps),
        }
