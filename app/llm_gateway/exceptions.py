"""LLM Gateway 异常。"""

LLM_UNAVAILABLE_ANSWER = (
    "当前智能助手暂时不可用。您可以稍后再试，"
    "或直接提供订单号查询进度，也可以回复「转人工」由专员继续处理。"
)


class TenantQuotaExceededError(Exception):
    """租户当日 LLM token 额度已用尽。"""

    def __init__(self, tenant_id: str, used: int, limit: int) -> None:
        self.tenant_id = tenant_id
        self.used = used
        self.limit = limit
        super().__init__(
            f"租户 {tenant_id} 当日 token 已达上限 used={used} limit={limit}"
        )


class LlmCircuitOpenError(Exception):
    """连续失败熔断已打开，跳过上游调用。"""

    def __init__(self, failures: int, cooldown_seconds: float) -> None:
        self.failures = failures
        self.cooldown_seconds = cooldown_seconds
        super().__init__(
            f"LLM 熔断已打开 consecutive_failures={failures} cooldown={cooldown_seconds}s"
        )


def is_llm_degradable(exc: BaseException) -> bool:
    """配额耗尽需单独提示；其余上游失败可由调用方降级。"""
    return not isinstance(exc, TenantQuotaExceededError)
