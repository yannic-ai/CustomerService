"""LLM Gateway：模型、超时、熔断、配额、短缓存与计价。"""

from pydantic import BaseModel, Field


class LLMSettings(BaseModel):
    """DeepSeek 与 Gateway 策略。新增 LLM 参数只改本文件。"""

    deepseek_api_key: str = ""
    deepseek_api_base: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    llm_timeout_seconds: float = Field(default=60, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    # 连续失败熔断：整次调用（含重试）失败累计；0 关闭。打开后冷却期内 fail-fast
    llm_circuit_failure_threshold: int = Field(default=3, ge=0, le=100)
    llm_circuit_cooldown_seconds: float = Field(default=30, ge=0)
    # 租户每日总 token（prompt+completion）；0 表示不限额
    llm_daily_token_limit: int = Field(default=500_000, ge=0)
    # 百万 token 单价（元）；默认按 deepseek-v4-flash 量级估算，可按账单改
    llm_prompt_price_per_mtok_yuan: float = Field(default=1.0, ge=0)
    llm_completion_price_per_mtok_yuan: float = Field(default=2.0, ge=0)
    llm_cache_enabled: bool = True
    llm_cache_ttl_seconds: int = Field(default=300, ge=0)
    llm_cache_max_chars: int = Field(default=4000, ge=0)
    llm_cache_allow_react: bool = False
