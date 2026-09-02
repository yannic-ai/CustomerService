"""组合各主题 mixin，对外仍是单一 ``Settings``。"""

from __future__ import annotations

from functools import lru_cache
from typing import Self

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config.capacity import CapacitySettings
from app.config.llm import LLMSettings
from app.config.memory import MemorySettings
from app.config.paths import DATA_DIR, KNOWLEDGE_DIR, ROOT_DIR
from app.config.rag import RAGSettings
from app.config.resilience import ResilienceSettings
from app.config.runtime import RuntimeSettings
from app.config.telemetry import TelemetrySettings


class Settings(
    LLMSettings,
    RAGSettings,
    RuntimeSettings,
    TelemetrySettings,
    CapacitySettings,
    MemorySettings,
    ResilienceSettings,
    BaseSettings,
):
    """运行时配置，优先读取项目根目录 ``.env``。

    不开启 ``validate_assignment``，便于测试原地 patch 单例字段。
    字段按主题拆在 ``app/config/*.py``；交叉校验与开关属性留在这里。
    """

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("deepseek_api_base", "default_tenant", "otel_service_name", mode="before")
    @classmethod
    def _strip_required_str(cls, value: object) -> object:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError("不能为空")
            return text
        return value

    @model_validator(mode="after")
    def _check_cross_constraints(self) -> Self:
        if self.context_discard_score > self.context_summarize_score:
            raise ValueError("context_discard_score 不能大于 context_summarize_score")
        if self.context_always_keep_recent > self.history_max_messages:
            raise ValueError("context_always_keep_recent 不能大于 history_max_messages")
        return self

    @property
    def llm_enabled(self) -> bool:
        """是否已配置 DeepSeek API Key，可用于真实大模型调用。"""
        return bool(self.deepseek_api_key.strip())

    @property
    def langfuse_enabled(self) -> bool:
        """是否已配置 Langfuse 公钥与私钥。"""
        return bool(self.langfuse_public_key.strip() and self.langfuse_secret_key.strip())

    @property
    def oss_enabled(self) -> bool:
        """OSS 四项凭证是否齐全；未配齐时走本地文件回退。"""
        return all(
            [
                self.oss_access_key_id,
                self.oss_access_key_secret,
                self.oss_bucket,
                self.oss_endpoint,
            ]
        )


@lru_cache
def get_settings() -> Settings:
    """返回全局单例配置，并确保数据目录存在。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    return Settings()
