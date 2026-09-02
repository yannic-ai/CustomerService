"""工具失败降级：结果缓存 TTL 与自动转人工阈值。"""

from pydantic import BaseModel, Field


class ResilienceSettings(BaseModel):
    """工具缓存与升级策略配置。"""

    tool_cache_enabled: bool = True
    tool_cache_order_ttl_seconds: int = Field(default=600, ge=0)
    tool_cache_course_ttl_seconds: int = Field(default=1800, ge=0)
    tool_escalation_enabled: bool = True
    tool_escalation_transient_threshold: int = Field(default=2, ge=1)
    tool_escalation_on_react_limit: bool = True
