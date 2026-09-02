"""进程入口：监听地址、日志、租户、数据库与 OSS。"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.config.paths import DATA_DIR

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["json", "text"]


class RuntimeSettings(BaseModel):
    """HTTP 进程与存储。新增运行参数只改本文件。"""

    database_url: str = f"sqlite:///{DATA_DIR / 'customer_service.db'}"

    oss_access_key_id: str = ""
    oss_access_key_secret: str = ""
    oss_bucket: str = ""
    oss_endpoint: str = ""
    oss_prefix: str = "customer-service/"

    app_host: str = "0.0.0.0"
    app_port: int = Field(default=8000, ge=1, le=65535)
    log_level: LogLevel = "INFO"
    log_format: LogFormat = "json"
    default_tenant: str = "demo"
    # 管理面 ingest API；留空则禁用 POST /admin/knowledge/ingest
    admin_ingest_token: str = ""

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("log_format", mode="before")
    @classmethod
    def _normalize_log_format(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value
