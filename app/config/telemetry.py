"""Langfuse 与 OpenTelemetry。"""

from pydantic import BaseModel


class TelemetrySettings(BaseModel):
    """轨迹上报。新增观测出口参数只改本文件。"""

    # Langfuse：两把钥匙都配齐才上报；未配置时零开销
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://127.0.0.1:3000"

    # OpenTelemetry：默认在 Langfuse 启用或显式配了 OTLP endpoint 时开启
    otel_enabled: bool = True
    otel_service_name: str = "customer-service"
    otel_environment: str = "development"
    # 留空则 Langfuse 启用时打到 {LANGFUSE_HOST}/api/public/otel/v1/traces
    otel_exporter_otlp_endpoint: str = ""
    # 额外 OTLP headers，逗号分隔 key=value（Authorization 一般由 Langfuse 密钥自动填）
    otel_exporter_otlp_headers: str = ""
