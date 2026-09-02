"""结构化 JSON 日志：自动带上 request_id / tenant_id / user_id / trace_id。"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from app.observability.context import get_request_id
from app.observability.otel import current_trace_ids
from app.tenancy import get_current_tenant, get_current_user

_CONFIGURED = False


class RequestContextFilter(logging.Filter):
    """给每条日志补上请求上下文与当前 OTel ids。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        record.tenant_id = get_current_tenant()
        record.user_id = get_current_user()
        trace_id, span_id = current_trace_ids()
        record.trace_id = trace_id
        record.span_id = span_id
        return True


class JsonFormatter(logging.Formatter):
    """一行一条 JSON；`extra={"cs": {...}}` 会合并进对象。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        request_id = getattr(record, "request_id", "") or ""
        if request_id:
            payload["request_id"] = request_id
        tenant_id = getattr(record, "tenant_id", None)
        if tenant_id:
            payload["tenant_id"] = tenant_id
        user_id = getattr(record, "user_id", None)
        if user_id:
            payload["user_id"] = user_id
        trace_id = getattr(record, "trace_id", "") or ""
        if trace_id:
            payload["trace_id"] = trace_id
        span_id = getattr(record, "span_id", "") or ""
        if span_id:
            payload["span_id"] = span_id
        extra = getattr(record, "cs", None)
        if isinstance(extra, dict):
            for key, value in extra.items():
                if key in payload or value is None:
                    continue
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """本地调试用文本格式，仍带 request_id；有 trace 时附上短 id。"""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s [%(request_id)s] %(name)s %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        if not getattr(record, "request_id", None):
            record.request_id = ""
        line = super().format(record)
        trace_id = getattr(record, "trace_id", "") or ""
        if trace_id:
            return f"{line} trace_id={trace_id}"
        return line


def log_event(logger: logging.Logger, msg: str, **fields: Any) -> None:
    """打一条带结构化字段的 info。"""
    logger.info(msg, extra={"cs": fields})


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """只配置 ``cs.*`` 日志树，避免覆盖 uvicorn 自身的 handler。"""
    global _CONFIGURED
    logger = logging.getLogger("cs")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestContextFilter())
    if (fmt or "json").lower() == "text":
        handler.setFormatter(TextFormatter())
    else:
        handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    _CONFIGURED = True
