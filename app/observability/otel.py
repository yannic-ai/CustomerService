"""OpenTelemetry：HTTP/DB/Redis 链路，OTLP 导出到 Langfuse（或自定义 endpoint）。

``/metrics`` 仍负责容量告警；本模块只补分布式 trace。未配置 Langfuse 且未设
``OTEL_EXPORTER_OTLP_ENDPOINT`` 时为零开销 noop。
"""

from __future__ import annotations

import base64
import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator
from urllib.parse import urljoin

logger = logging.getLogger("cs.otel")

_PROVIDER = None
_FASTAPI_INSTRUMENTED = False
_LIBS_INSTRUMENTED = False

TRACEPARENT_HEADER = "traceparent"
TRACESTATE_HEADER = "tracestate"


def otel_active() -> bool:
    """是否应启动 OTel（显式 endpoint 或已配 Langfuse）。"""
    from app.config import get_settings

    settings = get_settings()
    if not settings.otel_enabled:
        return False
    if (settings.otel_exporter_otlp_endpoint or "").strip():
        return True
    return settings.langfuse_enabled


def _resolve_endpoint_and_headers() -> tuple[str, dict[str, str]]:
    from app.config import get_settings

    settings = get_settings()
    explicit = (settings.otel_exporter_otlp_endpoint or "").strip()
    headers: dict[str, str] = {}
    raw_headers = (settings.otel_exporter_otlp_headers or "").strip()
    if raw_headers:
        for part in raw_headers.split(","):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            headers[key.strip()] = value.strip()

    if explicit:
        endpoint = explicit
        if settings.langfuse_enabled and "Authorization" not in headers:
            headers.update(_langfuse_auth_headers(settings))
        if settings.langfuse_enabled and "x-langfuse-ingestion-version" not in headers:
            headers["x-langfuse-ingestion-version"] = "4"
        return endpoint, headers

    # 默认打到自建 / 云端 Langfuse OTLP
    host = (settings.langfuse_host or "http://127.0.0.1:3000").rstrip("/") + "/"
    endpoint = urljoin(host, "api/public/otel/v1/traces")
    headers.update(_langfuse_auth_headers(settings))
    headers["x-langfuse-ingestion-version"] = "4"
    return endpoint, headers


def _langfuse_auth_headers(settings: Any) -> dict[str, str]:
    pk = settings.langfuse_public_key.strip()
    sk = settings.langfuse_secret_key.strip()
    token = base64.b64encode(f"{pk}:{sk}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def setup_otel() -> bool:
    """初始化 TracerProvider + 库级自动埋点。可重复调用，只生效一次。"""
    global _PROVIDER, _LIBS_INSTRUMENTED
    if _PROVIDER is not None:
        return True
    if not otel_active():
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.semconv.resource import ResourceAttributes
    except Exception as exc:
        logger.warning("OpenTelemetry SDK 不可用，跳过: %s", exc)
        return False

    from app.config import get_settings

    settings = get_settings()
    try:
        endpoint, headers = _resolve_endpoint_and_headers()
    except Exception as exc:
        logger.warning("解析 OTLP endpoint 失败: %s", exc)
        return False

    resource = Resource.create(
        {
            ResourceAttributes.SERVICE_NAME: settings.otel_service_name or "customer-service",
            ResourceAttributes.SERVICE_VERSION: "0.1.0",
            "deployment.environment": settings.otel_environment or "development",
        }
    )
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _PROVIDER = provider

    if not _LIBS_INSTRUMENTED:
        _instrument_libraries()
        _LIBS_INSTRUMENTED = True

    logger.info(
        "OpenTelemetry 已启用 service=%s endpoint=%s",
        settings.otel_service_name,
        endpoint,
    )
    return True


def _instrument_libraries() -> None:
    """SQLAlchemy / Redis / HTTPX 自动埋点；失败只告警。"""
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().instrument(enable_commenter=False)
    except Exception as exc:
        logger.warning("SQLAlchemy OTel 埋点跳过: %s", exc)
    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor

        RedisInstrumentor().instrument()
    except Exception as exc:
        logger.warning("Redis OTel 埋点跳过: %s", exc)
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
    except Exception as exc:
        logger.warning("HTTPX OTel 埋点跳过: %s", exc)


def instrument_fastapi(app: Any) -> bool:
    """给 FastAPI 打服务端 span（自动提取/关联 traceparent）。"""
    global _FASTAPI_INSTRUMENTED
    if not otel_active():
        return False
    # 先 setup：TestClient 多次进出 lifespan 时 provider 会重建，中间件可复用
    if not setup_otel():
        return False
    if _FASTAPI_INSTRUMENTED:
        return True
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except Exception as exc:
        logger.warning("FastAPI OTel 埋点不可用: %s", exc)
        return False
    try:
        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="health,ready,metrics,/health,/ready,/metrics",
        )
        # instrument 只替换 build_middleware_stack；已缓存的 stack 不会自动重建
        app.middleware_stack = None
        _FASTAPI_INSTRUMENTED = True
        return True
    except Exception as exc:
        logger.warning("FastAPI OTel 埋点失败: %s", exc)
        return False


def shutdown_otel() -> None:
    """进程退出前刷出并关闭 TracerProvider。"""
    global _PROVIDER
    provider = _PROVIDER
    _PROVIDER = None
    if provider is None:
        return
    try:
        provider.force_flush()
        provider.shutdown()
    except Exception as exc:
        logger.debug("OTel shutdown 跳过: %s", exc)


def _apply_span_attributes(span: Any, attrs: dict[str, Any]) -> None:
    """只写非空字符串，避免高基数 / None 污染。"""
    if span is None:
        return
    for key, value in attrs.items():
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        span.set_attribute(key, text)


def set_span_attributes(**attrs: Any) -> None:
    """给当前 active span 写低基数业务属性。"""
    if not attrs or _PROVIDER is None:
        return
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is None or not span.is_recording():
            return
        _apply_span_attributes(span, attrs)
    except Exception:
        return


def current_trace_ids() -> tuple[str, str]:
    """当前 recording span 的 ``(trace_id, span_id)`` hex；没有则空串。

    供 JSON 日志与 ``request_id`` 一起关联 SQLAlchemy / Redis 自动 span。
    """
    if _PROVIDER is None:
        return "", ""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is None:
            return "", ""
        ctx = span.get_span_context()
        if ctx is None or not getattr(ctx, "is_valid", False):
            return "", ""
        return f"{ctx.trace_id:032x}", f"{ctx.span_id:016x}"
    except Exception:
        return "", ""


@contextmanager
def start_span(name: str, **attrs: Any) -> Iterator[Any]:
    """在当前 trace 下开子 span；OTel 未启用时为零开销 noop。

    自动带上 ``cs.request_id``（若 contextvars 里有）。异常会 ``record_exception``。
    """
    if _PROVIDER is None or not (name or "").strip():
        yield None
        return
    try:
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode
    except Exception:
        yield None
        return

    from app.observability.context import get_request_id

    merged = dict(attrs)
    request_id = get_request_id()
    if request_id and "cs.request_id" not in merged:
        merged["cs.request_id"] = request_id

    tracer = trace.get_tracer("customer-service")
    with tracer.start_as_current_span(name.strip()) as span:
        _apply_span_attributes(span, merged)
        try:
            yield span
        except Exception as exc:
            try:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
            except Exception:
                pass
            raise


@contextmanager
def node_span(node: str, **attrs: Any) -> Iterator[Any]:
    """图节点：Prometheus ``cs_node_duration_seconds`` + OTel 子 span ``cs.graph.{node}``。"""
    from app.observability.metrics import observe_node

    name = (node or "").strip() or "unknown"
    started = time.monotonic()
    try:
        with start_span(f"cs.graph.{name}", **{"cs.node": name, **attrs}) as span:
            yield span
    finally:
        observe_node(name, time.monotonic() - started)


def current_trace_context() -> Any | None:
    """快照当前 span 的 Context（响应发出后 ASGI span 可能已结束）。"""
    if _PROVIDER is None:
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.trace import set_span_in_context

        span = trace.get_current_span()
        if span is None or not span.is_recording():
            return None
        return set_span_in_context(span)
    except Exception:
        return None


def inject_trace_context(headers: Any, context: Any | None = None) -> None:
    """把 trace context 注入响应头（traceparent / tracestate）。"""
    if _PROVIDER is None:
        return
    try:
        from opentelemetry import propagate

        carrier: dict[str, str] = {}
        if context is not None:
            propagate.inject(carrier, context=context)
        else:
            propagate.inject(carrier)
        for key, value in carrier.items():
            if value:
                headers[key] = value
    except Exception:
        return


def _reset_global_tracer_provider() -> None:
    """测试用：允许再次 ``set_tracer_provider``。生产路径只 ``shutdown``，不调用。"""
    try:
        from opentelemetry import trace as trace_api
        from opentelemetry.util._once import Once

        trace_api._TRACER_PROVIDER = None  # type: ignore[attr-defined]
        trace_api._TRACER_PROVIDER_SET_ONCE = Once()  # type: ignore[attr-defined]
    except Exception:
        return


def attach_span_exporter_for_tests(exporter: Any) -> bool:
    """测试用：挂到实际出 span 的 SDK TracerProvider（全局或本模块）。"""
    if exporter is None:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    except Exception:
        return False
    provider = _PROVIDER
    if provider is None or not isinstance(provider, SdkTracerProvider):
        candidate = trace.get_tracer_provider()
        if not isinstance(candidate, SdkTracerProvider):
            return False
        provider = candidate
    try:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        return True
    except Exception:
        return False


def reset_otel_for_tests(app: Any | None = None) -> None:
    """测试用：关闭 provider；可选卸载 FastAPI 埋点。"""
    global _PROVIDER, _FASTAPI_INSTRUMENTED, _LIBS_INSTRUMENTED
    if _PROVIDER is not None:
        try:
            _PROVIDER.shutdown()
        except Exception:
            pass
    _PROVIDER = None
    _reset_global_tracer_provider()
    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.uninstrument_app(app)
        except Exception:
            pass
        app.middleware_stack = None
        _FASTAPI_INSTRUMENTED = False
    # 库级 instrument 全局只做一次，测试间不反复 uninstrument
