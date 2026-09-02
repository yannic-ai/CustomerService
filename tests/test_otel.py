"""OpenTelemetry：启用条件、Langfuse OTLP endpoint、响应 traceparent。"""

from __future__ import annotations

import base64

from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export import SpanExportResult

from app.app import app
from app.config import Settings, get_settings
from app.observability.otel import (
    _resolve_endpoint_and_headers,
    otel_active,
    reset_otel_for_tests,
    setup_otel,
    shutdown_otel,
)


class _NoopSpanExporter:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def export(self, spans):
        return SpanExportResult.SUCCESS

    def shutdown(self, timeout_millis: int = 30_000) -> bool:
        return True

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def _settings(**overrides) -> Settings:
    return get_settings().model_copy(
        update={
            "otel_enabled": True,
            "langfuse_public_key": "",
            "langfuse_secret_key": "",
            "otel_exporter_otlp_endpoint": "",
            "otel_exporter_otlp_headers": "",
            **overrides,
        }
    )


def test_otel_inactive_without_langfuse_or_endpoint(monkeypatch) -> None:
    monkeypatch.setattr("app.config.get_settings", lambda: _settings(otel_enabled=True))
    assert otel_active() is False


def test_otel_disabled_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.config.get_settings",
        lambda: _settings(
            otel_enabled=False,
            langfuse_public_key="pk-test",
            langfuse_secret_key="sk-test",
        ),
    )
    assert otel_active() is False


def test_otel_active_with_langfuse_keys(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.config.get_settings",
        lambda: _settings(
            langfuse_public_key="pk-test",
            langfuse_secret_key="sk-test",
            langfuse_host="http://langfuse.example",
        ),
    )
    assert otel_active() is True


def test_resolve_langfuse_otlp_endpoint_and_auth(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.config.get_settings",
        lambda: _settings(
            langfuse_public_key="pk-lf-demo",
            langfuse_secret_key="sk-lf-demo",
            langfuse_host="http://127.0.0.1:3000",
        ),
    )
    endpoint, headers = _resolve_endpoint_and_headers()
    assert endpoint == "http://127.0.0.1:3000/api/public/otel/v1/traces"
    expected = base64.b64encode(b"pk-lf-demo:sk-lf-demo").decode("ascii")
    assert headers["Authorization"] == f"Basic {expected}"
    assert headers["x-langfuse-ingestion-version"] == "4"


def test_explicit_otlp_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.config.get_settings",
        lambda: _settings(
            otel_exporter_otlp_endpoint="http://collector:4318/v1/traces",
            otel_exporter_otlp_headers="x-custom=1",
        ),
    )
    assert otel_active() is True
    endpoint, headers = _resolve_endpoint_and_headers()
    assert endpoint == "http://collector:4318/v1/traces"
    assert headers["x-custom"] == "1"


def test_setup_otel_noop_when_inactive(monkeypatch) -> None:
    reset_otel_for_tests(app)
    monkeypatch.setattr(
        "app.config.get_settings",
        lambda: _settings(otel_enabled=False),
    )
    assert setup_otel() is False
    shutdown_otel()


def test_traceparent_echoed_when_otel_active(monkeypatch) -> None:
    """启用 OTel 后，响应回写 W3C traceparent，并延续入站 trace-id。"""
    reset_otel_for_tests(app)
    settings = _settings(
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        langfuse_host="http://127.0.0.1:3000",
    )
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
        _NoopSpanExporter,
    )

    inbound = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    try:
        with TestClient(app) as client:
            response = client.get(
                "/greet",
                headers={
                    "traceparent": inbound,
                    "X-Tenant-Id": "demo",
                },
            )
        assert response.status_code == 200
        tp = response.headers.get("traceparent")
        assert tp is not None
        assert tp.startswith("00-4bf92f3577b34da6a3ce929d0e0e4736-")
    finally:
        shutdown_otel()
        reset_otel_for_tests(app)


def _enable_sdk(monkeypatch, **overrides) -> None:
    reset_otel_for_tests(app)
    settings = _settings(
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        langfuse_host="http://127.0.0.1:3000",
        **overrides,
    )
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
        _NoopSpanExporter,
    )
    assert setup_otel()


def test_current_trace_ids_empty_when_inactive() -> None:
    reset_otel_for_tests(app)
    from app.observability.otel import current_trace_ids, start_span

    assert current_trace_ids() == ("", "")
    with start_span("cs.test.noop") as span:
        assert span is None
        assert current_trace_ids() == ("", "")


def test_json_log_omits_trace_when_inactive() -> None:
    import json
    import logging
    from io import StringIO

    from app.observability.logging import JsonFormatter, RequestContextFilter, log_event

    reset_otel_for_tests(app)
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.addFilter(RequestContextFilter())
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("cs.test.no_trace")
    logger.handlers[:] = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    log_event(logger, "no_trace")
    payload = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert payload["msg"] == "no_trace"
    assert "trace_id" not in payload
    assert "span_id" not in payload


def test_start_span_and_json_log_share_trace_id(monkeypatch) -> None:
    import json
    import logging
    from io import StringIO

    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from app.observability.logging import JsonFormatter, RequestContextFilter, log_event
    from app.observability.otel import (
        attach_span_exporter_for_tests,
        current_trace_ids,
        start_span,
    )

    _enable_sdk(monkeypatch)
    exporter = InMemorySpanExporter()
    assert attach_span_exporter_for_tests(exporter)
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.addFilter(RequestContextFilter())
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("cs.test.hello_trace")
    logger.handlers[:] = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        with start_span("cs.test.span", **{"cs.node": "test"}):
            tid, sid = current_trace_ids()
            assert len(tid) == 32
            assert len(sid) == 16
            log_event(logger, "hello_trace")
        payload = json.loads(buf.getvalue().strip().splitlines()[-1])
        assert payload["trace_id"] == tid
        assert payload["span_id"] == sid
        names = [span.name for span in exporter.get_finished_spans()]
        assert "cs.test.span" in names
    finally:
        shutdown_otel()
        reset_otel_for_tests(app)


def test_start_span_records_exception(monkeypatch) -> None:
    import pytest
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import StatusCode

    from app.observability.otel import attach_span_exporter_for_tests, start_span

    _enable_sdk(monkeypatch)
    exporter = InMemorySpanExporter()
    assert attach_span_exporter_for_tests(exporter)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            with start_span("cs.test.boom"):
                raise RuntimeError("boom")
        spans = [s for s in exporter.get_finished_spans() if s.name == "cs.test.boom"]
        assert spans
        assert spans[0].status.status_code == StatusCode.ERROR
        assert any(event.name == "exception" for event in spans[0].events)
    finally:
        shutdown_otel()
        reset_otel_for_tests(app)


def test_retrieve_span_is_child_of_rag_span(monkeypatch) -> None:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from app.agents.rag import retrieve_course_docs
    from app.observability.otel import attach_span_exporter_for_tests, start_span

    _enable_sdk(monkeypatch)
    exporter = InMemorySpanExporter()
    assert attach_span_exporter_for_tests(exporter)
    try:
        with start_span("cs.graph.rag", **{"cs.node": "rag"}):
            retrieve_course_docs("Python入门课包含哪些内容？", tenant_id="demo")
        by_name = {span.name: span for span in exporter.get_finished_spans()}
        assert "cs.graph.rag" in by_name
        assert "cs.rag.retrieve" in by_name
        retrieve = by_name["cs.rag.retrieve"]
        rag = by_name["cs.graph.rag"]
        assert retrieve.parent is not None
        assert retrieve.parent.span_id == rag.context.span_id
        assert retrieve.context.trace_id == rag.context.trace_id
        assert retrieve.attributes.get("cs.rag.empty") in {"true", "false"}
    finally:
        shutdown_otel()
        reset_otel_for_tests(app)


def test_course_chat_emits_graph_and_retrieve_spans(monkeypatch) -> None:
    """课程对话：LangGraph 节点与 FAISS 检索挂在同一条 HTTP trace 下。"""
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from app.observability import REQUEST_ID_HEADER
    from app.observability.otel import attach_span_exporter_for_tests

    reset_otel_for_tests(app)
    settings = _settings(
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        langfuse_host="http://127.0.0.1:3000",
        deepseek_api_key="",
    )
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    monkeypatch.setattr("app.agents.rag.get_settings", lambda: settings)
    monkeypatch.setattr("app.agents.intent.get_settings", lambda: settings)
    monkeypatch.setattr("app.graph.nodes.get_settings", lambda: settings)
    monkeypatch.setattr("app.graph.runner.get_settings", lambda: settings)
    monkeypatch.setattr("app.llm.get_settings", lambda: settings)
    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
        _NoopSpanExporter,
    )
    exporter = InMemorySpanExporter()
    try:
        with TestClient(app) as client:
            assert attach_span_exporter_for_tests(exporter)
            response = client.post(
                "/chat",
                json={"message": "Python入门课包含哪些内容？"},
                headers={REQUEST_ID_HEADER: "req-span-course"},
            )
            assert response.status_code == 200
        by_name = {span.name: span for span in exporter.get_finished_spans()}
        for name in (
            "cs.graph.security",
            "cs.graph.router",
            "cs.graph.rag",
            "cs.rag.retrieve",
            "cs.graph.respond",
        ):
            assert name in by_name, name
        retrieve = by_name["cs.rag.retrieve"]
        rag = by_name["cs.graph.rag"]
        assert retrieve.parent is not None
        assert retrieve.parent.span_id == rag.context.span_id
        cs_spans = [s for s in exporter.get_finished_spans() if s.name.startswith("cs.")]
        trace_ids = {s.context.trace_id for s in cs_spans}
        assert len(trace_ids) == 1
        assert rag.attributes.get("cs.request_id") == "req-span-course"
    finally:
        shutdown_otel()
        reset_otel_for_tests(app)
