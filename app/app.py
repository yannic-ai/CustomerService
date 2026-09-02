"""FastAPI 应用入口，提供 `/chat` 对话接口。"""

from contextlib import asynccontextmanager
from html import escape
import json
import logging
import time
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.concurrency import ChatBusyError, achat, achat_stream, shutdown_concurrency
from app.llm_gateway import TenantQuotaExceededError
from app.observability import (
    REQUEST_ID_HEADER,
    check_liveness,
    check_readiness,
    configure_logging,
    flush_langfuse,
    inc_chat_busy,
    inc_session_busy,
    log_event,
    new_request_id,
    render_prometheus,
    reset_request_id,
    set_request_id,
)
from app.observability.metrics import inc_http, observe_http
from app.observability.otel import (
    inject_trace_context,
    instrument_fastapi,
    set_span_attributes,
    setup_otel,
    shutdown_otel,
)
from app.session_cache import SessionBusyError
from app.db.seed import seed_if_empty
from app.db.session import get_session, init_db
from app.graph import get_graph, graph_mermaid
from app.rag.ingest_api import router as admin_knowledge_router
from app.rag.vectorstore import IndexUnavailableError, load_faiss_index, require_index_for_serve
from app.schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    GreetResponse,
    OrderReport,
)
from app.tenancy import TENANT_HEADER, USER_HEADER, get_tenant_id, get_user_id, resolve_request_user
from app.capabilities.order import query_order

STATIC_DIR = Path(__file__).resolve().parent / "static"
logger = logging.getLogger("cs.app")
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
_SKIP_HTTP_LOG = frozenset({"/health", "/ready", "/metrics"})


def _metric_path(path: str) -> str:
    """把高基数路径压成模板，避免 metrics label 爆炸。"""
    if path.startswith("/api/v1/orders/"):
        return "/api/v1/orders/{order_no}"
    if path.startswith("/static/"):
        return "/static"
    return path or "/"

DEFAULT_GREET = GreetResponse(
    greeting="您好，我是课程与订单智能客服，可以从下方选项开始，或直接输入问题。",
    options=[
        {"label": "课程咨询", "message": "Python入门课包含哪些内容？"},
        {"label": "订单查询", "message": "查询订单#20251114001的退款进度"},
        {"label": "转人工", "message": "转人工"},
    ],
)


def _render_graph_html(mermaid: str) -> str:
    """用 mermaid.js 渲染客服主图。"""
    body = escape(mermaid)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>客服 LangGraph 流程图</title>
  <style>
    body {{ margin: 0; font-family: "PingFang SC", "Noto Sans SC", sans-serif; background: #f4f6fb; color: #111827; }}
    main {{ max-width: 960px; margin: 32px auto; padding: 0 16px 48px; }}
    h1 {{ font-size: 22px; margin: 0 0 8px; }}
    p {{ color: #6b7280; }}
    .card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 16px; padding: 24px; overflow: auto; }}
  </style>
</head>
<body>
  <main>
    <h1>智能客服 LangGraph</h1>
    <p>安全检查 → 路由 → RAG / 订单工具 / ReACT → 输出脱敏</p>
    <div class="card">
      <pre class="mermaid">{body}</pre>
    </div>
  </main>
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
    mermaid.initialize({{ startOnLoad: true, theme: "neutral" }});
  </script>
</body>
</html>
"""


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时校验 Redis（若已配置）、建表、空库灌种，并加载已有 FAISS 索引。

    不在启动路径上 ingest / 全量重建；缺索引或 embedding 不兼容则拒绝启动。
    """
    from app.config import get_settings
    from app.session_cache import ensure_configured_redis

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    from app.observability.metrics import configure_metrics_multiprocess_from_settings

    configure_metrics_multiprocess_from_settings(settings)
    # OTel 须在 DB / Redis 客户端创建前初始化，才能挂上自动埋点
    setup_otel()
    instrument_fastapi(_app)
    ensure_configured_redis()
    from app.runtime import get_app_context

    get_app_context()
    init_db()
    from app.db.session import session_scope

    with session_scope() as session:
        seed_if_empty(session)
    require_index_for_serve()
    load_faiss_index()
    yield
    from app.observability.metrics import mark_metrics_process_dead

    mark_metrics_process_dead()
    flush_langfuse()
    shutdown_otel()
    shutdown_concurrency()


app = FastAPI(title="智能客服系统", version="0.1.0", lifespan=lifespan)
app.include_router(admin_knowledge_router)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ObservabilityMiddleware:
    """纯 ASGI：保留 OTel contextvars（BaseHTTPMiddleware 会丢）。

    写入 request_id，记录 HTTP 计数与耗时（SSE 记到 response.start / 首包），回写 traceparent。
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        request_id = new_request_id(headers.get(REQUEST_ID_HEADER))
        token = set_request_id(request_id)
        path = scope.get("path") or "/"
        method = scope.get("method") or "GET"
        tenant = (headers.get(TENANT_HEADER) or "").strip()
        user = (headers.get(USER_HEADER) or "").strip()
        set_span_attributes(
            **{
                "cs.request_id": request_id,
                "tenant.id": tenant or None,
                "user.id": user or None,
                "http.route": _metric_path(path),
            }
        )
        started = time.monotonic()
        status = 500
        recorded = False

        async def send_wrapper(message: Message) -> None:
            nonlocal status, recorded
            if message["type"] == "http.response.start":
                status = int(message.get("status", 500))
                out = MutableHeaders(scope=message)
                out[REQUEST_ID_HEADER] = request_id
                inject_trace_context(out)
                if not recorded:
                    recorded = True
                    elapsed = time.monotonic() - started
                    metric_path = _metric_path(path)
                    inc_http(method, metric_path, status)
                    observe_http(method, metric_path, status, elapsed)
                    if path not in _SKIP_HTTP_LOG:
                        log_event(
                            logger,
                            "http_request",
                            method=method,
                            path=metric_path,
                            status=status,
                            elapsed_ms=int(elapsed * 1000),
                        )
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            if not recorded:
                elapsed = time.monotonic() - started
                metric_path = _metric_path(path)
                inc_http(method, metric_path, status)
                observe_http(method, metric_path, status, elapsed)
                if path not in _SKIP_HTTP_LOG:
                    log_event(
                        logger,
                        "http_request",
                        method=method,
                        path=metric_path,
                        status=status,
                        elapsed_ms=int(elapsed * 1000),
                    )
            reset_request_id(token)


# add_middleware 后添加的更靠外；OTel 在 lifespan 里再包一层最外
app.add_middleware(ObservabilityMiddleware)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """返回演示聊天页。"""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "<h1>Customer Service API</h1><p>POST /chat（SSE）</p>"


@app.get("/health")
def health() -> dict[str, str]:
    """进程探活（liveness）。依赖检查见 ``GET /ready``。"""
    return check_liveness()


@app.get("/ready")
def ready() -> JSONResponse:
    """就绪检查：数据库、已配置时的 Redis、FAISS 索引文件。"""
    payload = check_readiness()
    status = 200 if payload["status"] == "ok" else 503
    return JSONResponse(payload, status_code=status)


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    """Prometheus 文本指标。"""
    return PlainTextResponse(render_prometheus(), media_type="text/plain; version=0.0.4")


@app.get("/greet", response_model=GreetResponse)
def greet(_tenant_id: str = Depends(get_tenant_id)) -> GreetResponse:
    """开场白：问候语与预置快捷问题。"""
    return DEFAULT_GREET


@app.get("/graph", response_class=HTMLResponse)
def graph_view() -> str:
    """在浏览器中展示 LangGraph 流程图。"""
    mermaid = graph_mermaid(get_graph())
    return _render_graph_html(mermaid)


@app.get("/graph.mmd")
def graph_mermaid_source() -> dict[str, str]:
    """返回流程图 Mermaid 源码。"""
    return {"mermaid": graph_mermaid(get_graph())}


def _sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _chat_user_id(
    payload: ChatRequest,
    x_user_id: str | None,
) -> str:
    """Header ``X-User-Id`` 优先于 JSON ``user_id``。"""
    return resolve_request_user(x_user_id, payload.user_id)


@app.post("/chat")
async def chat_api(
    payload: ChatRequest,
    tenant_id: str = Depends(get_tenant_id),
    x_user_id: Annotated[str | None, Header(alias=USER_HEADER)] = None,
) -> StreamingResponse:
    """对话接口：SSE 推送节点进度、中间答案与最终 ``done`` 事件。"""
    stream = achat_stream(
        payload.message,
        user_id=_chat_user_id(payload, x_user_id),
        tenant_id=tenant_id,
        session_id=payload.session_id,
    )
    try:
        first = await anext(stream)
    except ChatBusyError as exc:
        inc_chat_busy(exc.reason)
        log_event(logger, "chat_busy", reason=exc.reason)
        raise HTTPException(status_code=429, detail="对话繁忙，请稍后重试") from exc
    except SessionBusyError as exc:
        inc_session_busy()
        log_event(logger, "session_busy")
        raise HTTPException(status_code=409, detail="同一会话正在处理，请稍后重试") from exc
    except TenantQuotaExceededError as exc:
        log_event(logger, "llm_quota_exceeded")
        raise HTTPException(status_code=429, detail="今日大模型额度已用完，请明日再试") from exc

    async def _body():
        yield _sse(*first)
        try:
            async for event, data in stream:
                yield _sse(event, data)
        except Exception:
            log_event(logger, "sse_error")
            logger.exception("SSE 对话中断")
            yield _sse("error", {"detail": "服务内部错误"})

    return StreamingResponse(
        _body(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback_api(payload: FeedbackRequest) -> FeedbackResponse:
    """用户赞踩：记 Prometheus，并挂到 Langfuse ``user_feedback``（有 trace_id 时）。"""
    from app.observability.langfuse import score_user_feedback

    if payload.value not in (0, 1):
        raise HTTPException(status_code=422, detail="value 只能是 0（踩）或 1（赞）")
    if payload.value == 0:
        from app.evals.harvest import record_feedback_down

        record_feedback_down(payload.trace_id, comment=payload.comment)
    result = score_user_feedback(
        trace_id=payload.trace_id,
        value=float(payload.value),
        comment=payload.comment,
        request_id=payload.request_id,
        session_id=payload.session_id,
    )
    return FeedbackResponse(
        status="ok",
        recorded=bool(result.get("recorded", True)),
        langfuse=bool(result.get("langfuse")),
        reason=result.get("reason"),
    )


@app.post("/api/v1/chat", response_model=ChatResponse, include_in_schema=False)
async def chat_api_compat(
    payload: ChatRequest,
    tenant_id: str = Depends(get_tenant_id),
    x_user_id: Annotated[str | None, Header(alias=USER_HEADER)] = None,
) -> ChatResponse:
    """兼容旧路径 `/api/v1/chat`。"""
    try:
        return await achat(
            payload.message,
            user_id=_chat_user_id(payload, x_user_id),
            tenant_id=tenant_id,
            session_id=payload.session_id,
        )
    except ChatBusyError as exc:
        inc_chat_busy(exc.reason)
        log_event(logger, "chat_busy", reason=exc.reason)
        raise HTTPException(status_code=429, detail="对话繁忙，请稍后重试") from exc
    except TenantQuotaExceededError as exc:
        log_event(logger, "llm_quota_exceeded")
        raise HTTPException(status_code=429, detail="今日大模型额度已用完，请明日再试") from exc
    except SessionBusyError as exc:
        inc_session_busy()
        log_event(logger, "session_busy")
        raise HTTPException(status_code=409, detail="同一会话正在处理，请稍后重试") from exc


@app.get("/api/v1/orders/{order_no}", response_model=OrderReport)
def api_order(
    order_no: str,
    persist: bool = True,
    tenant_id: str = Depends(get_tenant_id),
    user_id: str = Depends(get_user_id),
    _db: Session = Depends(get_session),
) -> OrderReport:
    """订单进度 API，与订单能力共用同一套查询逻辑（按租户与下单人隔离）。"""
    try:
        return query_order(
            "",
            order_no,
            tenant_id=tenant_id,
            user_id=user_id,
            persist=persist,
            require_explicit_order_no=True,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="订单查询超时") from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def run() -> None:
    """以 uvicorn 启动 FastAPI 应用。"""
    import uvicorn

    from app.config import get_settings
    from app.session_cache import RedisUnavailableError, require_redis_for_serve

    try:
        require_redis_for_serve()
        require_index_for_serve()
    except (RedisUnavailableError, IndexUnavailableError) as exc:
        logging.getLogger("cs.serve").error("%s", exc)
        raise SystemExit(1) from exc
    settings = get_settings()
    workers = max(1, settings.uvicorn_workers)
    from app.observability.metrics import configure_metrics_multiprocess_from_settings

    configure_metrics_multiprocess_from_settings(settings)
    uvicorn.run(
        "app.app:app",
        host=settings.app_host,
        port=settings.app_port,
        workers=workers,
        reload=False,
    )


if __name__ == "__main__":
    run()
