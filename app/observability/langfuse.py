"""Langfuse 轨迹：未配密钥时 noop，不阻断对话。"""

from __future__ import annotations

import logging
import os
from typing import Any

from app.config import get_settings
from app.observability.context import get_request_id

logger = logging.getLogger("cs.langfuse")

_CLIENT_READY = False


def _ensure_env() -> bool:
    """把 Settings 同步到环境变量（SDK 读 env），两把钥匙齐全才启用。"""
    global _CLIENT_READY
    settings = get_settings()
    if not settings.langfuse_enabled:
        return False
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key.strip())
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key.strip())
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host.strip() or "http://127.0.0.1:3000")
    # 新版 SDK 也认 LANGFUSE_BASE_URL
    os.environ.setdefault("LANGFUSE_BASE_URL", os.environ["LANGFUSE_HOST"])
    _CLIENT_READY = True
    return True


def langfuse_callbacks() -> list[Any]:
    """返回 Langfuse LangChain CallbackHandler；未启用或导入失败时为空列表。"""
    if not _ensure_env():
        return []
    try:
        from langfuse.langchain import CallbackHandler
    except Exception as exc:
        logger.warning("Langfuse CallbackHandler 不可用: %s", exc)
        return []
    try:
        return [CallbackHandler()]
    except Exception as exc:
        logger.warning("创建 Langfuse CallbackHandler 失败: %s", exc)
        return []


def trace_metadata(
    *,
    user_id: str,
    tenant_id: str,
    session_id: str,
    thread_id: str,
) -> dict[str, Any]:
    """写入 LangChain config.metadata，供 Langfuse 识别 user/session/tags。"""
    meta: dict[str, Any] = {
        "request_id": get_request_id(),
        "tenant_id": tenant_id,
        "session_id": session_id,
        "thread_id": thread_id,
    }
    if not get_settings().langfuse_enabled:
        return meta
    meta["langfuse_user_id"] = user_id or "anonymous"
    meta["langfuse_session_id"] = thread_id
    meta["langfuse_tags"] = [f"tenant:{tenant_id}"]
    return meta


def score_turn(
    *,
    blocked: bool,
    intent: str,
    rag_empty: bool = False,
    outcome: str = "ok",
) -> None:
    """给当前 trace 打轻量自动 score；失败只记日志。

    自动信号：blocked / handoff / rag_empty / turn_ok（outcome==ok）。
    用户赞踩走 ``score_user_feedback``，不在这里写。
    """
    if not _ensure_env():
        return
    try:
        from langfuse import get_client

        client = get_client()
        client.score_current_trace(name="blocked", value=1.0 if blocked else 0.0)
        client.score_current_trace(name="handoff", value=1.0 if intent == "handoff" else 0.0)
        client.score_current_trace(name="rag_empty", value=1.0 if rag_empty else 0.0)
        client.score_current_trace(name="turn_ok", value=1.0 if outcome == "ok" else 0.0)
    except Exception as exc:
        logger.debug("Langfuse score 跳过: %s", exc)


def score_user_feedback(
    *,
    trace_id: str,
    value: float,
    comment: str | None = None,
    request_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """事后给指定 trace 打 ``user_feedback``（1=赞，0=踩）。

    未配 Langfuse 时仍记本地日志与 Prometheus，返回 ``langfuse=False``。
    """
    from app.observability.logging import log_event
    from app.observability.metrics import inc_user_feedback

    tid = (trace_id or "").strip()
    score = 1.0 if float(value) >= 0.5 else 0.0
    label = "up" if score >= 0.5 else "down"
    inc_user_feedback(label)
    log_event(
        logger,
        "user_feedback",
        trace_id=tid,
        value=score,
        label=label,
        request_id=(request_id or "").strip() or None,
        session_id=(session_id or "").strip() or None,
        comment=(comment or "")[:200] or None,
    )
    if not tid and not (session_id or "").strip():
        return {"recorded": True, "langfuse": False, "reason": "missing_trace_id"}
    if not _ensure_env():
        return {"recorded": True, "langfuse": False, "reason": "langfuse_disabled"}
    try:
        from langfuse import get_client

        meta: dict[str, Any] = {}
        if request_id:
            meta["request_id"] = request_id.strip()
        if session_id:
            meta["session_id"] = session_id.strip()
        kwargs: dict[str, Any] = {
            "name": "user_feedback",
            "value": score,
            "comment": (comment or "").strip() or None,
            "metadata": meta or None,
            "data_type": "NUMERIC",
        }
        if tid:
            kwargs["trace_id"] = tid
        if session_id:
            kwargs["session_id"] = session_id.strip()
        get_client().create_score(**kwargs)
        return {"recorded": True, "langfuse": True}
    except Exception as exc:
        logger.debug("Langfuse user_feedback 跳过: %s", exc)
        return {"recorded": True, "langfuse": False, "reason": "langfuse_error"}


# Langfuse Prompt Management 名称（未创建时用本地 fallback，改文案不必发版）
PROMPT_INTENT = "cs.intent"
PROMPT_REACT = "cs.react"
PROMPT_RAG_MARKDOWN = "cs.rag_markdown"
PROMPT_RAG_STRUCTURED = "cs.rag_structured"
PROMPT_SUMMARY = "cs.summary"


def get_text_prompt(name: str, fallback: str, *, label: str | None = None) -> str:
    """从 Langfuse 拉 text prompt；未启用 / 失败 / 空内容时回退本地常量。

    Langfuse UI 创建同名 Prompt 并打 ``production``（或指定 label）后即可热更新。
    SDK 自带短缓存（默认 60s）。
    """
    text = (fallback or "").strip()
    if not name or not _ensure_env():
        return text
    try:
        from langfuse import get_client

        client = get_client()
        kwargs: dict[str, Any] = {
            "fallback": fallback,
            "cache_ttl_seconds": 60,
            "type": "text",
        }
        if label:
            kwargs["label"] = label
        prompt = client.get_prompt(name, **kwargs)
        compiled = prompt.compile() if hasattr(prompt, "compile") else getattr(prompt, "prompt", "")
        out = (compiled or "").strip()
        return out or text
    except Exception as exc:
        logger.debug("Langfuse prompt %s 回退本地: %s", name, exc)
        return text


def flush_langfuse() -> None:
    """进程退出前刷出缓冲。"""
    if not get_settings().langfuse_enabled:
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as exc:
        logger.debug("Langfuse flush 跳过: %s", exc)
