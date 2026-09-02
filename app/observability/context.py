"""请求级关联 ID，经 contextvars 传到图线程池与工具线程池。"""

from __future__ import annotations

from contextvars import ContextVar, Token
from uuid import uuid4

REQUEST_ID_HEADER = "X-Request-Id"

_current_request_id: ContextVar[str] = ContextVar("current_request_id", default="")


def get_request_id() -> str:
    """当前请求的关联 ID；未设置时为空字符串。"""
    return _current_request_id.get() or ""


def set_request_id(request_id: str) -> Token[str]:
    """写入请求 ID，返回 token 供 reset。"""
    return _current_request_id.set((request_id or "").strip())


def reset_request_id(token: Token[str]) -> None:
    """恢复进入本请求前的 request_id。"""
    _current_request_id.reset(token)


def new_request_id(raw: str | None = None) -> str:
    """沿用客户端传入的 ID，否则生成 32 位 hex。"""
    value = (raw or "").strip()
    return value or uuid4().hex
