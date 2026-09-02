"""多租户上下文：解析与校验 X-Tenant-Id。"""

from __future__ import annotations

import re
from contextvars import ContextVar
from typing import Annotated

from fastapi import Header, HTTPException

from app.config import get_settings

TENANT_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
TENANT_HEADER = "X-Tenant-Id"
USER_HEADER = "X-User-Id"

_current_tenant: ContextVar[str] = ContextVar("current_tenant", default="demo")
_current_user: ContextVar[str] = ContextVar("current_user", default="anonymous")


def get_current_tenant() -> str:
    """读取当前请求/任务的租户 ID。"""
    return _current_tenant.get()


def set_current_tenant(tenant_id: str) -> None:
    """设置当前请求/任务的租户 ID（供 Tool / RAG 读取）。"""
    _current_tenant.set(normalize_tenant_id(tenant_id))


def get_current_user() -> str:
    """读取当前请求/任务的用户 ID。"""
    return _current_user.get() or "anonymous"


def set_current_user(user_id: str | None) -> None:
    """设置当前请求/任务的用户 ID（供订单工具按下单人过滤）。"""
    _current_user.set((user_id or "").strip() or "anonymous")


def normalize_tenant_id(raw: str | None) -> str:
    """校验租户 ID；空值回落到默认租户，非法格式抛出 ValueError。"""
    settings = get_settings()
    value = (raw or "").strip() or settings.default_tenant
    if not TENANT_RE.fullmatch(value):
        raise ValueError(
            f"非法租户 ID：{value!r}，仅允许字母、数字、下划线与连字符，长度 1-64"
        )
    return value


def get_tenant_id(
    x_tenant_id: Annotated[str | None, Header(alias=TENANT_HEADER)] = None,
) -> str:
    """FastAPI 依赖：从 Header 读取租户，缺省为 default_tenant。"""
    try:
        tenant = normalize_tenant_id(x_tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    set_current_tenant(tenant)
    return tenant


def resolve_request_user(*candidates: str | None) -> str:
    """按优先级取第一个非空用户 ID，并写入请求上下文；都空则 anonymous。"""
    for raw in candidates:
        value = (raw or "").strip()
        if value:
            set_current_user(value)
            return value
    set_current_user("anonymous")
    return "anonymous"


def get_user_id(
    x_user_id: Annotated[str | None, Header(alias=USER_HEADER)] = None,
) -> str:
    """FastAPI 依赖：从 Header 读取用户，缺省为 anonymous。"""
    return resolve_request_user(x_user_id)
