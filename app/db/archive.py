"""会话与用户档案冷存储：Redis 热缓存 miss 时从 MySQL/SQLite 回源。"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import select

from app.db.models import SessionArchive, UserProfile
from app.db.session import session_scope

logger = logging.getLogger("cs.archive")


def encode_messages(messages: list[BaseMessage] | None) -> list[dict[str, str]]:
    """把对话消息压成可演进 JSON（仅 human / ai）。"""
    encoded: list[dict[str, str]] = []
    for message in messages or []:
        if isinstance(message, HumanMessage):
            role = "human"
        elif isinstance(message, AIMessage):
            role = "ai"
        else:
            continue
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            content = str(content)
        encoded.append({"role": role, "content": content})
    return encoded


def decode_messages(raw: list[Any] | None) -> list[BaseMessage]:
    """从归档 JSON 还原 HumanMessage / AIMessage。"""
    messages: list[BaseMessage] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        content = item.get("content")
        text = content if isinstance(content, str) else str(content or "")
        if role == "human":
            messages.append(HumanMessage(content=text))
        elif role == "ai":
            messages.append(AIMessage(content=text))
    return messages


def build_session_state_payload(state: dict[str, Any]) -> dict[str, Any]:
    """从 CSState 抽取可归档的公共字段（identity.messages + memory）。"""
    from app.graph.state.views import MemoryView

    memory = MemoryView.from_state(state)
    payload: dict[str, Any] = {
        "messages": encode_messages(state.get("messages")),
        **memory.as_state(),
    }
    return payload


def upsert_user_profile(tenant_id: str, user_id: str, profile: dict[str, Any]) -> None:
    """写入或更新用户档案；失败只打日志。"""
    tenant = (tenant_id or "").strip() or "demo"
    user = (user_id or "").strip()
    if not user:
        return
    try:
        blob = json.dumps(profile, ensure_ascii=False)
        with session_scope() as session:
            row = session.scalar(
                select(UserProfile).where(
                    UserProfile.tenant_id == tenant,
                    UserProfile.user_id == user,
                )
            )
            if row is None:
                session.add(
                    UserProfile(tenant_id=tenant, user_id=user, profile_json=blob)
                )
            else:
                row.profile_json = blob
            session.commit()
    except Exception:
        logger.exception(
            "写入 user_profiles 失败 tenant=%s user=%s", tenant, user
        )


def load_user_profile(tenant_id: str, user_id: str) -> dict[str, Any] | None:
    """读取用户档案；失败或缺失返回 None。"""
    tenant = (tenant_id or "").strip() or "demo"
    user = (user_id or "").strip()
    if not user:
        return None
    try:
        with session_scope() as session:
            row = session.scalar(
                select(UserProfile).where(
                    UserProfile.tenant_id == tenant,
                    UserProfile.user_id == user,
                )
            )
            if row is None or not row.profile_json:
                return None
            data = json.loads(row.profile_json)
            return data if isinstance(data, dict) else None
    except Exception:
        logger.exception(
            "读取 user_profiles 失败 tenant=%s user=%s", tenant, user
        )
        return None


def upsert_session_archive(
    tenant_id: str,
    user_id: str,
    session_id: str,
    *,
    thread_id: str = "",
    state: dict[str, Any] | None = None,
) -> None:
    """写入或更新会话归档；失败只打日志。"""
    tenant = (tenant_id or "").strip() or "demo"
    user = (user_id or "").strip() or "anonymous"
    sid = (session_id or "").strip()
    if not sid:
        return
    payload = build_session_state_payload(state or {})
    try:
        blob = json.dumps(payload, ensure_ascii=False)
        with session_scope() as session:
            row = session.scalar(
                select(SessionArchive).where(
                    SessionArchive.tenant_id == tenant,
                    SessionArchive.user_id == user,
                    SessionArchive.session_id == sid,
                )
            )
            if row is None:
                session.add(
                    SessionArchive(
                        tenant_id=tenant,
                        user_id=user,
                        session_id=sid,
                        thread_id=thread_id or "",
                        state_json=blob,
                    )
                )
            else:
                row.state_json = blob
                if thread_id:
                    row.thread_id = thread_id
            session.commit()
    except Exception:
        logger.exception(
            "写入 session_archives 失败 tenant=%s user=%s session=%s",
            tenant,
            user,
            sid,
        )


def load_session_archive(
    tenant_id: str,
    user_id: str,
    session_id: str,
) -> dict[str, Any] | None:
    """读取会话归档；失败或缺失返回 None。"""
    tenant = (tenant_id or "").strip() or "demo"
    user = (user_id or "").strip() or "anonymous"
    sid = (session_id or "").strip()
    if not sid:
        return None
    try:
        with session_scope() as session:
            row = session.scalar(
                select(SessionArchive).where(
                    SessionArchive.tenant_id == tenant,
                    SessionArchive.user_id == user,
                    SessionArchive.session_id == sid,
                )
            )
            if row is None or not row.state_json:
                return None
            data = json.loads(row.state_json)
            return data if isinstance(data, dict) else None
    except Exception:
        logger.exception(
            "读取 session_archives 失败 tenant=%s user=%s session=%s",
            tenant,
            user,
            sid,
        )
        return None


def archive_to_graph_inputs(archived: dict[str, Any]) -> dict[str, Any]:
    """把归档记录转成可并入 graph inputs 的字段（不含本轮 query）。"""
    from app.graph.state.views import MemoryView

    return {
        "messages": decode_messages(archived.get("messages")),
        **MemoryView.from_state(archived).as_state(),
    }
