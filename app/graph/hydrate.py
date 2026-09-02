"""checkpoint miss 时从数据库回填会话公共状态。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.db.archive import archive_to_graph_inputs, load_session_archive

logger = logging.getLogger("cs.graph")


def checkpoint_exists(thread_id: str) -> bool:
    """当前 checkpointer 是否已有该 thread 的热状态。"""
    from app.runtime import get_app_context

    try:
        return get_app_context().checkpointer.get_tuple({"configurable": {"thread_id": thread_id}}) is not None
    except Exception:
        logger.exception("读取 checkpoint 失败 thread_id=%s", thread_id)
        return False


async def acheckpoint_exists(thread_id: str) -> bool:
    """异步探测热 checkpoint：优先 ``aget_tuple``，否则把同步 get 卸到线程。"""
    from app.runtime import get_app_context

    try:
        checkpointer = get_app_context().checkpointer
        aget = getattr(checkpointer, "aget_tuple", None)
        if callable(aget):
            result = await aget({"configurable": {"thread_id": thread_id}})
        else:
            result = await asyncio.to_thread(
                checkpointer.get_tuple,
                {"configurable": {"thread_id": thread_id}},
            )
        return result is not None
    except Exception:
        logger.exception("读取 checkpoint 失败 thread_id=%s", thread_id)
        return False


def _merge_archive(inputs: dict[str, Any], archived: Any, thread_id: str) -> None:
    if not archived:
        return
    inputs.update(archive_to_graph_inputs(archived))
    logger.info(
        "会话从 DB 回填 thread_id=%s messages=%s",
        thread_id,
        len(inputs.get("messages") or []),
    )


def hydrate_inputs_from_archive(
    inputs: dict[str, Any],
    *,
    tenant_id: str,
    user_id: str,
    session_id: str,
    thread_id: str,
) -> None:
    """checkpoint miss 时从 MySQL 回填公共状态；已有热状态则不合并，避免消息重复。"""
    if checkpoint_exists(thread_id):
        return
    try:
        archived = load_session_archive(tenant_id, user_id, session_id)
    except Exception:
        logger.exception(
            "读取会话归档失败，按新会话继续 thread_id=%s",
            thread_id,
        )
        return
    _merge_archive(inputs, archived, thread_id)


async def ahydrate_inputs_from_archive(
    inputs: dict[str, Any],
    *,
    tenant_id: str,
    user_id: str,
    session_id: str,
    thread_id: str,
) -> None:
    """异步 hydrate：checkpoint 走 aget_tuple / to_thread，归档走 to_thread。"""
    if await acheckpoint_exists(thread_id):
        return
    try:
        archived = await asyncio.to_thread(load_session_archive, tenant_id, user_id, session_id)
    except Exception:
        logger.exception(
            "读取会话归档失败，按新会话继续 thread_id=%s",
            thread_id,
        )
        return
    _merge_archive(inputs, archived, thread_id)
