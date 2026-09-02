"""把 LangGraph 流式帧转成 SSE 事件。"""

from __future__ import annotations

from typing import Any

from app.graph.state import NODE_STATUS

_STREAM_SKIP_KEYS = {"messages"}


def merge_public(accumulated: dict[str, Any], payload: dict[str, Any]) -> None:
    """把节点公开字段合并进累积状态；跳过 messages 以免 SSE 膨胀。"""
    for key, value in payload.items():
        if key in _STREAM_SKIP_KEYS:
            continue
        accumulated[key] = value


def events_from_custom(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    """把 LangGraph custom 流转成 SSE token 事件。"""
    if not isinstance(payload, dict):
        return []
    if payload.get("event") != "token":
        return []
    content = payload.get("content")
    if not isinstance(content, str) or not content:
        return []
    return [("token", {"content": content})]


def events_from_updates(
    chunk: Any,
    accumulated: dict[str, Any],
    nodes: list[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """把 LangGraph ``stream_mode=updates`` 的一帧转成 SSE 事件。"""
    events: list[tuple[str, dict[str, Any]]] = []
    if not isinstance(chunk, dict):
        return events
    for node, payload in chunk.items():
        if nodes is not None:
            nodes.append(str(node))
        events.append(
            (
                "status",
                {
                    "node": node,
                    "message": NODE_STATUS.get(str(node), str(node)),
                },
            )
        )
        if not isinstance(payload, dict):
            continue
        merge_public(accumulated, payload)
        answer = payload.get("answer")
        if answer:
            events.append(
                (
                    "answer",
                    {
                        "answer": answer,
                        "intent": accumulated.get("intent") or "chitchat",
                        "blocked": bool(accumulated.get("blocked")),
                    },
                )
            )
    return events


def events_from_stream_item(
    item: Any,
    accumulated: dict[str, Any],
    nodes: list[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """解析 ``stream_mode=['updates','custom']`` 的 (mode, data) 或旧式 updates dict。"""
    if isinstance(item, tuple) and len(item) == 2:
        mode, payload = item
        if mode == "custom":
            return events_from_custom(payload)
        if mode == "updates":
            return events_from_updates(payload, accumulated, nodes)
        return []
    return events_from_updates(item, accumulated, nodes)
