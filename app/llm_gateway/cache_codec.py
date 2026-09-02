"""LLM 短缓存编解码：LLK1 + msgpack + 显式 schema，不使用 pickle。"""

from __future__ import annotations

import inspect
from typing import Any

import ormsgpack
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

# 信封：LLK1 + msgpack。无此前缀视为 miss（含 TTL 内旧 pickle），不回退反序列化。
ENVELOPE_MAGIC = b"LLK1"
_SCHEMA_VERSION = 1


def _pydantic_schema(schema_type: Any) -> type[BaseModel] | None:
    """进程内绑定的结构化 schema；不从缓存字节里 import 类型。"""
    if schema_type is None:
        return None
    try:
        if not inspect.isclass(schema_type) or not issubclass(schema_type, BaseModel):
            return None
    except TypeError:
        return None
    if issubclass(schema_type, BaseMessage):
        return None
    return schema_type


def encode_llm_cache(result: Any) -> bytes | None:
    """把 invoke 结果编成 LLK1 + msgpack。无法按 schema 表达时返回 None。"""
    payload = _payload_from_result(result)
    if payload is None:
        return None
    try:
        return ENVELOPE_MAGIC + ormsgpack.packb(payload)
    except Exception:
        return None


def decode_llm_cache(raw: bytes, *, schema_type: Any = None) -> Any | None:
    """解码 LLM 短缓存。非 LLK1 或校验失败返回 None，不走 pickle。"""
    if not raw or not raw.startswith(ENVELOPE_MAGIC):
        return None
    try:
        loaded = ormsgpack.unpackb(raw[len(ENVELOPE_MAGIC) :])
    except Exception:
        return None
    if not isinstance(loaded, dict) or loaded.get("v") != _SCHEMA_VERSION:
        return None
    kind = loaded.get("kind")
    data = loaded.get("data")
    try:
        if kind == "ai_message":
            if not isinstance(data, dict):
                return None
            return AIMessage.model_validate(data)
        if kind == "pydantic":
            schema = _pydantic_schema(schema_type)
            if schema is None or not isinstance(data, dict):
                return None
            return schema.model_validate(data)
        if kind == "json":
            return data
    except Exception:
        return None
    return None


def _payload_from_result(result: Any) -> dict[str, Any] | None:
    if isinstance(result, AIMessage):
        return {
            "v": _SCHEMA_VERSION,
            "kind": "ai_message",
            "data": result.model_dump(mode="json"),
        }
    if isinstance(result, BaseModel):
        return {
            "v": _SCHEMA_VERSION,
            "kind": "pydantic",
            "data": result.model_dump(mode="json"),
        }
    if result is None or isinstance(result, (dict, list, str, int, float, bool)):
        return {"v": _SCHEMA_VERSION, "kind": "json", "data": result}
    return None
