"""能力注册表：计划查 tool 名，Supervisor 按 id 执行，ReACT 取工具列表。"""

from __future__ import annotations

from typing import Any

from app.capabilities.types import Capability, CapabilitySpec

_REGISTRY: dict[str, Capability] = {}
_FALLBACK_TOOL_NAME = {
    "order_query": "tool_order_query",
    "course_retrieve": "retrieve_course_knowledge",
}


def register(capability: Capability) -> Capability:
    """注册或覆盖一个能力。返回自身便于 ``ORDER = register(OrderCapability())``。"""
    spec = capability.spec
    if not spec.id:
        raise ValueError("CapabilitySpec.id 不能为空")
    _REGISTRY[spec.id] = capability
    return capability


def get_capability(capability_id: str) -> Capability:
    """按 id 取能力；未注册则抛 KeyError。"""
    try:
        return _REGISTRY[capability_id]
    except KeyError as exc:
        raise KeyError(f"未注册的能力：{capability_id}") from exc


def list_capabilities() -> tuple[Capability, ...]:
    """当前已注册能力（注册顺序）。"""
    return tuple(_REGISTRY.values())


def spec_for(capability_id: str) -> CapabilitySpec | None:
    cap = _REGISTRY.get(capability_id)
    return None if cap is None else cap.spec


def tool_name_for(capability_id: str) -> str:
    """计划 ``tools_to_call`` 用的可观察工具名。"""
    spec = spec_for(capability_id)
    if spec is not None:
        return spec.tool_name
    return _FALLBACK_TOOL_NAME.get(capability_id, capability_id)


def langchain_tools() -> list[Any]:
    """ReACT 挂载的全部工具；新增能力注册后自动出现。"""
    return [cap.as_langchain_tool() for cap in _REGISTRY.values()]


def unregister(capability_id: str) -> None:
    """测试用：从注册表移除一个能力。"""
    _REGISTRY.pop(capability_id, None)


def reset_registry() -> None:
    """测试用：清空注册表。随后需重新 import / register 内置能力。"""
    _REGISTRY.clear()
