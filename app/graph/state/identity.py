"""本轮身份与对话载体：谁在问、问了什么、消息列表。"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class IdentityState(TypedDict, total=False):
    """租户/用户/会话标识与本轮输入。新增身份字段写这里。"""

    query: str  # 本轮用户输入；安全检查后可能被脱敏改写
    user_id: str  # 当前用户标识
    tenant_id: str  # 租户 ID，隔离知识库与订单数据
    session_id: str  # 会话 ID；与 tenant_id、user_id 一起构成 thread_id
    messages: Annotated[list[BaseMessage], add_messages]  # 对话历史，经 add_messages 归约并按窗口裁剪
