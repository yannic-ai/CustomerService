"""会话线程键：tenant + user + session 三维隔离。"""

from __future__ import annotations


def _thread_part(value: str, fallback: str) -> str:
    """规范化 thread_id 片段，避免冒号拆键。"""
    text = (value or "").strip() or fallback
    return text.replace(":", "_")


def make_thread_id(tenant_id: str, session_id: str, user_id: str = "anonymous") -> str:
    """租户 + 用户 + 会话 三维隔离的线程键。

    Redis Cluster Hash Tag 注意：
        短期会话的所有相关 key 都把完整 thread_id 作为 hash tag，
        例如 ``cs:session:{tenant:user:session}`` / ``cs:session:lock:{tenant:user:session}``。
        这确保同一会话的 checkpoint + 互斥锁 路由到同一节点，
        便于未来 pipeline/事务/批量操作。

        相对地，长期记忆 key 使用 ``{tenant:user}`` 作为 hash tag
        （同一用户跨多个会话共享同一 slot），见 RedisUserStore._hash_tag_for_namespace。
    """
    tenant = _thread_part(tenant_id, "demo")
    user = _thread_part(user_id, "anonymous")
    session = _thread_part(session_id, "session")
    return f"{tenant}:{user}:{session}"
