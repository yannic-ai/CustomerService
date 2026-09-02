"""DeepSeek 聊天模型工厂，经 LLM Gateway 包装。"""

from app.config import get_settings
from app.llm_gateway import CallPolicy, GatewayChatModel, wrap_chat_model


def get_chat_model(
    temperature: float = 0.1,
    *,
    usage_tag: str = "default",
    cache_enabled: bool | None = None,
    enforce_quota: bool = True,
) -> GatewayChatModel:
    """创建经 Gateway 包装的 DeepSeek Chat 模型。

    Args:
        temperature: 采样温度，路由/结构化输出建议接近 0。
        usage_tag: 记账标签（intent / rag / react）。
        cache_enabled: 是否允许短响应缓存；None 时按 tag 默认（react 关闭）。
        enforce_quota: 是否执行租户日限额。

    Raises:
        RuntimeError: 未配置 `DEEPSEEK_API_KEY`。
    """
    settings = get_settings()
    if not settings.llm_enabled:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，无法调用大模型。")
    if cache_enabled is None:
        cache_enabled = usage_tag != "react" or settings.llm_cache_allow_react
    from langchain_deepseek import ChatDeepSeek

    inner = ChatDeepSeek(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        api_base=settings.deepseek_api_base,
        temperature=temperature,
        max_retries=0,
        timeout=settings.llm_timeout_seconds,
        extra_body={"thinking": {"type": "disabled"}},
    )
    policy = CallPolicy(
        usage_tag=usage_tag,
        cache_enabled=bool(cache_enabled),
        enforce_quota=enforce_quota,
        temperature=temperature,
        model=settings.deepseek_model,
    )
    return wrap_chat_model(inner, policy)
