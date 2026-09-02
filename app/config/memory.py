"""会话窗口、滚动摘要、长期记忆与 Redis checkpoint。"""

from pydantic import BaseModel, Field


class MemorySettings(BaseModel):
    """短期会话与长期档案。新增记忆 / 压缩参数只改本文件。"""

    # 会话历史按 token 裁剪；至少保留最近 min 条（与 context_always_keep_recent 对齐）
    history_token_budget: int = Field(default=8000, ge=256)
    history_min_messages: int = Field(default=10, ge=2)
    # token 估算：可指定 tiktoken 编码名；留空则按 deepseek_model 解析，失败回退 cl100k
    history_tokenizer_name: str = ""
    # 对估算结果的乘数校准（DeepSeek 与 cl100k 有偏差时可调，如 0.9 / 1.1）
    history_token_estimate_ratio: float = Field(default=1.0, gt=0, le=5)

    # 分级压缩：消息条数第二触发阈值；score 分级阈值；摘要开关
    history_max_messages: int = Field(default=50, ge=4)
    context_always_keep_recent: int = Field(default=10, ge=2)
    context_discard_score: float = Field(default=0.3, ge=0, le=1)
    context_summarize_score: float = Field(default=0.6, ge=0, le=1)
    # 默认 LLM 摘要；失败 / 纯闲聊 / 二次溢出仍走规则 fold_session_summary
    context_summary_use_llm: bool = True

    user_memory_enabled: bool = True
    user_memory_ttl_minutes: int = Field(default=60 * 24 * 30, ge=1)

    # 短期会话（槽位 + 消息 checkpoint）；留空且单进程则仅内存
    redis_url: str = ""
    session_ttl_seconds: int = Field(default=60 * 60 * 2, ge=60)  # 2 小时，滑动过期
    session_summary_max_chars: int = Field(default=800, ge=100, le=10000)
    # 同一 thread_id 互斥：覆盖整轮 graph.stream / astream，避免多 worker 交叉写会话
    session_lock_ttl_seconds: int = Field(default=180, ge=0)
    session_lock_wait_seconds: float = Field(default=10, ge=0)
    session_lock_retry_interval_seconds: float = Field(default=0.05, ge=0)
