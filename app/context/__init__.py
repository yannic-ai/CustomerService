"""会话短期上下文：窗口、摘要、槽位。

三块职责分开，避免再堆进单文件：

- ``window``：token 预算与消息截取（进 prompt 的最近对话）
- ``summary``：溢出轮次的结构化滚动摘要 + 分级压缩（见 ``summary/`` 子包）
- ``turn``：回合收尾（脱敏、课纲压缩、消息补丁、冷归档）
- ``slots``：跨轮业务指针（订单号 / 课名 / 意图）与追问回填

对外仍从 ``app.context`` 导入，调用方不必改路径。
"""

from app.context.slots import (
    BUSINESS_INTENTS,
    MISSING_COURSE_NAMES,
    SessionSlots,
    apply_session_slots,
    is_followup,
    slot_updates_after_turn,
)
from app.context.summary import (
    CompressResult,
    StructuredSummary,
    _clip_summary,
    _llm_batch_summarize,
    compact_rag_history,
    compress_by_importance,
    extract_summary_facts,
    fold_session_summary,
    score_message,
)
from app.context.turn import RespondTurnUpdate, finalize_respond_turn, persist_respond_turn, resolve_respond_answer
from app.context.window import (
    DEFAULT_HISTORY_MIN_MESSAGES,
    DEFAULT_HISTORY_TOKEN_BUDGET,
    SHORT_QUERY_CHARS,
    count_tokens,
    expand_query,
    format_history_for_prompt,
    message_text,
    order_no_from_history,
    overflow_messages,
    prompt_summary_line,
    prompt_summary_tokens,
    recent_messages,
    select_messages_by_tokens,
    token_estimate_strategy,
    trim_message_updates,
)

# 测试曾从单文件 import 这些私有名
_prompt_summary_line = prompt_summary_line
_prompt_summary_tokens = prompt_summary_tokens

__all__ = [
    "BUSINESS_INTENTS",
    "CompressResult",
    "DEFAULT_HISTORY_MIN_MESSAGES",
    "DEFAULT_HISTORY_TOKEN_BUDGET",
    "MISSING_COURSE_NAMES",
    "SHORT_QUERY_CHARS",
    "RespondTurnUpdate",
    "SessionSlots",
    "StructuredSummary",
    "_clip_summary",
    "_llm_batch_summarize",
    "apply_session_slots",
    "compact_rag_history",
    "compress_by_importance",
    "count_tokens",
    "expand_query",
    "extract_summary_facts",
    "finalize_respond_turn",
    "fold_session_summary",
    "format_history_for_prompt",
    "is_followup",
    "message_text",
    "order_no_from_history",
    "overflow_messages",
    "persist_respond_turn",
    "recent_messages",
    "resolve_respond_answer",
    "score_message",
    "select_messages_by_tokens",
    "slot_updates_after_turn",
    "token_estimate_strategy",
    "trim_message_updates",
]
