"""滚动摘要与分级压缩：窗口裁掉的旧轮留下结构化事实，而不是原文。

子模块：
- ``facts``：结构化摘要、规则抽取与合并
- ``scoring``：消息重要性评分与轮次分组
- ``rag_compact``：课纲长回复就地压缩
- ``llm``：LLM 摘要触发与异步调用
- ``compress``：分级压缩入口（异步）
"""

from app.context.summary.compress import CompressResult, compress_by_importance
from app.context.summary.facts import (
    StructuredSummary,
    extract_summary_facts,
    fold_session_summary,
)
from app.context.summary.facts import finalize_summary as _finalize_summary
from app.context.summary.facts import _clip_summary
from app.context.summary.llm import _llm_batch_summarize, should_invoke_llm_summary
from app.context.summary.rag_compact import compact_rag_history
from app.context.summary.scoring import score_message

# 测试与旧代码兼容的私有别名
_should_invoke_llm_summary = should_invoke_llm_summary

__all__ = [
    "CompressResult",
    "StructuredSummary",
    "_clip_summary",
    "_llm_batch_summarize",
    "_should_invoke_llm_summary",
    "compact_rag_history",
    "compress_by_importance",
    "extract_summary_facts",
    "fold_session_summary",
    "score_message",
]
