"""指标名、HELP 文本与低基数 label 白名单。

新增指标时：在此登记 HELP，再到 ``instruments.py`` 写 inc/observe。
"""

from __future__ import annotations

METRIC_HELP: dict[str, str] = {
    "cs_http_requests_total": "HTTP 请求数",
    "cs_http_duration_seconds": "HTTP 处理耗时（SSE 为到首包，非首 token）",
    "cs_chat_requests_total": "对话轮次数",
    "cs_chat_duration_seconds": "整轮对话图耗时",
    "cs_chat_busy_total": "对话并发打满次数（HTTP 429；reason=cluster|tenant）",
    "cs_session_busy_total": "同一会话锁冲突次数（HTTP 409）",
    "cs_llm_requests_total": "LLM Gateway 调用次数",
    "cs_llm_tokens_total": "LLM token 累计（按 direction=prompt|completion 分列）",
    "cs_llm_cost_yuan_total": "LLM 估算成本累计（元，按配置单价）",
    "cs_llm_duration_seconds": "LLM 调用耗时",
    "cs_llm_errors_total": "LLM 调用失败次数",
    "cs_llm_circuit_total": "LLM 熔断事件（event=open|reject）",
    "cs_rag_empty_total": "课程检索空召回次数（含低分拒答）",
    "cs_rag_route_upgrade_total": "RAG 空召回且问句含订单信号、升级到 ReACT 的次数",
    "cs_order_access_deny_total": "订单权限拒绝次数（按原因+上下文+租户分桶）",
    "cs_tool_calls_total": "工具调用次数（按 tool+outcome；order_query / course_retrieve；outcome=ok|not_found|deny|timeout|busy|error）",
    "cs_tool_pool_rejected_total": "工具线程池背压拒绝次数（pool=order|course）",
    "cs_tool_pool_queue_seconds": "工具线程池 submit 前的排队等待耗时（pool=order|course）",
    "cs_tool_pool_in_flight": "工具线程池当前占用 + 排队槽数（pool=order|course）",
    "cs_agent_runaway_total": "Agent 跑飞次数（high_llm_calls / react_limit）",
    "cs_chat_in_flight": "当前占用的对话并发槽（有 Redis 时为集群合计）",
    "cs_node_duration_seconds": "LangGraph 节点耗时",
    "cs_ttft_seconds": "对话开始到首条 SSE token 的等待",
    "cs_llm_calls_per_turn": "每轮对话 LLM 调用次数（含 cache hit）",
    "cs_react_limit_hit_total": "ReACT 工具调用撞 run_limit 次数",
    "cs_tool_cache_hit_total": "工具失败缓存回退次数（tool+source=session|redis）",
    "cs_tool_escalation_total": "工具失败自动转人工次数（reason=transient_exhausted|react_limit）",
    "cs_user_feedback_total": "用户赞踩次数（value=up|down）",
}

# 查单工具 outcome 白名单（低基数）
TOOL_ORDER_OUTCOMES = frozenset({"ok", "not_found", "deny", "timeout", "busy", "error"})
RUNAWAY_REASONS = frozenset({"high_llm_calls", "react_limit"})
ESCALATION_REASONS = frozenset({"transient_exhausted", "react_limit"})
TOOL_CACHE_SOURCES = frozenset({"session", "redis"})
CHAT_BUSY_REASONS = frozenset({"cluster", "tenant"})
TOOL_POOL_LABELS = frozenset({"order", "course"})

DURATION_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, float("inf"))
LLM_CALLS_BUCKETS = (1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 12.0, 20.0, float("inf"))
QUEUE_BUCKETS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, float("inf"))
FLUSH_INTERVAL_SECONDS = 0.05
