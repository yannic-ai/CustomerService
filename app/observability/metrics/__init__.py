"""Prometheus 文本指标。

- ``names.py``：指标名与 HELP
- ``store.py``：进程内计数
- ``multiproc.py``：跨 worker 快照
- ``instruments.py``：业务 inc/observe（新增指标主要改这里）
- ``render.py``：``/metrics`` 文本

``UVICORN_WORKERS>1`` 或显式 ``PROMETHEUS_MULTIPROC_DIR`` 时，每个 worker
把快照写到共享目录；刮任意一个 ``/metrics`` 都会合并所有存活 pid。
"""

from app.observability.metrics.instruments import (
    estimate_llm_cost_yuan,
    inc_agent_runaway,
    inc_chat,
    inc_chat_busy,
    inc_http,
    inc_llm,
    inc_llm_circuit,
    inc_llm_error,
    inc_order_access_deny,
    inc_rag_empty,
    inc_rag_route_upgrade,
    inc_react_limit_hit,
    inc_session_busy,
    inc_tool_call,
    inc_tool_cache_hit,
    inc_tool_escalation,
    inc_tool_pool_rejected,
    inc_user_feedback,
    maybe_inc_runaway_llm_calls,
    observe_chat,
    observe_http,
    observe_llm_calls,
    observe_node,
    observe_tool_pool_queue,
    observe_ttft,
)
from app.observability.metrics.multiproc import (
    _pid_alive,
    configure_metrics_multiprocess,
    configure_metrics_multiprocess_from_settings,
    disable_metrics_multiprocess,
    flush_metrics_if_multiprocess,
    mark_metrics_process_dead,
    metrics_multiproc_enabled,
)
from app.observability.metrics.names import (
    CHAT_BUSY_REASONS,
    METRIC_HELP,
    RUNAWAY_REASONS,
    TOOL_ORDER_OUTCOMES,
    TOOL_POOL_LABELS,
)
from app.observability.metrics.render import render_prometheus
from app.observability.metrics.store import (
    _COUNTERS,
    _HISTOGRAMS,
    _inc,
    _observe,
    reset_metrics,
)

_HELP = METRIC_HELP

__all__ = [
    "CHAT_BUSY_REASONS",
    "METRIC_HELP",
    "RUNAWAY_REASONS",
    "TOOL_ORDER_OUTCOMES",
    "TOOL_POOL_LABELS",
    "configure_metrics_multiprocess",
    "configure_metrics_multiprocess_from_settings",
    "disable_metrics_multiprocess",
    "estimate_llm_cost_yuan",
    "flush_metrics_if_multiprocess",
    "inc_agent_runaway",
    "inc_chat",
    "inc_chat_busy",
    "inc_http",
    "inc_llm",
    "inc_llm_circuit",
    "inc_llm_error",
    "inc_order_access_deny",
    "inc_rag_empty",
    "inc_rag_route_upgrade",
    "inc_react_limit_hit",
    "inc_session_busy",
    "inc_tool_call",
    "inc_tool_cache_hit",
    "inc_tool_escalation",
    "inc_tool_pool_rejected",
    "inc_user_feedback",
    "mark_metrics_process_dead",
    "maybe_inc_runaway_llm_calls",
    "metrics_multiproc_enabled",
    "observe_chat",
    "observe_http",
    "observe_llm_calls",
    "observe_node",
    "observe_tool_pool_queue",
    "observe_ttft",
    "render_prometheus",
    "reset_metrics",
]
