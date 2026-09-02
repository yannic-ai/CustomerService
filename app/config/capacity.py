"""并发槽、线程池、worker 与连接池。"""

from pydantic import BaseModel, Field


class CapacitySettings(BaseModel):
    """容量与背压。新增限流 / worker / 超时只改本文件。"""

    # 集群同时跑对话图的上限（有 Redis 时跨 worker 共享）；打满返回 HTTP 429
    chat_concurrency: int = Field(default=8, ge=1)
    # 单个租户同时占用的对话槽上限，防止吵闹租户占满全集群；不超过 chat_concurrency
    chat_tenant_concurrency: int = Field(default=4, ge=1)
    # 槽位租约 TTL（秒）；进程崩溃后过期回收，避免永久占槽
    chat_slot_ttl_seconds: int = Field(default=180, ge=1)
    graph_workers: int = Field(default=8, ge=0)
    order_tool_workers: int = Field(default=4, ge=0)
    # 订单工具线程池单次执行超时（秒）；0 表示不限制
    order_tool_timeout_seconds: float = Field(default=10, ge=0)
    # 工具线程池队列深度 = workers × tool_pool_queue_factor（建议 1–3；=1 拒绝更激进更安全）
    tool_pool_queue_factor: int = Field(default=2, ge=1, le=10)
    # 队列满后调用方最多再等几秒；等不到抛 ToolPoolBusyError → 上层降级
    tool_pool_queue_wait_seconds: float = Field(default=1.5, ge=0)
    # 单轮 LLM 调用超过该次数记 cs_agent_runaway_total{reason=high_llm_calls}；0 关闭
    agent_runaway_llm_calls: int = Field(default=8, ge=0)
    uvicorn_workers: int = Field(default=1, ge=1)
    # 留空且 workers=1：只记本进程。workers>1 时默认 data/prometheus_multiproc
    prometheus_multiproc_dir: str = ""
    db_pool_size: int = Field(default=5, ge=1)
    db_max_overflow: int = Field(default=10, ge=0)
