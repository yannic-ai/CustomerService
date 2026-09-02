<!--
Sync Impact Report
- Version change: (unratified template) → 1.0.0
- Modified principles:
  - [PRINCIPLE_1_NAME] → I. RoutingPolicy 为唯一分流权威
  - [PRINCIPLE_2_NAME] → II. 安全与租户隔离不可绕过
  - [PRINCIPLE_3_NAME] → III. 测试优先且离线可证 (NON-NEGOTIABLE)
  - [PRINCIPLE_4_NAME] → IV. 失败可见、可观测、可降级
  - [PRINCIPLE_5_NAME] → V. 对话体验一致且可预测
- Added sections:
  - 技术与运行约束
  - 质量门禁
  - Governance（首次批准）
- Removed sections: none (placeholders replaced)
- Follow-up TODOs: none
-->

# CustomerService Constitution

## Core Principles

### I. RoutingPolicy 为唯一分流权威
分流规则 MUST 只写在 `app/agents/policy.py` 的 `RoutingPolicy`（及同模块的升级/计划决策）。
意图识别可以发生在 `intent.py`，但识别之后的图目标、RAG 升级、Supervisor 澄清 MUST NOT
再在 Router 边、RAG 节点、Supervisor 或图 builder 里各写一套。

不可谈判的分流语义：
- 明确课程咨询且高置信 MUST 直达 RAG；明确订单且能绑上单号 MUST 直达 Tool。
- `mixed`、`ambiguous`、低置信 RAG MUST 交给 Supervisor，MUST NOT 猜测执行。
- 已进入 RAG 但本轮问句含订单信号时，MUST 升级 Supervisor，MUST NOT 生成课程答案，MUST NOT 重跑 Router。
- 转人工与闲聊 MUST 直达对应节点，MUST NOT 借道 ReACT。

新增场景时 MUST 先改 Policy 与对应黄金集，再改图或节点。禁止在节点里硬编码「再分流一次」。

### II. 安全与租户隔离不可绕过
每个用户请求 MUST 先经过安全中间件，再进入 Router。注入检测命中 MUST 拦截；身份证、银行卡、手机号、邮箱 MUST 脱敏后再进入下游与日志。输入超长 MUST 拒绝，不得截断后静默继续。

多租户 MUST 行级隔离：
- 知识库按租户 FAISS 子索引检索，禁止跨租户召回。
- 登录用户 MUST 只能查询 `Order.user_id` 匹配的订单；禁止用他人会话记忆回填单号。
- HTTP 身份以 `X-Tenant-Id` / `X-User-Id` 为准；`POST /chat` 中 Header 优先于 JSON `user_id`。
- 不同用户即使复用同一 `session_id` MUST NOT 串话。`anonymous` MUST NOT 写入长期用户记忆。

密钥、`.env`、OSS/LLM 凭证 MUST NOT 提交到仓库。演示数据可以公开，生产凭证不可以。

### III. 测试优先且离线可证 (NON-NEGOTIABLE)
行为变更 MUST 先有失败测试（或黄金集用例），再改实现。本地与 CI 的默认路径 MUST 离线：
清空真实 `DEEPSEEK_API_KEY`，评测会话改用 ngram 索引，不打真实 LLM。

覆盖要求：
- 路由、RAG 忠实度、订单权限、ReACT、安全 MUST 有 `evals/*.jsonl` 黄金集，并有对应 `tests/test_eval_*.py`。
- 改 Policy / 租户 / 安全过滤 / SSE 协议时，MUST 同步更新黄金集或契约测试。
- 真实 LLM 评测（`@pytest.mark.llm` / `RUN_LLM_EVAL=1`）是 nightly 可选项，MUST NOT 成为合并门禁。

禁止以「需要 Key 才能测」为由跳过回归。未覆盖的分流或权限路径视为未完成。

### IV. 失败可见、可观测、可降级
依赖失败 MUST 显式、可度量，禁止静默回退到错误模式：
- 已配置 `REDIS_URL` 时启动 ping 失败 MUST 退出，MUST NOT 悄悄改用内存。
- 多 worker MUST 配置 Redis；配额、会话槽、checkpoint 的跨进程语义不得在单进程近似上假装集群安全。
- LLM 连续失败 MUST 打开熔断（fail-fast），课程走检索模板、ReACT 走离线拼接，`respond` 在仍无答案时回退固定话术。
- 工具瞬态失败（`timeout` / `busy` / 连接类）MUST 先尝试会话 structured + Redis TTL 缓存回退，并在文案中明示陈旧；`not_found`、缺单号、鉴权拒绝、空召回 MUST NOT 读缓存或自动转人工。
- 同轮同工具+同 key 瞬态失败耗尽（默认 ≥2 次）且缓存未命中，或 ReACT 撞 `run_limit`，MAY 自动升级 `handoff`；用户主动说转人工仍走关键词路由，与自动升级并列。
- 非法配置 MUST 在启动时被 Pydantic 拒绝，不得带错值运行。

可观测分工不可混淆：`/metrics` + JSON 日志负责容量与告警；Langfuse / OTel 负责轨迹与 prompt。
新的业务路径（节点、工具、失败态）MUST 暴露可告警指标（次数、outcome、耗时或跑飞），并保持 `request_id` / `trace_id` 可关联。`/ready` MUST 探活数据库、已配置的 Redis 与 FAISS；MUST NOT 把 Langfuse 当作就绪依赖。

### V. 对话体验一致且可预测
用户可感知行为 MUST 稳定、可解释，避免「有时查单、有时编造」：
- 缺单号的订单意图 MUST `ask_user` 澄清，MUST NOT 调订单工具或编造进度。
- 课程咨询 MUST 输出结构化结果（模块 + 学习路径 + 来源），MUST NOT 在空召回时臆造课纲。
- RAG 只补充命中切片的邻域，MUST NOT 把整份源文件倒进上下文。
- 跨轮追问 MUST 回填本会话槽位（如 `last_order_no`）；绑不上单号/课名才允许 ReACT 兜底。
- HTTP 对话主路径是 `POST /chat` SSE（`meta` / `status` / `token` / `answer` / `done`）。改事件名或必填字段视为破坏性变更。
- Supervisor 热路径 MUST 直调 `capabilities`（订单/课程），MUST NOT 为了「也能跑」而把明确计划塞进 ReACT。

闲聊 MUST 说明系统能力边界（课程咨询与查单），MUST NOT 假装能办理退款或修改订单。

## 技术与运行约束

- 语言与包：Python 3.11+；核心框架为 LangChain ≥ 1.0 与 LangGraph；HTTP 为 FastAPI。
- 检索：课程知识默认 FAISS + jieba BM25 + RRF；更换 embedding 或维度后 MUST `python main.py ingest` 重建，否则拒绝加载。
- `serve` / `chat` MUST 只加载已有索引，MUST NOT 在启动时自动 ingest；缺索引 MUST 直接退出。
- 数据：开发默认 SQLite；生产订单库为 MySQL 8。OSS 未配齐时本地回退 `data/oss/`，路径仍按租户隔离。
- 容量：对话并发槽、租户槽、工具线程池、LLM 超时/日限额是一等配置。打满并发 MUST 返回 HTTP 429，MUST NOT 无限排队拖垮事件循环。
- 代码结构：图编排在 `app/graph/`，能力在 `app/capabilities/`，工具适配在 `app/tools/`，配置按主题 mixin 拆分。新功能 MUST 放入已有边界，禁止在 `app.py` 堆积业务规则。
- 复杂度：能用 Policy + 确定性计划解决的，MUST NOT 引入新的 Agent 循环。新增依赖 MUST 有明确失败与降级说明。

## 质量门禁

合并前 MUST 通过与 `.github/workflows/test.yml` 等价的检查：

1. `ruff check app tests`
2. `Settings()` 能以默认/测试配置实例化
3. `mypy app/config`
4. `pytest -q`（离线、无真实 LLM）

建议（非合并阻断，但行为变更时应跑）：路由 / RAG / 忠实度 / 订单 / ReACT / 安全黄金集。覆盖率报告可选用，不得用降低断言来换绿。

代码评审 MUST 核对：Policy 是否仍是分流单点、租户/PII 是否被绕过、指标与黄金集是否同步、SSE/配置是否破坏兼容。无法离线证明的行为不得合并。

## Governance

本宪章高于风格偏好、临时快捷方式和未记录的口头约定。与宪章冲突的 PR、脚本或 Agent 生成代码 MUST 先改宪章（并升版本）或先改实现以符合宪章。

修订程序：
- 任何原则的增删、重新定义或把 MUST 降为 SHOULD，MUST 更新本文件并写出迁移影响（测试、指标、配置、文档）。
- 版本策略：MAJOR = 删除或不相容地重定义原则；MINOR = 新增原则/章节或实质性加严；PATCH = 澄清、措辞、笔误。
- `Last Amended` MUST 为实际修订日（ISO 8601）。批准日（`Ratified`）保持首次通过日期，除非发生 MAJOR 重写。
- 合规检查：CI 质量门禁是自动基线；架构与安全原则由评审人工确认。发现违规 MUST 在合并前修复，或显式记录豁免期限与回滚条件。
- 运行时开发以仓库 `README.md` 与本宪章为准；二者冲突时以本宪章为准，并 MUST 随后修正 README。

**Version**: 1.0.0 | **Ratified**: 2026-08-28 | **Last Amended**: 2026-08-28
