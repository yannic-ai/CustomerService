# 智能客服系统

面向课程教育的智能客服系统，识别多种意图。

未配置 `DEEPSEEK_API_KEY` 时仍可本地演示（关键词路由 + 检索模板 + 查库）。Key 已配但上游持续失败时，网关熔断后降到同一套离线能力。

```mermaid
flowchart TD
    START([用户请求]) --> security[安全中间件]
    security --> router[Router Agent]
    router -->|课程咨询 高置信| rag[RAG Agent]
    router -->|订单查询 有单号| tool[Tool Agent]
    router -->|mixed / ambiguous / 低置信 RAG| supervisor[Supervisor Agent]
    router -->|转人工| handoff[人工转接]
    router -->|闲聊| chitchat[闲聊]
    router -->|安全拦截| respond[输出脱敏]
    rag -->|订单信号误入| supervisor
    rag --> respond
    tool --> respond
    supervisor --> respond
    handoff --> respond
    chitchat --> respond
    respond --> END([返回答复])
```

明确课程 / 订单 / 转人工直达。`mixed`（多意图确定）与 `ambiguous`（缺单号、过短问句、问候混业务词）交给 Supervisor：能绑上单号或课名则调能力（无依赖可并行，课名来自订单则先查单）；需要澄清则 `ask_user`；否则 ReACT 兜底。已进入 RAG 但本轮含订单信号时，升 Supervisor，不重跑 Router。

流程图也可在服务启动后打开 [http://127.0.0.1:8000/graph](http://127.0.0.1:8000/graph)，或执行 `python main.py graph`。

## 核心场景

| 场景 | 示例 | 流程 |
| --- | --- | --- |
| 课程咨询 | Python入门课包含哪些内容？ | 高置信直达 RAG：检索大纲 → 模块 + 学习路径 |
| 订单查询 | 查询订单#20251114001的退款进度 | 有单号直达 Tool：订单 API → 进度报告 |
| 混合（串行） | 订单#20251114001 这门课讲什么，退款进度呢 | Supervisor：先查单拿 `product_name` 再检索 |
| 混合（并行） | 订单20251114001买的入门课包含哪些内容？ | Supervisor：单号与课名都能绑定，查单与检索并发 |
| 缺单号澄清 | 查一下我的订单 / 退款进度怎么样 | Supervisor `ask_user` 要完整单号，不调工具 |
| 跨轮追问 | （已查过单）那退款呢 | 槽位回填 `last_order_no`，不必重说单号 |
| 绑不上兜底 | 那个怎么样了 | Supervisor 走 ReACT |
| 转人工 | 转人工 / 我要找人工客服 | 直达 handoff |
| 闲聊 | 你好 / 今天天气怎么样 | 直达 chitchat |

## 快速开始

需要 Python 3.11+。默认 SQLite，不必起 Docker。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # 可选：写入 DEEPSEEK_API_KEY

python main.py ingest         # 首次或更换 embedding 后必须；日常增量 ingest
python main.py seed
python main.py serve          # 缺索引会直接退出，不会在启动时 ingest
```

浏览器打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)。

```bash
python main.py chat "Python入门课包含哪些内容？"
python main.py chat "查询订单#20251114001的退款进度"
```

全量重建索引：`python main.py ingest --rebuild`。

MySQL 8 与 Redis（多进程 / 重启后续聊）用 Docker：

```bash
docker compose up -d
# .env 中设置：
# DATABASE_URL=mysql+pymysql://cs:cs@127.0.0.1:3306/customer_service?charset=utf8mb4
# REDIS_URL=redis://127.0.0.1:6379/0
```

账号 `cs` / `cs`，库名 `customer_service`。只起 Redis：`docker compose up -d redis`。

## 配置

复制 `.env.example` 为 `.env`。未列出的项有默认值，完整清单见 `.env.example`。

| 变量 | 说明 |
| --- | --- |
| `DEEPSEEK_API_KEY` | DeepSeek 密钥；不配则走离线路由与检索模板 |
| `DATABASE_URL` | 默认 SQLite；生产用 `mysql+pymysql://cs:cs@127.0.0.1:3306/customer_service?charset=utf8mb4` |
| `REDIS_URL` | 会话、档案、配额与并发槽。单进程演示可留空；多 worker / 生产必须配置。填了但 ping 失败会拒绝启动 |
| OSS 四项 | `OSS_ACCESS_KEY_ID` / `SECRET` / `BUCKET` / `ENDPOINT` 配齐后上传阿里云，否则写 `data/oss/` |
| `EMBEDDING_BACKEND` | 默认 `huggingface`（`BAAI/bge-small-zh-v1.5`）。离线可改 `ngram`。换模型或维度后必须 `python main.py ingest` |

### Langfuse（可选）

与应用的 MySQL / Redis 分开，避免抢 `6379`：

```bash
cp langfuse.env.example langfuse.env
docker compose -f docker-compose.langfuse.yml -p cs-langfuse --env-file langfuse.env up -d
```

UI [http://127.0.0.1:3000](http://127.0.0.1:3000)。把 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` 写入应用 `.env`。未配置时服务照常运行；`/ready` 不探活 Langfuse。

## 多租户

共享一个库，行级隔离；课程知识按租户 FAISS 子索引检索。

- HTTP：`X-Tenant-Id`（缺省 `demo`）、`X-User-Id`（缺省 `anonymous`；`POST /chat` 中 Header 优先于 JSON `user_id`）
- CLI：`python main.py --tenant demo chat "..." --user u10001`
- 示例订单 `20251114001` 属于 `demo` / `u10001`；换租户或登录用户 `alice` 应 404
- 匿名可凭本轮单号查询（演示）；登录用户只能查自己的订单
- 知识库：`data/knowledge/*.md` → `demo`；`data/knowledge/<tenant>/*.md` → 对应租户

旧 SQLite  schema 不兼容时，删除 `data/customer_service.db` 后重新 `python main.py seed`。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 进程探活 |
| `GET` | `/ready` | 就绪：数据库、已配置的 Redis、FAISS；失败 503（不含 Langfuse） |
| `GET` | `/metrics` | Prometheus 文本指标 |
| `GET` | `/graph` | LangGraph 流程图 |
| `GET` | `/greet` | 开场白与预置选项 |
| `POST` | `/chat` | 对话（SSE：`meta` / `status` / `token` / `answer` / `done`） |
| `POST` | `/feedback` | 赞踩（`trace_id` + `value` 0\|1） |
| `GET` | `/api/v1/orders/{order_no}` | 订单进度（按租户与下单人隔离） |
| `POST` | `/admin/knowledge/ingest` | 管理面知识库 ingest（需 `ADMIN_INGEST_TOKEN`） |

`POST /api/v1/chat` 为兼容旧路径，返回一整段 JSON。

```bash
curl -N -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-Id: demo' \
  -d '{"message":"Python入门课包含哪些内容？","session_id":"s1"}'
```

同一 `tenant_id` + `user_id` + `session_id` 续聊；未传 `session_id` 则新建会话。不同用户复用同一 `session_id` 也不会串话。稳定的 `user_id`（非 `anonymous`）会写入长期记忆，换 session 也能回填。

## 可观测

- **容量与告警**：`/metrics` + JSON 日志（带 `request_id` / `trace_id` / `span_id`）。告警规则在 [`deploy/prometheus/`](deploy/prometheus/)，加载 `rules/cs.yml`。本仓库不提供 Grafana。
- **轨迹与 Prompt**：配齐 Langfuse 后上报自动 score（`blocked` / `handoff` / `rag_empty` / `turn_ok`）与用户赞踩；Prompt 名称 `cs.intent` / `cs.react` / `cs.rag_markdown` / `cs.rag_structured` / `cs.summary`（未创建则用代码内 fallback）。
- **链路**：OpenTelemetry 打 HTTP / LangGraph 节点 / FAISS / DB / Redis，OTLP 进同一 Langfuse。演示页 `/` 每条回复可点「有用 / 无用」。

## 目录

```
app/
  agents/        Router / RAG / Tool / Supervisor / ReACT
  graph/         LangGraph 主流程
  rag/           课程切片、FAISS、混合检索、ingest
  capabilities/  订单 / 课程能力（Graph、Supervisor、ReACT 共用）
  security/      安全过滤与中间件
  llm_gateway/   超时 / 重试 / 熔断 / 配额
  observability/ 日志、Prometheus、OTel、就绪检查
  config/        运行配置
  app.py         FastAPI 入口
data/knowledge/  课程大纲 Markdown（根目录=demo，子目录=租户）
evals/           黄金评测集（JSONL）
deploy/prometheus/  告警规则
docker-compose.yml            本地 MySQL 8 + Redis
docker-compose.langfuse.yml   自建 Langfuse（仅暴露 3000）
```

## 测试

默认离线：改用 ngram 索引并清空 `DEEPSEEK_API_KEY`，不读本机密钥。请用项目虚拟环境里的解释器。

```bash
.venv/bin/pytest -q
.venv/bin/ruff check app tests
```

可选：

```bash
.venv/bin/pytest -q --cov=app --cov-report=term-missing
RUN_LLM_EVAL=1 .venv/bin/pytest -q -m llm          # 需 DEEPSEEK_API_KEY
python main.py eval                                # 黄金集
python main.py eval --llm
python main.py eval harvest                        # 失败样本 → evals/pending/
python main.py eval merge-pending --apply          # 写入正式 evals
```

RAGAS nightly：`pip install -e ".[dev,eval]"` 后执行 `RUN_LLM_EVAL=1 .venv/bin/pytest -q -m llm tests/test_eval_ragas.py`。

## 技术栈

| 模块 | 选型 | 说明 |
| --- | --- | --- |
| 核心框架 | LangChain ≥ 1.0 | `create_agent` + LangGraph |
| 向量库 | FAISS 1.8.0 + BM25/RRF | 课程大纲检索 |
| 大模型 | DeepSeek API | `langchain-deepseek` / `deepseek-chat` |
| 数据存储 | MySQL 8.0+ / SQLite + OSS | 订单与报告归档 |
| 安全中间件 | Callback + AgentMiddleware | 适配 LangChain v1.0 |
