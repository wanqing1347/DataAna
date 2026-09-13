# DataAna

DataAna 是一个面向数据分析场景的 Python Agent 全栈项目。当前主实现使用 **FastAPI + LangGraph + SQLAlchemy + sqlglot + MCP**，前端由仓库内的轻量静态页面提供，并由 FastAPI 同源托管。

项目重点展示的不是普通 CRUD，而是完整的数据分析 Agent 工程链路：工具按需发现、Text-to-SQL、SQL 安全治理、数据权限与脱敏、MCP 图表、durable checkpoint，以及 BIRD Dev 评测。

## 核心能力

- **LangGraph ReAct Agent**：支持多轮 tool loop、checkpoint、恢复与运行时约束。
- **Tool Registry + tool_search**：业务工具采用 deferred loading，首轮只暴露工具发现能力，降低初始 tool schema 上下文。
- **Text-to-SQL**：支持表结构探查、业务术语口径、SQL 生成、校验、执行和结果分析。
- **SQL 安全治理**：只读 SQL、表白名单、LIMIT、JOIN 数量限制、数据权限改写和敏感字段脱敏。
- **MCP 图表链路**：LangGraph Agent → mcp-echarts → MinIO → 浏览器可访问图表 URL。
- **BIRD Dev Eval**：保留 direct ChatModel baseline 与 Agent 评测入口，复用仓库内 runner 做 execution-result scoring。

> Python 评测成绩只以最新 fresh-run 报告为准。仓库不会把历史 Java 成绩冒充当前 Python 成绩。

## 仓库结构

```text
DataAna/
├── frontend/                  # 当前 Web 前端，FastAPI 直接托管
│   ├── index.html
│   ├── login.html
│   ├── css/
│   └── js/
├── python_backend/            # FastAPI + LangGraph 主后端
│   ├── app/
│   ├── tests/
│   ├── evals/
│   ├── scripts/
│   ├── pyproject.toml
│   └── uv.lock
├── schema/                    # Agent 使用的 schema / glossary
├── deploy/                    # Docker Compose、MCP、Nginx 配置
├── scripts/                   # BIRD 评测 runner
├── sql/                       # 数据库初始化脚本与参考 SQL
├── skills/                    # Agent skills
└── README.md
```

旧 Java/Spring AI AgentX 实现仅作为本地迁移参考，不属于当前 GitHub 发布结构。

## 本地启动

### 1. 准备 Python 环境

推荐使用 `uv`：

```bash
cd python_backend
cp .env.example .env
uv sync --extra dev
```

根据自己的环境修改 `python_backend/.env`，至少配置数据库连接和模型 API Key。真实 `.env` 不应提交到 Git。

### 2. 初始化数据库

项目 Docker 部署会自动挂载 `sql/init.sql` 初始化 MySQL。若使用本机 MySQL，也可以手动导入该脚本。

### 3. 启动 FastAPI

```bash
cd python_backend
uv run uvicorn app.main:app --host 0.0.0.0 --port 8889 --reload
```

打开：

```text
http://127.0.0.1:8889/
```

FastAPI 会直接提供 `frontend/` 下的页面和静态资源，因此前后端不需要分别启动，也不需要额外配置 CORS。

## Docker 一键启动

```bash
cd deploy
cp .env.example .env
# 填写 MySQL、MinIO、DeepSeek、JWT 等真实配置

docker compose up -d --build
```

主要服务：

```text
Browser
   ↓
FastAPI :8889
   ├── frontend
   ├── Agent API / SSE
   └── /health
        ↓
LangGraph Agent
   ├── MySQL
   └── MCP ECharts → MinIO
```

健康检查：

```bash
curl http://127.0.0.1:8889/health
```

如需公网 HTTPS 部署，可继续使用 `deploy/nginx-dataana.conf` 作为反向代理参考。

## 测试

```bash
cd python_backend
uv run pytest
```

当前测试覆盖包括：

- SQL 白名单、安全校验和 LIMIT guard
- 数据权限 AST rewrite
- durable checkpoint / resume
- Tool Registry / tool_search / deferred loading
- planner-critic 与工具调用约束
- 前端 MCP 图表 URL contract
- BIRD baseline / Agent 评测 contract
- MCP live integration test（显式 opt-in）

## BIRD 评测

本地 benchmark 时才建议开启：

```env
BIRD_EVAL_ENABLED=true
```

主要接口：

```text
POST /bird/eval/baseline
POST /bird/eval/question
```

相关 runner 位于：

```text
scripts/bird_eval_runner.py
scripts/reproduce_bird_resume.py
```

完整评测机制和口径见 `python_backend/README.md` 与 `python_backend/INTERVIEW_GUIDE.md`。

## 配置与安全

- 不要提交 `python_backend/.env` 或 `deploy/.env`。
- 不要提交本地 BIRD 数据库、模型文件、构建产物或运行时 checkpoint。
- `deploy/.env.example` 与 `python_backend/.env.example` 只保留占位配置。
- 默认数据分析表白名单定义在 Python 配置中，schema/glossary 位于 `schema/dodo_agentx.yml`。
- BIRD 本地 SQLite 评测接口默认关闭，避免生产环境暴露本地文件读取入口。

## 技术栈

- Python 3.13
- FastAPI
- LangGraph / LangChain
- SQLAlchemy
- sqlglot
- MySQL 8
- MCP / mcp-echarts
- MinIO
- Docker Compose
- Vue 3 CDN + Vanilla JavaScript
