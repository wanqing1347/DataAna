# DataAna Python（LangGraph 版）

这是对原 Java/Spring AI AgentX 数据分析链路的 Python 重构。目标不是逐行翻译，而是把项目做成一个能在 Agent 开发岗位面试中完整解释 **Agent runtime、工具按需发现、MCP、durable execution、Text-to-SQL、安全治理、可观测性和评测** 的工程案例。

当前版本：**0.4.0（简历能力对齐阶段）**。

## 本阶段已经对齐的三条能力

### 1. Tool Registry + tool_search + deferred tools

生产主 Agent 不再把所有工具 schema 一次性 bind 给模型。

当前工具分层：

~~~text
Always visible
└── tool_search

Deferred local tools
├── listTables
├── describeTables
├── lookupGlossary
├── queryData
├── validateSql
├── executeSql
└── calculate

Deferred MCP tools
└── mcp-echarts 暴露的 chart tool（MCP 连接成功后才注册）
~~~

执行流程：

~~~text
START
  |
  v
model
  |
  | first round: bind(tool_search only)
  v
tool_search("需要查询业务数据")
  |
  | registry search
  | result.loaded_tools = ["queryData", ...]
  v
LangGraph state.loaded_tools
  |
  v
model
  |
  | next round:
  | bind(tool_search + discovered tools only)
  v
ToolNode(allow-listed visible tools)
  |
  v
model / END
~~~

这里不是只在 prompt 中要求“先搜索工具”。

ToolRegistry 同时参与：

1. 工具元数据注册；
2. tool_search 检索；
3. 每轮动态 bind_tools；
4. ToolNode 执行白名单。

因此模型即使猜到一个 deferred tool 名，也不能在尚未搜索加载时直接执行。

#### 初始 tool schema context 指标

运行：

~~~bash
cd python_backend
uv run python scripts/tool_context_metrics.py
~~~

当前 7 个真实本地业务工具的测量结果：

| 方案 | 首轮可见 schema 数 | schema chars | chars/4 近似 tokens |
| --- | ---: | ---: | ---: |
| eager：tool_search + 7 个业务工具 | 8 | 2330 | 583 |
| deferred：仅 tool_search | 1 | 379 | 95 |

按 schema 字符量计算，首轮工具上下文减少 **83.73%**。

> approxTokens 是公开透明的 chars / 4 proxy，只用于仓库内相对对比，不声称是模型 tokenizer 或计费 token 的精确值。

相关实现：

- app/tool_registry.py
- app/agent_graph.py
- tests/test_tool_registry.py
- tests/test_agent_graph_deferred.py
- scripts/tool_context_metrics.py

### 2. MCP ECharts + MinIO + 浏览器展示链路

Python Agent 使用 langchain-mcp-adapters 连接 mcp-echarts：

~~~text
LangGraph Agent
      |
      | tool_search("chart / echarts / 可视化")
      v
deferred MCP chart tool
      |
      v
mcp-echarts :3033/mcp
      |
      | upload chart object
      v
MinIO dataana-charts bucket
      |
      | public/object URL
      v
Nginx /dataana-charts/
      |
      v
browser <img>
~~~

MCP 启动状态不再静默吞错。GET /health 会返回：

~~~json
{
  "status": "ok",
  "agent": "langgraph",
  "language": "python",
  "mcp": {
    "enabled": true,
    "connected": true,
    "url": "http://mcp-echarts:3033/mcp",
    "tools": ["..."],
    "error": null
  }
}
~~~

如果 MCP 未连接，connected=false 且保留错误信息；MCP 工具不会进入 Tool Registry。

仓库已有前端会识别 chart tool result 中的：

- HTTP/HTTPS URL
- Markdown image
- JSON url/imageUrl/src
- PNG base64

HTTPS 页面遇到同域 HTTP MinIO URL 时，会重写成 Nginx 的同源 HTTPS /dataana-charts/... 地址。

相关实现：

- app/agent_service.py
- scripts/mcp_e2e_check.py
- tests/integration/test_mcp_echarts_e2e.py
- tests/test_frontend_chart_contract.py
- deploy/docker-compose.yml
- deploy/python_backend.Dockerfile
- frontend/js/chat.js
- deploy/nginx-dataana.conf

#### Python Docker 部署

在 deploy/：

~~~bash
cp .env.example .env
# 填写 MySQL / MinIO / DeepSeek / JWT 等真实配置

docker compose up -d --build
~~~

docker-compose.yml 直接构建并启动 Python/FastAPI 服务，并设置：

~~~env
CHART_MCP_ENABLED=true
CHART_MCP_URL=http://mcp-echarts:3033/mcp
~~~

部署后先检查：

~~~bash
curl http://127.0.0.1:8889/health
~~~

再做 MCP → MinIO URL integration check：

~~~bash
cd ../python_backend
RUN_MCP_E2E=1 uv run pytest tests/integration/test_mcp_echarts_e2e.py -q
~~~

也可以直接运行诊断脚本：

~~~bash
uv run python scripts/mcp_e2e_check.py
~~~

可选环境变量：

~~~env
MCP_E2E_URL=http://127.0.0.1:3033/mcp
MCP_E2E_TOOL_NAME=<实际 chart tool name>
MCP_E2E_ARGS_JSON=<需要覆盖默认参数时填写 JSON>
~~~

integration test 的通过条件包括：

1. Python MCP adapter 能发现 chart tool；
2. chart tool 调用成功；
3. 返回内容中能提取 HTTP/HTTPS URL；
4. 对该 URL 发起 GET 得到 HTTP 200 且响应体非空。

前端渲染契约由 test_frontend_chart_contract.py 单独回归。完整浏览器视觉验收仍应在真实部署环境打开页面执行一次。

### 3. Python BIRD Dev Text-to-SQL Eval

仓库原有 scripts/bird_eval_runner.py 的数据加载、并发请求、SQLite execution-result scoring 和报告生成逻辑继续复用。

Python FastAPI 新增与旧 runner 兼容的两个接口：

~~~text
POST /bird/eval/baseline
POST /bird/eval/question
~~~

评测接口默认关闭：

~~~env
BIRD_EVAL_ENABLED=false
~~~

只在本地 benchmark 时开启，避免生产服务暴露任意本地 SQLite 路径读取入口。

#### Direct ChatModel baseline

/bird/eval/baseline：

~~~text
question + evidence + complete SQLite schema
                  |
                  v
        ChatModel structured output
                  |
                  v
                 SQL
~~~

baseline 不使用 Agent tool 或 planner/critic，用于衡量“单次模型直出 SQL”的基线。

#### LangGraph ReAct（Java AgentX 机制迁移）

/bird/eval/question：

~~~text
question + evidence + BirdAgentOutput schema + <no_think>
                         |
                         v
                       model
                    /         \
             tool calls      structured JSON
                |                 |
 listTables / describeTables / verifySql
                |                 |
                +----> model <----+
                         |
                         v
                    END -> SQL

maxRounds 命中且仍有 pending tool call：
skip pending tools -> one force-final model call -> END
~~~

当前 Python/LangGraph 版本保留 AgentX 的 ReAct 轨迹语义，同时把“停止条件”和 verifier 能力边界拆开：模型没有 tool call 时仍按 AgentX 正常结束；普通工具执行后继续回到模型；final-mode `verifySql(sql, requestedColumns)` 先经过现有 `BirdSqlVerifier` 的 execution/structural gate，再经过启动 ReAct 前由 `question + evidence + schema` 生成的 immutable `TaskContract` semantic gate。只有 `structuralPassed=true` 且 `semantic.passed=true` 时才写入 `last_passed_sql` 并由 LangGraph deterministic `END`，因此 Agent 自己缩减 `requestedColumns` 不能再把缺失输出或缺失条件伪装成 verifier pass。达到 `maxRounds` 时，pending tool 仍不执行，而是写入 `Agent maximum rounds reached. Tool execution skipped.` 后做一次 force-final；生产 semantic gate 开启时，未经过双 Gate 的 force-final/text-only SQL 不会被标记为批准结果。

`verifySql` 默认把完整执行结果交回模型（`BIRD_EVAL_TOOL_RESULT_MAX_ROWS=0`），用于真实值、日期表示、join 粒度和聚合范围自检；exploratory probe 默认最多 8 次（`BIRD_EVAL_MAX_PROBE_CALLS=8`）。一旦 probe budget 用完，probe-mode 调用会被拒绝；一旦 final SQL 结构通过但 semantic contract 失败，也立即进入 targeted-repair mode，即使尚有 probe 预算也不再开放自由探索，只允许最多 `BIRD_EVAL_MAX_SEMANTIC_REPAIRS=3` 次 corrected final verify。semantic gate 先执行确定性 contract checks（输出数量/answer shape/显式 predicate/source/grouping 等），必要时再用一个独立、无工具、temperature=0 的 semantic critic 审核难以程序化的语义；critic 故障采用 fail-closed，而不是把 structural pass 当作最终正确。Python 端仍保留只读 SQLite、`PRAGMA query_only=ON` 与执行 timeout；生成与 semantic review 阶段均不读取 gold SQL 或 gold result。

相关实现：

- app/bird_eval.py
- app/bird_semantic.py
- app/bird_verifier.py
- tests/test_bird_eval.py
- ../scripts/bird_eval_runner.py

#### 运行 BIRD 对照评测

为了复现简历里历史 120 题口径，先从 BIRD 官方 Dev 下载源准备真实数据：

~~~bash
cd ..
python scripts/prepare_bird_dev.py
~~~

脚本会下载官方 `dev.zip` 到 `data/bird/`，解压后自动定位 `dev.json + dev_databases`，并校验历史 120 题的数据指纹：

~~~text
simple=73
moderate=39
challenging=8
total=120
~~~

同时生成 `data/bird/manifest.resume-120.json`，记录数据源、SHA-256、题数和首 120 题难度分布。大型 benchmark 数据已加入 `.gitignore`，不会误提交进仓库。

也可以直接一条命令启动 Python BIRD 服务并 fresh-run 两组 120 题：

~~~bash
python scripts/reproduce_bird_resume.py
~~~

它会依次生成：

~~~text
scripts/report/pred.python.baseline.jsonl
scripts/report/pred.python.baseline.report.md
scripts/report/pred.python.agent.jsonl
scripts/report/pred.python.agent.report.md
~~~

并自动和历史简历目标比较：

~~~text
baseline: 70 / 120 = 58.33%
agent:    83 / 120 = 69.17%
~~~

如果当前模型供应商对同一个非版本化模型别名做过更新，fresh run 可能与历史分数发生漂移；此时应以新生成的 `pred.python.*.report.md` 为当前 Python/LangGraph 实测成绩，而不能把历史目标硬写成当前结果。

手动运行时，准备 BIRD Dev 解压目录，要求：

~~~text
<DATA_ROOT>/
├── dev.json
└── dev_databases/
~~~

启动 Python 服务：

~~~bash
cd python_backend
# .env 中填写真实 DEEPSEEK_API_KEY
# 本地评测时开启：
# BIRD_EVAL_ENABLED=true

uv run uvicorn app.main:app --host 127.0.0.1 --port 8889
~~~

从仓库根目录分别跑：

~~~bash
python scripts/bird_eval_runner.py \
  --mode baseline \
  --data-root <DATA_ROOT> \
  --max-questions 120
~~~

~~~bash
python scripts/bird_eval_runner.py \
  --mode agent \
  --data-root <DATA_ROOT> \
  --max-questions 120
~~~

默认输出：

~~~text
scripts/report/pred.python.baseline.jsonl
scripts/report/pred.python.baseline.report.md
scripts/report/pred.python.agent.jsonl
scripts/report/pred.python.agent.report.md
~~~

报告中的 total EX 才是可以写进 Python 简历的实测 execution accuracy。

> 仓库现有 scripts/report/pred.report.md 的 **69.17%** 和 pred.baseline.report.md 的 **58.33%** 是迁移前 Java/AgentX 历史评测记录。它们只能作为历史对照，**不能当作 Python/LangGraph 当前成绩**。Python 简历数字必须以 pred.python.*.report.md 的重新实测结果为准。

---

## 技术栈

- **FastAPI**：HTTP / SSE API
- **LangGraph**：StateGraph、ToolNode、动态工具绑定、checkpoint/resume
- **LangChain / ChatOpenAI**：DeepSeek OpenAI-compatible 模型、StructuredTool
- **langchain-mcp-adapters**：MCP tool adapter
- **langgraph-checkpoint-sqlite + aiosqlite**：本地持久化 checkpoint
- **SQLAlchemy + sqlglot**：业务数据库访问、SQL AST 安全校验和权限改写
- **SQLite**：BIRD benchmark 数据库与只读 execution probe
- **MySQL**：业务会话、run 控制面、工具幂等结果
- **MinIO + Nginx**：图表对象存储与浏览器访问
- **Pydantic Settings**：配置管理
- **pytest**：unit / integration contract / opt-in live E2E

## Durable execution 与安全治理

### Memory 和 checkpoint 分离

~~~text
py_agent_turn
  -> ConversationMemory
  -> 下一次新 run 的 prompt context

AsyncSqliteSaver
  -> LangGraph execution state
  -> 同一个 run 的 stop/resume/crash recovery
~~~

每个逻辑 run 使用：

~~~text
user:<user_id>:run:<run_id>
~~~

而不是直接把 conversationId 当 execution thread。

### Crash recovery 有控制面索引

图运行前先把 py_agent_run 保存成 running，然后才执行 LangGraph。进程异常退出后，stale running 记录仍能把 user/conversation/run 与 durable checkpoint 对应起来。

### 工具幂等 + bounded retry

py_agent_tool_result 保存：

- idempotency key
- tool call id
- args hash
- result/status
- attempts
- last error

已成功的逻辑调用在 resume 时复用结果；相同 call id 被不同参数复用时 fail closed。

SQL retry 只针对典型 transient DB failure，不对语法、权限、安全校验等确定性错误无脑重试。

### 生产 SQL 安全边界

~~~text
LLM / planner / critic
        |
        v
SqlSafetyGuard
        |
DataScopeRewriter
        |
       DB
        |
SensitiveFilter
~~~

LLM critic 只负责软语义判断，永远不是权限或 SQL 安全边界。

## 业务 Text-to-SQL 子图

生产 queryData 仍使用独立 bounded workflow：

~~~text
planner
  -> deterministic SQL guard
  -> semantic critic
  -> reject? bounded replan
  -> executor
  -> guard again
  -> scope rewrite
  -> idempotent DB execution
  -> sensitive filter
~~~

这里与 BIRD eval 子图是两个不同用途：

- 生产 queryData 强调业务安全、数据权限和受控执行；
- BIRD eval 强调 benchmark execution accuracy，使用 BIRD 自己的 SQLite schema，不复用业务白名单/权限改写。

## 配置

核心 runtime：

~~~env
CHECKPOINT_PATH=.data/langgraph_checkpoints.sqlite3
TOOL_RETRY_MAX_ATTEMPTS=3
TOOL_RETRY_BASE_DELAY_MS=100
SQL_PLANNER_MAX_ATTEMPTS=3
~~~

MCP：

~~~env
CHART_MCP_ENABLED=false
CHART_MCP_URL=http://localhost:3033/mcp
~~~

BIRD：

~~~env
BIRD_EVAL_ENABLED=false
BIRD_EVAL_TEMPERATURE=0.0
BIRD_EVAL_MAX_ROUNDS=50
BIRD_EVAL_SQL_TIMEOUT_SECONDS=20
BIRD_EVAL_TOOL_RESULT_MAX_ROWS=0
BIRD_EVAL_MAX_PROBE_CALLS=8
~~~

## 本地启动

~~~bash
cd python_backend
cp .env.example .env
uv sync --extra dev
uv run uvicorn app.main:app --host 0.0.0.0 --port 8889 --reload
~~~

Python 后端直接服务仓库根目录 `frontend/` 前端。

## 测试

~~~bash
cd python_backend
uv run pytest
~~~

当前回归覆盖：

- SQL mutation / 白名单 / LIMIT guard
- 数据权限 AST rewrite
- conversation memory budget
- runtime budget / telemetry
- durable checkpoint 跨 manager 重开后 resume
- persisted idempotency + bounded retry / args hash conflict
- 生产 SQL planner-critic reject/replan
- tool_search search / deferred loading / dynamic LangGraph bind
- 首轮 tool schema context reduction contract
- frontend MCP chart result → image URL contract
- BIRD SQLite read-only probe
- BIRD direct ChatModel baseline contract
- BIRD LangGraph ReAct / AgentX terminal-state parity contract
- MCP live integration test（显式 opt-in）

本阶段全量本地回归：**31 passed，1 skipped**。

其中 skipped 项是 RUN_MCP_E2E=1 才执行的真实 MCP/MinIO integration test。Windows 本机若全部用例通过后只在 pytest 退出清理 Temp 目录时出现 PermissionError，属于已知临时目录权限问题，不代表测试用例失败。

## 当前简历表述边界

### 已经可以直接写

**工具按需加载**：可以写 Tool Registry、tool_search、deferred binding，以及 7 个真实本地业务工具场景下首轮 schema context 从 2330 chars 降到 379 chars（-83.73%）；该数字不包含运行时 MCP schema，token 只有注明为近似 proxy 时才写。

### 完成真实环境验收后再写“完整验证”

**MCP 集成**：代码、Docker 配置、health、integration test 和前端 contract 已具备；只有 live MCP → MinIO URL → 浏览器实际加载通过后，才建议写“完整验证 E2E 链路”。

### Python 重新实测后再写数字

**Agent 评测**：direct baseline、Java-equivalent LangGraph ReAct/tool loop、runner 和 execution-result 报告链路已经迁移；Python accuracy 只认 fresh-run 的 `pred.python.*.report.md`，不能把历史 Java 69.17%/58.33% 直接冒充当前 Python 成绩。

更完整的面试问答与推荐表述见 INTERVIEW_GUIDE.md。

## 后续还可以强化

- AsyncPostgresSaver + 多实例 checkpoint
- OpenTelemetry / LangSmith trace
- prompt/model/version 固化后的 BIRD 全量多次重复实验
- latency / token / tool-call / retry / failure-rate eval
- HITL SQL approval / time travel
- 真正副作用工具的 transactional outbox / compensation
- 文件 RAG / web router
