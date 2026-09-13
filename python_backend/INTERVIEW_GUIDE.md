# DataAna Python：Agent 开发岗位面试讲解提纲（0.4.0 简历能力对齐阶段）

## 30 秒项目介绍

我把一个 Java/Spring AI 数据分析项目重构成了 Python + FastAPI + LangGraph 的 Agent runtime。核心不是换语言，而是把 LLM 的不确定决策与确定性工程边界分开：主 Agent 负责理解问题和按需发现工具，业务 Text-to-SQL 用 planner/critic 子图，真正的数据访问仍由 SQL AST guard、数据权限、脱敏、幂等和 retry 强制治理；执行状态用持久化 checkpoint 支持 resume。第四阶段又补上 Tool Registry + tool_search + deferred tools、mcp-echarts/MinIO E2E 验证链路，以及 BIRD Dev 的 direct ChatModel baseline 和 Java AgentX 机制等价的 Python LangGraph ReAct 评测入口。

## 1. 为什么用 LangGraph，而不是普通 AgentExecutor？

回答重点：

1. ReAct 循环是显式图，状态和边界可观察；
2. 可以在 model/tools 之间插入 budget、tool discovery、critic、approval、retry；
3. checkpoint 以图执行边界持久化，适合 fault recovery；
4. SQL planner -> critic -> executor 可以自然做成 bounded 子图；
5. loaded_tools 也是图状态，deferred tool discovery 能真实影响后续 bind；
6. 后续可继续做 interrupt、time travel、subgraph、multi-agent。

不要只说“LangGraph 更流行”。

## 2. Memory 和 checkpoint 到底有什么区别？

一句话：

- **Memory**：下一次业务请求应该记住什么；
- **Checkpoint**：同一次执行中断后应该从哪里继续。

本项目：

~~~text
py_agent_turn -> ConversationMemory -> 下一次新 run 的 prompt context
AsyncSqliteSaver -> LangGraph state/checkpoint -> 同一个 run 的 resume
~~~

为什么 checkpoint thread 不直接用 conversationId？

因为一个 conversation 会有多个独立 run。把整个 conversation 共用一个 execution thread，会把“新请求”和“恢复旧执行”混在一起，也更难做 run 级审计和故障恢复。

## 3. 真正的 crash recovery 为什么不能只换一个持久化 Saver？

因为还需要控制面索引。

~~~text
persist py_agent_run = running
        |
        v
execute LangGraph
        |
        v
durable checkpoint
~~~

如果进程在运行中崩溃，数据库里的 stale running 记录仍能把 user / conversation / runId 与 checkpoint 对应起来。

所以 durable execution 不是“加个 SQLite 文件”，而是：

~~~text
execution state plane + run control plane
~~~

## 4. LangGraph error resume 为什么传 None？

因为这是继续已有 thread 的待执行节点，不是提交新的用户输入。

~~~text
same thread_id
+ existing checkpoint
+ graph.ainvoke/astream(None, config)
~~~

如果重新传原始 input，会从 start 重新执行，可能重复已经成功的节点。

## 5. 为什么 resume 一定要考虑工具幂等？

checkpoint 保证状态可恢复，不保证外部世界副作用不会重复。

典型窗口：

~~~text
tool side effect succeeded
        |
process crashed before next checkpoint
        |
resume
        |
tool may run again
~~~

当前核心执行是只读 SELECT，天然幂等；同时 py_agent_tool_result 还提供 persisted result cache。

如果未来是扣款、发短信或写 SaaS，则要用上游 idempotency key、transactional outbox 或 compensation，而不能宣称本地 cache 提供 exactly-once。

## 6. 工具幂等 key 怎么设计？

当前逻辑键：

~~~text
hash(user_id | conversation_id | tool_name | tool_call_id)
~~~

同时保存 args hash。

理由：

- tool_call_id 表示同一个逻辑调用身份；
- args hash 防止同一 call id 被错误复用于另一组参数；
- 冲突时 fail closed，抛 IdempotencyConflict。

## 7. Retry 为什么不能 catch Exception 后统一重试？

Transient failure：

- DB 连接短暂断开
- connection invalidated
- 典型网络/数据库瞬时异常

Deterministic failure：

- SQL 语法错误
- 非白名单表
- mutation
- 参数错误
- 权限拒绝

确定性错误重试只会放大延迟、负载和副作用风险。当前 SQL executor 只对典型瞬时 SQLAlchemy DB 异常做 bounded exponential backoff。

## 8. 为什么把业务 Text-to-SQL 做成 planner -> critic -> executor 子图？

生产 queryData：

~~~text
planner
  -> deterministic SqlSafetyGuard
  -> semantic critic
  -> reject? replan
  -> executor
  -> guard again
  -> DataScopeRewriter
  -> DB
  -> SensitiveFilter
~~~

好处：

1. Text-to-SQL 是明确 bounded workflow；
2. planner/critic 可以独立测试；
3. critic feedback 能定向驱动 replan；
4. 主 Agent 只决定“何时需要查询”；
5. SQL 安全仍由 deterministic executor 兜底。

## 9. LLM critic 和 SQL guard 有什么区别？

LLM critic 擅长语义：

- 查询粒度是否回答问题
- 聚合是否合理
- 时间范围是否匹配
- JOIN 是否可能重复计数

SQL guard 擅长硬规则：

- 只允许 SELECT/WITH
- 表白名单
- JOIN 上限
- mutation 拒绝
- 危险函数
- LIMIT

因此：

~~~text
semantic correctness -> LLM critic
hard safety          -> deterministic guard
~~~

即使 critic approve，executor 仍再次 guard。

## 10. Text-to-SQL 越权怎么防？

三层：

1. SqlSafetyGuard：白名单 + 只读 AST + JOIN/LIMIT；
2. DataScopeRewriter：根据 SELF / DEPT / DEPT_AND_SUB / ALL 注入 AST 条件；
3. SensitiveFilter：结果集敏感列脱敏。

关键面试句：**数据权限不能让模型自己拼 WHERE。**

## 11. 为什么 planner/critic 显式使用 function_calling structured output？

项目连接的是 DeepSeek OpenAI-compatible endpoint。

“兼容 OpenAI API protocol”不等于“完整支持 OpenAI 所有 Structured Outputs 特性”，所以显式使用：

~~~python
with_structured_output(..., method="function_calling")
~~~

这是兼容性和可迁移性判断，不是单纯语法偏好。

## 12. 什么叫 tool_search / deferred tools？为什么不是 prompt 技巧？

0.4.0 之前主 Agent 会：

~~~text
model.bind_tools(all_tools)
~~~

也就是说 listTables、describeTables、lookupGlossary、queryData、calculate、底层 SQL 工具和 MCP chart schema 会全部进入模型首轮上下文。

现在：

~~~text
first turn:
model.bind_tools([tool_search])

tool_search result:
loaded_tools = ["queryData"]

next turn:
model.bind_tools([tool_search, queryData])
~~~

Tool Registry 既控制搜索，也控制 bind 和 ToolNode 执行白名单。

所以 deferred 的含义是：

- schema 没有预先进入模型上下文；
- 工具没搜索出来前也不能执行；
- 搜索结果通过 LangGraph state 持久到后续轮次。

## 13. tool_search 怎么做检索？为什么不用 embedding/vector DB？

当前工具规模很小，Registry 使用：

- tool name
- description
- source
- curated aliases
- ASCII token
- 中文 n-gram

做轻量确定性排名。

理由：

1. 工具集合只有个位数到十几；
2. embedding 会引入网络依赖、成本和额外 failure mode；
3. 当前目标是验证 deferred loading 架构，不是打造通用 tool marketplace。

当工具规模上百后，再考虑 BM25 / embedding / hybrid retrieval / capability taxonomy。

## 14. 如何证明按需加载真的降低了上下文？

不是用“应该更省 token”这种口头描述，而是把真实 StructuredTool schema 序列化后测量。

脚本：

~~~text
python_backend/scripts/tool_context_metrics.py
~~~

结果：

| 方案 | schema 数 | schema chars | chars/4 proxy |
| --- | ---: | ---: | ---: |
| eager | 8 | 2330 | 583 |
| first-turn deferred | 1 | 379 | 95 |

schema 字符量下降 **83.73%**。

面试时要主动说明：583 / 95 是 chars÷4 的透明近似，不是 tokenizer 精确 token；最可靠的硬指标是 2330 → 379 chars。

## 15. 为什么 MCP chart 也应该 deferred？

图表不是所有数据分析请求都需要。

如果每轮都提前注入 MCP chart schema，会产生：

- 无意义 context cost；
- 工具选择干扰；
- MCP server 不可用时的额外复杂度。

所以当前流程是：

~~~text
user needs visualization
  -> tool_search("chart / echarts")
  -> registry only returns MCP tool if MCP connected
  -> next round bind chart tool
~~~

MCP 本身也变成能力注册源，而不是永远在线的硬依赖。

## 16. Python MCP 链路怎么接？

~~~text
LangGraph
  -> langchain-mcp-adapters
  -> mcp-echarts /mcp
  -> MinIO object
  -> object/public URL
  -> Nginx /dataana-charts/
  -> frontend image
~~~

AgentService startup 会调用 MultiServerMCPClient.get_tools()。

与旧实现不同，失败不再静默：

- /health 返回 enabled / connected / url / tools / error；
- MCP 连接失败时不向 Registry 注册 chart tool；
- 日志保留连接失败原因。

## 17. MCP E2E 怎么验证才算完整？

分四层，不要只测“能列出工具”：

1. adapter discovery：Python 能发现 MCP chart tool；
2. invocation：真实调用 mcp-echarts；
3. object URL：返回值中能提取 URL，GET 为 HTTP 200 且非空；
4. browser contract：前端能把 tool result 转成 img URL，同域 HTTP URL 能经 Nginx 改成 HTTPS。

仓库：

~~~text
scripts/mcp_e2e_check.py
tests/integration/test_mcp_echarts_e2e.py
tests/test_frontend_chart_contract.py
~~~

完整浏览器视觉验收仍应在 live Docker/服务器环境打开页面实际看一次，不把静态 contract test 冒充真实浏览器 E2E。

## 18. 为什么 MCP integration test 默认 skip？

因为它依赖真实外部基础设施：

- mcp-echarts
- MinIO
- 可访问 object URL

默认 unit suite 应该稳定、可离线运行。

所以只有设置：

~~~env
RUN_MCP_E2E=1
~~~

才执行 live integration test。

这体现了 test pyramid 和 hermetic tests 的边界。

## 19. BIRD 为什么要做 direct ChatModel baseline？

没有 baseline，只报告 Agent accuracy，无法说明提升来自 Agent 架构还是模型本身。

当前两条路径使用同一套模型配置和相同 BIRD question/evidence/schema 信息：

~~~text
baseline:
full schema -> direct structured ChatModel -> SQL

agent:
question + evidence + output schema + <no_think>
  -> LangGraph ReAct
  -> listTables / describeTables / verifySql
  -> structured BirdAgentOutput -> SQL
~~~

最后都由同一个 runner 做 execution-result scoring。

## 20. Python BIRD Agent 如何对齐 Java ReactAgent？

这里不是把 Java 代码逐行翻译成 Python，而是迁移它真正影响模型轨迹的 runtime 语义：

- Java `RunnableParams.outputType(BirdAgentOutput.class)` 会把结构化输出 schema 追加到用户消息，Python 同样追加 `BirdAgentOutput` JSON schema；
- Java ThinkingMode 默认关闭时会追加 `<no_think>`，Python 保持一致；
- 模型无 tool call 时 ReAct 正常结束，而不是由 Python 框架强制继续调用 verifier；
- 普通工具执行后回到模型；final-mode `verifySql` 的 Java-style execution/structural verifier 不再单独拥有终止权，而是先通过 `BirdSqlVerifier`，再通过 ReAct 启动前由 question/evidence/schema 固化的 immutable `TaskContract` semantic gate；只有两个 Gate 都通过才写 `last_passed_sql` 并立即 `END`；
- `requestedColumns` 降级为候选声明，必须与 `TaskContract.expectedOutputs` 一致，不能由 Agent 通过少报输出字段来改变最终 verifier 真值；
- semantic gate 先做确定性 output/answer-shape/predicate/source/grouping 检查，复杂语义再交给独立、无工具、temperature=0 的 critic；critic 故障 fail-closed；
- exploratory probe 默认最多 8 次；达到上限后 probe-mode 调用被拒绝。若 final structural pass 但 semantic fail，则提前进入 targeted-repair mode，不再开放 probe，只允许最多 3 次 corrected final verify；
- `maxRounds` 命中且仍有 pending tool call 时，跳过该工具并做一次 force-final；
- `verifySql` 默认把完整执行结果交给模型，而不是只截取 preview rows；
- 模型轮调用失败按 AgentX 默认语义配置最多 3 次 retry。

Python 唯一刻意保留的强化边界是 SQLite 只读与 timeout；它们用于保护 benchmark endpoint，不改写合法 SELECT 的结果。

## 21. BIRD eval 如何防止 gold leakage？

生成阶段只接收：

- question
- evidence
- SQLite schema
- candidate SQL 的只读执行结果与 verifier diagnostics

不会传：

- gold SQL
- gold result
- correct/incorrect label

gold SQL 只在 runner 的最终评分阶段执行，用于 execution-result equivalence。

## 22. BIRD endpoint 为什么默认关闭？

请求包含 sqlitePath。

如果生产环境开放，就可能变成本地文件探查入口。因此：

~~~env
BIRD_EVAL_ENABLED=false
~~~

默认 fail closed，只在本地 benchmark 环境显式开启。

这是“评测代码也要考虑安全边界”的面试点。

## 23. execution accuracy 怎么算？为什么不能预设 69.17% / 58.33%？

现有 runner 会分别执行：

- predicted SQL
- gold SQL

然后比较结果集合是否一致，生成 total EX。

迁移前仓库已有：

- Java Agent 历史：69.17%
- Java baseline 历史：58.33%

这些值证明旧 Java 评测曾跑过，但**不是 Python/LangGraph 结果**。

Python 只有生成：

~~~text
pred.python.baseline.report.md
pred.python.agent.report.md
~~~

之后，才能把新的 total EX 写进简历。

面试时如果还没有重新实测，宁可说“评测链路已迁移，当前正在固定模型/数据版本做重测”，也不要偷用历史数字。

## 24. 为什么 BIRD benchmark 子图和生产 queryData 子图不是同一个？

两者优化目标不同。

生产 queryData：

- MySQL
- 业务表白名单
- DataScope 权限
- SensitiveFilter
- SQL LIMIT/安全策略
- 幂等 retry

BIRD eval：

- 每题独立 SQLite
- benchmark schema
- execution accuracy
- 不应注入业务权限条件
- 不应改写候选 SQL 来迎合生产约束

如果强行复用同一执行器，反而会污染 benchmark。

## 25. 当前测试证据是什么？

0.4.0 本地全量：

~~~text
31 passed, 1 skipped
~~~

覆盖：

- SQL AST guard
- data scope rewrite
- memory
- checkpoint resume
- idempotency / retry
- production SQL critic replan
- Tool Registry search
- LangGraph dynamic deferred bind
- tool context reduction
- frontend chart contract
- BIRD read-only SQLite probe
- BIRD baseline
- BIRD ReAct / AgentX terminal-state parity

1 个 skipped 是显式 opt-in 的 live MCP/MinIO E2E。

Windows 测试结束后的 pytest Temp cleanup PermissionError 是宿主机临时目录权限问题，不代表用例失败。

## 26. 白板时可以画的总架构

~~~text
                         +-----------------------+
User -> FastAPI -------->| AgentService          |
                         | run / stop / resume   |
                         +-----------+-----------+
                                     |
                                     v
                         +-----------------------+
                         | Main LangGraph        |
                         |                       |
                         | model                 |
                         |   |                   |
                         | tool_search           |
                         |   | loaded_tools      |
                         | dynamic bind          |
                         |   |                   |
                         | ToolNode allow-list   |
                         +-----+-----------+-----+
                               |           |
                  queryData ---+           +--- MCP chart
                      |                         |
                      v                         v
             +------------------+        mcp-echarts
             | planner          |             |
             | critic           |           MinIO
             | executor         |             |
             +--------+---------+          public URL
                      |                         |
                SqlSafetyGuard                 v
                      |                     Browser
               DataScopeRewriter
                      |
             IdempotentToolRunner
                      |
                     DB
                      |
               SensitiveFilter

Durability:
- SQLite checkpoint: execution state
- MySQL py_agent_run: run control plane
- MySQL py_agent_tool_result: idempotency/retry
- MySQL py_agent_turn: business memory

Evaluation:
- direct ChatModel baseline
- BIRD LangGraph ReAct（Java AgentX 机制迁移）
- shared execution-result runner
~~~

## 27. 如果面试官问“这一阶段最重要的工程判断是什么？”

可以概括成三点：

第一，我没有把“工具很多”只当作 prompt 长的问题，而是把工具注册、检索、模型可见 schema 和执行白名单统一进 Tool Registry，让 deferred loading 成为运行时语义。

第二，MCP 不是“能连接 server 就算集成”，而是把 discovery、tool invocation、MinIO object URL、Nginx 浏览器访问和前端渲染拆成可验证链路。

第三，Agent eval 不能只报一个漂亮分数。先建立同模型 direct baseline，再评 Java-equivalent Python LangGraph ReAct，并且只认重新实测的 execution accuracy，不把 Java 历史结果冒充 Python 结果。

## 28. 当前推荐简历表述

### 工具按需加载——现在可以直接使用

**工具按需加载：**围绕数据分析场景设计 Tool Registry 与 tool_search，将 listTables、describeTables、lookupGlossary、queryData、calculate 及 MCP 图表能力注册为 deferred tools；LangGraph 根据搜索结果动态 bind 工具并以 ToolNode 白名单约束执行。在 7 个真实本地业务工具场景下，首轮 tool schema context 从 2330 chars 降至 379 chars，减少 83.73%（该测量不包含运行时 MCP schema）。

### MCP 集成——live E2E 通过后使用完整版

当前代码已经支持的保守表述：

**MCP 集成：**通过 langchain-mcp-adapters 将 mcp-echarts 接入 Python 数据分析 Agent，打通 chart tool 注册、MinIO 对象 URL、Nginx 浏览器访问与前端图表解析，并补充 MCP health、Docker 部署配置及可重复 integration test。

只有真实环境 MCP → MinIO → browser 验收通过后，再把“打通”升级成“完整验证端到端链路”。

### Agent 评测——必须填 Python 实测数字

在 Python report 产生之前：

**Agent 评测：**迁移 BIRD Dev Text-to-SQL 评测链路，将 Java AgentX 的结构化终态、ReAct 工具循环、max-round force-final 与完整 `verifySql` execution feedback 迁移到 LangGraph，并复用同一 runner 与 direct ChatModel baseline 做 execution-result equivalence 对照。

重测完成后再追加：

~~~text
LangGraph ReAct EX = <PYTHON_AGENT_EX>%
Direct ChatModel baseline EX = <PYTHON_BASELINE_EX>%
提升 = <DELTA> 个百分点
~~~

禁止直接填历史 Java 的 69.17% / 58.33%。

## 29. 如果岗位更偏高级 Agent 平台，下一步怎么讲？

优先级：

- AsyncPostgresSaver + 多实例 checkpoint
- OpenTelemetry / LangSmith
- prompt/model/version 固化后的 BIRD 重复实验与置信区间
- token / latency / tool-call / retry / failure-rate eval
- HITL SQL approval
- checkpoint retention / time travel
- 真正副作用工具的 outbox / compensation
- 大规模工具库的 embedding / hybrid tool retrieval
