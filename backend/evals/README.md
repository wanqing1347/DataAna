# Agent Evaluation Contracts

0.4.0 继续坚持“先确定性指标，再 LLM judge”，并把 BIRD Dev execution-result benchmark 纳入 Python 评测链路。

`cases.json` 为典型问题声明：

- `requiredTools`：该类问题预期经过的关键工具；
- `forbiddenTools`：安全场景中不应该出现的工具；
- `app.evaluation.score_tool_path`：根据结构化 tool trace 计算 coverage 与 pass/fail。

生产主 SQL 路径已经从低层 `validateSql -> executeSql` 升级为：

~~~text
queryData
  -> planner
  -> critic
  -> executor
~~~

因此 eval contract 也改为优先要求 `queryData`，而不是强制主 Agent 暴露 SQL 子图内部的每个实现步骤。

当前建议的分层评测：

1. **Tool path contract**：required / forbidden tools；
2. **SQL safety contract**：AST 白名单、只读、JOIN、LIMIT；
3. **Runtime contract**：tool budget、checkpoint/resume、idempotency/retry；
4. **Text-to-SQL contract**：critic reject 后是否 bounded replan；
5. **Tool discovery contract**：tool_search 命中、deferred bind、首轮 schema context；
6. **BIRD execution accuracy**：direct ChatModel baseline 与 Java AgentX 机制等价的 LangGraph ReAct/tool loop 使用同一 runner 做 SQLite execution-result 对照；
7. **System metrics**：latency、tool-call count、retry count、failure rate；
8. **LLM judge**：最终答案完整性、表达质量，作为软指标而不是唯一门禁。

BIRD Python 结果只认 `scripts/report/pred.python.*.report.md`。仓库旧 `pred.report.md` / `pred.baseline.report.md` 是迁移前 Java 历史结果，不能作为 Python 当前成绩。

运行时 trace 可通过：

~~~text
GET /session/{conversation_id}/runs
~~~

获取，适合作为离线 eval 和回归门禁输入。
