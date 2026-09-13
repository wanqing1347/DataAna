from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_chunk_to_message,
)
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from .bird_schema import SqliteSchemaProvider
from .bird_semantic import BirdSemanticGate, BirdTaskContract
from .bird_verifier import BirdSqlVerifier
from .config import Settings


COMMON_INSTRUCTIONS = """# BIRD SQL Evaluation

You are solving one BIRD-SQL Dev question. BIRD scores by execution-result matching:
every result row is an ordered tuple, so extra columns, wrong column order, inferred
filters, or reshaped identifiers make an otherwise correct answer wrong. Output one
executable SQLite `SELECT` or `WITH` statement.

## SQL rules

- Return only the values the question asks for, in the wording order. No helper or
  explanation columns.
- Decide the row grain first: one row per requested entity or group, one detail row,
  or one aggregate row. Do not aggregate at another grain unless the question asks.
- Join with declared foreign keys or exact stable identifiers, never a display name.
  Use `LEFT JOIN` when the attached information is optional ("if any").
- Compare text identifiers exactly as stored: leading zeros and length matter; do not
  pad, cast, trim, or slice them unless evidence explicitly requires it.
- Do not assume a date/time format: follow the schema examples/descriptions and the
  question/evidence conventions.
- Apply every formula, mapping, exclusion, and date convention from the question and
  evidence literally; do not substitute a similar-looking field. If evidence defines
  an aggregate formula (for example SUM(x)/COUNT(...)), the final predicate must keep
  that aggregate structure; never collapse it into a row-level x > threshold test.
  Treat explicit question/evidence constraints as invariants: exploratory probes may
  add diagnostics, but the final SQL must not silently drop a required filter, mapping,
  grouping key, formula, or join condition that was already established.
- When similar concepts exist in multiple tables, compare descriptions, scope, and
  examples, then pick the source matching the requested entity and measure.
- For geographic or organizational names, establish whether the schema means city,
  county, district, state, or another scope. If the same place name can match multiple
  scopes, compare the candidate columns/values before committing; do not assume the
  first plausible physical-location field is the intended scope.
- Preserve literal semantic distinctions from the question/evidence when the schema has
  near-synonym columns (for example offered vs served, opened vs closed, school vs
  district). Probe results may disambiguate values, but equal/non-empty execution
  results are not evidence to replace a wording-faithful mapping with a near synonym.
- When multiple columns could supply a requested attribute, inspect their descriptions
  and table grain before choosing. A syntactically valid or non-null result does not
  prove that the selected concept is the one requested.
- Do not infer filters from field names or common sense; do not use a BIRD metadata
  field marked `unuseful` unless the question or evidence names it.
- Check whether a one-to-many join expanded rows or changed the aggregate scope.
"""

AGENT_INSTRUCTIONS = """## Agent workflow

1. Call `listTables` to see tables and foreign keys, then `describeTables` for the
   relevant tables. Treat column descriptions, value mappings, and examples as
   authoritative metadata.
2. The runtime supplies an immutable TaskContract before exploration. Derive
   `requestedColumns` from that contract: exactly the values the question asks to
   return (exact field names or contract-approved aliases), no helper columns and none
   missing, in the final SELECT order. You may repair SQL/requestedColumns, but you may
   not weaken, rewrite, or replace the framework-owned TaskContract.
3. Before broad probing, write down the literal schema commitments implied by the
   wording/evidence (for example `offers` -> an offered-grade field, not a served-grade
   field). Treat every TaskContract `ambiguities[]` entry as an explicit obligation to
   compare the listed candidate columns/sources before final verification; never resolve
   it by simply taking the first candidate. If a place phrase has no explicit scope noun,
   enumerate every plausible described scope column (city, county, district, state, and
   measure-table equivalents) that can match it; do not stop after testing only one or two
   scopes. If a phrase is ambiguous across near-synonym or geographic columns, use a small
   discriminating probe that compares the competing interpretations directly. If the
   contract contains `requiredAggregateFormulas`, preserve the evidence-defined aggregate
   expression (for example SUM(...)/COUNT(...)) and keep aggregate thresholds in HAVING;
   never collapse them into row-level filters such as `AvgScrMath > 400`.
   Do not switch away from a wording-faithful mapping merely because another candidate
   also returns rows or happens to return the same result. Prefer a coherent dimension
   source when one table contains the requested entity label, requested categorical
   attribute, and a plausible scope field at the same grain; join the measure table by
   its stable key rather than mixing similarly named dimensions across tables without
   a semantic reason.
4. Probe with `verifySql(sql, [])` to inspect real values, identifier formats, dates,
   join direction, row grain, or why a draft is empty. Exploratory probes have a
   separate finite runtime budget; once that budget is exhausted, do not issue more
   probe-mode calls and move directly to a final-mode `verifySql` with non-empty
   `requestedColumns`. When a join only attaches an attribute to the question's subject
   entity, compare the joined row count against the un-joined filtered count of the
   subject table; a big drop means entities are being lost and the join may need to be
   a LEFT JOIN. A probe-mode `passed=true` is never approval to terminate.
5. Before the final check, audit a compact constraint ledger against the SQL: requested
   outputs, every explicit question/evidence filter or mapping, geographic/organizational
   scope, formulas, grouping grain, and required joins. If a condition appeared in a
   correct discriminating probe but is absent from the candidate final SQL, restore it
   unless the question explicitly made it irrelevant.
6. Final-check with `verifySql(sql, requestedColumns)`. You MUST make at least one
   non-empty `requestedColumns` final-mode call before considering the answer complete.
   The tool first runs structural/execution verification and then the immutable semantic
   TaskContract gate; it never rewrites SQL or injects LIMIT/filters. If `passed=false`,
   apply each `errors[].fix` and re-verify. A semantic failure switches the runtime into
   targeted-repair mode: exploratory probes remain closed and only corrected final-mode
   verification is allowed. Only structuralPassed=true AND semantic.passed=true yields
   top-level `passed=true`; only that combined result terminates the graph.

Return exactly the SQL that passed the last final-mode `verifySql` call. No markdown,
no alternatives.
"""

BASELINE_INSTRUCTIONS = """## No-tool baseline

Database tools are unavailable; the complete schema is embedded below. Apply the SQL
rules above to the embedded schema only. Do not invent tables, columns, relationships,
or values. Return exactly one JSON object matching the structured-output contract:
one statement, no markdown, no alternatives.
"""


class BirdEvalRequest(BaseModel):
    questionId: str = ""
    dbId: str = ""
    question: str
    evidence: str = ""
    sqlitePath: str
    maxRounds: int | None = None


class BirdAgentOutput(BaseModel):
    # Jackson can deserialize a missing Java field as null; use a blank default so
    # Python accepts the same payload shape and lets the service apply its blank-SQL check.
    sql: str = Field(
        default="",
        description=(
            "本题最终 SQLite SELECT/WITH 查询 SQL。必须是单条可直接执行的查询，"
            "不要 markdown，不要分号。"
        ),
    )
    requestedColumns: list[str] = Field(
        default_factory=list,
        description=(
            "Values explicitly requested by the question/evidence, listed in final SELECT "
            "order. Use the exact schema field name when one exists, or a stable output alias "
            "otherwise. Include a rank measure only when needed; never add explanatory entity "
            "names or helper columns."
        ),
    )


_JSON_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"

_LIST_TABLES_ARGS_SCHEMA: dict[str, Any] = {
    "$schema": _JSON_SCHEMA_DRAFT,
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}

_DESCRIBE_TABLES_ARGS_SCHEMA: dict[str, Any] = {
    "$schema": _JSON_SCHEMA_DRAFT,
    "type": "object",
    "properties": {
        "tableNames": {
            "type": "array",
            "items": {"type": "string"},
            "description": '要查看的表名列表，如 ["customer", "rental"]',
        }
    },
    "required": ["tableNames"],
    "additionalProperties": False,
}

_VERIFY_SQL_ARGS_SCHEMA: dict[str, Any] = {
    "$schema": _JSON_SCHEMA_DRAFT,
    "type": "object",
    "properties": {
        "sql": {
            "type": "string",
            "description": "SQLite SELECT/WITH statement to execute exactly as written",
        },
        "requestedColumns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Empty for a probe; otherwise exact requested output fields in SELECT order",
        },
    },
    "required": ["sql", "requestedColumns"],
    "additionalProperties": False,
}


class BirdAgentState(MessagesState):
    model_rounds: int
    max_rounds: int
    tool_rounds: int
    probe_calls: int
    semantic_repairs: int
    repair_mode: bool
    semantic_repair_exhausted: bool
    task_contract: dict[str, Any]
    last_passed_sql: str
    last_final_sql: str
    last_requested_columns: list[str]
    last_verification: dict[str, Any]
    last_semantic_verification: dict[str, Any]
    last_error: str
    tool_history: list[dict[str, Any]]


class _AgentXCompatibleChatOpenAI(ChatOpenAI):
    """Preserve the DeepSeekV4ChatModel chat-completions wire format used by AgentX."""

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        wire_messages = payload.get("messages")
        if not isinstance(wire_messages, list):
            return payload

        source_messages = self._convert_input(input_).to_messages()
        tool_names = {
            message.tool_call_id: message.name
            for message in source_messages
            if isinstance(message, ToolMessage) and message.name
        }
        for message in wire_messages:
            if not isinstance(message, dict):
                continue
            if (
                message.get("role") == "assistant"
                and message.get("tool_calls")
                and message.get("content") is None
            ):
                # DeepSeekV4ChatModel normalizes null assistant text to an empty string.
                message["content"] = ""
            if message.get("role") == "tool":
                tool_name = tool_names.get(str(message.get("tool_call_id") or ""))
                if tool_name:
                    # Spring AI DeepSeekApi sends both name and tool_call_id for tool responses.
                    message["name"] = tool_name
        return payload


class BirdEvalService:
    def __init__(self, settings: Settings, model=None, semantic_model=None):
        self.settings = settings
        owns_model = model is None
        self.model = model or _AgentXCompatibleChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=settings.bird_eval_temperature,
            timeout=300,
            # AgentX ReactAgent retries an LLM round up to three times.
            max_retries=3,
        )
        if semantic_model is not None:
            self.semantic_model = semantic_model
        elif owns_model and settings.bird_eval_semantic_gate_enabled:
            # Deliberately separate from the ReAct model/tool loop: no tools, temperature 0,
            # and a fresh message context for both task-contract construction and criticism.
            self.semantic_model = _AgentXCompatibleChatOpenAI(
                model=settings.deepseek_model,
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                temperature=0.0,
                timeout=120,
                max_retries=2,
            )
        else:
            # Test doubles and explicit injected agent models keep a deterministic fallback
            # unless the caller also injects a semantic_model.
            self.semantic_model = None

    async def eval_baseline(self, req: BirdEvalRequest) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            path = self._validate_request(req)
            provider = SqliteSchemaProvider(path)
            schema = await asyncio.to_thread(provider.describe_all_tables)
            system_prompt = (
                COMMON_INSTRUCTIONS
                + "\n\n"
                + BASELINE_INSTRUCTIONS
                + f"\n\n## Current database\n\nDatabase ID: {req.dbId}"
                + "\n\n## Complete schema\n\n"
                + schema
                + "\n\n## Structured output\n\n"
                + _structured_output_format()
            )
            output = await self.model.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=_build_query(req)),
                ]
            )
            sql = _parse_sql_output(output)
            if not sql:
                raise ValueError("baseline returned blank SQL")
            return self._response(
                started,
                success=True,
                sql=sql,
                mode="baseline",
                attempts=1,
                approved=None,
            )
        except Exception as exc:
            return self._response(
                started,
                success=False,
                sql="",
                error=f"{type(exc).__name__}: {exc}",
                mode="baseline",
                attempts=1,
                approved=None,
            )

    async def eval_agent(self, req: BirdEvalRequest) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            path = self._validate_request(req)
            provider = SqliteSchemaProvider(path)
            verifier = BirdSqlVerifier(
                path,
                request_context=_build_review_context(req),
                timeout_seconds=self.settings.bird_eval_sql_timeout_seconds,
                schema_provider=provider,
            )
            semantic_gate = BirdSemanticGate(
                model=(
                    self.semantic_model
                    if self.settings.bird_eval_semantic_gate_enabled
                    else None
                ),
                provider=provider,
                question=req.question,
                evidence=req.evidence,
                critic_enabled=self.settings.bird_eval_semantic_critic_enabled,
            )
            task_contract = (
                await semantic_gate.build_contract()
                if self.settings.bird_eval_semantic_gate_enabled
                else BirdTaskContract(source="disabled")
            )
            max_rounds = int(req.maxRounds or self.settings.bird_eval_max_rounds)
            max_rounds = max(1, max_rounds)
            graph = self._build_agent_graph(
                req,
                provider,
                verifier,
                semantic_gate,
                task_contract,
            )
            final = await graph.ainvoke(
                {
                    "messages": [HumanMessage(content=_build_agent_query(req))],
                    "model_rounds": 0,
                    "max_rounds": max_rounds,
                    "tool_rounds": 0,
                    "probe_calls": 0,
                    "semantic_repairs": 0,
                    "repair_mode": False,
                    "semantic_repair_exhausted": False,
                    "task_contract": task_contract.model_dump(),
                    "last_passed_sql": "",
                    "last_final_sql": "",
                    "last_requested_columns": [],
                    "last_verification": {},
                    "last_semantic_verification": {},
                    "last_error": "",
                    "tool_history": [],
                },
                config={"recursion_limit": max_rounds * 3 + 16},
            )
            verification = dict(final.get("last_verification") or {})
            semantic_verification = dict(
                final.get("last_semantic_verification") or {}
            )
            semantic_repairs = int(final.get("semantic_repairs") or 0)
            last_passed_sql = str(final.get("last_passed_sql") or "").strip()
            final_messages = list(final.get("messages") or [])
            output = _parse_agent_output(final_messages[-1] if final_messages else None)
            if last_passed_sql:
                # A passed final verify is already the deterministic terminal result.
                # Do not depend on an additional model turn to restate it as JSON.
                sql = last_passed_sql
                requested_columns = list(final.get("last_requested_columns") or [])
                approved = True
            else:
                sql = (output.sql if output is not None else "").strip()
                requested_columns = (
                    list(output.requestedColumns)
                    if output is not None
                    else list(final.get("last_requested_columns") or [])
                )
                approved = verification.get("passed") if verification else None
            if (
                self.settings.bird_eval_semantic_gate_enabled
                and self.semantic_model is not None
                and not last_passed_sql
            ):
                return self._response(
                    started,
                    success=False,
                    sql="",
                    error=(
                        str(final.get("last_error") or "").strip()
                        or "FINAL_SEMANTIC_VERIFICATION_REQUIRED"
                    ),
                    mode="langgraph-react-tools",
                    attempts=int(final.get("model_rounds") or 0),
                    approved=False,
                    critic=json.dumps(
                        verification.get("errors", []), ensure_ascii=False
                    ),
                    probe=json.dumps(
                        verification.get("execution", {}), ensure_ascii=False, default=str
                    ),
                    tool_rounds=int(final.get("tool_rounds") or 0),
                    probe_calls=int(final.get("probe_calls") or 0),
                    semantic_repairs=semantic_repairs,
                    requested_columns=requested_columns,
                    task_contract=task_contract.model_dump(),
                    semantic_verification=semantic_verification,
                    tool_history=list(final.get("tool_history") or []),
                )
            if not sql:
                return self._response(
                    started,
                    success=False,
                    sql="",
                    error=(
                        str(final.get("last_error") or "").strip()
                        or (
                            "ValueError: agent returned blank or invalid BirdAgentOutput "
                            "structured output"
                        )
                    ),
                    mode="langgraph-react-tools",
                    attempts=int(final.get("model_rounds") or 0),
                    approved=approved,
                    critic=json.dumps(
                        verification.get("errors", []), ensure_ascii=False
                    ),
                    probe=json.dumps(
                        verification.get("execution", {}), ensure_ascii=False, default=str
                    ),
                    tool_rounds=int(final.get("tool_rounds") or 0),
                    probe_calls=int(final.get("probe_calls") or 0),
                    semantic_repairs=semantic_repairs,
                    requested_columns=requested_columns,
                    task_contract=task_contract.model_dump(),
                    semantic_verification=semantic_verification,
                    tool_history=list(final.get("tool_history") or []),
                )
            return self._response(
                started,
                success=True,
                sql=sql,
                mode="langgraph-react-tools",
                attempts=int(final.get("model_rounds") or 0),
                approved=approved,
                critic=json.dumps(verification.get("errors", []), ensure_ascii=False),
                probe=json.dumps(
                    verification.get("execution", {}), ensure_ascii=False, default=str
                ),
                tool_rounds=int(final.get("tool_rounds") or 0),
                probe_calls=int(final.get("probe_calls") or 0),
                semantic_repairs=semantic_repairs,
                requested_columns=requested_columns,
                task_contract=task_contract.model_dump(),
                semantic_verification=semantic_verification,
                tool_history=list(final.get("tool_history") or []),
            )
        except Exception as exc:
            return self._response(
                started,
                success=False,
                sql="",
                error=f"{type(exc).__name__}: {exc}",
                mode="langgraph-react-tools",
                attempts=0,
                approved=False,
            )

    def _build_agent_graph(
        self,
        req: BirdEvalRequest,
        provider: SqliteSchemaProvider,
        verifier: BirdSqlVerifier,
        semantic_gate: BirdSemanticGate,
        task_contract: BirdTaskContract,
    ):
        list_tables_tool = StructuredTool.from_function(
            name="listTables",
            description=(
                "列出当前数据库的全部表和视图，含描述、关联表（外键指向的其它表）。"
                "回答数据问题的第一步：根据用户问题与表描述，挑出相关的表，"
                "再用 describeTables 查看这些表的字段详情。"
                "关联表能帮你快速判断要 JOIN 哪些表"
            ),
            func=provider.list_tables,
            args_schema=_LIST_TABLES_ARGS_SCHEMA,
        )
        describe_tables_tool = StructuredTool.from_function(
            name="describeTables",
            description=(
                "查看指定表/视图的字段详情（字段名/类型/可空/键/注释/外键/示例值）。"
                "写 SQL 前必须先调本工具看真实结构，严禁凭记忆写 SQL。"
                "建议先调 listTables 挑出相关表，再把表名传给本工具。"
                "字段是否当指标/维度/时间，由你（LLM）根据字段注释 + 类型 + 示例值自行判断。"
            ),
            func=provider.describe_tables,
            args_schema=_DESCRIBE_TABLES_ARGS_SCHEMA,
        )
        verify_sql_tool = StructuredTool.from_function(
            name="verifySql",
            description=(
                "Execute one SQLite SELECT/WITH statement exactly as written, then return its complete "
                "result and structured semantic diagnostics. Pass requestedColumns=[] to probe real values, "
                "join shape, or an empty result. Pass the non-empty exact output field list in SELECT order "
                "to verify a final answer. The tool never rewrites SQL or injects LIMIT, permissions, "
                "read-only transformation, timeout, formatting, or row truncation."
            ),
            func=lambda sql, requestedColumns: json.dumps(
                verifier.verify(sql, requestedColumns).as_tool_dict(
                    self.settings.bird_eval_tool_result_max_rows
                ),
                ensure_ascii=False,
                default=str,
            ),
            args_schema=_VERIFY_SQL_ARGS_SCHEMA,
        )
        tools = [list_tables_tool, describe_tables_tool, verify_sql_tool]
        # AgentX does not disable provider-side parallel tool calls; keep the provider default.
        bound_model = self.model.bind_tools(tools)

        max_probe_calls = max(0, int(self.settings.bird_eval_max_probe_calls))
        max_semantic_repairs = max(
            1,
            int(self.settings.bird_eval_max_semantic_repairs),
        )
        probe_budget_prompt = (
            f"\n\n## Exploratory probe budget\n\nAt most {max_probe_calls} probe-mode "
            "verifySql calls may execute. Once the budget is exhausted, only final-mode "
            "verifySql calls with non-empty requestedColumns are allowed. After any semantic "
            "final-check failure, exploratory probes stay disabled even if budget remains."
            if max_probe_calls > 0
            else (
                "\n\n## Semantic repair mode\n\nAfter any semantic final-check failure, "
                "exploratory probes are disabled; only corrected final-mode verification is allowed."
            )
        )
        contract_prompt = (
            "\n\n## Immutable TaskContract\n\n"
            + json.dumps(
                task_contract.model_dump(exclude={"source"}),
                ensure_ascii=False,
                indent=2,
            )
            + "\n\nThis contract is framework-owned and cannot be weakened by changing requestedColumns. "
            f"At most {max_semantic_repairs} targeted semantic repair attempts are allowed."
        )
        system_prompt = (
            COMMON_INSTRUCTIONS
            + "\n\n"
            + AGENT_INSTRUCTIONS
            + probe_budget_prompt
            + contract_prompt
            + f"\n\n## Current database\n\nDatabase ID: {req.dbId}"
        )

        async def call_model(state: BirdAgentState):
            rounds = int(state.get("model_rounds", 0))
            if rounds >= int(state.get("max_rounds", 1)):
                return {"last_error": "MAX_ROUNDS_REACHED"}
            response = await _invoke_agent_model_streaming(
                bound_model,
                [SystemMessage(content=system_prompt), *state["messages"]],
            )
            response = _sanitize_ai_tool_calls_for_history(
                response,
                round_number=rounds + 1,
            )
            return {
                "messages": [response],
                "model_rounds": rounds + 1,
            }

        async def call_tools(state: BirdAgentState):
            last = state["messages"][-1]
            tool_calls = list(getattr(last, "tool_calls", None) or [])
            invalid_tool_calls = list(getattr(last, "invalid_tool_calls", None) or [])
            updates: dict[str, Any] = {
                "tool_rounds": int(state.get("tool_rounds", 0)) + 1,
                "last_error": "",
            }
            history = list(state.get("tool_history") or [])
            probe_calls = int(state.get("probe_calls", 0))
            semantic_repairs = int(state.get("semantic_repairs", 0))
            repair_mode = bool(state.get("repair_mode", False))

            async def execute_one(
                call: dict[str, Any],
                *,
                fallback_index: int,
                probe_count: int,
                repair_active: bool,
                semantic_repair_count: int,
            ) -> dict[str, Any]:
                name = str(call.get("name") or "")
                call_id = str(call.get("id") or f"tool-{fallback_index}")
                trace: dict[str, Any] = {"name": name}
                item_updates: dict[str, Any] = {}
                probe_increment = 0
                semantic_repair_increment = 0
                enter_repair_mode = False
                repair_exhausted = False
                final_passed = False
                try:
                    args = _coerce_tool_args(call.get("args"))
                    if name == "listTables":
                        result_text = await asyncio.to_thread(provider.list_tables)
                    elif name == "describeTables":
                        table_names = args.get("tableNames")
                        trace["tableNames"] = table_names
                        if not isinstance(table_names, list):
                            raise ValueError("tableNames must be a list of exact table names")
                        result_text = await asyncio.to_thread(
                            provider.describe_tables, table_names
                        )
                    elif name == "verifySql":
                        sql = str(args.get("sql") or "").strip()
                        requested = args.get("requestedColumns")
                        if requested is None:
                            raise ValueError(
                                "requestedColumns is required; use [] explicitly for probe mode"
                            )
                        if not isinstance(requested, list):
                            raise ValueError("requestedColumns must be a list")
                        requested = [str(value or "").strip() for value in requested]
                        trace["requestedColumns"] = requested
                        trace["mode"] = "final" if requested else "probe"
                        trace["sql"] = sql[:500]
                        if not requested and repair_active:
                            payload = {
                                "passed": False,
                                "structuralPassed": False,
                                "semantic": {"passed": False, "errors": []},
                                "execution": {
                                    "ok": False,
                                    "error": "SEMANTIC_REPAIR_MODE",
                                    "columns": [],
                                    "rows": [],
                                    "rowCount": 0,
                                    "durationMs": 0,
                                    "rowsTruncated": False,
                                },
                                "errors": [
                                    {
                                        "rule": "SEMANTIC_REPAIR_MODE",
                                        "message": (
                                            "A final candidate already failed the immutable semantic contract; "
                                            "exploratory probes are now closed."
                                        ),
                                        "fix": (
                                            "Apply the semantic diagnostics directly and call final-mode "
                                            "verifySql(sql, requestedColumns) again."
                                        ),
                                    }
                                ],
                            }
                            result_text = json.dumps(payload, ensure_ascii=False)
                            trace["passed"] = False
                            trace["structuralPassed"] = False
                            trace["semanticPassed"] = False
                            trace["rules"] = ["SEMANTIC_REPAIR_MODE"]
                            trace["rowCount"] = 0
                            item_updates["last_verification"] = payload
                        elif (
                            not requested
                            and max_probe_calls > 0
                            and probe_count >= max_probe_calls
                        ):
                            payload = {
                                "passed": False,
                                "execution": {
                                    "ok": False,
                                    "error": "PROBE_BUDGET_EXHAUSTED",
                                    "columns": [],
                                    "rows": [],
                                    "rowCount": 0,
                                    "durationMs": 0,
                                    "rowsTruncated": False,
                                },
                                "errors": [
                                    {
                                        "rule": "PROBE_BUDGET_EXHAUSTED",
                                        "message": (
                                            f"Probe budget of {max_probe_calls} exploratory calls is exhausted."
                                        ),
                                        "fix": (
                                            "Stop probing. Derive the non-empty requestedColumns contract from "
                                            "the question/evidence and call verifySql(sql, requestedColumns) now."
                                        ),
                                    }
                                ],
                            }
                            result_text = json.dumps(payload, ensure_ascii=False)
                            trace["passed"] = False
                            trace["rules"] = ["PROBE_BUDGET_EXHAUSTED"]
                            trace["rowCount"] = 0
                            item_updates["last_verification"] = payload
                        else:
                            verification = await asyncio.to_thread(
                                verifier.verify, sql, requested
                            )
                            payload = verification.as_tool_dict(
                                self.settings.bird_eval_tool_result_max_rows
                            )
                            structural_passed = bool(verification.passed)
                            semantic_payload: dict[str, Any] = {
                                "passed": True,
                                "errors": [],
                                "criticUsed": False,
                            }
                            if (
                                requested
                                and structural_passed
                                and self.settings.bird_eval_semantic_gate_enabled
                            ):
                                semantic_result = await semantic_gate.verify(
                                    contract=task_contract,
                                    sql=sql,
                                    requested_columns=requested,
                                    result_columns=list(verification.execution.columns),
                                )
                                semantic_payload = semantic_result.as_dict()

                            semantic_passed = bool(semantic_payload.get("passed", True))
                            combined_passed = structural_passed and (
                                semantic_passed if requested else True
                            )
                            semantic_errors = list(semantic_payload.get("errors") or [])
                            if semantic_errors:
                                payload["errors"] = [
                                    *list(payload.get("errors") or []),
                                    *semantic_errors,
                                ]
                            payload["structuralPassed"] = structural_passed
                            payload["semantic"] = semantic_payload
                            payload["passed"] = combined_passed

                            result_text = json.dumps(
                                payload, ensure_ascii=False, default=str
                            )
                            trace["passed"] = combined_passed
                            trace["structuralPassed"] = structural_passed
                            trace["semanticPassed"] = semantic_passed if requested else None
                            trace["rules"] = [
                                str(error.get("rule") or "")
                                for error in payload.get("errors", [])
                                if isinstance(error, dict)
                            ]
                            trace["rowCount"] = verification.execution.row_count
                            item_updates["last_verification"] = payload
                            if requested:
                                item_updates["last_final_sql"] = sql
                                item_updates["last_requested_columns"] = requested
                                item_updates["last_semantic_verification"] = semantic_payload
                                if combined_passed:
                                    item_updates["last_passed_sql"] = sql
                                    final_passed = True
                                elif structural_passed and not semantic_passed:
                                    semantic_repair_increment = 1
                                    enter_repair_mode = True
                                    next_repair_count = (
                                        semantic_repair_count + semantic_repair_increment
                                    )
                                    if next_repair_count >= max_semantic_repairs:
                                        repair_exhausted = True
                                        item_updates["semantic_repair_exhausted"] = True
                                        item_updates["last_error"] = (
                                            "SEMANTIC_REPAIR_BUDGET_EXHAUSTED"
                                        )
                            else:
                                probe_increment = 1
                    else:
                        raise ValueError(f"Unknown BIRD tool: {name}")
                except Exception as exc:
                    result_text = json.dumps(
                        {
                            "passed": False,
                            "errors": [
                                {
                                    "rule": "TOOL_CALL_ERROR",
                                    "message": f"{type(exc).__name__}: {exc}",
                                    "fix": (
                                        "Correct the tool name/arguments and call the tool again. "
                                        "Do not guess the result."
                                    ),
                                }
                            ],
                        },
                        ensure_ascii=False,
                    )
                    item_updates["last_error"] = f"{type(exc).__name__}: {exc}"
                    trace["error"] = f"{type(exc).__name__}: {exc}"
                    trace["rules"] = ["TOOL_CALL_ERROR"]

                return {
                    "message": ToolMessage(
                        content=result_text,
                        tool_call_id=call_id,
                        name=name,
                    ),
                    "trace": trace,
                    "updates": item_updates,
                    "probe_increment": probe_increment,
                    "semantic_repair_increment": semantic_repair_increment,
                    "enter_repair_mode": enter_repair_mode,
                    "repair_exhausted": repair_exhausted,
                    "final_passed": final_passed,
                }

            # Semantic-gated final checks must be admitted sequentially so a failed final
            # immediately closes probes and a passed final can stop sibling tool calls.
            # Pure legacy mode (no probe cap and semantic gate disabled) keeps AgentX-style
            # concurrent tool execution compatibility.
            if max_probe_calls > 0 or self.settings.bird_eval_semantic_gate_enabled:
                executed: list[dict[str, Any]] = []
                for index, call in enumerate(tool_calls, start=1):
                    item = await execute_one(
                        call,
                        fallback_index=index,
                        probe_count=probe_calls,
                        repair_active=repair_mode,
                        semantic_repair_count=semantic_repairs,
                    )
                    executed.append(item)
                    probe_calls += int(item["probe_increment"])
                    semantic_repairs += int(item["semantic_repair_increment"])
                    if item["enter_repair_mode"]:
                        repair_mode = True
                    if item["final_passed"] or item["repair_exhausted"]:
                        # The final SQL is either fully approved or the bounded repair
                        # budget is exhausted. Do not execute later sibling calls.
                        break
            else:
                executed = list(
                    await asyncio.gather(
                        *(
                            execute_one(
                                call,
                                fallback_index=index,
                                probe_count=probe_calls,
                                repair_active=repair_mode,
                                semantic_repair_count=semantic_repairs,
                            )
                            for index, call in enumerate(tool_calls, start=1)
                        )
                    )
                )
                probe_calls += sum(int(item["probe_increment"]) for item in executed)
                semantic_repairs += sum(
                    int(item["semantic_repair_increment"]) for item in executed
                )
                repair_mode = repair_mode or any(
                    bool(item["enter_repair_mode"]) for item in executed
                )

            terminal_item = next(
                (item for item in executed if item["final_passed"]),
                None,
            )

            messages: list[ToolMessage] = []
            for item in executed:
                messages.append(item["message"])
                history.append(item["trace"])
                for key, value in item["updates"].items():
                    if key == "last_error":
                        if value:
                            updates["last_error"] = value
                    else:
                        updates[key] = value

            for invalid in invalid_tool_calls:
                name = str(invalid.get("name") or "invalidToolCall")
                call_id = str(invalid.get("id") or "")
                if not call_id:
                    raise ValueError("malformed tool call is missing tool_call_id")
                error = str(invalid.get("error") or "Malformed tool arguments")
                payload = {
                    "passed": False,
                    "errors": [
                        {
                            "rule": "TOOL_CALL_MALFORMED",
                            "message": error,
                            "fix": (
                                "Call the tool again with valid JSON arguments matching its schema. "
                                "Do not reuse the malformed arguments."
                            ),
                        }
                    ],
                }
                history.append(
                    {
                        "name": name,
                        "error": error,
                        "rules": ["TOOL_CALL_MALFORMED"],
                    }
                )
                messages.append(
                    ToolMessage(
                        content=json.dumps(payload, ensure_ascii=False),
                        tool_call_id=call_id,
                        name=name,
                    )
                )
                updates["last_error"] = error

            if terminal_item is not None:
                # In the explicit unlimited-probe compatibility mode tools may still
                # execute concurrently. Re-apply the passed final verification last so
                # later sibling tool results cannot overwrite the deterministic terminal state.
                for key, value in terminal_item["updates"].items():
                    if key != "last_error":
                        updates[key] = value

            updates["messages"] = messages
            updates["tool_history"] = history
            updates["probe_calls"] = probe_calls
            updates["semantic_repairs"] = semantic_repairs
            updates["repair_mode"] = repair_mode
            return updates

        async def force_final(state: BirdAgentState):
            """Mirror AgentX: skip pending tool execution, then request one final response."""
            last = state["messages"][-1]
            pending_calls = [
                *list(getattr(last, "tool_calls", None) or []),
                *list(getattr(last, "invalid_tool_calls", None) or []),
            ]
            skipped_messages: list[ToolMessage] = []
            history = list(state.get("tool_history") or [])
            for call in pending_calls:
                name = str(call.get("name") or "invalidToolCall")
                call_id = str(call.get("id") or f"tool-{len(skipped_messages) + 1}")
                skipped_messages.append(
                    ToolMessage(
                        content="Agent maximum rounds reached. Tool execution skipped.",
                        tool_call_id=call_id,
                        name=name,
                    )
                )
                history.append({"name": name, "skipped": "MAX_ROUNDS"})

            response = await _invoke_agent_model_streaming(
                bound_model,
                [
                    SystemMessage(content=system_prompt),
                    *state["messages"],
                    *skipped_messages,
                ],
            )
            return {
                "messages": [*skipped_messages, response],
                "tool_history": history,
                "last_error": "MAX_ROUNDS_FORCE_FINAL",
            }

        def route_after_tools(state: BirdAgentState):
            if str(state.get("last_passed_sql") or "").strip():
                return END
            if bool(state.get("semantic_repair_exhausted", False)):
                return END
            return "model"

        def route_after_model(state: BirdAgentState):
            last = state["messages"][-1]
            has_tool_calls = bool(
                getattr(last, "tool_calls", None)
                or getattr(last, "invalid_tool_calls", None)
            )
            if not has_tool_calls:
                # AgentX completes a normal ReAct turn as soon as the model emits text only.
                return END
            if int(state.get("model_rounds", 0)) >= int(state.get("max_rounds", 1)):
                return "force_final"
            return "tools"

        builder = StateGraph(BirdAgentState)
        builder.add_node("model", call_model)
        builder.add_node("tools", call_tools)
        builder.add_node("force_final", force_final)
        builder.add_edge(START, "model")
        builder.add_conditional_edges(
            "model",
            route_after_model,
            {"tools": "tools", "force_final": "force_final", END: END},
        )
        builder.add_conditional_edges(
            "tools",
            route_after_tools,
            {"model": "model", END: END},
        )
        builder.add_edge("force_final", END)
        return builder.compile(checkpointer=False)

    def _validate_request(self, req: BirdEvalRequest) -> Path:
        if not self.settings.bird_eval_enabled:
            raise RuntimeError("BIRD eval endpoint is disabled; set BIRD_EVAL_ENABLED=true")
        if not req.question.strip():
            raise ValueError("question is blank")
        path = Path(req.sqlitePath).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"SQLite file not found: {path}")
        if path.suffix.lower() not in {".sqlite", ".db", ".sqlite3"}:
            raise ValueError(f"unsupported SQLite file extension: {path.suffix}")
        return path

    @staticmethod
    def _response(
        started: float,
        *,
        success: bool,
        sql: str,
        mode: str,
        attempts: int,
        approved: bool | None,
        error: str | None = None,
        critic: str = "",
        probe: str = "",
        tool_rounds: int = 0,
        probe_calls: int = 0,
        semantic_repairs: int = 0,
        requested_columns: list[str] | None = None,
        task_contract: dict[str, Any] | None = None,
        semantic_verification: dict[str, Any] | None = None,
        tool_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "success": success,
            "sql": sql,
            "error": error,
            "durationMs": round((time.perf_counter() - started) * 1000),
            "mode": mode,
            "attempts": attempts,
            "approved": approved,
            "critic": critic,
            "probe": probe,
            "toolRounds": tool_rounds,
            "probeCalls": probe_calls,
            "semanticRepairs": semantic_repairs,
            "requestedColumns": requested_columns or [],
            "taskContract": task_contract or {},
            "semanticVerification": semantic_verification or {},
            "toolHistory": tool_history or [],
        }


def describe_sqlite_schema(path: Path) -> str:
    """Compatibility wrapper used by existing tests and callers."""
    return SqliteSchemaProvider(path).describe_all_tables()


def probe_sqlite_query(path: Path, sql: str, timeout_seconds: float) -> dict[str, Any]:
    """Compatibility probe; verifier itself retains the complete execution result."""
    verifier = BirdSqlVerifier(
        path,
        request_context="",
        timeout_seconds=timeout_seconds,
    )
    execution = verifier.execute(sql)
    return {
        "ok": execution.ok,
        "columns": list(execution.columns),
        "previewRows": list(execution.rows[:5]),
        "previewCount": min(5, execution.row_count),
        "rowCount": execution.row_count,
        "error": execution.error,
    }


def _build_review_context(req: BirdEvalRequest) -> str:
    return (
        f"Question: {req.question}\nEvidence: "
        + (req.evidence if req.evidence.strip() else "(none)")
    )


def _build_query(req: BirdEvalRequest) -> str:
    evidence = req.evidence if req.evidence.strip() else "(none)"
    return (
        "Generate one SQLite SELECT/WITH query for the following BIRD question.\n\n"
        f"Question:\n{req.question}\n\nEvidence:\n{evidence}\n"
    )


def _structured_output_format() -> str:
    """Match Spring AI 1.1.0 BeanOutputConverter.getFormat() for BirdAgentOutput."""
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "requestedColumns": {
                "description": (
                    "Values explicitly requested by the question/evidence, listed in final SELECT "
                    "order. Use the exact schema field name when one exists, or a stable output alias "
                    "otherwise. Include a rank measure only when needed; never add explanatory entity "
                    "names or helper columns."
                ),
                "type": "array",
                "items": {"type": "string"},
            },
            "sql": {
                "type": "string",
                "description": (
                    "本题最终 SQLite SELECT/WITH 查询 SQL。必须是单条可直接执行的查询，"
                    "不要 markdown，不要分号。"
                ),
            },
        },
        "additionalProperties": False,
    }
    pretty_schema = json.dumps(schema, ensure_ascii=False, indent=2)
    # Jackson's DefaultPrettyPrinter emits a space on both sides of ':'.
    pretty_schema = re.sub(r'(".*?"):', r'\1 :', pretty_schema)
    return (
        "Your response should be in JSON format.\n"
        "Do not include any explanations, only provide a RFC8259 compliant JSON response "
        "following this format without deviation.\n"
        "Do not include markdown code blocks in your response.\n"
        "Remove the ```json markdown from the output.\n"
        "Here is the JSON Schema instance your output must adhere to:\n"
        "```"
        + pretty_schema
        + "```\n"
    )


def _build_agent_query(req: BirdEvalRequest) -> str:
    # LoopMessageBuilder appends BeanOutputConverter format and <no_think> when
    # RunnableParams.outputType is present and ThinkingMode is disabled.
    return _build_query(req) + "\n" + _structured_output_format() + "\n<no_think>"


def _coerce_tool_args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("tool arguments must be a JSON object")


async def _invoke_agent_model_streaming(
    model: Any,
    messages: list[Any],
    *,
    max_retries: int = 3,
    retry_delay_seconds: float = 10.0,
) -> AIMessage:
    """Mirror AgentX reasoning rounds, including whole-stream retry semantics.

    AgentX retries a reasoning/force-final round when the reactive stream fails, even
    after some chunks have already arrived. ``ChatOpenAI(max_retries=...)`` only covers
    provider/request retries and cannot reliably recover a stream that terminates after
    it has started. Retrying here discards partial chunks and replays the whole model
    round, matching ``LlmInvoker.handleStreamError``.

    Production ChatOpenAI exposes ``astream`` and yields AIMessageChunk objects. Test
    doubles that only implement ``ainvoke`` keep using the fallback so deterministic
    unit tests do not need to emulate provider chunking.
    """
    retries = max(0, int(max_retries))
    delay = max(0.0, float(retry_delay_seconds))

    for retry_attempt in range(retries + 1):
        try:
            astream = getattr(model, "astream", None)
            if not callable(astream):
                response = await model.ainvoke(messages)
                return response

            merged: Any | None = None
            async for chunk in astream(messages):
                merged = chunk if merged is None else merged + chunk

            if merged is None:
                return AIMessage(content="")
            if isinstance(merged, AIMessage):
                return merged
            return message_chunk_to_message(merged)
        except Exception:
            if retry_attempt >= retries:
                raise
            if delay:
                await asyncio.sleep(delay)

    raise AssertionError("unreachable retry loop")


def _sanitize_ai_tool_calls_for_history(
    message: AIMessage,
    *,
    round_number: int,
) -> AIMessage:
    """Mirror AgentX ToolCallExecutor.sanitizeToolCalls before history replay.

    langchain-openai keeps malformed function arguments in invalid_tool_calls.
    Replaying that assistant message verbatim to some OpenAI-compatible providers can
    make the next request fail before the model sees the tool response. AgentX avoids
    this by replacing malformed arguments with {} before appending the assistant
    tool-call message to history. Normalize missing/duplicate ids at the same boundary
    so every replayed tool call has exactly one addressable ToolMessage.
    """
    valid_calls = list(getattr(message, "tool_calls", None) or [])
    invalid_calls = list(getattr(message, "invalid_tool_calls", None) or [])
    if not valid_calls and not invalid_calls:
        return message

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def append_call(call: dict[str, Any], *, malformed: bool) -> None:
        index = len(normalized) + 1
        call_id = str(call.get("id") or "").strip()
        if not call_id or call_id in seen_ids:
            call_id = f"bird-tool-{round_number}-{index}"
            while call_id in seen_ids:
                index += 1
                call_id = f"bird-tool-{round_number}-{index}"
        seen_ids.add(call_id)

        name = str(call.get("name") or "invalidToolCall")
        if malformed:
            args: dict[str, Any] = {}
        else:
            try:
                args = _coerce_tool_args(call.get("args"))
            except Exception:
                args = {}

        normalized.append(
            {
                "name": name,
                "args": args,
                "id": call_id,
                "type": "tool_call",
            }
        )

    for call in valid_calls:
        append_call(dict(call), malformed=False)
    for call in invalid_calls:
        append_call(dict(call), malformed=True)

    return message.model_copy(
        update={
            "tool_calls": normalized,
            "invalid_tool_calls": [],
        }
    )


def _message_text(output: Any) -> str:
    content = getattr(output, "content", output)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _agentx_fix_json(text: str) -> str:
    """Port AgentX JsonRepairUtil.fixJson used by ReactAgent.call(outputType=...)."""
    original = text or ""
    if not original.strip():
        return "{}"

    fixed = original.strip()

    # JsonRepairUtil.extractJsonFromMarkdown: use the first fenced block when present.
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", fixed, flags=re.IGNORECASE)
    if fenced:
        fixed = fenced.group(1).strip()

    # JsonRepairUtil.removeLeadingTrailingGarbage: keep the widest JSON-looking span.
    starts = [index for index in (fixed.find("{"), fixed.find("[")) if index >= 0]
    start = min(starts) if starts else -1
    end = max(fixed.rfind("}"), fixed.rfind("]"))
    if start >= 0 and end >= start:
        fixed = fixed[start : end + 1]

    # JsonRepairUtil.fixQuotes: normalize smart quotes, then convert structural single
    # quotes only while outside a double-quoted JSON string. This intentionally leaves
    # SQL literals such as WHERE name = 'Ada' untouched inside a JSON string value.
    fixed = fixed.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    quoted: list[str] = []
    in_string = False
    escaped = False
    for char in fixed:
        if in_string:
            quoted.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
                quoted.append(char)
            elif char == "'":
                quoted.append('"')
            else:
                quoted.append(char)
    fixed = "".join(quoted)

    fixed = re.sub(r",\s*}", "}", fixed)
    fixed = re.sub(r",\s*]", "]", fixed)
    fixed = re.sub(
        r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:",
        r'\1"\2":',
        fixed,
    )
    fixed = re.sub(r"(?<!\\)\n", " ", fixed)
    fixed = re.sub(r"(?<!\\)\r", " ", fixed)
    fixed = re.sub(r"(?<!\\)\t", " ", fixed)

    try:
        json.loads(fixed)
        return fixed
    except (json.JSONDecodeError, TypeError):
        # JsonRepairUtil.wrapAsSimpleJson returns a valid object containing the original
        # text. BirdAgentOutput then has blank sql, which preserves Java's failure mode
        # for genuinely unrecoverable/truncated output instead of inventing a SQL fallback.
        return json.dumps({"content": original}, ensure_ascii=False)


def _parse_agent_output(output: Any) -> BirdAgentOutput | None:
    """Mirror ReactAgent.call(outputType) JsonRepairUtil + BirdEvalService readValue."""
    if isinstance(output, BirdAgentOutput):
        return output
    if isinstance(output, dict):
        try:
            return BirdAgentOutput.model_validate(output)
        except Exception:
            return None

    text = _agentx_fix_json(_message_text(output))
    try:
        candidate = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(candidate, dict):
        return None
    try:
        return BirdAgentOutput.model_validate(candidate)
    except Exception:
        return None


def _clean_bean_output_text(text: str) -> str:
    """Mirror Spring AI 1.1.0 BeanOutputConverter default ResponseTextCleaner chain."""
    cleaned = (text or "").strip()
    for pattern in (
        r"(?s)<thinking>.*?</thinking>\s*",
        r"(?s)<think>.*?</think>\s*",
        r"(?s)<reasoning>.*?</reasoning>\s*",
        r"(?s)```thinking.*?```\s*",
        r"(?s)<!--\s*thinking:.*?-->\s*",
    ):
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = cleaned.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        parts = cleaned.split("\n", 1)
        opening = parts[0].strip()
        if opening.lower().startswith("```"):
            if len(opening) > 3:
                cleaned = parts[1] if len(parts) > 1 else ""
            else:
                cleaned = cleaned[3:]
        else:
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
    return cleaned.strip()


def _parse_sql_output(output: Any) -> str:
    """Mirror Java baseline BeanOutputConverter.convert(content)."""
    text = _clean_bean_output_text(_message_text(output))
    try:
        candidate = json.loads(text)
    except json.JSONDecodeError:
        return ""
    if not isinstance(candidate, dict):
        return ""
    try:
        parsed = BirdAgentOutput.model_validate(candidate)
    except Exception:
        return ""
    return (parsed.sql or "").strip()
