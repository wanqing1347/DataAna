import csv
import json
import sqlite3
import threading

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.bird_eval import (
    BirdEvalRequest,
    BirdEvalService,
    _AgentXCompatibleChatOpenAI,
    _invoke_agent_model_streaming,
    describe_sqlite_schema,
    probe_sqlite_query,
)
from app.bird_schema import SqliteSchemaProvider
from app.bird_verifier import BirdSqlVerifier
from app.config import Settings


class BaselineModel:
    async def ainvoke(self, messages):
        return AIMessage(content='{"sql":"SELECT name FROM people"}')


class SequenceToolModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.bound_tools = []
        self.calls = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("no fake model response left")
        return self.responses.pop(0)


class SequenceSemanticModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("no fake semantic-model response left")
        response = self.responses.pop(0)
        if isinstance(response, AIMessage):
            return response
        return AIMessage(content=response)


class StreamingOnlyModel:
    def __init__(self):
        self.stream_calls = 0

    async def astream(self, messages):
        self.stream_calls += 1
        yield AIMessageChunk(content='{"sql":"SELECT ')
        yield AIMessageChunk(content='name FROM people","requestedColumns":["name"]}')

    async def ainvoke(self, messages):
        raise AssertionError("agent reasoning must use astream when available")


class StreamingReasoningModel:
    async def astream(self, messages):
        yield AIMessageChunk(
            content="",
            additional_kwargs={"reasoning_content": "private-step-1"},
        )
        yield AIMessageChunk(
            content='{"sql":"SELECT ',
            additional_kwargs={"reasoning_content": "private-step-2"},
        )
        yield AIMessageChunk(
            content='name FROM people","requestedColumns":["name"]}'
        )


class FlakyStreamingModel:
    def __init__(self):
        self.stream_calls = 0

    async def astream(self, messages):
        self.stream_calls += 1
        yield AIMessageChunk(content='{"sql":"SELECT ')
        if self.stream_calls < 3:
            raise RuntimeError("upstream stream ended before terminal chunk")
        yield AIMessageChunk(content='name FROM people","requestedColumns":["name"]}')


@pytest.mark.asyncio
async def test_bird_agent_reasoning_prefers_streaming_and_merges_chunks():
    model = StreamingOnlyModel()

    response = await _invoke_agent_model_streaming(model, [])

    assert model.stream_calls == 1
    assert response.content == '{"sql":"SELECT name FROM people","requestedColumns":["name"]}'


@pytest.mark.asyncio
async def test_bird_agent_streaming_keeps_deepseek_reasoning_out_of_final_content():
    response = await _invoke_agent_model_streaming(StreamingReasoningModel(), [])

    assert response.content == '{"sql":"SELECT name FROM people","requestedColumns":["name"]}'
    assert "private-step" not in response.content
    assert response.additional_kwargs.get("reasoning_content") == "private-step-1private-step-2"


@pytest.mark.asyncio
async def test_bird_agent_streaming_retries_whole_round_after_partial_stream_failure():
    model = FlakyStreamingModel()

    response = await _invoke_agent_model_streaming(
        model,
        [],
        max_retries=3,
        retry_delay_seconds=0,
    )

    assert model.stream_calls == 3
    assert response.content == '{"sql":"SELECT name FROM people","requestedColumns":["name"]}'


def test_agentx_compatible_chat_payload_preserves_tool_name_and_empty_assistant_content():
    model = _AgentXCompatibleChatOpenAI(
        model="deepseek/deepseek-v4-flash",
        api_key="test-key",
        base_url="https://example.invalid/v1",
        temperature=0,
    )
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "verifySql",
                    "args": {"sql": "SELECT 1", "requestedColumns": []},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='{"passed":true}',
            tool_call_id="call-1",
            name="verifySql",
        ),
    ]

    payload = model._get_request_payload(messages)

    assert payload["messages"][0]["content"] == ""
    assert payload["messages"][1]["name"] == "verifySql"
    assert payload["messages"][1]["tool_call_id"] == "call-1"


@pytest.fixture
def sqlite_file(tmp_path):
    path = tmp_path / "demo.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
    connection.executemany("INSERT INTO people(name) VALUES (?)", [("Ada",), ("Linus",)])
    connection.commit()
    connection.close()
    return path


@pytest.fixture
def verifier_db(tmp_path):
    path = tmp_path / "verifier.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE region (
            id TEXT PRIMARY KEY,
            name TEXT
        );
        CREATE TABLE left_t (
            id INTEGER PRIMARY KEY,
            region_id TEXT,
            region_name TEXT,
            code TEXT,
            a INTEGER,
            b INTEGER,
            event_date TEXT,
            rtype TEXT,
            FOREIGN KEY(region_id) REFERENCES region(id)
        );
        CREATE TABLE right_t (
            id INTEGER PRIMARY KEY,
            region_id TEXT,
            region_name TEXT,
            FOREIGN KEY(region_id) REFERENCES region(id)
        );
        INSERT INTO region VALUES ('01', 'East');
        INSERT INTO left_t(region_id, region_name, code, a, b, event_date, rtype)
            VALUES ('01', 'East', '00123', 3, 2, '1994-03-03', 'S');
        INSERT INTO right_t(region_id, region_name) VALUES ('01', 'East');
        """
    )
    connection.commit()
    connection.close()

    desc_dir = tmp_path / "database_description"
    desc_dir.mkdir()
    with (desc_dir / "left_t.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "original_column_name",
                "column_name",
                "column_description",
                "data_format",
                "value_description",
                "extra",
            ]
        )
        writer.writerow(["rtype", "record type", "metadata discriminator", "", "unuseful", ""])
        writer.writerow(["event_date", "event date", "stored event date", "", "ISO date", ""])
    return path


def request_for(path):
    return BirdEvalRequest(
        questionId="1",
        dbId="demo",
        question="Return every person's name.",
        evidence="",
        sqlitePath=str(path),
    )


def rules(result):
    return {error.rule for error in result.errors}


def test_sqlite_schema_and_probe_are_read_only_and_keep_row_count(sqlite_file):
    schema = describe_sqlite_schema(sqlite_file)
    assert "Table `people`" in schema
    assert "`name` text" in schema

    ok = probe_sqlite_query(sqlite_file, "SELECT name FROM people", 2)
    assert ok["ok"] is True
    assert ok["columns"] == ["name"]
    assert ok["rowCount"] == 2

    rejected = probe_sqlite_query(sqlite_file, "DELETE FROM people", 2)
    assert rejected["ok"] is False
    assert rejected["error"] == "ONLY_SELECT_OR_WITH_ALLOWED"

    connection = sqlite3.connect(sqlite_file)
    count = connection.execute("SELECT COUNT(*) FROM people").fetchone()[0]
    connection.close()
    assert count == 2


def test_schema_provider_includes_fk_descriptions_unuseful_and_examples(verifier_db):
    provider = SqliteSchemaProvider(verifier_db)
    listing = provider.list_tables()
    description = provider.describe_tables(["left_t", "region"])

    assert "`left_t`" in listing
    assert "`left_t`.`region_id` → `region`.`id`" in listing
    assert "metadata discriminator" in description
    assert "Values: unuseful. Do not filter" in description
    assert "Examples: [S]" in description
    assert "→ region.id" in description


def test_verify_sql_probe_mode_executes_without_final_semantic_checks(verifier_db):
    verifier = BirdSqlVerifier(
        verifier_db,
        request_context="Question: Return code.",
        timeout_seconds=2,
    )
    result = verifier.verify("SELECT code FROM left_t WHERE rtype = 'S'", [])

    assert result.passed is True
    assert result.execution.row_count == 1
    assert result.execution.rows == (("00123",),)


def test_verify_sql_projection_count_name_and_order(verifier_db):
    verifier = BirdSqlVerifier(
        verifier_db,
        request_context="Question: Return region_name, then code.",
        timeout_seconds=2,
    )

    count = verifier.verify("SELECT region_name FROM left_t", ["region_name", "code"])
    assert "PROJECTION_COUNT" in rules(count)

    name = verifier.verify(
        "SELECT region_name, code AS wrong_name FROM left_t",
        ["region_name", "code"],
    )
    assert "PROJECTION_NAME" in rules(name)

    order = verifier.verify(
        "SELECT code, region_name FROM left_t",
        ["region_name", "code"],
    )
    assert "PROJECTION_ORDER" in rules(order)


def test_verify_sql_inferred_null_date_and_unuseful_metadata(verifier_db):
    verifier = BirdSqlVerifier(
        verifier_db,
        request_context="Question: Return code.",
        timeout_seconds=2,
    )

    inferred_null = verifier.verify(
        "SELECT code FROM left_t WHERE code IS NOT NULL",
        ["code"],
    )
    assert "INFERRED_NULL_FILTER" in rules(inferred_null)

    date = verifier.verify(
        "SELECT code FROM left_t WHERE event_date = '1994/03/03'",
        ["code"],
    )
    assert "DATE_LITERAL_FORMAT" in rules(date)

    unuseful = verifier.verify(
        "SELECT code FROM left_t WHERE rtype = 'S'",
        ["code"],
    )
    assert "UNUSEFUL_METADATA_COLUMN" in rules(unuseful)


@pytest.mark.parametrize(
    ("sql", "expected_rule"),
    [
        ("SELECT code FROM left_t WHERE '00' || code = '0000123'", "TEXT_IDENTIFIER_PADDING"),
        ("SELECT code FROM left_t WHERE CAST(code AS INTEGER) = 123", "TEXT_IDENTIFIER_CAST"),
        ("SELECT code FROM left_t WHERE substr(code, 1, 2) = '00'", "TEXT_IDENTIFIER_SUBSTR"),
        ("SELECT code FROM left_t WHERE ltrim(code, '0') = '123'", "TEXT_IDENTIFIER_TRIM"),
    ],
)
def test_verify_sql_text_identifier_rules(verifier_db, sql, expected_rule):
    verifier = BirdSqlVerifier(
        verifier_db,
        request_context="Question: Return code.",
        timeout_seconds=2,
    )
    result = verifier.verify(sql, ["code"])
    assert expected_rule in rules(result)


def test_verify_sql_integer_division_and_display_name_join(verifier_db):
    verifier = BirdSqlVerifier(
        verifier_db,
        request_context="Question: Return the requested value.",
        timeout_seconds=2,
    )

    division = verifier.verify("SELECT a / b AS ratio FROM left_t", ["ratio"])
    assert "INTEGER_DIVISION" in rules(division)

    display_join = verifier.verify(
        "SELECT l.code FROM left_t l JOIN right_t r "
        "ON l.region_name = r.region_name",
        ["code"],
    )
    assert "JOIN_DISPLAY_NAME" in rules(display_join)


def test_verify_sql_tool_payload_retains_full_count_when_rows_are_compressed(sqlite_file):
    verifier = BirdSqlVerifier(
        sqlite_file,
        request_context="Question: Return name.",
        timeout_seconds=2,
    )
    result = verifier.verify("SELECT name FROM people", [])
    payload = result.as_tool_dict(max_rows=1)

    assert payload["execution"]["rowCount"] == 2
    assert payload["execution"]["rowsTruncated"] is True
    assert len(payload["execution"]["rows"]) == 1

    java_parity = result.as_tool_dict(max_rows=0)
    assert java_parity["execution"]["rowsTruncated"] is False
    assert len(java_parity["execution"]["rows"]) == 2


@pytest.mark.asyncio
async def test_bird_baseline_is_direct_chatmodel_with_local_json_parsing(sqlite_file):
    settings = Settings(bird_eval_enabled=True)
    service = BirdEvalService(settings, model=BaselineModel())

    result = await service.eval_baseline(request_for(sqlite_file))

    assert result["success"] is True
    assert result["mode"] == "baseline"
    assert result["sql"] == "SELECT name FROM people"
    assert result["attempts"] == 1


@pytest.mark.asyncio
async def test_bird_langgraph_agent_uses_real_tools_then_emits_structured_final(sqlite_file):
    model = SequenceToolModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "listTables", "args": {}, "id": "call-1", "type": "tool_call"}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "describeTables",
                        "args": {"tableNames": ["people"]},
                        "id": "call-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "verifySql",
                        "args": {
                            "sql": "SELECT name FROM people WHERE name IS NOT NULL",
                            "requestedColumns": ["name"],
                        },
                        "id": "call-3",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "verifySql",
                        "args": {
                            "sql": "SELECT name FROM people",
                            "requestedColumns": ["name"],
                        },
                        "id": "call-4",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content='{"sql":"SELECT name FROM people","requestedColumns":["name"]}'
            ),
        ]
    )
    settings = Settings(
        bird_eval_enabled=True,
        bird_eval_max_rounds=10,
        bird_eval_sql_timeout_seconds=2,
    )
    service = BirdEvalService(settings, model=model)

    result = await service.eval_agent(request_for(sqlite_file))

    assert [tool.name for tool in model.bound_tools] == [
        "listTables",
        "describeTables",
        "verifySql",
    ]
    wire_schemas = {
        tool.name: convert_to_openai_tool(tool)["function"]["parameters"]
        for tool in model.bound_tools
    }
    assert wire_schemas["listTables"] == {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    assert wire_schemas["describeTables"]["required"] == ["tableNames"]
    assert wire_schemas["describeTables"]["additionalProperties"] is False
    assert wire_schemas["verifySql"]["required"] == ["sql", "requestedColumns"]
    assert wire_schemas["verifySql"]["additionalProperties"] is False
    assert (
        wire_schemas["verifySql"]["properties"]["requestedColumns"]["description"]
        == "Empty for a probe; otherwise exact requested output fields in SELECT order"
    )
    assert result["success"] is True
    assert result["mode"] == "langgraph-react-tools"
    assert result["attempts"] == 4
    assert len(model.calls) == 4
    assert len(model.responses) == 1
    assert result["toolRounds"] == 4
    assert result["approved"] is True
    assert result["requestedColumns"] == ["name"]
    assert result["sql"] == "SELECT name FROM people"


@pytest.mark.asyncio
async def test_bird_agent_parallel_tool_calls_execute_concurrently_and_replay_in_order(
    sqlite_file, monkeypatch
):
    original_list_tables = SqliteSchemaProvider.list_tables
    original_describe_tables = SqliteSchemaProvider.describe_tables
    both_started = threading.Event()
    lock = threading.Lock()
    started: list[str] = []

    def rendezvous(label: str) -> None:
        with lock:
            started.append(label)
            if len(started) == 2:
                both_started.set()
        if not both_started.wait(timeout=2):
            raise AssertionError("parallel BIRD tool calls did not overlap")

    def gated_list_tables(self):
        rendezvous("listTables")
        return original_list_tables(self)

    def gated_describe_tables(self, table_names):
        rendezvous("describeTables")
        return original_describe_tables(self, table_names)

    monkeypatch.setattr(SqliteSchemaProvider, "list_tables", gated_list_tables)
    monkeypatch.setattr(SqliteSchemaProvider, "describe_tables", gated_describe_tables)

    model = SequenceToolModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "listTables",
                        "args": {},
                        "id": "parallel-1",
                        "type": "tool_call",
                    },
                    {
                        "name": "describeTables",
                        "args": {"tableNames": ["people"]},
                        "id": "parallel-2",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(
                content='{"sql":"SELECT name FROM people","requestedColumns":["name"]}'
            ),
        ]
    )
    service = BirdEvalService(
        Settings(
            bird_eval_enabled=True,
            bird_eval_max_rounds=3,
            bird_eval_max_probe_calls=0,
        ),
        model=model,
    )

    result = await service.eval_agent(request_for(sqlite_file))

    assert result["success"] is True
    assert set(started) == {"listTables", "describeTables"}
    assert [item["name"] for item in result["toolHistory"][:2]] == [
        "listTables",
        "describeTables",
    ]
    replayed_tool_messages = model.calls[1][-2:]
    assert [message.tool_call_id for message in replayed_tool_messages] == [
        "parallel-1",
        "parallel-2",
    ]


@pytest.mark.asyncio
async def test_bird_agent_default_probe_budget_allows_normal_short_exploration(sqlite_file):
    model = SequenceToolModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "verifySql",
                        "args": {"sql": "SELECT name FROM people", "requestedColumns": []},
                        "id": "unlimited-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "verifySql",
                        "args": {"sql": "SELECT id FROM people", "requestedColumns": []},
                        "id": "unlimited-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "verifySql",
                        "args": {"sql": "SELECT name FROM people", "requestedColumns": ["name"]},
                        "id": "unlimited-3",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content='{"sql":"SELECT name FROM people","requestedColumns":["name"]}'
            ),
        ]
    )
    settings = Settings(
        bird_eval_enabled=True,
        bird_eval_max_rounds=6,
        bird_eval_sql_timeout_seconds=2,
    )
    service = BirdEvalService(settings, model=model)

    result = await service.eval_agent(request_for(sqlite_file))

    assert result["success"] is True
    assert result["probeCalls"] == 2
    assert not any(
        "PROBE_BUDGET_EXHAUSTED" in item.get("rules", [])
        for item in result["toolHistory"]
    )


@pytest.mark.asyncio
async def test_bird_agent_probe_budget_forces_final_verification(sqlite_file):
    model = SequenceToolModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "listTables", "args": {}, "id": "budget-1", "type": "tool_call"}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "describeTables",
                        "args": {"tableNames": ["people"]},
                        "id": "budget-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "verifySql",
                        "args": {
                            "sql": "SELECT name FROM people",
                            "requestedColumns": [],
                        },
                        "id": "budget-3",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "verifySql",
                        "args": {
                            "sql": "SELECT id FROM people",
                            "requestedColumns": [],
                        },
                        "id": "budget-4",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "verifySql",
                        "args": {
                            "sql": "SELECT name FROM people",
                            "requestedColumns": ["name"],
                        },
                        "id": "budget-5",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content='{"sql":"SELECT name FROM people","requestedColumns":["name"]}'
            ),
        ]
    )
    settings = Settings(
        bird_eval_enabled=True,
        bird_eval_max_rounds=6,
        bird_eval_max_probe_calls=1,
        bird_eval_sql_timeout_seconds=2,
    )
    service = BirdEvalService(settings, model=model)

    result = await service.eval_agent(request_for(sqlite_file))

    assert result["success"] is True
    assert result["probeCalls"] == 1
    assert result["sql"] == "SELECT name FROM people"
    assert any(
        "PROBE_BUDGET_EXHAUSTED" in item.get("rules", [])
        for item in result["toolHistory"]
    )


@pytest.mark.asyncio
async def test_bird_agent_rejects_unverified_final_text(sqlite_file):
    model = SequenceToolModel([AIMessage(content="SELECT name FROM people")])
    settings = Settings(bird_eval_enabled=True, bird_eval_max_rounds=1)
    service = BirdEvalService(settings, model=model)

    result = await service.eval_agent(request_for(sqlite_file))

    assert result["success"] is False
    assert "invalid BirdAgentOutput structured output" in result["error"]


@pytest.mark.asyncio
async def test_bird_agent_matches_java_terminal_semantics_without_forced_verify(sqlite_file):
    model = SequenceToolModel(
        [AIMessage(content='{"sql":"SELECT name FROM people","requestedColumns":["name"]}')]
    )
    settings = Settings(bird_eval_enabled=True, bird_eval_max_rounds=1)
    service = BirdEvalService(settings, model=model)

    result = await service.eval_agent(request_for(sqlite_file))

    assert result["success"] is True
    assert result["approved"] is None
    assert result["toolRounds"] == 0
    assert result["sql"] == "SELECT name FROM people"
    prompt = model.calls[0][-1].content
    assert "requestedColumns" in prompt
    assert "Do not include any explanations, only provide a RFC8259 compliant JSON response" in prompt
    assert '"$schema" : "https://json-schema.org/draft/2020-12/schema"' in prompt
    assert '"additionalProperties" : false' in prompt
    assert '"required"' not in prompt
    assert prompt.index('"requestedColumns" :') < prompt.index('"sql" :')
    assert prompt.endswith("<no_think>")


@pytest.mark.asyncio
async def test_bird_agent_normal_final_applies_agentx_json_repair(sqlite_file):
    model = SequenceToolModel(
        [
            AIMessage(
                content=(
                    "Result follows:\n```json\n"
                    '{sql: "SELECT name FROM people", requestedColumns: ["name"],}\n'
                    "```"
                )
            )
        ]
    )
    settings = Settings(bird_eval_enabled=True, bird_eval_max_rounds=1)
    service = BirdEvalService(settings, model=model)

    result = await service.eval_agent(request_for(sqlite_file))

    assert result["success"] is True
    assert result["approved"] is None
    assert result["sql"] == "SELECT name FROM people"
    assert result["requestedColumns"] == ["name"]


@pytest.mark.asyncio
async def test_bird_agent_final_verify_passed_ends_without_model_json_tail(sqlite_file):
    model = SequenceToolModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "verifySql",
                        "args": {
                            "sql": "SELECT name FROM people",
                            "requestedColumns": ["name"],
                        },
                        "id": "verify-final-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content=(
                    'Final: {sql: "SELECT name FROM people", '
                    'requestedColumns: ["name"],}'
                )
            ),
        ]
    )
    settings = Settings(
        bird_eval_enabled=True,
        bird_eval_max_rounds=2,
        bird_eval_sql_timeout_seconds=2,
    )
    service = BirdEvalService(settings, model=model)

    result = await service.eval_agent(request_for(sqlite_file))

    assert result["success"] is True
    assert result["approved"] is True
    assert result["attempts"] == 1
    assert result["toolRounds"] == 1
    assert result["sql"] == "SELECT name FROM people"
    assert result["requestedColumns"] == ["name"]
    assert len(model.calls) == 1
    assert len(model.responses) == 1


@pytest.mark.asyncio
async def test_bird_agent_semantic_gate_rejects_self_declared_output_contract_then_repairs(
    sqlite_file,
):
    model = SequenceToolModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "verifySql",
                        "args": {
                            "sql": "SELECT name FROM people",
                            "requestedColumns": ["name"],
                        },
                        "id": "semantic-output-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "verifySql",
                        "args": {
                            "sql": "SELECT name, id FROM people",
                            "requestedColumns": ["name", "id"],
                        },
                        "id": "semantic-output-2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    contract = {
        "expectedOutputs": [
            {"name": "name", "acceptedNames": [], "strictName": True},
            {"name": "id", "acceptedNames": [], "strictName": True},
        ],
        "answerShape": "rows",
        "grain": "one row per person",
        "aggregation": "",
        "requiredPredicates": [],
        "requiredSources": ["people"],
        "requiredGroupBy": [],
        "semanticRequirements": [],
    }
    semantic_model = SequenceSemanticModel(
        [
            json.dumps(contract),
            json.dumps({"passed": True, "errors": []}),
        ]
    )
    service = BirdEvalService(
        Settings(
            bird_eval_enabled=True,
            bird_eval_max_rounds=4,
            bird_eval_sql_timeout_seconds=2,
        ),
        model=model,
        semantic_model=semantic_model,
    )
    request = BirdEvalRequest(
        questionId="semantic-output",
        dbId="demo",
        question="Return every person's name and id.",
        evidence="",
        sqlitePath=str(sqlite_file),
    )

    result = await service.eval_agent(request)

    assert result["success"] is True
    assert result["approved"] is True
    assert result["semanticRepairs"] == 1
    assert result["requestedColumns"] == ["name", "id"]
    assert result["sql"] == "SELECT name, id FROM people"
    assert "REQUESTED_COLUMNS_CONTRACT_MISMATCH" in result["toolHistory"][0]["rules"]
    assert result["toolHistory"][0]["structuralPassed"] is True
    assert result["toolHistory"][0]["semanticPassed"] is False
    assert result["toolHistory"][1]["semanticPassed"] is True
    assert len(semantic_model.calls) == 2


@pytest.mark.asyncio
async def test_bird_agent_semantic_failure_closes_probes_and_requires_targeted_final_repair(
    sqlite_file,
):
    model = SequenceToolModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "verifySql",
                        "args": {
                            "sql": "SELECT name FROM people",
                            "requestedColumns": ["name"],
                        },
                        "id": "semantic-filter-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "verifySql",
                        "args": {
                            "sql": "SELECT id FROM people",
                            "requestedColumns": [],
                        },
                        "id": "semantic-filter-probe",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "verifySql",
                        "args": {
                            "sql": "SELECT name FROM people WHERE name = 'Ada'",
                            "requestedColumns": ["name"],
                        },
                        "id": "semantic-filter-2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    contract = {
        "expectedOutputs": [
            {"name": "name", "acceptedNames": [], "strictName": True},
        ],
        "answerShape": "rows",
        "grain": "the matching person",
        "aggregation": "",
        "requiredPredicates": [
            {
                "table": "people",
                "column": "name",
                "operator": "=",
                "value": "Ada",
                "description": "Explicit person filter",
            }
        ],
        "requiredSources": ["people"],
        "requiredGroupBy": [],
        "semanticRequirements": [],
    }
    semantic_model = SequenceSemanticModel(
        [
            json.dumps(contract),
            json.dumps({"passed": True, "errors": []}),
        ]
    )
    service = BirdEvalService(
        Settings(
            bird_eval_enabled=True,
            bird_eval_max_rounds=5,
            bird_eval_sql_timeout_seconds=2,
        ),
        model=model,
        semantic_model=semantic_model,
    )
    request = BirdEvalRequest(
        questionId="semantic-filter",
        dbId="demo",
        question="Return the name of the person named Ada.",
        evidence="name = 'Ada'",
        sqlitePath=str(sqlite_file),
    )

    result = await service.eval_agent(request)

    assert result["success"] is True
    assert result["approved"] is True
    assert result["semanticRepairs"] == 1
    assert result["probeCalls"] == 0
    assert "MISSING_REQUIRED_FILTER" in result["toolHistory"][0]["rules"]
    assert "SEMANTIC_REPAIR_MODE" in result["toolHistory"][1]["rules"]
    assert result["sql"] == "SELECT name FROM people WHERE name = 'Ada'"
    assert len(semantic_model.calls) == 2


@pytest.mark.asyncio
async def test_bird_agent_semantic_gate_enforces_scalar_answer_shape(sqlite_file):
    model = SequenceToolModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "verifySql",
                        "args": {
                            "sql": "SELECT name FROM people",
                            "requestedColumns": ["name"],
                        },
                        "id": "semantic-shape-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "verifySql",
                        "args": {
                            "sql": "SELECT COUNT(*) AS person_count FROM people",
                            "requestedColumns": ["person_count"],
                        },
                        "id": "semantic-shape-2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    contract = {
        "expectedOutputs": [
            {
                "name": "person_count",
                "acceptedNames": ["count", "COUNT(*)"],
                "strictName": False,
            }
        ],
        "answerShape": "scalar_aggregate",
        "grain": "one aggregate row",
        "aggregation": "COUNT people",
        "requiredPredicates": [],
        "requiredSources": ["people"],
        "requiredGroupBy": [],
        "semanticRequirements": [],
    }
    semantic_model = SequenceSemanticModel(
        [
            json.dumps(contract),
            json.dumps({"passed": True, "errors": []}),
        ]
    )
    service = BirdEvalService(
        Settings(
            bird_eval_enabled=True,
            bird_eval_max_rounds=4,
            bird_eval_sql_timeout_seconds=2,
        ),
        model=model,
        semantic_model=semantic_model,
    )
    request = BirdEvalRequest(
        questionId="semantic-shape",
        dbId="demo",
        question="How many people are there?",
        evidence="",
        sqlitePath=str(sqlite_file),
    )

    result = await service.eval_agent(request)

    assert result["success"] is True
    assert result["semanticRepairs"] == 1
    assert "ANSWER_SHAPE_MISMATCH" in result["toolHistory"][0]["rules"]
    assert result["sql"] == "SELECT COUNT(*) AS person_count FROM people"


@pytest.mark.asyncio
async def test_bird_agent_semantic_repair_budget_fails_closed(sqlite_file):
    model = SequenceToolModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "verifySql",
                        "args": {
                            "sql": "SELECT name FROM people",
                            "requestedColumns": ["name"],
                        },
                        "id": "semantic-budget-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "verifySql",
                        "args": {
                            "sql": "SELECT name FROM people WHERE name = 'Linus'",
                            "requestedColumns": ["name"],
                        },
                        "id": "semantic-budget-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="unused"),
        ]
    )
    contract = {
        "expectedOutputs": [
            {"name": "name", "acceptedNames": [], "strictName": True},
        ],
        "answerShape": "rows",
        "grain": "the matching person",
        "aggregation": "",
        "requiredPredicates": [
            {
                "table": "people",
                "column": "name",
                "operator": "=",
                "value": "Ada",
                "description": "Explicit person filter",
            }
        ],
        "requiredSources": ["people"],
        "requiredGroupBy": [],
        "semanticRequirements": [],
    }
    semantic_model = SequenceSemanticModel([json.dumps(contract)])
    service = BirdEvalService(
        Settings(
            bird_eval_enabled=True,
            bird_eval_max_rounds=5,
            bird_eval_max_semantic_repairs=2,
            bird_eval_sql_timeout_seconds=2,
        ),
        model=model,
        semantic_model=semantic_model,
    )
    request = BirdEvalRequest(
        questionId="semantic-budget",
        dbId="demo",
        question="Return the name of the person named Ada.",
        evidence="name = 'Ada'",
        sqlitePath=str(sqlite_file),
    )

    result = await service.eval_agent(request)

    assert result["success"] is False
    assert result["approved"] is False
    assert result["semanticRepairs"] == 2
    assert result["error"] == "SEMANTIC_REPAIR_BUDGET_EXHAUSTED"
    assert len(model.calls) == 2
    assert len(model.responses) == 1


@pytest.mark.asyncio
async def test_bird_agent_default_probe_budget_blocks_ninth_probe_then_allows_final(sqlite_file):
    probes = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "verifySql",
                    "args": {"sql": "SELECT name FROM people", "requestedColumns": []},
                    "id": f"probe-{index}",
                    "type": "tool_call",
                }
            ],
        )
        for index in range(1, 10)
    ]
    model = SequenceToolModel(
        [
            *probes,
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "verifySql",
                        "args": {
                            "sql": "SELECT name FROM people",
                            "requestedColumns": ["name"],
                        },
                        "id": "final-after-budget",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="unused model tail"),
        ]
    )
    settings = Settings(
        bird_eval_enabled=True,
        bird_eval_max_rounds=12,
        bird_eval_sql_timeout_seconds=2,
    )
    service = BirdEvalService(settings, model=model)

    result = await service.eval_agent(request_for(sqlite_file))

    assert settings.bird_eval_max_probe_calls == 8
    assert result["success"] is True
    assert result["approved"] is True
    assert result["probeCalls"] == 8
    assert result["attempts"] == 10
    assert result["sql"] == "SELECT name FROM people"
    assert any(
        "PROBE_BUDGET_EXHAUSTED" in item.get("rules", [])
        for item in result["toolHistory"]
    )
    assert len(model.calls) == 10
    assert len(model.responses) == 1
    system_prompt = model.calls[0][0].content
    assert "At most 8 probe-mode verifySql calls may execute" in system_prompt
    assert "only final-mode verifySql calls" in system_prompt


@pytest.mark.asyncio
async def test_bird_agent_max_rounds_skips_pending_tool_and_forces_final(sqlite_file):
    model = SequenceToolModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "listTables", "args": {}, "id": "max-1", "type": "tool_call"}
                ],
            ),
            AIMessage(content='{"sql":"SELECT name FROM people","requestedColumns":["name"]}'),
        ]
    )
    settings = Settings(bird_eval_enabled=True, bird_eval_max_rounds=1)
    service = BirdEvalService(settings, model=model)

    result = await service.eval_agent(request_for(sqlite_file))

    assert result["success"] is True
    assert result["attempts"] == 1
    assert result["toolRounds"] == 0
    assert any(item.get("skipped") == "MAX_ROUNDS" for item in result["toolHistory"])
    assert "Agent maximum rounds reached. Tool execution skipped." in model.calls[1][-1].content
    assert result["sql"] == "SELECT name FROM people"


@pytest.mark.asyncio
async def test_bird_agent_force_final_applies_same_agentx_json_repair(sqlite_file):
    model = SequenceToolModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "listTables", "args": {}, "id": "max-repair-1", "type": "tool_call"}
                ],
            ),
            AIMessage(
                content=(
                    "```json\n"
                    '{sql: "SELECT name FROM people", requestedColumns: ["name"],}\n'
                    "```"
                )
            ),
        ]
    )
    settings = Settings(bird_eval_enabled=True, bird_eval_max_rounds=1)
    service = BirdEvalService(settings, model=model)

    result = await service.eval_agent(request_for(sqlite_file))

    assert result["success"] is True
    assert result["toolRounds"] == 0
    assert any(item.get("skipped") == "MAX_ROUNDS" for item in result["toolHistory"])
    assert result["sql"] == "SELECT name FROM people"


@pytest.mark.asyncio
async def test_bird_agent_sanitizes_malformed_tool_args_before_history_replay(sqlite_file):
    model = SequenceToolModel(
        [
            AIMessage(
                content="",
                invalid_tool_calls=[
                    {
                        "name": "verifySql",
                        "args": "{bad-json",
                        "id": "malformed-1",
                        "error": "Invalid JSON tool arguments",
                        "type": "invalid_tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "verifySql",
                        "args": {
                            "sql": "SELECT name FROM people",
                            "requestedColumns": ["name"],
                        },
                        "id": "malformed-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content='{"sql":"SELECT name FROM people","requestedColumns":["name"]}'
            ),
        ]
    )
    settings = Settings(
        bird_eval_enabled=True,
        bird_eval_max_rounds=3,
        bird_eval_sql_timeout_seconds=2,
    )
    service = BirdEvalService(settings, model=model)

    result = await service.eval_agent(request_for(sqlite_file))

    assert result["success"] is True
    assert result["sql"] == "SELECT name FROM people"
    assert any("TOOL_CALL_ERROR" in item.get("rules", []) for item in result["toolHistory"])
    replayed_assistant = model.calls[1][-2]
    assert replayed_assistant.invalid_tool_calls == []
    assert replayed_assistant.tool_calls[0]["id"] == "malformed-1"
    assert replayed_assistant.tool_calls[0]["args"] == {}
    assert model.calls[1][-1].tool_call_id == "malformed-1"


@pytest.mark.asyncio
async def test_bird_agent_force_final_closes_sanitized_tool_call_with_missing_id(sqlite_file):
    model = SequenceToolModel(
        [
            AIMessage(
                content="",
                invalid_tool_calls=[
                    {
                        "name": "verifySql",
                        "args": "{bad-json",
                        "id": None,
                        "error": "Invalid JSON tool arguments",
                        "type": "invalid_tool_call",
                    }
                ],
            ),
            AIMessage(content='{"sql":"SELECT name FROM people","requestedColumns":["name"]}'),
        ]
    )
    settings = Settings(bird_eval_enabled=True, bird_eval_max_rounds=1)
    service = BirdEvalService(settings, model=model)

    result = await service.eval_agent(request_for(sqlite_file))

    assert result["success"] is True
    assert result["toolRounds"] == 0
    replayed_assistant = model.calls[1][-2]
    skipped_tool = model.calls[1][-1]
    assert replayed_assistant.invalid_tool_calls == []
    assert replayed_assistant.tool_calls[0]["args"] == {}
    assert replayed_assistant.tool_calls[0]["id"] == "bird-tool-1-1"
    assert skipped_tool.tool_call_id == "bird-tool-1-1"
    assert skipped_tool.content == "Agent maximum rounds reached. Tool execution skipped."


@pytest.mark.asyncio
async def test_bird_eval_is_disabled_by_default(sqlite_file):
    settings = Settings(bird_eval_enabled=False)
    service = BirdEvalService(settings, model=SequenceToolModel([]))

    result = await service.eval_agent(request_for(sqlite_file))

    assert result["success"] is False
    assert "BIRD eval endpoint is disabled" in result["error"]
