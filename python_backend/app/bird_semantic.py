from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .bird_schema import SqliteSchemaProvider


class ContractOutput(BaseModel):
    name: str
    acceptedNames: list[str] = Field(default_factory=list)
    strictName: bool = True
    description: str = ""


class ContractPredicate(BaseModel):
    table: str = ""
    column: str
    operator: str = "="
    value: str
    description: str = ""


class ContractAmbiguity(BaseModel):
    concept: str
    candidateColumns: list[str] = Field(default_factory=list)
    reason: str = ""


class AggregateFormulaRequirement(BaseModel):
    description: str
    requireSum: bool = True
    requireCount: bool = True
    requireDivision: bool = True
    predicatePlacement: Literal["having", "any"] = "any"
    thresholdOperator: str = ""
    thresholdValue: str = ""


class BirdTaskContract(BaseModel):
    expectedOutputs: list[ContractOutput] = Field(default_factory=list)
    answerShape: Literal["scalar_aggregate", "grouped", "rows", "unknown"] = "unknown"
    grain: str = ""
    aggregation: str = ""
    requiredPredicates: list[ContractPredicate] = Field(default_factory=list)
    requiredSources: list[str] = Field(default_factory=list)
    requiredGroupBy: list[str] = Field(default_factory=list)
    requiredAggregateFormulas: list[AggregateFormulaRequirement] = Field(default_factory=list)
    ambiguities: list[ContractAmbiguity] = Field(default_factory=list)
    semanticRequirements: list[str] = Field(default_factory=list)
    source: str = "fallback"


class SemanticGateError(BaseModel):
    rule: str
    message: str
    fix: str


class SemanticGateResult(BaseModel):
    passed: bool
    errors: list[SemanticGateError] = Field(default_factory=list)
    criticUsed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump()


_CONTRACT_SYSTEM_PROMPT = """You are the immutable task-contract builder for a BIRD Text-to-SQL evaluator.
Do not write SQL. Convert the question, evidence, and database schema into a compact semantic contract that a later verifier can enforce.

Rules:
- Capture only requirements supported by the question/evidence/schema; do not invent filters.
- expectedOutputs are exactly the values that must appear in the final SELECT, in order. Include every requested entity/attribute/aggregate and no helper columns.
- For schema-backed outputs, use the actual schema column name as name and put legitimate aliases/synonyms in acceptedNames. For derived aggregates, set strictName=false.
- answerShape=scalar_aggregate only when the final answer is one aggregate value with no grouping; grouped when the final answer is one row per group; rows for ordinary detail/entity rows; unknown only when genuinely unclear.
- requiredPredicates must preserve every explicit question/evidence filter or mapped value. Resolve evidence mappings to one schema field only when the wording/schema makes that mapping unambiguous.
- Never invent a scope noun. If the question merely says "in Riverside" (or another bare place/organization name) without saying city/county/district/state, do NOT lock it to County/City/District. Put the unresolved choice in ambiguities instead.
- If a requested concept matches multiple real schema columns (for example generic "funding type", "school name", or a geographic scope), preserve those alternatives in ambiguities rather than silently choosing one source.
- If evidence defines a formula such as SUM(x) / COUNT(y), preserve the aggregate formula exactly. Do NOT rewrite an aggregate threshold as a row-level predicate such as x > threshold.
- requiredSources lists only tables that are definitely necessary regardless of unresolved alternatives. Do not force one ambiguous source table merely because it was your first guess.
- requiredGroupBy lists schema fields that unambiguously define the requested grouping grain.
- requiredAggregateFormulas lists evidence-defined aggregate formula requirements. Use predicatePlacement="having" when the question compares the aggregate to a threshold.
- ambiguities lists semantic mappings the SQL agent must actively resolve from schema/probes before final verification.
- semanticRequirements contains any important requirement that is difficult to express as a simple predicate, source, grouping, formula, or ambiguity check.
- The contract is immutable after this step. A later SQL agent is not allowed to weaken it.

Return only one RFC8259 JSON object with these keys:
{
  "expectedOutputs": [{"name":"...","acceptedNames":["..."],"strictName":true,"description":"..."}],
  "answerShape":"scalar_aggregate|grouped|rows|unknown",
  "grain":"...",
  "aggregation":"...",
  "requiredPredicates":[{"table":"...","column":"...","operator":"=","value":"...","description":"..."}],
  "requiredSources":["..."],
  "requiredGroupBy":["..."],
  "requiredAggregateFormulas":[{"description":"...","requireSum":true,"requireCount":true,"requireDivision":true,"predicatePlacement":"having|any","thresholdOperator":">","thresholdValue":"400"}],
  "ambiguities":[{"concept":"...","candidateColumns":["table.column"],"reason":"..."}],
  "semanticRequirements":["..."]
}
"""


_CRITIC_SYSTEM_PROMPT = """You are an independent semantic SQL critic. You do not generate SQL and you have no tools.
The candidate SQL has already passed execution/structural verification. Decide whether it completely satisfies the immutable task contract and the original question/evidence.

Do not treat successful execution, non-empty rows, or matching self-declared requestedColumns as proof of semantic correctness. Fail the candidate for any missing requested output, missing explicit filter/mapping, wrong semantic source, wrong aggregate/grain/grouping, or wording-faithfulness error. Do not add requirements that are absent from the task contract/question/evidence.

Return only one RFC8259 JSON object:
{
  "passed": true,
  "errors": []
}
or
{
  "passed": false,
  "errors": [
    {"rule":"SEMANTIC_CONSTRAINT_MISSING","message":"...","fix":"..."}
  ]
}
"""


class BirdSemanticGate:
    def __init__(
        self,
        *,
        model: Any | None,
        provider: SqliteSchemaProvider,
        question: str,
        evidence: str,
        critic_enabled: bool = True,
    ) -> None:
        self.model = model
        self.provider = provider
        self.question = question.strip()
        self.evidence = evidence.strip()
        self.critic_enabled = bool(critic_enabled)
        self._schema_text: str | None = None

    def schema_text(self) -> str:
        if self._schema_text is None:
            self._schema_text = self.provider.describe_all_tables()
        return self._schema_text

    async def build_contract(self) -> BirdTaskContract:
        if self.model is None:
            return self._fallback_contract()

        prompt = (
            f"Question:\n{self.question}\n\n"
            f"Evidence:\n{self.evidence or '(none)'}\n\n"
            f"Database schema:\n{self.schema_text()}"
        )
        try:
            response = await self.model.ainvoke(
                [
                    SystemMessage(content=_CONTRACT_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )
            payload = _parse_json_object(_message_text(response))
            contract = BirdTaskContract.model_validate(payload)
            contract = self._augment_contract(contract)
            return contract.model_copy(update={"source": "semantic_model"})
        except Exception:
            return self._augment_contract(self._fallback_contract())

    async def verify(
        self,
        *,
        contract: BirdTaskContract,
        sql: str,
        requested_columns: list[str],
        result_columns: list[str] | tuple[str, ...],
    ) -> SemanticGateResult:
        deterministic = _deterministic_contract_errors(
            contract=contract,
            sql=sql,
            requested_columns=requested_columns,
            result_columns=list(result_columns),
        )
        if deterministic:
            return SemanticGateResult(passed=False, errors=deterministic, criticUsed=False)

        if self.model is None or not self.critic_enabled:
            return SemanticGateResult(passed=True, errors=[], criticUsed=False)

        prompt = (
            f"Question:\n{self.question}\n\n"
            f"Evidence:\n{self.evidence or '(none)'}\n\n"
            f"Immutable task contract:\n{json.dumps(contract.model_dump(exclude={'source'}), ensure_ascii=False, indent=2)}\n\n"
            f"Database schema:\n{self.schema_text()}\n\n"
            f"Candidate SQL:\n{sql}\n\n"
            f"Agent requestedColumns:\n{json.dumps(requested_columns, ensure_ascii=False)}\n\n"
            f"Actual result columns:\n{json.dumps(list(result_columns), ensure_ascii=False)}"
        )
        try:
            response = await self.model.ainvoke(
                [
                    SystemMessage(content=_CRITIC_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )
            payload = _parse_json_object(_message_text(response))
            passed = bool(payload.get("passed"))
            raw_errors = payload.get("errors") or []
            errors: list[SemanticGateError] = []
            for raw in raw_errors:
                if not isinstance(raw, dict):
                    continue
                errors.append(
                    SemanticGateError(
                        rule=str(raw.get("rule") or "SEMANTIC_CONTRACT_FAILED"),
                        message=str(raw.get("message") or "Semantic contract check failed."),
                        fix=str(raw.get("fix") or "Repair the SQL to satisfy the immutable task contract, then final-verify again."),
                    )
                )
            if not passed and not errors:
                errors.append(
                    SemanticGateError(
                        rule="SEMANTIC_CONTRACT_FAILED",
                        message="The independent semantic critic rejected the candidate SQL.",
                        fix="Repair the SQL to satisfy the immutable task contract, then final-verify again.",
                    )
                )
            return SemanticGateResult(passed=passed and not errors, errors=errors, criticUsed=True)
        except Exception as exc:
            # A critic outage must never turn structural success into semantic approval.
            return SemanticGateResult(
                passed=False,
                criticUsed=True,
                errors=[
                    SemanticGateError(
                        rule="SEMANTIC_CRITIC_ERROR",
                        message=f"Independent semantic critic failed: {type(exc).__name__}: {exc}",
                        fix="Retry the final semantic verification; do not treat structural success as final approval.",
                    )
                ],
            )

    def _augment_contract(self, contract: BirdTaskContract) -> BirdTaskContract:
        """Add deterministic facts the contract model is not allowed to erase.

        The model is useful for semantic decomposition, but explicit evidence formulae and
        schema-level ambiguity are safer to derive mechanically. This post-pass therefore
        strengthens the immutable contract without consulting gold SQL/results.
        """
        formulas = list(contract.requiredAggregateFormulas)
        predicates = list(contract.requiredPredicates)
        ambiguities = list(contract.ambiguities)
        requirements = list(contract.semanticRequirements)

        evidence_lower = self.evidence.lower()
        has_sum_count_formula = bool(
            re.search(r"\bsum\s*\(", evidence_lower)
            and re.search(r"\bcount\s*\(", evidence_lower)
            and "/" in evidence_lower
        )
        if has_sum_count_formula:
            operator, threshold = _comparison_threshold(self.question)
            if not formulas:
                formulas.append(
                    AggregateFormulaRequirement(
                        description=self.evidence,
                        predicatePlacement="having" if operator and threshold else "any",
                        thresholdOperator=operator,
                        thresholdValue=threshold,
                    )
                )
            formula_requirement = (
                "Evidence defines an aggregate SUM(...)/COUNT(...) formula. Preserve that "
                "aggregate expression; do not replace it with a row-level measure filter."
            )
            if formula_requirement not in requirements:
                requirements.append(formula_requirement)

            # If the contract model incorrectly converted the aggregate threshold into a
            # direct row predicate on the aggregated measure, remove that weaker predicate.
            aggregation_normalized = _normalize_name(contract.aggregation)
            filtered: list[ContractPredicate] = []
            for predicate in predicates:
                comparison = predicate.operator.strip().lower()
                value_matches_threshold = bool(
                    threshold and _normalize_name(predicate.value) == _normalize_name(threshold)
                )
                column_is_aggregate_measure = bool(
                    aggregation_normalized
                    and _normalize_name(predicate.column) in aggregation_normalized
                )
                if (
                    comparison in {">", "<", ">=", "<=", "gt", "lt", "gte", "lte"}
                    and value_matches_threshold
                    and column_is_aggregate_measure
                ):
                    continue
                filtered.append(predicate)
            predicates = filtered

        # Preserve schema ambiguity for generic requested concepts. If two or more actual
        # columns match the output name/aliases, the SQL agent must inspect alternatives
        # instead of assuming the first source is authoritative.
        schema_columns: dict[str, list[str]] = {}
        for table, columns in self.provider.column_types().items():
            for column in columns:
                schema_columns.setdefault(_normalize_name(column), []).append(
                    f"{table}.{column}"
                )
        existing_ambiguity_keys = {
            (ambiguity.concept, tuple(sorted(ambiguity.candidateColumns)))
            for ambiguity in ambiguities
        }
        for expected in contract.expectedOutputs:
            candidates: set[str] = set()
            for name in [expected.name, *expected.acceptedNames]:
                candidates.update(schema_columns.get(_normalize_name(name), []))
            if len(candidates) < 2:
                continue
            ordered = sorted(candidates)
            key = (f"output:{expected.name}", tuple(ordered))
            if key in existing_ambiguity_keys:
                continue
            ambiguities.append(
                ContractAmbiguity(
                    concept=f"output:{expected.name}",
                    candidateColumns=ordered,
                    reason=(
                        "Multiple real schema columns match this requested concept. Inspect the "
                        "candidate table/column semantics before choosing the final source."
                    ),
                )
            )
            existing_ambiguity_keys.add(key)

        return contract.model_copy(
            update={
                "requiredPredicates": predicates,
                "requiredAggregateFormulas": formulas,
                "ambiguities": ambiguities,
                "semanticRequirements": requirements,
            }
        )

    def _fallback_contract(self) -> BirdTaskContract:
        """Conservative no-model fallback used for outages and unit-test doubles.

        It intentionally enforces only high-confidence requirements so a contract-builder
        outage cannot invent benchmark semantics. The production path uses the independent
        semantic model above.
        """
        question_lower = self.question.lower()
        answer_shape: Literal["scalar_aggregate", "grouped", "rows", "unknown"] = "unknown"
        if re.match(r"^\s*how many\b", question_lower) and not re.search(
            r"\b(each|per|for each|by)\b", question_lower
        ):
            answer_shape = "scalar_aggregate"

        required_predicates: list[ContractPredicate] = []
        evidence = self.evidence
        assignment_pattern = re.compile(
            r"(?P<column>`[^`]+`|[A-Za-z_][A-Za-z0-9_ ]{0,60}?)\s*=\s*"
            r"(?P<value>'[^']*'|\"[^\"]*\"|-?\d+(?:\.\d+)?)"
        )
        known_columns = {
            _normalize_name(column): (table, column)
            for table, columns in self.provider.column_types().items()
            for column in columns
        }
        for match in assignment_pattern.finditer(evidence):
            raw_column = match.group("column").strip("` ")
            normalized = _normalize_name(raw_column)
            resolved = known_columns.get(normalized)
            if resolved is None:
                continue
            table, column = resolved
            raw_value = match.group("value").strip()
            value = raw_value[1:-1] if raw_value[:1] in {"'", '"'} else raw_value
            required_predicates.append(
                ContractPredicate(
                    table=table,
                    column=column,
                    operator="=",
                    value=value,
                    description="Explicit evidence assignment",
                )
            )

        return BirdTaskContract(
            answerShape=answer_shape,
            requiredPredicates=required_predicates,
            requiredSources=sorted({p.table for p in required_predicates if p.table}),
            source="fallback",
        )


def _deterministic_contract_errors(
    *,
    contract: BirdTaskContract,
    sql: str,
    requested_columns: list[str],
    result_columns: list[str],
) -> list[SemanticGateError]:
    errors: list[SemanticGateError] = []

    if contract.expectedOutputs:
        if len(requested_columns) != len(contract.expectedOutputs):
            errors.append(
                SemanticGateError(
                    rule="REQUESTED_COLUMNS_CONTRACT_MISMATCH",
                    message=(
                        f"Task contract requires {len(contract.expectedOutputs)} output(s), but "
                        f"agent requestedColumns declares {len(requested_columns)}."
                    ),
                    fix=(
                        "Restore every output required by TaskContract.expectedOutputs and use that "
                        "same output order in requestedColumns and the final SELECT."
                    ),
                )
            )
        elif len(result_columns) != len(contract.expectedOutputs):
            errors.append(
                SemanticGateError(
                    rule="EXPECTED_OUTPUT_COUNT_MISMATCH",
                    message=(
                        f"Task contract requires {len(contract.expectedOutputs)} output(s), but the "
                        f"SQL returns {len(result_columns)}."
                    ),
                    fix="Change the final SELECT to return exactly the task-contract outputs.",
                )
            )
        else:
            for index, (declared, expected) in enumerate(
                zip(requested_columns, contract.expectedOutputs), start=1
            ):
                if not expected.strictName:
                    continue
                allowed = [expected.name, *expected.acceptedNames]
                if not any(_names_equivalent(declared, candidate) for candidate in allowed):
                    errors.append(
                        SemanticGateError(
                            rule="REQUESTED_COLUMNS_CONTRACT_MISMATCH",
                            message=(
                                f"Output {index} is declared as {declared!r}, but the immutable "
                                f"contract requires {expected.name!r}."
                            ),
                            fix=(
                                f"Return {expected.name!r} in output position {index} (or one of its "
                                "contract-approved aliases) and keep requestedColumns aligned."
                            ),
                        )
                    )

    sql_lower = sql.lower()
    if contract.answerShape == "scalar_aggregate":
        if len(result_columns) != 1 or not re.search(
            r"\b(count|sum|avg|min|max)\s*\(", sql_lower
        ) or re.search(r"\bgroup\s+by\b", sql_lower):
            errors.append(
                SemanticGateError(
                    rule="ANSWER_SHAPE_MISMATCH",
                    message=(
                        "Task contract requires one scalar aggregate value, but the candidate SQL "
                        "does not have scalar-aggregate shape."
                    ),
                    fix=(
                        "Return exactly one aggregate value (for example COUNT/SUM/AVG as required) "
                        "without detail columns or GROUP BY unless the task contract says otherwise."
                    ),
                )
            )
    elif contract.answerShape == "grouped" and not re.search(r"\bgroup\s+by\b", sql_lower):
        errors.append(
            SemanticGateError(
                rule="ANSWER_SHAPE_MISMATCH",
                message="Task contract requires grouped output, but the candidate SQL has no GROUP BY.",
                fix="Group at the grain required by TaskContract.requiredGroupBy/grain.",
            )
        )

    for source in contract.requiredSources:
        if source and not _source_is_referenced(sql, source):
            errors.append(
                SemanticGateError(
                    rule="MISSING_REQUIRED_SOURCE",
                    message=f"Task contract requires source table {source!r}, but the SQL does not reference it.",
                    fix=f"Reference {source!r} through the schema-supported join/source needed by the contract.",
                )
            )

    for formula in contract.requiredAggregateFormulas:
        missing_parts: list[str] = []
        if formula.requireSum and not re.search(r"\bsum\s*\(", sql_lower):
            missing_parts.append("SUM(...)")
        if formula.requireCount and not re.search(r"\bcount\s*\(", sql_lower):
            missing_parts.append("COUNT(...)")
        if formula.requireDivision and "/" not in sql:
            missing_parts.append("division (/)")
        having_region = _having_region(sql)
        if formula.predicatePlacement == "having" and not having_region:
            missing_parts.append("HAVING aggregate predicate")
        if formula.thresholdValue:
            threshold_region = having_region if formula.predicatePlacement == "having" else sql
            if _normalize_name(formula.thresholdValue) not in _normalize_name(threshold_region):
                missing_parts.append(f"threshold {formula.thresholdValue}")
        if missing_parts:
            errors.append(
                SemanticGateError(
                    rule="AGGREGATE_FORMULA_MISMATCH",
                    message=(
                        "Task evidence requires an aggregate formula, but the candidate SQL is "
                        "missing: " + ", ".join(missing_parts) + "."
                    ),
                    fix=(
                        "Implement the evidence-defined aggregate with SUM(...)/COUNT(...), keep "
                        "the aggregate threshold in HAVING when required, and final-verify again."
                    ),
                )
            )

    predicate_region = _predicate_region(sql)
    for predicate in contract.requiredPredicates:
        if not _predicate_is_present(predicate_region, predicate):
            rendered = f"{predicate.column} {predicate.operator} {predicate.value}"
            errors.append(
                SemanticGateError(
                    rule="MISSING_REQUIRED_FILTER",
                    message=f"Required contract predicate is absent from the SQL filter logic: {rendered}.",
                    fix=f"Restore the explicit predicate {rendered} using the correct table/column mapping.",
                )
            )

    if contract.requiredGroupBy:
        group_region = _group_by_region(sql)
        for field in contract.requiredGroupBy:
            if field and _normalize_name(field) not in _normalize_name(group_region):
                errors.append(
                    SemanticGateError(
                        rule="MISSING_REQUIRED_GROUPING",
                        message=f"Task contract requires grouping by {field!r}, but it is absent from GROUP BY.",
                        fix=f"Group by {field!r} at the contract grain.",
                    )
                )

    return _dedupe_errors(errors)


def _source_is_referenced(sql: str, table: str) -> bool:
    escaped = re.escape(table)
    return bool(
        re.search(
            rf"\b(?:from|join)\s+(?:[`\"\[])?{escaped}(?:[`\"\]])?(?:\s|$)",
            sql,
            flags=re.IGNORECASE,
        )
    )


def _predicate_region(sql: str) -> str:
    match = re.search(r"\b(where|having)\b", sql, flags=re.IGNORECASE)
    return sql[match.start() :] if match else ""


def _having_region(sql: str) -> str:
    match = re.search(r"\bhaving\b", sql, flags=re.IGNORECASE)
    if not match:
        return ""
    tail = sql[match.end() :]
    stop = re.search(
        r"\b(order\s+by|limit|union|except|intersect)\b",
        tail,
        flags=re.IGNORECASE,
    )
    return tail[: stop.start()] if stop else tail


def _group_by_region(sql: str) -> str:
    match = re.search(r"\bgroup\s+by\b", sql, flags=re.IGNORECASE)
    if not match:
        return ""
    tail = sql[match.end() :]
    stop = re.search(r"\b(order\s+by|limit|having|union|except|intersect)\b", tail, flags=re.IGNORECASE)
    return tail[: stop.start()] if stop else tail


def _predicate_is_present(region: str, predicate: ContractPredicate) -> bool:
    if not region:
        return False
    normalized_region = _normalize_name(region)
    if _normalize_name(predicate.column) not in normalized_region:
        return False

    value = str(predicate.value or "").strip()
    if not value:
        return True
    value_normalized = _normalize_name(value)
    if value_normalized and value_normalized in normalized_region:
        return True

    # Numeric values should not disappear merely because SQL renders 1 as 1.0.
    try:
        numeric = float(value)
    except ValueError:
        return False
    numeric_pattern = rf"(?<![\d.]){re.escape(str(int(numeric)) if numeric.is_integer() else str(numeric))}(?:\.0+)?(?![\d.])"
    return bool(re.search(numeric_pattern, region))


def _comparison_threshold(question: str) -> tuple[str, str]:
    text = str(question or "")
    patterns = (
        (r"\b(?:greater|higher|more)\s+than\s+(-?\d+(?:\.\d+)?)", ">"),
        (r"\b(?:less|lower|fewer)\s+than\s+(-?\d+(?:\.\d+)?)", "<"),
        (r"\bat\s+least\s+(-?\d+(?:\.\d+)?)", ">="),
        (r"\bat\s+most\s+(-?\d+(?:\.\d+)?)", "<="),
        (r"\b(?:>=|≥)\s*(-?\d+(?:\.\d+)?)", ">="),
        (r"\b(?:<=|≤)\s*(-?\d+(?:\.\d+)?)", "<="),
        (r"\b>\s*(-?\d+(?:\.\d+)?)", ">"),
        (r"\b<\s*(-?\d+(?:\.\d+)?)", "<"),
    )
    for pattern, operator in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return operator, match.group(1)
    return "", ""


def _names_equivalent(left: str, right: str) -> bool:
    a = _normalize_name(left)
    b = _normalize_name(right)
    if not a or not b:
        return False
    return a == b or a.replace("count", "") == b.replace("count", "")


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _dedupe_errors(errors: list[SemanticGateError]) -> list[SemanticGateError]:
    seen: set[tuple[str, str]] = set()
    result: list[SemanticGateError] = []
    for error in errors:
        key = (error.rule, error.message)
        if key in seen:
            continue
        seen.add(key)
        result.append(error)
    return result


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


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, flags=re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start : end + 1]
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("semantic model response must be a JSON object")
    return parsed
