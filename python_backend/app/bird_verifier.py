from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bird_schema import SqliteSchemaProvider


LINE_COMMENT = re.compile(r"--[^\r\n]*")
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
STRING_LITERAL = re.compile(r"'(?:''|[^'])*'")
NON_ISO_DATE_LITERAL = re.compile(
    r"'(\d{4}/\d{1,2}(?:/\d{1,2})?|\d{1,2}/\d{1,2}/\d{4})'"
)
ABSENCE_REQUEST = re.compile(
    r"\b(?:is null|not null|missing|without|empty|absent)\b", re.IGNORECASE
)
NULL_CONDITION = re.compile(r"\bIS\s+(?:NOT\s+)?NULL\b", re.IGNORECASE)
CAST_AS_REAL = re.compile(r"CAST\s*\([^()]*AS\s+REAL\s*\)", re.IGNORECASE)
REAL_PROMOTION = re.compile(r"[\d.]+\s*\*\s*$")
TABLE_REF = re.compile(
    r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:AS\s+)?"
    r"([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
JOIN_EQUALITY = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*(?:`([^`]+)`|\"([^\"]+)\"|"
    r"([A-Za-z_][A-Za-z0-9_]*))\s*=\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*(?:`([^`]+)`|\"([^\"]+)\"|"
    r"([A-Za-z_][A-Za-z0-9_]*))",
)
DIVISION = re.compile(
    r"(?:(?:`([^`]+)`|\"([^\"]+)\"|([A-Za-z_][A-Za-z0-9_]*))\.)?"
    r"(?:`([^`]+)`|\"([^\"]+)\"|([A-Za-z_][A-Za-z0-9_]*))"
    r"\s*/\s*"
    r"(?:(?:`([^`]+)`|\"([^\"]+)\"|([A-Za-z_][A-Za-z0-9_]*))\.)?"
    r"(?:`([^`]+)`|\"([^\"]+)\"|([A-Za-z_][A-Za-z0-9_]*))"
)


@dataclass(frozen=True)
class ValidationDiagnostic:
    rule: str
    message: str
    fix: str

    def as_dict(self) -> dict[str, str]:
        return {"rule": self.rule, "message": self.message, "fix": self.fix}


@dataclass(frozen=True)
class SqlExecutionResult:
    ok: bool
    error: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    row_count: int
    duration_ms: int

    def as_tool_dict(self, max_rows: int) -> dict[str, Any]:
        max_rows = int(max_rows)
        if max_rows <= 0 or len(self.rows) <= max_rows:
            rows = list(self.rows)
            truncated = False
        else:
            if max_rows < 4:
                rows = list(self.rows[:max_rows])
            else:
                head_count = max_rows - 2
                rows = [*self.rows[:head_count], *self.rows[-2:]]
            truncated = True
        return {
            "ok": self.ok,
            "error": self.error or None,
            "columns": list(self.columns),
            "rows": rows,
            "rowCount": self.row_count,
            "durationMs": self.duration_ms,
            "rowsTruncated": truncated,
        }


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    execution: SqlExecutionResult
    errors: tuple[ValidationDiagnostic, ...]

    def as_tool_dict(self, max_rows: int) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "execution": self.execution.as_tool_dict(max_rows),
            "errors": [error.as_dict() for error in self.errors],
        }


class BirdSqlVerifier:
    """Execute a BIRD SQL draft and apply the Java verifier's deterministic rules."""

    def __init__(
        self,
        sqlite_path: Path | str,
        *,
        request_context: str,
        timeout_seconds: float = 20.0,
        schema_provider: SqliteSchemaProvider | None = None,
    ):
        self.path = Path(sqlite_path).expanduser().resolve()
        self.request_context = request_context or ""
        self.timeout_seconds = float(timeout_seconds)
        self.schema_provider = schema_provider or SqliteSchemaProvider(self.path)

        column_types = self.schema_provider.column_types()
        self.text_column_names = {
            name
            for columns in column_types.values()
            for name, col_type in columns.items()
            if "TEXT" in (col_type or "").upper()
        }
        self.integer_column_names = {
            name
            for columns in column_types.values()
            for name, col_type in columns.items()
            if "INT" in (col_type or "").upper()
        }
        self.fk_columns_by_table: dict[str, set[str]] = {}
        self.fk_targets_by_table: dict[str, set[str]] = {}
        for fk in self.schema_provider.foreign_keys():
            self.fk_columns_by_table.setdefault(fk.table, set()).add(fk.column)
            self.fk_targets_by_table.setdefault(fk.table, set()).add(
                f"{fk.ref_table}.{fk.ref_column}"
            )

    def verify(self, sql: str, requested_columns: list[str] | None) -> VerificationResult:
        sql = (sql or "").strip()
        if not sql:
            execution = SqlExecutionResult(
                False, "The SQL statement is blank.", tuple(), tuple(), 0, 0
            )
            return VerificationResult(
                False,
                execution,
                (
                    ValidationDiagnostic(
                        "BLANK_SQL",
                        "The SQL statement is blank.",
                        "Write one SQLite SELECT/WITH statement.",
                    ),
                ),
            )

        execution = self.execute(sql)
        if not execution.ok:
            return VerificationResult(
                False,
                execution,
                (
                    ValidationDiagnostic(
                        "SQLITE_EXECUTION",
                        execution.error,
                        "Fix the reported SQLite error with schema information, then call verifySql again.",
                    ),
                ),
            )

        requested = [str(value or "").strip() for value in (requested_columns or [])]
        if not requested:
            return VerificationResult(True, execution, tuple())

        errors: list[ValidationDiagnostic] = []
        inspectable_sql = _strip_literals_and_comments(sql)
        self._add_projection_errors(requested, list(execution.columns), errors)
        self._add_inferred_null_error(requested, inspectable_sql, errors)
        self._add_date_literal_errors(sql, errors)
        self._add_unuseful_column_errors(inspectable_sql, errors)
        self._add_text_identifier_errors(sql, errors)
        self._add_integer_division_errors(inspectable_sql, errors)
        self._add_display_name_join_errors(inspectable_sql, errors)
        return VerificationResult(not errors, execution, tuple(errors))

    def execute(self, sql: str) -> SqlExecutionResult:
        sql = (sql or "").strip()
        if not sql:
            return SqlExecutionResult(False, "EMPTY_SQL", tuple(), tuple(), 0, 0)
        first = sql.lstrip().split(None, 1)[0].lower() if sql.lstrip() else ""
        if first not in {"select", "with"}:
            return SqlExecutionResult(
                False, "ONLY_SELECT_OR_WITH_ALLOWED", tuple(), tuple(), 0, 0
            )

        started = time.perf_counter()
        connection: sqlite3.Connection | None = None
        timed_out = False
        try:
            uri = self.path.as_uri() + "?mode=ro"
            connection = sqlite3.connect(
                uri, uri=True, timeout=min(self.timeout_seconds, 30.0)
            )
            connection.execute("PRAGMA query_only=ON")
            deadline = time.monotonic() + self.timeout_seconds

            def progress() -> int:
                nonlocal timed_out
                if time.monotonic() > deadline:
                    timed_out = True
                    return 1
                return 0

            connection.set_progress_handler(progress, 1000)
            cursor = connection.execute(sql)
            columns = tuple(item[0] for item in (cursor.description or []))
            rows = tuple(tuple(row) for row in cursor.fetchall())
            return SqlExecutionResult(
                True,
                "",
                columns,
                rows,
                len(rows),
                round((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:
            error = (
                f"TIMEOUT after {self.timeout_seconds}s"
                if timed_out
                else f"{type(exc).__name__}: {exc}"
            )
            return SqlExecutionResult(
                False,
                error,
                tuple(),
                tuple(),
                0,
                round((time.perf_counter() - started) * 1000),
            )
        finally:
            if connection is not None:
                connection.close()

    def _add_projection_errors(
        self,
        requested: list[str],
        result_columns: list[str],
        errors: list[ValidationDiagnostic],
    ) -> None:
        if any(not field for field in requested):
            errors.append(
                ValidationDiagnostic(
                    "PROJECTION_CONTRACT",
                    "requestedColumns contains a blank field name.",
                    "Use exact schema field names or stable output aliases, then call verifySql again.",
                )
            )
            return

        if len(result_columns) != len(requested):
            errors.append(
                ValidationDiagnostic(
                    "PROJECTION_COUNT",
                    f"The SELECT returns {len(result_columns)} column(s), but requestedColumns has "
                    f"{len(requested)}.",
                    "Change SELECT to output exactly ["
                    + ", ".join(requested)
                    + "] in that order.",
                )
            )
            return

        positions: list[int] = []
        for field in requested:
            position = _unique_position(_normalize(field), result_columns)
            if position < 0:
                errors.append(
                    ValidationDiagnostic(
                        "PROJECTION_NAME",
                        f"No unique result column matches requested field {field}.",
                        f"Alias that output value as {field}, or correct requestedColumns to "
                        "the value requested by the question.",
                    )
                )
            else:
                positions.append(position)
        if len(positions) == len(requested) and len(requested) > 1:
            if any(position != index for index, position in enumerate(positions)):
                errors.append(
                    ValidationDiagnostic(
                        "PROJECTION_ORDER",
                        "The SELECT column order differs from requestedColumns.",
                        "Reorder SELECT to [" + ", ".join(requested) + "].",
                    )
                )

    def _add_inferred_null_error(
        self,
        requested_columns: list[str],
        sql: str,
        errors: list[ValidationDiagnostic],
    ) -> None:
        request = (
            self.request_context
            + " Requested values: "
            + ", ".join(requested_columns)
        )
        if ABSENCE_REQUEST.search(request) or not NULL_CONDITION.search(sql):
            return
        errors.append(
            ValidationDiagnostic(
                "INFERRED_NULL_FILTER",
                "The SQL adds an IS NULL or IS NOT NULL condition not required by the question/evidence.",
                "Remove that condition unless absence is explicitly requested.",
            )
        )

    def _add_date_literal_errors(
        self, sql: str, errors: list[ValidationDiagnostic]
    ) -> None:
        match = NON_ISO_DATE_LITERAL.search(sql)
        if not match:
            return
        errors.append(
            ValidationDiagnostic(
                "DATE_LITERAL_FORMAT",
                f"Date literal '{match.group(1)}' uses slashes; BIRD databases usually store dates "
                "as ISO-8601 text 'YYYY-MM-DD' and SQLite compares TEXT lexicographically, so a "
                "slash literal usually excludes matching rows.",
                "Probe SELECT DISTINCT <date_col> to confirm the stored format, then normalize the "
                "literal to 'YYYY-MM-DD' or compare years with STRFTIME('%Y', column).",
            )
        )

    def _add_unuseful_column_errors(
        self, sql: str, errors: list[ValidationDiagnostic]
    ) -> None:
        for table, descriptions in self.schema_provider.descriptions.items():
            for column, description in descriptions.items():
                normalized = _normalize(description.value_description)
                if normalized not in {"unuseful", "notuseful"}:
                    continue
                if _contains_identifier(sql, column) and not _contains_identifier(
                    self.request_context, column
                ):
                    errors.append(
                        ValidationDiagnostic(
                            "UNUSEFUL_METADATA_COLUMN",
                            f"The SQL uses BIRD metadata-unuseful column {table}.{column}.",
                            "Remove that filter, grouping, or branch unless the question/evidence names it.",
                        )
                    )

    def _add_text_identifier_errors(
        self, sql: str, errors: list[ValidationDiagnostic]
    ) -> None:
        for column in self.text_column_names:
            token = _identifier_alternation(column)
            qualified = r"(?:[A-Za-z_][A-Za-z0-9_]*\.)?"
            checks = (
                (
                    "TEXT_IDENTIFIER_PADDING",
                    rf"['\"]0+['\"]\s*\|\|\s*{qualified}{token}|"
                    rf"{qualified}{token}\s*\|\|\s*['\"]0+['\"]",
                    f"Text identifier {column} is padded with leading zeroes.",
                    "Compare the identifier as stored and remove the padding.",
                ),
                (
                    "TEXT_IDENTIFIER_CAST",
                    rf"CAST\s*\(\s*{qualified}{token}\s+AS\s+(?:INTEGER|INT)\s*\)",
                    f"Text identifier {column} is cast to INTEGER.",
                    "Remove the CAST and compare the original text value.",
                ),
                (
                    "TEXT_IDENTIFIER_SUBSTR",
                    rf"(?:SUBSTR|SUBSTRING)\s*\(\s*{qualified}{token}\s*,",
                    f"Text identifier {column} is reshaped with SUBSTR/SUBSTRING.",
                    "Remove the substring operation unless evidence explicitly requires it.",
                ),
                (
                    "TEXT_IDENTIFIER_TRIM",
                    rf"(?:LTRIM|TRIM)\s*\(\s*{qualified}{token}\s*,\s*['\"]0['\"]\s*\)",
                    f"Leading zeroes are removed from text identifier {column}.",
                    "Remove LTRIM/TRIM and compare the original text value.",
                ),
            )
            for rule, pattern, message, fix in checks:
                if re.search(pattern, sql, re.IGNORECASE):
                    errors.append(ValidationDiagnostic(rule, message, fix))

    def _add_integer_division_errors(
        self, sql: str, errors: list[ValidationDiagnostic]
    ) -> None:
        real_spans = [match.span() for match in CAST_AS_REAL.finditer(sql)]
        for match in DIVISION.finditer(sql):
            left = _first_group(match, 4, 5, 6)
            right = _first_group(match, 10, 11, 12)
            if (
                left not in self.integer_column_names
                or right not in self.integer_column_names
            ):
                continue
            if REAL_PROMOTION.search(sql[: match.start()]):
                continue
            if any(start <= match.start() < end for start, end in real_spans):
                continue
            errors.append(
                ValidationDiagnostic(
                    "INTEGER_DIVISION",
                    f"Column {left} / {right} divides two INTEGER columns, so SQLite truncates "
                    "the quotient toward zero.",
                    "Wrap the numerator in CAST(... AS REAL) or multiply it by 1.0, unless an "
                    "integer quotient is intended.",
                )
            )
            return

    def _add_display_name_join_errors(
        self, sql: str, errors: list[ValidationDiagnostic]
    ) -> None:
        alias_to_table: dict[str, str] = {}
        for match in TABLE_REF.finditer(sql):
            alias_to_table[match.group(2).lower()] = match.group(1)

        for match in JOIN_EQUALITY.finditer(sql):
            alias_a = match.group(1)
            column_a = _first_group(match, 2, 3, 4)
            alias_b = match.group(5)
            column_b = _first_group(match, 6, 7, 8)
            table_a = alias_to_table.get(alias_a.lower(), alias_a)
            table_b = alias_to_table.get(alias_b.lower(), alias_b)

            if table_a.lower() == table_b.lower():
                continue
            if (
                column_a not in self.text_column_names
                or column_b not in self.text_column_names
            ):
                continue
            if column_a in self.fk_columns_by_table.get(
                table_a, set()
            ) or column_b in self.fk_columns_by_table.get(table_b, set()):
                continue

            targets_a = self.fk_targets_by_table.get(table_a, set())
            targets_b = self.fk_targets_by_table.get(table_b, set())
            direct_fk = any(
                target.startswith(table_b + ".") for target in targets_a
            ) or any(target.startswith(table_a + ".") for target in targets_b)
            shared = next(iter(targets_a & targets_b), None)
            if not direct_fk and shared is None:
                continue

            evidence = (
                f"A declared foreign key connects {table_a} and {table_b}"
                if direct_fk
                else f"Tables {table_a} and {table_b} both declare foreign keys onto {shared}"
            )
            errors.append(
                ValidationDiagnostic(
                    "JOIN_DISPLAY_NAME",
                    f"{evidence}, but this SQL joins them on text columns "
                    f"{alias_a}.{column_a} = {alias_b}.{column_b}.",
                    "Join through the declared code columns instead; display-name joins drop or "
                    "duplicate rows whenever names repeat. Probe both join shapes to compare row "
                    "counts if unsure.",
                )
            )
            return


def _first_group(match: re.Match[str], *groups: int) -> str:
    for group in groups:
        value = match.group(group)
        if value is not None:
            return value
    return ""


def _unique_position(requested: str, result_columns: list[str]) -> int:
    positions = [
        index
        for index, column in enumerate(result_columns)
        if _normalize(column) == requested
    ]
    return positions[0] if len(positions) == 1 else -1


def _identifier_alternation(identifier: str) -> str:
    quoted = re.escape(identifier)
    return rf"(?:`{quoted}`|\"{quoted}\"|\[{quoted}\]|{quoted})"


def _contains_identifier(sql: str, identifier: str) -> bool:
    token = _identifier_alternation(identifier)
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9_]){token}(?![A-Za-z0-9_])",
            sql,
            re.IGNORECASE,
        )
    )


def _strip_literals_and_comments(sql: str) -> str:
    without_comments = BLOCK_COMMENT.sub(" ", LINE_COMMENT.sub(" ", sql))
    return STRING_LITERAL.sub("''", without_comments)


def _normalize(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


