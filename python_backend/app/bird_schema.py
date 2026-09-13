from __future__ import annotations

import csv
import io
import sqlite3
from dataclasses import dataclass
from pathlib import Path


MAX_EXAMPLE_VALUES = 5
MAX_EXAMPLE_LENGTH = 40


@dataclass(frozen=True)
class ColumnDescription:
    alias: str = ""
    description: str = ""
    value_description: str = ""

    def formatted_comment(self, original_name: str) -> str:
        parts: list[str] = []
        if self.description.strip():
            parts.append(self.description.strip())
        if self.alias.strip() and _normalize(self.alias) != _normalize(original_name):
            parts.append(f"alias: {self.alias.strip()}")
        if self.value_description.strip():
            value = self.value_description.strip()
            if _normalize(value) == "unuseful":
                parts.append(
                    "Values: unuseful. Do not filter, group by, or branch on this field "
                    "unless the question or evidence explicitly requires it."
                )
            else:
                parts.append(f"Values: {value}")
        return "\n".join(parts)


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    col_type: str
    not_null: bool
    primary_key: bool
    description: ColumnDescription
    examples: tuple[str, ...]


@dataclass(frozen=True)
class ForeignKeyInfo:
    table: str
    column: str
    ref_table: str
    ref_column: str


class SqliteSchemaProvider:
    """BIRD SQLite schema exploration aligned with the Java SqliteSchemaProvider."""

    def __init__(self, sqlite_path: Path | str):
        self.path = Path(sqlite_path).expanduser().resolve()
        self.db_id = self.path.stem
        self._descriptions = load_database_descriptions(self.path)
        self._tables: dict[str, tuple[ColumnInfo, ...]] | None = None
        self._foreign_keys: tuple[ForeignKeyInfo, ...] | None = None

    @property
    def descriptions(self) -> dict[str, dict[str, ColumnDescription]]:
        return self._descriptions

    def list_tables(self) -> str:
        self._ensure_schema()
        assert self._tables is not None
        if not self._tables:
            return "Schema is empty. The SQLite file may contain no tables or the path may be wrong."
        lines = [f"Database: {self.db_id}", "Tables:"]
        lines.extend(f"  {_quote_tick(table)}" for table in self._tables)
        fk_lines = self._format_foreign_keys(set(self._tables))
        if fk_lines:
            lines.extend(["", "Foreign keys (table.column → ref_table.ref_column):"])
            lines.extend(f"  {line}" for line in fk_lines)
        return "\n".join(lines).strip()

    def describe_tables(self, table_names: list[str] | tuple[str, ...] | None) -> str:
        self._ensure_schema()
        assert self._tables is not None
        if not table_names:
            return "tableNames is empty. Provide exact table names, or call listTables first."

        found: list[str] = []
        missing: list[str] = []
        for raw in table_names:
            name = str(raw or "").strip()
            if name in self._tables:
                found.append(name)
            elif name:
                missing.append(name)
        if not found:
            return (
                "No matching tables found: "
                + ", ".join(missing)
                + ". Call listTables and use exact table names."
            )

        text = self._format_tables(found)
        if missing:
            text += (
                "\n\nMissing tables skipped: "
                + ", ".join(missing)
                + ". Call listTables to verify exact names."
            )
        return text.strip()

    def describe_all_tables(self) -> str:
        self._ensure_schema()
        assert self._tables is not None
        if not self._tables:
            return "Schema is empty. The SQLite file may contain no tables or the path may be wrong."
        return self._format_tables(list(self._tables))

    def column_types(self) -> dict[str, dict[str, str]]:
        self._ensure_schema()
        assert self._tables is not None
        return {
            table: {column.name: column.col_type for column in columns}
            for table, columns in self._tables.items()
        }

    def foreign_keys(self) -> tuple[ForeignKeyInfo, ...]:
        self._ensure_schema()
        assert self._foreign_keys is not None
        return self._foreign_keys

    def _format_tables(self, table_names: list[str]) -> str:
        assert self._tables is not None
        lines = [f"Database: {self.db_id}", ""]
        for table in table_names:
            lines.append(f"Table {_quote_tick(table)}")
            for column in self._tables[table]:
                tags: list[str] = []
                if column.primary_key:
                    tags.append("PK")
                elif column.not_null:
                    tags.append("NOT NULL")
                prefix = f"[{' '.join(tags)}] " if tags else ""
                fk_suffix = self._fk_suffix(table, column.name)
                line = f"  {prefix}{_quote_tick(column.name)} {_simple_type(column.col_type)}"
                if fk_suffix:
                    line += f" {fk_suffix}"
                comment = column.description.formatted_comment(column.name)
                if comment:
                    lines.append(line)
                    lines.extend(f"      {part}" for part in comment.splitlines() if part.strip())
                    if column.examples:
                        lines[-1] += "  Examples: [" + ", ".join(column.examples) + "]"
                else:
                    if column.examples:
                        line += "  Examples: [" + ", ".join(column.examples) + "]"
                    lines.append(line)
            lines.append("")

        scope = set(table_names)
        fk_lines = self._format_foreign_keys(scope)
        if fk_lines:
            lines.append("Foreign keys (table.column → ref_table.ref_column):")
            lines.extend(f"  {line}" for line in fk_lines)
        return "\n".join(lines).strip()

    def _fk_suffix(self, table: str, column: str) -> str:
        assert self._foreign_keys is not None
        targets = [
            f"→ {fk.ref_table}.{fk.ref_column}"
            for fk in self._foreign_keys
            if fk.table == table and fk.column == column
        ]
        return "; ".join(targets)

    def _format_foreign_keys(self, scope: set[str]) -> list[str]:
        assert self._foreign_keys is not None
        selected = [
            fk
            for fk in self._foreign_keys
            if fk.table in scope and fk.ref_table in scope
        ]
        selected.sort(
            key=lambda fk: (
                fk.table.lower(),
                fk.column.lower(),
                fk.ref_table.lower(),
                fk.ref_column.lower(),
            )
        )
        return [
            f"{_quote_tick(fk.table)}.{_quote_tick(fk.column)} → "
            f"{_quote_tick(fk.ref_table)}.{_quote_tick(fk.ref_column)}"
            for fk in selected
        ]

    def _ensure_schema(self) -> None:
        if self._tables is not None:
            return
        uri = self.path.as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.execute("PRAGMA query_only=ON")
        try:
            table_names = [
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            tables: dict[str, tuple[ColumnInfo, ...]] = {}
            foreign_keys: list[ForeignKeyInfo] = []
            for table in table_names:
                quoted = _quote_ident(table)
                columns: list[ColumnInfo] = []
                for _, name, col_type, not_null, _, pk in connection.execute(
                    f"PRAGMA table_info({quoted})"
                ):
                    description = self._descriptions.get(table, {}).get(
                        name, ColumnDescription()
                    )
                    examples = tuple(
                        _sample_text_values(connection, table, name, col_type or "")
                    )
                    columns.append(
                        ColumnInfo(
                            name=str(name),
                            col_type=str(col_type or ""),
                            not_null=bool(not_null),
                            primary_key=bool(pk),
                            description=description,
                            examples=examples,
                        )
                    )
                tables[table] = tuple(columns)

                for fk in connection.execute(f"PRAGMA foreign_key_list({quoted})"):
                    ref_table = fk[2]
                    from_col = fk[3]
                    to_col = fk[4] or _primary_key_column(connection, ref_table)
                    if ref_table and from_col and to_col:
                        foreign_keys.append(
                            ForeignKeyInfo(table, from_col, ref_table, to_col)
                        )
            self._tables = tables
            self._foreign_keys = tuple(foreign_keys)
        finally:
            connection.close()


def load_database_descriptions(
    sqlite_path: Path | str,
) -> dict[str, dict[str, ColumnDescription]]:
    path = Path(sqlite_path)
    directory = path.parent / "database_description"
    if not directory.is_dir():
        return {}

    result: dict[str, dict[str, ColumnDescription]] = {}
    for file in sorted(directory.glob("*.csv")):
        try:
            raw = file.read_bytes()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
            rows = list(csv.reader(io.StringIO(text)))
        except (OSError, csv.Error):
            continue
        if len(rows) <= 1:
            continue

        columns: dict[str, ColumnDescription] = {}
        for row in rows[1:]:
            original = _cell(row, 0)
            if not original:
                continue
            value_description = "\n".join(
                value for value in (_cell(row, 4), _cell(row, 5)) if value
            )
            columns[original] = ColumnDescription(
                alias=_cell(row, 1),
                description=_cell(row, 2),
                value_description=value_description,
            )
        if columns:
            result[file.stem] = columns
    return result


def _sample_text_values(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    col_type: str,
) -> list[str]:
    if not _is_text_type(col_type):
        return []
    sql = (
        f"SELECT DISTINCT {_quote_ident(column)} FROM {_quote_ident(table)} "
        f"WHERE {_quote_ident(column)} IS NOT NULL LIMIT {MAX_EXAMPLE_VALUES}"
    )
    try:
        values: list[str] = []
        for (value,) in connection.execute(sql):
            if value is None:
                continue
            text = str(value)
            if text:
                if len(text) > MAX_EXAMPLE_LENGTH:
                    text = text[: MAX_EXAMPLE_LENGTH - 3] + "..."
                values.append(text)
        return values
    except sqlite3.Error:
        return []


def _primary_key_column(connection: sqlite3.Connection, table: str) -> str | None:
    for row in connection.execute(f"PRAGMA table_info({_quote_ident(table)})"):
        if row[5]:
            return str(row[1])
    return None


def _is_text_type(col_type: str | None) -> bool:
    text = (col_type or "").upper()
    return not text or any(token in text for token in ("CHAR", "TEXT", "CLOB"))


def _simple_type(col_type: str | None) -> str:
    text = (col_type or "").strip().lower()
    if not text:
        return "unknown"
    if "int" in text:
        return "integer"
    if any(token in text for token in ("real", "double", "float", "num")):
        return "real"
    if any(token in text for token in ("char", "text", "clob")):
        return "text"
    if "blob" in text:
        return "blob"
    return text


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _quote_tick(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def _cell(row: list[str], index: int) -> str:
    return row[index].strip() if index < len(row) and row[index] is not None else ""


def _normalize(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())
