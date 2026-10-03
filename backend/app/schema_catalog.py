from __future__ import annotations

from typing import Any

import yaml

MAX_EXAMPLES = 5
MAX_EXAMPLE_LENGTH = 30


class SchemaCatalog:
    def __init__(self, path, allowed_tables: list[str]):
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        self.tables: dict[str, Any] = raw.get("tables", {})
        self.glossary: list[dict[str, Any]] = raw.get("glossary", []) or []
        self.allowed = {t.lower() for t in allowed_tables}
        self._term_index: dict[str, dict[str, Any]] = {}
        self._synonym_index: dict[str, dict[str, Any]] = {}
        for item in self.glossary:
            term = str(item.get("term") or "").strip()
            if term:
                self._term_index[term.lower()] = item
            for synonym in item.get("synonyms") or []:
                key = str(synonym).strip().lower()
                if key:
                    self._synonym_index.setdefault(key, item)

    def list_tables(self) -> str:
        lines = []
        for name, meta in self.tables.items():
            if name.lower() not in self.allowed:
                continue
            related = []
            for fk in meta.get("foreignKeys", []) or []:
                ref = fk.get("refTable")
                if ref:
                    related.append(ref)
            suffix = f" | 关联: {', '.join(sorted(set(related)))}" if related else ""
            lines.append(f"- {name}: {meta.get('description', '')}{suffix}")
        return "\n".join(lines) if lines else "没有配置可用的数据分析表。"

    def describe_tables(self, names: list[str]) -> str:
        blocks: list[str] = []
        for name in names:
            key = name.strip()
            if key.lower() not in self.allowed:
                blocks.append(f"## {key}\n不在数据分析白名单中。")
                continue
            meta = self.tables.get(key)
            if not meta:
                blocks.append(f"## {key}\nSchema 字典中不存在该表。")
                continue
            lines = [f"## {key}", meta.get("description", ""), "", "字段："]
            for col, info in (meta.get("columns") or {}).items():
                line = (
                    f"- {col} | {info.get('type', '')} | "
                    f"{info.get('comment', '')} | key={info.get('key', '') or '-'} | "
                    f"nullable={info.get('nullable', True)}"
                )
                examples = _format_examples(info.get("examples"))
                if examples:
                    line += f" | Examples: [{examples}]"
                lines.append(line)
            fks = meta.get("foreignKeys") or []
            if fks:
                lines.append("外键：")
                for fk in fks:
                    lines.append(
                        f"- {fk.get('column')} -> {fk.get('refTable')}.{fk.get('refColumn')}"
                    )
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def glossary_catalog(self) -> str:
        """参照实现的 LookupGlossaryTool 风格：把术语清单（含同义词）渲染成紧凑文本。"""
        lines: list[str] = []
        for item in self.glossary:
            term = str(item.get("term") or "")
            synonyms = [str(x) for x in (item.get("synonyms") or []) if str(x).strip()]
            line = f"- {term}"
            if synonyms:
                line += f"（同义词：{'、'.join(synonyms)}）"
            lines.append(line)
        return "\n".join(lines)

    def lookup_glossary(self, term: str) -> str:
        """精确匹配术语名或同义词（口径必须一术语一标准答案，模糊检索会出错）。"""
        query = (term or "").strip()
        if not query:
            return (
                "入参为空，请必须传入具体术语名（如 \"活跃客户\"），"
                "参考工具说明中的可用术语列表，再次尝试调用本工具。"
            )
        key = query.lower()
        hit = self._term_index.get(key) or self._synonym_index.get(key)
        if hit is not None:
            return _format_glossary(hit)
        if not self.glossary:
            return (
                f"未找到术语「{query}」的标准口径。当前 glossary 为空——"
                "可能是 schema 字典未正确加载，请检查 DataAna.yml 的 glossary section。"
            )
        return "\n".join(
            [
                f"未找到术语「{query}」的标准口径。",
                "",
                "当前已登记的术语（用准确名或同义词重试）：",
                self.glossary_catalog(),
            ]
        ).strip()


def _format_glossary(item: dict[str, Any]) -> str:
    parts = [f"【术语】{item.get('term')}"]
    description = str(item.get("description") or "").strip()
    if description:
        parts.append(f"【标准口径】{description}")
    sql_fragment = str(item.get("sqlFragment") or "").strip()
    if sql_fragment:
        parts.append(f"【SQL 片段（可直接复用）】\n{sql_fragment}")
    synonyms = [str(x) for x in (item.get("synonyms") or []) if str(x).strip()]
    if synonyms:
        parts.append(f"【同义词】{'、'.join(synonyms)}")
    return "\n\n".join(parts)


def _format_examples(values: Any) -> str:
    if not values:
        return ""
    if isinstance(values, str):
        values = [values]
    rendered: list[str] = []
    for value in list(values)[:MAX_EXAMPLES]:
        text = str(value)
        if len(text) > MAX_EXAMPLE_LENGTH:
            text = text[: MAX_EXAMPLE_LENGTH - 3] + "..."
        rendered.append(text)
    return ", ".join(rendered)
