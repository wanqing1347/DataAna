from __future__ import annotations

from typing import Any

import yaml


class SchemaCatalog:
    def __init__(self, path, allowed_tables: list[str]):
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        self.tables: dict[str, Any] = raw.get("tables", {})
        self.glossary: list[dict[str, Any]] = raw.get("glossary", []) or []
        self.allowed = {t.lower() for t in allowed_tables}

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
                lines.append(
                    f"- {col} | {info.get('type', '')} | "
                    f"{info.get('comment', '')} | key={info.get('key', '') or '-'} | "
                    f"nullable={info.get('nullable', True)}"
                )
            fks = meta.get("foreignKeys") or []
            if fks:
                lines.append("外键：")
                for fk in fks:
                    lines.append(
                        f"- {fk.get('column')} -> {fk.get('refTable')}.{fk.get('refColumn')}"
                    )
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def lookup_glossary(self, keyword: str) -> str:
        k = keyword.strip().lower()
        hits: list[str] = []
        for item in self.glossary:
            term = str(item.get("term") or "")
            description = str(item.get("description") or "")
            sql_fragment = str(item.get("sqlFragment") or "")
            synonyms = [str(x) for x in (item.get("synonyms") or [])]
            haystack = " ".join([term, description, sql_fragment, *synonyms]).lower()
            if k in haystack or any(s.lower() in k for s in [term, *synonyms] if s):
                block = [f"- {term}: {description}"]
                if sql_fragment:
                    block.append(f"  SQL口径: {sql_fragment}")
                if synonyms:
                    block.append(f"  同义词: {', '.join(synonyms)}")
                hits.append("\n".join(block))
        return "\n".join(hits[:20]) if hits else f"未找到与“{keyword}”匹配的业务口径。"
