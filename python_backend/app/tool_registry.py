from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from langchain_core.tools import BaseTool, StructuredTool


DEFAULT_SEARCH_TERMS: dict[str, tuple[str, ...]] = {
    "listTables": ("schema", "tables", "table discovery", "表", "数据表", "有哪些表"),
    "describeTables": ("schema", "columns", "fields", "table structure", "字段", "列", "表结构"),
    "lookupGlossary": ("glossary", "metric", "terminology", "business definition", "指标", "术语", "口径"),
    "queryData": ("query", "sql", "data analysis", "text to sql", "查询", "数据", "分析", "取数"),
    "validateSql": ("validate sql", "sql guard", "safety", "校验", "安全", "检查sql"),
    "executeSql": ("execute sql", "raw sql", "debug sql", "执行sql", "底层sql"),
    "calculate": ("math", "calculate", "arithmetic", "ratio", "计算", "比例", "环比", "贡献度"),
}


@dataclass(frozen=True, slots=True)
class ToolSpec:
    tool: BaseTool
    source: str = "local"
    search_terms: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.tool.name

    @property
    def description(self) -> str:
        return self.tool.description or ""

    @property
    def search_text(self) -> str:
        extras = " ".join(self.search_terms)
        return f"{self.name} {self.description} {self.source} {extras}".lower()


@dataclass(frozen=True, slots=True)
class ToolContextMetrics:
    tool_count: int
    schema_chars: int
    approx_tokens: int

    def as_dict(self) -> dict[str, int]:
        return {
            "toolCount": self.tool_count,
            "schemaChars": self.schema_chars,
            "approxTokens": self.approx_tokens,
        }


class ToolRegistry:
    """
    Registry for deferred Agent tools.

    The model sees only tool_search on the first turn. Deferred tool schemas
    are bound only after a successful search result names them. The registry is
    also the execution allow-list, so a guessed tool name cannot bypass
    discovery.
    """

    SEARCH_TOOL_NAME = "tool_search"

    def __init__(self, specs: Iterable[ToolSpec]):
        items = list(specs)
        duplicate_names = _duplicates(spec.name for spec in items)
        if duplicate_names:
            raise ValueError(f"duplicate tool names: {sorted(duplicate_names)}")
        self._specs = {spec.name: spec for spec in items}
        self._search_tool = StructuredTool.from_function(
            self._tool_search,
            name=self.SEARCH_TOOL_NAME,
            description=(
                "按任务意图搜索并加载延迟工具。首次需要数据库探索、SQL 查询、计算或图表时先调用本工具；"
                "返回的 loaded_tools 会在下一轮加入模型可见工具 schema。"
            ),
        )

    @classmethod
    def from_tools(
        cls,
        local_tools: Iterable[BaseTool],
        *,
        mcp_tools: Iterable[BaseTool] = (),
    ) -> "ToolRegistry":
        specs: list[ToolSpec] = []
        for tool in local_tools:
            specs.append(
                ToolSpec(
                    tool=tool,
                    source="local",
                    search_terms=DEFAULT_SEARCH_TERMS.get(tool.name, ()),
                )
            )
        for tool in mcp_tools:
            specs.append(
                ToolSpec(
                    tool=tool,
                    source="mcp",
                    search_terms=(
                        "mcp",
                        "echarts",
                        "chart",
                        "visualization",
                        "plot",
                        "图表",
                        "可视化",
                        "画图",
                    ),
                )
            )
        return cls(specs)

    @property
    def deferred_tools(self) -> list[BaseTool]:
        return [spec.tool for spec in self._specs.values()]

    @property
    def search_tool(self) -> BaseTool:
        return self._search_tool

    def initial_tools(self) -> list[BaseTool]:
        return [self._search_tool]

    def bindable_tools(self, loaded_names: Iterable[str] = ()) -> list[BaseTool]:
        tools: list[BaseTool] = [self._search_tool]
        seen = {self.SEARCH_TOOL_NAME}
        for name in loaded_names:
            if name in seen:
                continue
            spec = self._specs.get(name)
            if spec is None:
                continue
            tools.append(spec.tool)
            seen.add(name)
        return tools

    def search(self, query: str, max_results: int = 4) -> list[dict[str, Any]]:
        query = (query or "").strip().lower()
        if not query:
            return []
        max_results = min(max(1, int(max_results)), 8)
        query_terms = _search_units(query)

        ranked: list[tuple[int, ToolSpec]] = []
        for spec in self._specs.values():
            score = _score(query, query_terms, spec)
            if score > 0:
                ranked.append((score, spec))
        ranked.sort(key=lambda item: (-item[0], item[1].name))

        return [
            {
                "name": spec.name,
                "description": spec.description,
                "source": spec.source,
                "score": score,
            }
            for score, spec in ranked[:max_results]
        ]

    def _tool_search(self, query: str, max_results: int = 4) -> str:
        """搜索与当前任务最相关的延迟工具；query 应描述所需能力而不是猜工具参数。"""
        matches = self.search(query, max_results=max_results)
        return json.dumps(
            {
                "query": query,
                "loaded_tools": [item["name"] for item in matches],
                "matches": matches,
            },
            ensure_ascii=False,
        )

    def context_metrics(self, loaded_names: Iterable[str] = ()) -> ToolContextMetrics:
        return tool_context_metrics(self.bindable_tools(loaded_names))

    def eager_context_metrics(self) -> ToolContextMetrics:
        return tool_context_metrics([self._search_tool, *self.deferred_tools])


def extract_loaded_tool_names(messages: Iterable[Any]) -> list[str]:
    loaded: list[str] = []
    seen: set[str] = set()
    for message in messages:
        if getattr(message, "name", None) != ToolRegistry.SEARCH_TOOL_NAME:
            continue
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        names = payload.get("loaded_tools") if isinstance(payload, dict) else None
        if not isinstance(names, list):
            continue
        for name in names:
            if isinstance(name, str) and name not in seen:
                loaded.append(name)
                seen.add(name)
    return loaded


def tool_context_metrics(tools: Iterable[BaseTool]) -> ToolContextMetrics:
    payload = [_tool_schema_payload(tool) for tool in tools]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    chars = len(encoded)
    approx_tokens = (chars + 3) // 4
    return ToolContextMetrics(len(payload), chars, approx_tokens)


def _tool_schema_payload(tool: BaseTool) -> dict[str, Any]:
    schema: dict[str, Any] = {}
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None:
        model_json_schema = getattr(args_schema, "model_json_schema", None)
        if callable(model_json_schema):
            schema = model_json_schema()
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": schema,
    }


def _score(query: str, query_terms: set[str], spec: ToolSpec) -> int:
    text = spec.search_text
    score = 0
    if query in text:
        score += 12
    name = spec.name.lower()
    if name in query or query in name:
        score += 16

    text_terms = _search_units(text)
    for term in query_terms:
        if term in text_terms:
            score += 4
        elif len(term) >= 2 and term in text:
            score += 2

    for alias in spec.search_terms:
        alias_lower = alias.lower()
        if alias_lower and alias_lower in query:
            score += 8
    return score


def _search_units(text: str) -> set[str]:
    ascii_terms = set(re.findall(r"[a-z0-9_]+", text.lower()))
    cjk_chunks = re.findall(r"[\u4e00-\u9fff]+", text)
    cjk_units: set[str] = set()
    for chunk in cjk_chunks:
        if len(chunk) == 1:
            cjk_units.add(chunk)
            continue
        for size in (2, 3, 4):
            if len(chunk) < size:
                continue
            cjk_units.update(chunk[i : i + size] for i in range(len(chunk) - size + 1))
    return ascii_terms | cjk_units


def _duplicates(names: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    return duplicates
