from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.request
from typing import Any



DEFAULT_OPTION = {
    "title": {"text": "DataAna MCP E2E"},
    "tooltip": {},
    "xAxis": {"type": "category", "data": ["A", "B", "C"]},
    "yAxis": {"type": "value"},
    "series": [{"name": "value", "type": "bar", "data": [1, 3, 2]}],
}


def _schema(tool) -> dict[str, Any]:
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None:
        fn = getattr(args_schema, "model_json_schema", None)
        if callable(fn):
            schema = fn()
            if schema.get("properties"):
                return schema

    args = getattr(tool, "args", None)
    if isinstance(args, dict) and args:
        required = [
            name
            for name, spec in args.items()
            if isinstance(spec, dict) and "default" not in spec
        ]
        return {
            "type": "object",
            "properties": args,
            "required": required,
        }
    return {}


def _value_for_schema(name: str, schema: dict[str, Any]) -> Any:
    lowered = name.lower()
    schema_type = schema.get("type")
    if lowered in {"option", "options", "config", "echartsoption", "echarts_option"}:
        return json.dumps(DEFAULT_OPTION, ensure_ascii=False) if schema_type == "string" else DEFAULT_OPTION
    if lowered in {"outputtype", "output_type", "format"}:
        enum = schema.get("enum") or []
        for preferred in ("png", "url"):
            if preferred in enum:
                return preferred
        return enum[0] if enum else "png"
    if lowered in {"width", "height"}:
        return 640 if lowered == "width" else 360
    if "title" in lowered:
        return "DataAna MCP E2E"
    if lowered in {"data", "dataset"}:
        if schema_type == "string":
            return json.dumps([{"name": "A", "value": 1}, {"name": "B", "value": 3}], ensure_ascii=False)
        return [{"name": "A", "value": 1}, {"name": "B", "value": 3}]
    if schema_type == "array":
        return []
    if schema_type == "integer":
        return 1
    if schema_type == "number":
        return 1.0
    if schema_type == "boolean":
        return False
    if schema_type == "object":
        return {}
    return "DataAna MCP E2E"


def build_default_args(tool) -> dict[str, Any]:
    schema = _schema(tool)
    properties = schema.get("properties") or {}
    required = schema.get("required") or []
    args: dict[str, Any] = {}

    option_key = next(
        (
            name
            for name in properties
            if name.lower() in {"option", "options", "config", "echartsoption", "echarts_option"}
        ),
        None,
    )
    if option_key:
        args[option_key] = _value_for_schema(option_key, properties[option_key])

    for name in required:
        if name not in args:
            args[name] = _value_for_schema(name, properties.get(name, {}))

    for name in ("outputType", "output_type", "format", "width", "height"):
        if name in properties and name not in args:
            args[name] = _value_for_schema(name, properties[name])
    return args


def choose_tool(tools, explicit_name: str | None = None):
    by_name = {tool.name: tool for tool in tools}
    if explicit_name:
        if explicit_name not in by_name:
            raise RuntimeError(
                f"MCP tool {explicit_name!r} not found; available={sorted(by_name)}"
            )
        return by_name[explicit_name]

    preferred = ("generate_echarts", "generate_chart", "echarts")
    for name in preferred:
        if name in by_name:
            return by_name[name]
    for tool in tools:
        normalized = tool.name.lower()
        if "echart" in normalized or "chart" in normalized:
            return tool
    raise RuntimeError(f"No chart-like MCP tool found; available={sorted(by_name)}")


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def extract_http_url(value: Any) -> str | None:
    text = _text(value)
    match = re.search(r"https?://[^\s<>()\"']+", text)
    if not match:
        return None
    return match.group(0).rstrip("),.;")


def verify_url(url: str, timeout: float = 20.0) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read(32)
        content_type = response.headers.get("Content-Type", "")
        status = int(response.status)
    if status != 200:
        raise RuntimeError(f"chart URL returned HTTP {status}: {url}")
    if not body:
        raise RuntimeError(f"chart URL returned an empty body: {url}")
    return {"status": status, "contentType": content_type, "prefixBytes": len(body)}


async def run_check() -> dict[str, Any]:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    url = os.getenv("MCP_E2E_URL", "http://127.0.0.1:3033/mcp")
    explicit_tool = os.getenv("MCP_E2E_TOOL_NAME") or None
    raw_args = os.getenv("MCP_E2E_ARGS_JSON")

    client = MultiServerMCPClient(
        {"charts": {"transport": "http", "url": url}},
        handle_tool_errors=False,
    )
    tools = await client.get_tools()
    tool = choose_tool(tools, explicit_tool)

    if raw_args:
        args = json.loads(raw_args)
        if not isinstance(args, dict):
            raise RuntimeError("MCP_E2E_ARGS_JSON must decode to a JSON object")
    else:
        args = build_default_args(tool)

    result = await tool.ainvoke(args)
    chart_url = extract_http_url(result)
    if not chart_url:
        raise RuntimeError(
            "MCP tool did not return an HTTP URL. "
            "MinIO may be disabled/unreachable or the selected tool/args may be unsuitable. "
            f"tool={tool.name} args={args} result={_text(result)[:800]}"
        )

    url_check = await asyncio.to_thread(verify_url, chart_url)
    return {
        "mcpUrl": url,
        "toolName": tool.name,
        "availableTools": [item.name for item in tools],
        "args": args,
        "chartUrl": chart_url,
        "urlCheck": url_check,
    }


def main() -> None:
    result = asyncio.run(run_check())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
