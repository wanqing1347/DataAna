from __future__ import annotations

import asyncio


async def main() -> None:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "charts": {
                "transport": "stdio",
                "command": "npx.cmd",
                "args": ["-y", "mcp-echarts@0.7.1", "-t", "stdio"],
            }
        },
        handle_tool_errors=False,
    )

    tools = await client.get_tools()
    names = [tool.name for tool in tools]
    print("MCP_TOOL_NAMES=" + repr(names))
    assert tools, "mcp-echarts returned no tools"
    assert any("chart" in name.lower() or "echart" in name.lower() for name in names), names

    for tool in tools:
        schema_fn = getattr(getattr(tool, "args_schema", None), "model_json_schema", None)
        schema = schema_fn() if callable(schema_fn) else {}
        print(f"MCP_TOOL={tool.name}")
        print(f"MCP_DESCRIPTION={tool.description}")
        print(f"MCP_ARGS={getattr(tool, 'args', None)}")
        print(f"MCP_SCHEMA={schema}")

    print("MCP_STDIO_DISCOVERY_OK")


if __name__ == "__main__":
    asyncio.run(main())
