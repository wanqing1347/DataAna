from __future__ import annotations

import asyncio
import json
import os


OPTION = {
    "title": {"text": "DataAna MCP Smoke"},
    "xAxis": {"type": "category", "data": ["A", "B", "C"]},
    "yAxis": {"type": "value"},
    "series": [{"type": "bar", "data": [1, 3, 2]}],
}


async def main() -> None:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "charts": {
                "transport": "stdio",
                "command": "npx.cmd",
                "args": ["-y", "mcp-echarts@0.7.1", "-t", "stdio"],
                "env": {
                    key: os.environ[key]
                    for key in (
                        "MINIO_ENDPOINT",
                        "MINIO_PORT",
                        "MINIO_USE_SSL",
                        "MINIO_ACCESS_KEY",
                        "MINIO_SECRET_KEY",
                        "MINIO_BUCKET_NAME",
                        "DEBUG_MCP_ECHARTS",
                    )
                    if key in os.environ
                },
            }
        },
        handle_tool_errors=False,
    )
    tools = await client.get_tools()
    tool = next(item for item in tools if item.name == "generate_echarts")

    args = {
        "echartsOption": json.dumps(OPTION, ensure_ascii=False),
        "width": 640,
        "height": 360,
        "theme": "default",
        "outputType": "png",
    }
    print("CALL_TOOL=" + tool.name)
    print("CALL_ARGS=" + repr(args))

    result = await tool.ainvoke(args)
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)

    print("RESULT_TYPE=" + type(result).__name__)
    print("RESULT_PREFIX=" + text[:1200])
    print("RESULT_LENGTH=" + str(len(text)))

    lowered = text.lower()
    if "http://" in lowered or "https://" in lowered:
        print("MCP_RENDER_HTTP_URL_OK")
    elif "base64" in lowered or "data:image" in lowered:
        print("MCP_RENDER_BASE64_OK")
    else:
        print("MCP_RENDER_RETURNED_NON_URL_OUTPUT")


if __name__ == "__main__":
    asyncio.run(main())
