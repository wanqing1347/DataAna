import os

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MCP_E2E") != "1",
    reason="set RUN_MCP_E2E=1 after starting mcp-echarts + MinIO",
)


@pytest.mark.asyncio
async def test_python_mcp_echarts_minio_public_url_e2e():
    from scripts.mcp_e2e_check import run_check

    result = await run_check()

    assert result["availableTools"]
    assert result["chartUrl"].startswith(("http://", "https://"))
    assert result["urlCheck"]["status"] == 200
