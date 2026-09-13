from pathlib import Path

import yaml


def test_python_compose_enables_mcp_and_uses_python_app_image():
    root = Path(__file__).resolve().parents[2]
    compose = yaml.safe_load(
        (root / "deploy/docker-compose.yml").read_text(encoding="utf-8")
    )

    app = compose["services"]["app"]
    assert app["build"]["dockerfile"] == "deploy/python_backend.Dockerfile"
    assert app["environment"]["CHART_MCP_ENABLED"] == "true"
    assert app["environment"]["CHART_MCP_URL"] == "http://mcp-echarts:3033/mcp"
    assert app["environment"]["BIRD_EVAL_ENABLED"] == "false"

    assert "mcp-echarts" in compose["services"]
    assert "minio" in compose["services"]
    assert "minio-init" in compose["services"]
    assert "minio-public-proxy" in compose["services"]

    minio_init = compose["services"]["minio-init"]
    assert "mc anonymous set download" in minio_init["command"]
    assert (
        compose["services"]["mcp-echarts"]["depends_on"]["minio-init"]["condition"]
        == "service_completed_successfully"
    )


def test_python_dockerfile_runs_fastapi_uvicorn():
    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "deploy/python_backend.Dockerfile").read_text(encoding="utf-8")

    assert "python:3.13-slim" in dockerfile
    assert "uv sync --frozen" in dockerfile
    assert "uvicorn" in dockerfile
    assert "app.main:app" in dockerfile
    assert "COPY frontend /opt/dataana/frontend" in dockerfile
    assert "COPY schema /opt/dataana/schema" in dockerfile
