from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def main() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        print("HEALTH_STATUS=" + str(health.status_code))
        print("HEALTH_JSON=" + repr(health.json()))
        assert health.status_code == 200
        payload = health.json()
        assert payload.get("status") == "ok"
        assert payload.get("agent") == "langgraph"
        assert payload.get("language") == "python"

        index = client.get("/")
        print("INDEX_STATUS=" + str(index.status_code))
        print("INDEX_CONTENT_TYPE=" + str(index.headers.get("content-type")))
        assert index.status_code == 200
        assert "text/html" in str(index.headers.get("content-type", ""))

    print("FASTAPI_SERVICE_OK")


if __name__ == "__main__":
    main()
