from pathlib import Path


def test_frontend_has_mcp_chart_url_rendering_contract():
    root = Path(__file__).resolve().parents[2]
    chat_js = (root / "frontend/js/chat.js").read_text(encoding="utf-8")

    assert "const isChartToolName" in chat_js
    assert "generateecharts" in chat_js.lower()
    assert "const parseChartUrl" in chat_js
    assert "appendChart(aiMsg, parseChartUrl(resultText))" in chat_js
    assert "window.location.origin + parsed.pathname + parsed.search" in chat_js
