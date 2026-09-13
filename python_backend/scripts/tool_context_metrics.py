from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.models import DataScope, DataScopeContext
from app.schema_catalog import SchemaCatalog
from app.security import DataScopeRewriter, SensitiveFilter, SqlSafetyGuard
from app.tool_registry import ToolRegistry
from app.tools import build_tools


def measure() -> dict:
    settings = Settings()
    catalog = SchemaCatalog(settings.resolved_schema_path(), settings.allowed_tables)
    local_tools = build_tools(
        MagicMock(),
        catalog,
        SqlSafetyGuard(settings.allowed_tables, settings.max_rows, settings.max_joins),
        DataScopeRewriter(),
        SensitiveFilter(settings.sensitive_filter_enabled, settings.sensitive_fields),
        DataScopeContext(user_id=1, scope=DataScope.ALL, dept_ids=[]),
        conversation_id="metrics",
        session_store=MagicMock(),
        model=MagicMock(),
        tool_retry_max_attempts=settings.tool_retry_max_attempts,
        tool_retry_base_delay_ms=settings.tool_retry_base_delay_ms,
        sql_planner_max_attempts=settings.sql_planner_max_attempts,
    )
    registry = ToolRegistry.from_tools(local_tools)
    initial = registry.context_metrics()
    eager = registry.eager_context_metrics()
    reduction = 0.0 if eager.schema_chars == 0 else 1 - initial.schema_chars / eager.schema_chars
    return {
        "deferredToolNames": [tool.name for tool in registry.deferred_tools],
        "initial": initial.as_dict(),
        "eager": eager.as_dict(),
        "schemaCharReductionPct": round(reduction * 100, 2),
        "note": "approxTokens uses transparent chars/4 proxy, not tokenizer-exact billing tokens",
    }


if __name__ == "__main__":
    print(json.dumps(measure(), ensure_ascii=False, indent=2))
