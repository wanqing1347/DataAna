from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolPathEvalCase:
    case_id: str
    question: str
    required_tools: tuple[str, ...]
    forbidden_tools: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ToolPathEvalCase":
        return cls(
            case_id=str(payload["id"]),
            question=str(payload["question"]),
            required_tools=tuple(payload.get("requiredTools") or ()),
            forbidden_tools=tuple(payload.get("forbiddenTools") or ()),
        )


def score_tool_path(case: ToolPathEvalCase, actual_tools: list[str]) -> dict[str, Any]:
    actual = set(actual_tools)
    missing = [name for name in case.required_tools if name not in actual]
    forbidden_hits = [name for name in case.forbidden_tools if name in actual]
    required_count = len(case.required_tools)
    coverage = 1.0 if required_count == 0 else (required_count - len(missing)) / required_count
    return {
        "caseId": case.case_id,
        "passed": not missing and not forbidden_hits,
        "coverage": round(coverage, 4),
        "missingRequiredTools": missing,
        "forbiddenToolHits": forbidden_hits,
        "actualTools": actual_tools,
    }


def score_trace(case: ToolPathEvalCase, trace: dict[str, Any]) -> dict[str, Any]:
    actual_tools = [
        str(item.get("name"))
        for item in (trace.get("tools") or [])
        if item.get("name")
    ]
    result = score_tool_path(case, actual_tools)
    result["runId"] = trace.get("runId")
    result["runStatus"] = trace.get("status")
    return result
