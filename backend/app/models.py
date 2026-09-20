from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DataScope(str, Enum):
    SELF = "SELF"
    DEPT = "DEPT"
    DEPT_AND_SUB = "DEPT_AND_SUB"
    ALL = "ALL"

    @classmethod
    def max_scope(cls, values: list[str]) -> "DataScope":
        rank = {
            cls.SELF: 0,
            cls.DEPT: 1,
            cls.DEPT_AND_SUB: 2,
            cls.ALL: 3,
        }
        parsed = [cls(v) if v in cls._value2member_map_ else cls.SELF for v in values]
        return max(parsed or [cls.SELF], key=rank.get)


@dataclass(slots=True)
class Role:
    id: int
    code: str
    name: str
    data_scope: str


@dataclass(slots=True)
class Department:
    id: int
    name: str
    parent_id: int | None = None
    ancestors: str = ""


@dataclass(slots=True)
class User:
    id: int
    username: str
    nickname: str | None
    status: str
    roles: list[Role] = field(default_factory=list)
    departments: list[Department] = field(default_factory=list)
    data_scope: DataScope = DataScope.SELF
    profile: dict[str, Any] | None = None


@dataclass(slots=True)
class DataScopeContext:
    user_id: int
    scope: DataScope
    dept_ids: list[int]
