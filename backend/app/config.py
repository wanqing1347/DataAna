from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "mysql+pymysql://root:123456@127.0.0.1:3306/dataana?charset=utf8mb4"

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    deepseek_temperature: float = 0.2

    jwt_secret: str = "dataana-dev-secret-change-me"
    jwt_expire_hours: int = 24

    schema_path: Path = Path("../schema/DataAna.yml")
    skills_dir: Path = Path("../skills")
    # 与参照实现对齐：Sakila 全量 + 系统权限表，共 20 张（见 schema/DataAna.yml）。
    allowed_tables: Annotated[list[str], NoDecode] = [
        "actor",
        "category",
        "language",
        "country",
        "city",
        "address",
        "customer",
        "film",
        "inventory",
        "rental",
        "payment",
        "film_actor",
        "film_category",
        "film_text",
        "sys_user",
        "sys_dept",
        "sys_role",
        "sys_user_role",
        "sys_user_dept",
        "user_profile",
    ]
    max_rows: int = 200
    max_joins: int = 3

    # 与参照实现 ReactAgent.maxRounds(100) 对齐；TodoWrite 的创建/逐项更新也会占用轮次。
    agent_max_tool_rounds: int = 100
    # LangGraph 每个 tool round = model + tools 两个 super-step，需覆盖 100 轮。
    agent_recursion_limit: int = 210
    memory_max_turns: int = 6
    memory_max_chars: int = 12000

    checkpoint_path: Path = Path(".data/langgraph_checkpoints.sqlite3")
    tool_retry_max_attempts: int = 3
    tool_retry_base_delay_ms: int = 100

    explain_precheck_enabled: bool = True
    explain_precheck_timeout_seconds: int = 5
    explain_precheck_max_estimated_rows: int = 100000

    data_permission_enabled: bool = True
    sensitive_filter_enabled: bool = True
    sensitive_fields: Annotated[list[str], NoDecode] = [
        "sys_user.password",
        "user_profile.id_card",
        "user_profile.home_address",
    ]

    chart_mcp_enabled: bool = False
    chart_mcp_url: str = "http://localhost:3033/mcp"

    bird_eval_enabled: bool = False
    bird_eval_temperature: float = 0.0
    bird_eval_max_rounds: int = 50
    bird_eval_sql_timeout_seconds: float = 20.0
    # 0 keeps the complete verifySql result, matching the reference BirdVerifySqlTool.
    # Set a positive value only for constrained local experiments.
    bird_eval_tool_result_max_rows: int = 0
    # Keep exploratory verification bounded so long ReAct trajectories converge to a final check.
    # Set 0 only when explicitly reproducing the old unlimited-probe behavior.
    bird_eval_max_probe_calls: int = 8
    # Structural/execution success is not sufficient for termination. Build an immutable
    # task contract before ReAct, then require an independent semantic gate to pass too.
    bird_eval_semantic_gate_enabled: bool = True
    bird_eval_semantic_critic_enabled: bool = True
    # After a final candidate fails semantic verification, probes stay closed and the
    # agent gets only this many targeted final-SQL repair attempts.
    bird_eval_max_semantic_repairs: int = 3

    @field_validator("allowed_tables", "sensitive_fields", mode="before")
    @classmethod
    def split_csv(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    def resolved_schema_path(self) -> Path:
        if self.schema_path.is_absolute():
            return self.schema_path
        return (Path(__file__).resolve().parents[1] / self.schema_path).resolve()

    def resolved_skills_dir(self) -> Path:
        if self.skills_dir.is_absolute():
            return self.skills_dir
        return (Path(__file__).resolve().parents[1] / self.skills_dir).resolve()

    def resolved_checkpoint_path(self) -> Path:
        if self.checkpoint_path.is_absolute():
            return self.checkpoint_path
        return (Path(__file__).resolve().parents[1] / self.checkpoint_path).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
