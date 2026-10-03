from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from langchain_core.tools import StructuredTool


SKILL_FILE = "SKILL.md"
FRONTMATTER_DELIMITER = "---"


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    path: Path

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description}


def parse_frontmatter(text: str) -> dict[str, Any]:
    """Parse the leading YAML frontmatter block (``--- ... ---``).

    Returns an empty dict when the block is missing or malformed, so a broken
    skill can never take down skill discovery.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_DELIMITER:
        return {}
    end = next(
        (index for index in range(1, len(lines)) if lines[index].strip() == FRONTMATTER_DELIMITER),
        None,
    )
    if end is None:
        return {}
    try:
        data = yaml.safe_load("\n".join(lines[1:end])) or {}
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def strip_frontmatter(text: str) -> str:
    """Return the SKILL.md body without the leading frontmatter block."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_DELIMITER:
        return text
    end = next(
        (index for index in range(1, len(lines)) if lines[index].strip() == FRONTMATTER_DELIMITER),
        None,
    )
    if end is None:
        return text
    return "\n".join(lines[end + 1 :]).lstrip("\n")


class SkillRegistry:
    """Discover SKILL.md skills and expose them to the agent on demand.

    Mirrors the Java ``SkillManager`` + ``SkillsTool`` split: the registry owns
    discovery/content loading, while the agent gets one always-visible ``skill``
    tool implementing progressive disclosure (list catalog -> load full SKILL.md).
    """

    TOOL_NAME = "skill"

    def __init__(self, skills_dir: Path | str | None):
        self.skills_dir = Path(skills_dir).expanduser() if skills_dir else None
        self._skills: dict[str, Skill] = {}
        self.reload()
        self._tool = StructuredTool.from_function(
            self._skill_tool,
            name=self.TOOL_NAME,
            description=(
                "加载专业 Skill 流程。无 name 时返回可用 Skill 清单（name + description）；"
                "传入 name 时返回该 Skill 的完整 SKILL.md 内容。"
            ),
        )

    def reload(self) -> None:
        skills: dict[str, Skill] = {}
        directory = self.skills_dir
        if directory and directory.is_dir():
            for entry in sorted(directory.iterdir()):
                if not entry.is_dir():
                    continue
                skill_file = entry / SKILL_FILE
                if not skill_file.is_file():
                    continue
                try:
                    text = skill_file.read_text(encoding="utf-8")
                except OSError:
                    continue
                meta = parse_frontmatter(text)
                name = str(meta.get("name") or entry.name).strip()
                if not name:
                    continue
                skills[name] = Skill(
                    name=name,
                    description=str(meta.get("description") or "").strip(),
                    path=skill_file,
                )
        self._skills = skills

    def list(self) -> list[Skill]:
        return list(self._skills.values())

    def get(self, name: str) -> Skill | None:
        return self._skills.get((name or "").strip())

    def has_skills(self) -> bool:
        return bool(self._skills)

    def catalog_text(self) -> str:
        """Name + description lines for the system prompt, or ``""`` when empty."""
        if not self._skills:
            return ""
        return "\n".join(f"- {skill.name}: {skill.description}" for skill in self._skills.values())

    @property
    def tool(self) -> StructuredTool:
        return self._tool

    def _skill_tool(self, name: str = "") -> str:
        """无 name 返回 Skill 清单；带 name 返回该 Skill 的完整 SKILL.md 流程。"""
        if not self._skills:
            return "当前没有可用的 Skill，请直接使用通用工具完成任务。"
        requested = (name or "").strip()
        if not requested:
            return json.dumps(
                {"skills": [skill.as_dict() for skill in self._skills.values()]},
                ensure_ascii=False,
            )
        skill = self._skills.get(requested)
        if skill is None:
            available = ", ".join(self._skills)
            return json.dumps(
                {"error": f"未找到 Skill「{requested}」。可用 Skill：{available}"},
                ensure_ascii=False,
            )
        try:
            body = skill.path.read_text(encoding="utf-8")
        except OSError as exc:
            return json.dumps({"error": f"读取 Skill 失败：{exc}"}, ensure_ascii=False)
        return f"# Skill: {skill.name}\n\n{strip_frontmatter(body).strip()}"
