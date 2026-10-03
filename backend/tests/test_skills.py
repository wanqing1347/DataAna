import json
from pathlib import Path

from langchain_core.tools import StructuredTool

from app.prompts import build_system_prompt
from app.skills import SkillRegistry, parse_frontmatter, strip_frontmatter
from app.tool_registry import ToolRegistry


def write_skill(root: Path, name: str, description: str, body: str = "# Steps\n1. do it") -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return directory


def make_tool(name: str, description: str) -> StructuredTool:
    def echo(value: str = "") -> str:
        return value

    return StructuredTool.from_function(echo, name=name, description=description)


def test_parse_and_strip_frontmatter():
    text = "---\nname: demo\ndescription: 演示 skill——用于测试\n---\n\n# Body\n"
    assert parse_frontmatter(text)["name"] == "demo"
    assert parse_frontmatter(text)["description"] == "演示 skill——用于测试"
    assert strip_frontmatter(text).startswith("# Body")


def test_registry_discovers_only_dirs_with_skill_md(tmp_path):
    write_skill(tmp_path, "data-analysis", "数据分析任务")
    write_skill(tmp_path, "other", "其他任务")
    (tmp_path / "not-a-skill").mkdir()

    registry = SkillRegistry(tmp_path)
    assert [skill.name for skill in registry.list()] == ["data-analysis", "other"]
    assert registry.has_skills()


def test_empty_registry_is_graceful(tmp_path):
    registry = SkillRegistry(tmp_path)
    assert not registry.has_skills()
    assert registry.catalog_text() == ""
    assert registry.tool.invoke({"name": ""}).startswith("当前没有可用")


def test_skill_tool_lists_then_loads_body(tmp_path):
    write_skill(tmp_path, "data-analysis", "数据分析任务", "# Steps\n先探查 Schema")
    registry = SkillRegistry(tmp_path)

    catalog = json.loads(registry.tool.invoke({"name": ""}))
    assert catalog["skills"][0]["name"] == "data-analysis"

    loaded = registry.tool.invoke({"name": "data-analysis"})
    assert "先探查 Schema" in loaded
    assert "description:" not in loaded


def test_skill_tool_reports_unknown_skill(tmp_path):
    write_skill(tmp_path, "data-analysis", "数据分析任务")
    registry = SkillRegistry(tmp_path)
    payload = json.loads(registry.tool.invoke({"name": "missing"}))
    assert "error" in payload
    assert "data-analysis" in payload["error"]


def test_tool_registry_keeps_always_on_tool_visible_before_search():
    always = make_tool("skill", "加载 Skill 流程")
    deferred = make_tool("queryData", "根据问题生成并执行 SQL 查询数据")
    registry = ToolRegistry.from_tools([deferred], always_tools=[always])

    assert [tool.name for tool in registry.initial_tools()] == ["skill", "tool_search"]
    assert [tool.name for tool in registry.bindable_tools()] == ["skill", "tool_search"]
    assert "queryData" in [tool.name for tool in registry.bindable_tools(["queryData"])]


def test_build_system_prompt_injects_skill_catalog_and_principle():
    prompt = build_system_prompt("用户上下文", "- data-analysis: 数据分析任务")
    assert "## Skills 机制（优先）" in prompt
    assert "- data-analysis: 数据分析任务" in prompt
    assert prompt.rstrip().endswith("用户上下文")


def test_build_system_prompt_without_skills_omits_section():
    prompt = build_system_prompt("用户上下文")
    assert "## Skills 机制" not in prompt
    assert prompt.rstrip().endswith("用户上下文")
