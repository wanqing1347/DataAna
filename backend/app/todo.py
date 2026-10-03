from __future__ import annotations

import json
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field


# 与 Java agentx 常驻工具同名（com.agentx.ai.core.tools.TodoWriteTool）。
TODO_TOOL_NAME = "TodoWrite"

TODO_STATUSES = ("pending", "in_progress", "completed")

# 描述文本与 Java TodoWriteTool 的工具描述保持一致（含强制规则、使用/不使用场景、校验规则和示例）。
TODO_TOOL_DESCRIPTION = """创建和管理结构化任务列表，用于跟踪多步骤任务的进度。

## 强制执行规则
1. 收到多步骤任务时，必须先调用此工具创建任务列表（全部 pending），然后再执行任何实际操作
2. 每开始一个任务前，必须先调用此工具将其标记为 in_progress
3. 每完成一个任务后，必须立即调用此工具将其标记为 completed，禁止批量更新
4. 执行过程中发现新步骤时，立即添加到列表

正确流程：创建列表 → 标记任务1 in_progress → 执行任务1 → 标记任务1 completed → 标记任务2 in_progress → 执行任务2 → ...

## 使用场景
- 复杂的多步骤任务（3步以上）
- 用户提供了多个任务（编号或逗号分隔）
- 用户明确要求使用任务列表

## 不使用场景
- 单一简单任务、信息性问答、3步以内的简单操作

## 校验规则
- 同一时间只能有一个 in_progress 任务
- content 和 activeForm 不能为空
- 状态值必须是 pending、in_progress 或 completed

## 参数说明
每个任务项包含：
- content：祈使形式，描述需要做什么（如"运行测试"）
- activeForm：现在进行时形式，执行时显示（如"正在运行测试"）
- status：pending（未开始）、in_progress（执行中）、completed（已完成）

## 示例
<example>
用户：帮我添加深色模式开关，完成后运行测试
助手：
1. 调用 TodoWrite 创建列表：[添加深色模式开关(pending), 实现主题切换(pending), 运行测试(pending)]
2. 调用 TodoWrite 标记：[添加深色模式开关(in_progress), 实现主题切换(pending), 运行测试(pending)]
3. 执行添加开关的操作
4. 调用 TodoWrite 更新：[添加深色模式开关(completed), 实现主题切换(in_progress), 运行测试(pending)]
5. 执行主题切换操作
6. 以此类推，逐步完成所有任务
</example>
"""


class TodoItem(BaseModel):
    """单个任务项。字段与 Java TodoWriteTool 的 todos[] item 一致。"""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(description='任务内容，祈使形式（如"运行测试"）')
    activeForm: str = Field(description='执行时显示的现在进行时形式（如"正在运行测试"）')
    status: Literal["pending", "in_progress", "completed"] = Field(
        description="任务状态：pending（未开始）、in_progress（执行中）、completed（已完成）"
    )


class TodoWriteInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    todos: list[TodoItem] = Field(description="任务列表，包含所有任务的当前状态")


def _error(message: str) -> str:
    return json.dumps(
        {
            "ok": False,
            "error": message,
            "suggestion": "请修正后重新调用 TodoWrite，并传入包含全部任务的最新完整列表（不是增量）。",
        },
        ensure_ascii=False,
    )


def todo_write(todos: list[TodoItem]) -> str:
    """校验任务列表并回显最新状态（对齐 Java TodoWriteTool 的校验规则）。"""
    if not todos:
        return _error("todos 不能为空，至少需要包含一个任务")
    for index, item in enumerate(todos, start=1):
        if not item.content.strip():
            return _error(f"第 {index} 个任务的 content 不能为空")
        if not item.activeForm.strip():
            return _error(f"第 {index} 个任务的 activeForm 不能为空")

    in_progress = [item for item in todos if item.status == "in_progress"]
    if len(in_progress) > 1:
        items = "、".join(item.content for item in in_progress)
        return _error(f"同一时间只能有一个 in_progress 任务，当前有 {len(in_progress)} 个：{items}")

    pending = sum(1 for item in todos if item.status == "pending")
    completed = sum(1 for item in todos if item.status == "completed")
    return json.dumps(
        {
            "ok": True,
            "message": (
                f"任务列表已更新：共 {len(todos)} 项"
                f"（pending {pending} / in_progress {len(in_progress)} / completed {completed}）"
            ),
            "total": len(todos),
            "pending": pending,
            "inProgress": len(in_progress),
            "completed": completed,
            "todos": [
                {"content": item.content, "activeForm": item.activeForm, "status": item.status}
                for item in todos
            ],
        },
        ensure_ascii=False,
    )


def build_todo_tool() -> StructuredTool:
    """构建常驻的 TodoWrite 工具（与 skill / tool_search 同层，始终对模型可见）。"""
    return StructuredTool.from_function(
        todo_write,
        name=TODO_TOOL_NAME,
        description=TODO_TOOL_DESCRIPTION,
        args_schema=TodoWriteInput,
    )
