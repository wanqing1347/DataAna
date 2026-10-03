from __future__ import annotations

import ast
import json
import math
import operator
from typing import Annotated

from langchain_core.tools import InjectedToolCallId, StructuredTool

from .db import Database
from .explain_precheck import ExplainPrecheck
from .models import DataScopeContext
from .schema_catalog import SchemaCatalog
from .security import DataScopeRewriter, SensitiveFilter, SqlSafetyGuard
from .session_store import SessionStore
from .sql_execution import SqlExecutionService
from .tool_runtime import IdempotentToolRunner

LOOKUP_GLOSSARY_DESCRIPTION_TEMPLATE = """查询业务术语的标准口径（含定义 + 可复用 SQL 片段 + 同义词）。

何时使用本工具：
- 用户问题涉及"活跃客户""VIP""热门影片""高消费客户""租金"等业务指标术语时
- 用户问题涉及"近 N 个月""最近""本月""上月""今年"等相对时间时
  （历史样本库所有相对时间必须基于数据最大时间，而非系统当前时间）

<可用术语列表>
%s
</可用术语列表>

传入上述术语名或其同义词即可命中（精确匹配）。未命中会返回全部术语供你重试。
"""

CALCULATOR_DESCRIPTION = (
    "数学表达式计算器。传表达式 + 变量 JSON，后端安全求值并返回结果。"
    "用于同环比、占比、贡献度、加权平均等数据分析的最终计算。"
    "支持运算符：+ - * / // % ^(幂，等价 **)；"
    "内置函数：sin/cos/tan/asin/acos/atan/sinh/cosh/tanh/log(自然对数)/ln/log2/log10/"
    "exp/sqrt/cbrt/abs/ceil/floor/max/min；"
    "额外函数：round(x, n) 保留 n 位小数。"
    "聚合（求和/均值/最大最小）请推到 SQL 用 GROUP BY + SUM/AVG 完成，本工具只算最终标量公式。"
)


class SafeCalculator:
    """受限表达式求值器（对齐参照实现 CalculateTool 的表达式能力）。

    支持变量绑定和常用数学函数，禁止属性访问、下标、lambda 等任意代码执行。
    """

    OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    FUNCTIONS = {
        "abs": abs,
        "ceil": math.ceil,
        "floor": math.floor,
        "sqrt": math.sqrt,
        "cbrt": lambda x: math.copysign(abs(x) ** (1.0 / 3.0), x),
        "exp": math.exp,
        "log": math.log,  # 与参照实现一致：自然对数
        "ln": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "max": max,
        "min": min,
        "round": lambda x, n=0: round(x, int(n)),
    }

    MAX_EXPRESSION_LENGTH = 2000
    MAX_VARIABLES = 50

    def __init__(self, variables: dict[str, float] | None = None):
        self.variables = variables or {}

    def eval(self, expression: str) -> float | int:
        if not expression or not expression.strip():
            raise ValueError("表达式不能为空")
        if len(expression) > self.MAX_EXPRESSION_LENGTH:
            raise ValueError(f"表达式过长（>{self.MAX_EXPRESSION_LENGTH} 字符）")
        node = ast.parse(expression, mode="eval").body
        return self._eval(node)

    def _eval(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(
            node.value, bool
        ):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.variables:
                return self.variables[node.id]
            raise ValueError(f"未定义变量 {node.id}（变量名区分大小写）")
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.BitXor):
                return operator.pow(self._eval(node.left), self._eval(node.right))
            if type(node.op) in self.OPS:
                return self.OPS[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self.OPS:
            return self.OPS[type(node.op)](self._eval(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return self._call(node.func.id, node.args, node.keywords)
        raise ValueError("不支持的表达式语法，只允许数字、变量、运算符和内置函数")

    def _call(self, name: str, arg_nodes: list, keyword_nodes: list):
        if keyword_nodes:
            raise ValueError(f"函数 {name}() 不支持关键字参数")
        fn = self.FUNCTIONS.get(name.lower())
        if fn is None:
            raise ValueError(f"不支持的函数 {name}()")
        args = [self._eval(arg) for arg in arg_nodes]
        try:
            return fn(*args)
        except TypeError as exc:
            raise ValueError(f"函数 {name}() 参数错误：{exc}") from exc


def _parse_variables(raw: str | None) -> dict[str, float]:
    if not raw or not raw.strip():
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("variablesJson 必须是 JSON 对象，如 {\"current\":3298,\"previous\":3105}")
    variables: dict[str, float] = {}
    for key, value in data.items():
        if isinstance(value, bool):
            raise ValueError(f"变量 {key} 的值不是数字：{value}")
        if isinstance(value, (int, float)):
            variables[str(key)] = value
        elif isinstance(value, str):
            try:
                variables[str(key)] = float(value.strip())
            except ValueError as exc:
                raise ValueError(f"变量 {key} 的值不是数字：{value}") from exc
        else:
            raise ValueError(f"变量 {key} 的值不是数字：{value}")
    return variables


def _normalize_number(value: float | int) -> float | int:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def build_tools(
    db: Database,
    catalog: SchemaCatalog,
    guard: SqlSafetyGuard,
    rewriter: DataScopeRewriter,
    sensitive_filter: SensitiveFilter,
    scope_ctx: DataScopeContext,
    *,
    conversation_id: str,
    session_store: SessionStore,
    tool_retry_max_attempts: int,
    tool_retry_base_delay_ms: int,
    explain_precheck: ExplainPrecheck | None = None,
):
    calculator = SafeCalculator()
    runner = IdempotentToolRunner(
        session_store,
        user_id=scope_ctx.user_id,
        conversation_id=conversation_id,
        max_attempts=tool_retry_max_attempts,
        base_delay_ms=tool_retry_base_delay_ms,
    )
    sql_executor = SqlExecutionService(
        db,
        guard,
        rewriter,
        sensitive_filter,
        scope_ctx,
        runner,
        explain_precheck,
    )

    def list_tables() -> str:
        """列出当前 DataAna 允许查询的业务表及其说明。"""
        return catalog.list_tables()

    def describe_tables(table_names: list[str]) -> str:
        """查看指定表的真实字段、类型、注释、外键和示例值，写 SQL 前必须先调用。"""
        return catalog.describe_tables(table_names)

    def lookup_glossary(term: str) -> str:
        return catalog.lookup_glossary(term)

    def validate_sql(sql: str) -> str:
        """只读 SQL 校验：AST 白名单、只读性、JOIN 和强制 LIMIT。生成 SQL 后、执行前必须调用。"""
        return guard.validate(sql).as_json()

    def execute_sql(
        sql: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> str:
        """执行只读 SELECT/WITH 查询；内部做安全校验、数据权限注入、EXPLAIN 预检和敏感字段脱敏。"""
        return sql_executor.execute(
            sql,
            tool_name="executeSql",
            tool_call_id=tool_call_id,
            idempotency_arguments={"sql": sql},
        )

    def calculate(expression: str, variablesJson: str = "") -> str:
        try:
            variables = _parse_variables(variablesJson)
        except Exception as exc:  # noqa: BLE001
            return json.dumps(
                {"error": f"variablesJson 解析失败：{exc}", "suggestion": '正确格式：{"var1":1,"var2":2.5}'},
                ensure_ascii=False,
            )
        if len(variables) > SafeCalculator.MAX_VARIABLES:
            return json.dumps(
                {"error": f"变量数量超过上限 {SafeCalculator.MAX_VARIABLES}"}, ensure_ascii=False
            )
        try:
            result = SafeCalculator(variables).eval(expression)
        except ZeroDivisionError:
            return json.dumps({"error": "算术异常：除数为 0"}, ensure_ascii=False)
        except ValueError as exc:
            return json.dumps({"error": f"表达式无效：{exc}"}, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": f"计算失败：{exc}"}, ensure_ascii=False)
        if isinstance(result, float) and math.isnan(result):
            return json.dumps({"error": "结果为 NaN（例如 0/0）"}, ensure_ascii=False)
        if isinstance(result, float) and math.isinf(result):
            return json.dumps({"error": "结果为无穷大（例如 1/0）"}, ensure_ascii=False)
        return json.dumps(
            {
                "expression": expression.strip(),
                "variables": variables,
                "result": _normalize_number(result),
            },
            ensure_ascii=False,
        )

    glossary_terms = catalog.glossary_catalog() or "（当前没有登记的术语）"
    return [
        StructuredTool.from_function(list_tables, name="listTables"),
        StructuredTool.from_function(describe_tables, name="describeTables"),
        StructuredTool.from_function(
            lookup_glossary,
            name="lookupGlossary",
            description=LOOKUP_GLOSSARY_DESCRIPTION_TEMPLATE % glossary_terms,
        ),
        StructuredTool.from_function(validate_sql, name="validateSql"),
        StructuredTool.from_function(execute_sql, name="executeSql"),
        StructuredTool.from_function(calculate, name="calculate", description=CALCULATOR_DESCRIPTION),
    ]
