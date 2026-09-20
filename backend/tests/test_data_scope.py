import sqlglot

from app.models import DataScope, DataScopeContext
from app.security import DataScopeRewriter


def normalize(sql: str) -> str:
    return sqlglot.parse_one(sql, read="mysql").sql(dialect="mysql")


def test_self_scope_injects_user_id():
    sql = "SELECT payment_id, amount FROM payment WHERE amount > 5 LIMIT 20"
    out = DataScopeRewriter().rewrite(
        sql,
        DataScopeContext(user_id=7, scope=DataScope.SELF, dept_ids=[]),
    )
    assert "payment.user_id = 7" in out


def test_dept_scope_injects_dept_ids():
    sql = "SELECT rental_id FROM rental LIMIT 20"
    out = DataScopeRewriter().rewrite(
        sql,
        DataScopeContext(user_id=7, scope=DataScope.DEPT_AND_SUB, dept_ids=[10, 11]),
    )
    assert "rental.dept_id IN (10, 11)" in out
