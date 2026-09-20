from app.security import SqlSafetyGuard


def guard():
    return SqlSafetyGuard(["payment", "rental", "customer", "film"], max_rows=200, max_joins=3)


def test_rejects_mutation():
    result = guard().validate("DELETE FROM payment")
    assert result.valid is False


def test_rejects_non_whitelisted_table():
    result = guard().validate("SELECT * FROM sys_user")
    assert result.valid is False
    assert "白名单" in result.reason


def test_caps_limit():
    result = guard().validate("SELECT payment_id FROM payment LIMIT 999")
    assert result.valid is True
    assert "LIMIT 200" in result.safe_sql.upper()


def test_keeps_small_limit():
    result = guard().validate("SELECT payment_id FROM payment LIMIT 10")
    assert result.valid is True
    assert "LIMIT 10" in result.safe_sql.upper()
