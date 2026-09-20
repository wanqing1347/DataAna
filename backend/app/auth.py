from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import Header, HTTPException
from sqlalchemy import bindparam, text

from .config import Settings
from .db import Database
from .models import DataScope, Department, Role, User


class AuthService:
    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings

    def login(self, username: str, password: str) -> tuple[str, User]:
        with self.db.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT id, username, password, nickname, status "
                    "FROM sys_user WHERE username = :username LIMIT 1"
                ),
                {"username": username},
            ).mappings().first()
        if not row:
            raise ValueError("用户名不存在")
        if row["password"] != password:
            raise ValueError("密码错误")
        if row["status"] != "ACTIVE":
            raise ValueError("账号已被禁用，请联系管理员")

        user = self.get_user(int(row["id"]))
        token = self._encode_token(user.id)
        return token, user

    def get_user(self, user_id: int) -> User:
        with self.db.connect() as conn:
            base = conn.execute(
                text(
                    "SELECT id, username, nickname, status "
                    "FROM sys_user WHERE id = :id LIMIT 1"
                ),
                {"id": user_id},
            ).mappings().first()
            if not base:
                raise ValueError(f"用户不存在: {user_id}")

            role_rows = conn.execute(
                text(
                    "SELECT r.id, r.code, r.name, r.data_scope "
                    "FROM sys_user_role ur JOIN sys_role r ON r.id = ur.role_id "
                    "WHERE ur.user_id = :uid AND r.status = 'ACTIVE'"
                ),
                {"uid": user_id},
            ).mappings().all()
            dept_rows = conn.execute(
                text(
                    "SELECT d.id, d.name, d.parent_id, d.ancestors "
                    "FROM sys_user_dept ud JOIN sys_dept d ON d.id = ud.dept_id "
                    "WHERE ud.user_id = :uid AND d.status = 'ACTIVE'"
                ),
                {"uid": user_id},
            ).mappings().all()
            profile = conn.execute(
                text(
                    "SELECT real_name, id_card, age, education, home_address "
                    "FROM user_profile WHERE user_id = :uid LIMIT 1"
                ),
                {"uid": user_id},
            ).mappings().first()

        roles = [
            Role(int(r["id"]), r["code"], r["name"], r["data_scope"])
            for r in role_rows
        ]
        departments = [
            Department(int(d["id"]), d["name"], d["parent_id"], d["ancestors"] or "")
            for d in dept_rows
        ]
        return User(
            id=int(base["id"]),
            username=base["username"],
            nickname=base["nickname"],
            status=base["status"],
            roles=roles,
            departments=departments,
            data_scope=DataScope.max_scope([r.data_scope for r in roles]),
            profile=dict(profile) if profile else None,
        )

    def get_user_from_token(self, token: str) -> User:
        try:
            payload = jwt.decode(token, self.settings.jwt_secret, algorithms=["HS256"])
            user_id = int(payload["sub"])
        except Exception as exc:
            raise HTTPException(status_code=401, detail="未登录或登录已过期") from exc
        return self.get_user(user_id)

    def _encode_token(self, user_id: int) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "sub": str(user_id),
            "iat": now,
            "exp": now + timedelta(hours=self.settings.jwt_expire_hours),
        }
        return jwt.encode(payload, self.settings.jwt_secret, algorithm="HS256")


def user_to_frontend(user: User, token: str | None = None) -> dict[str, Any]:
    result = {
        "id": user.id,
        "username": user.username,
        "nickname": user.nickname,
        "status": user.status,
        "dataScope": user.data_scope.value,
        "roles": [{"id": r.id, "code": r.code, "name": r.name} for r in user.roles],
        "depts": [
            {"id": d.id, "name": d.name, "path": [d.name]}
            for d in user.departments
        ],
        "profile": user.profile,
    }
    if token:
        result["token"] = token
    return result
