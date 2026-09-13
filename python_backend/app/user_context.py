from __future__ import annotations

from sqlalchemy import text

from .db import Database
from .models import DataScope, DataScopeContext, User
from .security import SensitiveFilter


class UserContextService:
    def __init__(self, db: Database, sensitive_filter: SensitiveFilter, permission_enabled: bool):
        self.db = db
        self.sensitive_filter = sensitive_filter
        self.permission_enabled = permission_enabled

    def resolve_scope(self, user: User) -> DataScopeContext:
        if not self.permission_enabled:
            return DataScopeContext(user.id, DataScope.ALL, [])
        if user.data_scope in (DataScope.ALL, DataScope.SELF):
            return DataScopeContext(user.id, user.data_scope, [])

        own = {d.id for d in user.departments}
        if user.data_scope == DataScope.DEPT:
            return DataScopeContext(user.id, user.data_scope, sorted(own))

        with self.db.connect() as conn:
            rows = conn.execute(
                text("SELECT id, ancestors FROM sys_dept WHERE status = 'ACTIVE'")
            ).mappings().all()
        visible = set(own)
        for row in rows:
            ancestors = {
                int(x) for x in (row["ancestors"] or "").split(",")
                if x.strip().isdigit() and int(x) > 0
            }
            if own & ancestors:
                visible.add(int(row["id"]))
        return DataScopeContext(user.id, user.data_scope, sorted(visible))

    def build_prompt_context(self, user: User) -> str:
        role_text = ", ".join(f"{r.name}({r.code})" for r in user.roles) or "未分配"
        dept_text = ", ".join(f"{d.name}({d.id})" for d in user.departments) or "未挂载"
        lines = [
            "",
            "## 当前用户上下文",
            f"- 用户ID：{user.id}",
            f"- 用户名：{user.username}",
            f"- 昵称：{user.nickname or ''}",
            f"- 角色：{role_text}",
            f"- 部门：{dept_text}",
            f"- 数据范围：{user.data_scope.value}",
            f"- 数据权限过滤：{'已开启' if self.permission_enabled else '未开启'}",
        ]
        if user.profile:
            profile = dict(user.profile)
            for key in ("id_card", "home_address"):
                if key in profile and key in self.sensitive_filter.columns:
                    profile[key] = SensitiveFilter.MASK
            lines.extend(
                [
                    "",
                    "### 个人档案",
                    f"- 真实姓名：{profile.get('real_name') or ''}",
                    f"- 身份证号：{profile.get('id_card') or ''}",
                    f"- 年龄：{profile.get('age') or ''}",
                    f"- 学历：{profile.get('education') or ''}",
                    f"- 家庭住址：{profile.get('home_address') or ''}",
                ]
            )
        lines.extend(
            [
                "",
                "### 用户问“我的 xxx”时的指代",
                f"- 指当前用户（ID={user.id}）的数据。",
                "- 权限条件由系统确定性注入，Agent 不要自行绕过或重复拼接 user_id/dept_id。",
            ]
        )
        return "\n".join(lines)
