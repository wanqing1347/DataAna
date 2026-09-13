from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .agent_graph import AgentGraphFactory
from .agent_service import AgentService
from .bird_eval import BirdEvalRequest, BirdEvalService
from .auth import AuthService, user_to_frontend
from .checkpointing import DurableCheckpointManager
from .config import get_settings
from .db import Database
from .schema_catalog import SchemaCatalog
from .security import DataScopeRewriter, SensitiveFilter, SqlSafetyGuard
from .session_store import SessionStore
from .user_context import UserContextService


settings = get_settings()
db = Database(settings.database_url)
auth_service = AuthService(db, settings)
catalog = SchemaCatalog(settings.resolved_schema_path(), settings.allowed_tables)
guard = SqlSafetyGuard(settings.allowed_tables, settings.max_rows, settings.max_joins)
rewriter = DataScopeRewriter()
sensitive_filter = SensitiveFilter(settings.sensitive_filter_enabled, settings.sensitive_fields)
scope_service = UserContextService(db, sensitive_filter, settings.data_permission_enabled)
session_store = SessionStore(db)
checkpoint_manager = DurableCheckpointManager(settings.resolved_checkpoint_path())
graph_factory = AgentGraphFactory(settings)
bird_eval_service = BirdEvalService(settings)
agent_service = AgentService(
    settings,
    db,
    catalog,
    graph_factory,
    scope_service,
    session_store,
    guard,
    rewriter,
    sensitive_filter,
    checkpoint_manager,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = PROJECT_ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    session_store.initialize()
    await agent_service.startup()
    try:
        yield
    finally:
        await agent_service.shutdown()
        db.dispose()


app = FastAPI(title="DataAna Python", version="0.4.0", lifespan=lifespan)


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    query: str
    conversationId: str | None = None
    online: bool | None = False
    fileIds: list[str] | None = None


class ResumeRequest(BaseModel):
    conversationId: str
    runId: str


def ok(data=None, msg: str = "操作成功"):
    return {"code": 200, "msg": msg, "data": data}


def current_user(satoken: str | None):
    if not satoken:
        raise HTTPException(status_code=401, detail="未登录")
    return auth_service.get_user_from_token(satoken)


@app.post("/auth/login")
def login(req: LoginRequest):
    try:
        token, user = auth_service.login(req.username, req.password)
        return ok(user_to_frontend(user, token), "登录成功")
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"code": 400, "msg": str(exc), "data": None})


@app.post("/auth/logout")
def logout():
    return ok()


@app.get("/auth/me")
def me(satoken: str | None = Header(default=None)):
    user = current_user(satoken)
    return ok(user_to_frontend(user))


@app.post("/agent/stream")
async def stream_agent(req: ChatRequest, satoken: str | None = Header(default=None)):
    user = current_user(satoken)
    if req.online or req.fileIds:
        raise HTTPException(status_code=400, detail="DataAna Python 当前仅支持业务数据库数据分析")
    conversation_id = req.conversationId or ("dodo_conv_" + uuid4().hex)
    return StreamingResponse(
        agent_service.stream(req.query, conversation_id, user),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/agent/stop")
async def stop_agent(conversationId: str, satoken: str | None = Header(default=None)):
    user = current_user(satoken)
    run_id = agent_service.stop(conversationId, user.id)
    has_state = bool(run_id and await agent_service.has_checkpoint(user.id, run_id))
    return ok(
        {
            "conversationId": conversationId,
            "runId": run_id,
            "interrupted": run_id is not None,
            "hasState": has_state,
        }
    )


@app.post("/agent/resume")
async def resume_agent(req: ResumeRequest, satoken: str | None = Header(default=None)):
    user = current_user(satoken)
    return StreamingResponse(
        agent_service.resume(req.conversationId, req.runId, user),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/session/list")
def list_sessions(
    page: int = 0,
    size: int = 10,
    satoken: str | None = Header(default=None),
):
    user = current_user(satoken)
    return ok(session_store.list_conversations(user.id, page, size))


@app.get("/session/{conversation_id}")
def session_detail(conversation_id: str, satoken: str | None = Header(default=None)):
    user = current_user(satoken)
    detail = session_store.detail(user.id, conversation_id)
    if detail is None:
        return JSONResponse(status_code=404, content={"code": 404, "msg": "会话不存在", "data": None})
    return ok(detail)


@app.get("/session/{conversation_id}/runs")
def session_runs(
    conversation_id: str,
    limit: int = 20,
    satoken: str | None = Header(default=None),
):
    user = current_user(satoken)
    return ok(session_store.list_runs(user.id, conversation_id, limit))


@app.delete("/session/{conversation_id}")
async def delete_session(conversation_id: str, satoken: str | None = Header(default=None)):
    user = current_user(satoken)
    if not await agent_service.delete_session(conversation_id, user.id):
        return JSONResponse(
            status_code=404,
            content={"code": 404, "msg": "会话不存在或仍有运行中的 Agent", "data": None},
        )
    return ok()


@app.post("/bird/eval/baseline")
async def bird_eval_baseline(req: BirdEvalRequest):
    return await bird_eval_service.eval_baseline(req)


@app.post("/bird/eval/question")
async def bird_eval_question(req: BirdEvalRequest):
    return await bird_eval_service.eval_agent(req)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "agent": "langgraph",
        "language": "python",
        "mcp": agent_service.mcp_health(),
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/{asset_path:path}")
def static_asset(asset_path: str):
    target = (STATIC_DIR / asset_path).resolve()
    if STATIC_DIR.resolve() not in target.parents and target != STATIC_DIR.resolve():
        raise HTTPException(status_code=404)
    if not target.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(target)
