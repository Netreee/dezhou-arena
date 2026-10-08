from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4
from shared_logging import context_scope, get_logger

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from poker.gui.interfaces import GuiSessionManager
from poker.gui.models import (
    CommandAcceptance, GuiCommandConflict, GuiSessionId, GuiSessionNotFound, GuiSnapshot,
)


class JoinBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=64)
    table_id: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class CommandBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    line: str = Field(min_length=1, max_length=256)


class BrowserDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event: Literal["gui.render_failed", "gui.request_failed"]
    message: str = Field(max_length=512)


def create_app(manager: GuiSessionManager, *, assets_dir: Path | None = None) -> FastAPI:
    """HTTP only calls the manager; it never accesses a game table or repository."""
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            manager.close_all()

    app = FastAPI(title="Local Hold'em GUI", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def diagnostics(request: Request, call_next: object) -> Response:
        from collections.abc import Awaitable, Callable
        from typing import cast
        correlation = uuid4().hex
        request.state.correlation_id = correlation
        started = perf_counter()
        with context_scope(correlation_id=correlation):
            try:
                response = await cast(Callable[[Request], Awaitable[Response]], call_next)(request)
            except Exception:
                get_logger("gui.http").exception("gui.http_failed", "HTTP request failed")
                raise
            get_logger("gui.http").emit("WARNING" if response.status_code >= 400 else "DEBUG", "gui.http_rejected" if response.status_code >= 400 else "gui.http_completed", "HTTP request completed",
                                        {"method": request.method, "status": response.status_code, "duration_ms": (perf_counter() - started) * 1000})
            return response

    @app.exception_handler(GuiSessionNotFound)
    async def absent(request: Request, error: GuiSessionNotFound) -> JSONResponse:
        return JSONResponse(
            {"error": {"code": "gui_session_not_found", "message": str(error)}},
            status_code=404,
        )

    @app.exception_handler(GuiCommandConflict)
    async def conflicting(request: Request, error: GuiCommandConflict) -> JSONResponse:
        return JSONResponse({"error": {"code": "gui_command_pending", "message": str(error)}}, status_code=409)

    @app.exception_handler(ValueError)
    async def invalid(request: Request, error: ValueError) -> JSONResponse:
        return JSONResponse({"error": {"code": "gui_invalid_input", "message": str(error)}}, status_code=400)

    @app.post("/api/gui/sessions", status_code=201)
    def create(body: JoinBody, request: Request) -> GuiSnapshot:
        with context_scope(correlation_id=request.state.correlation_id):
            return manager.create(body.name, body.table_id)

    @app.get("/api/gui/sessions/{session_id}")
    def snapshot(session_id: str, request: Request) -> GuiSnapshot:
        with context_scope(correlation_id=request.state.correlation_id):
            return manager.snapshot(GuiSessionId(session_id))

    @app.post("/api/gui/sessions/{session_id}/commands", status_code=202)
    def submit(session_id: str, body: CommandBody, request: Request) -> CommandAcceptance:
        with context_scope(correlation_id=request.state.correlation_id):
            return manager.submit(GuiSessionId(session_id), body.line)

    @app.delete("/api/gui/sessions/{session_id}", status_code=204)
    def close(session_id: str) -> Response:
        manager.close(GuiSessionId(session_id))
        return Response(status_code=204)

    @app.post("/api/gui/diagnostics", status_code=204)
    def browser_diagnostic(body: BrowserDiagnostic) -> Response:
        get_logger("gui.browser", origin_role="browser").emit("ERROR", body.event, body.message)
        return Response(status_code=204)

    if assets_dir is not None:
        assets_dir = assets_dir.resolve()
        if not (assets_dir / "index.html").is_file():
            raise FileNotFoundError("先在 gui/web 执行 npm run build，或指定有效的 assets_dir")
        app.mount("/assets", StaticFiles(directory=assets_dir / "assets"), name="assets")

        @app.get("/", include_in_schema=False)
        def home() -> FileResponse:
            assert assets_dir is not None
            return FileResponse(assets_dir / "index.html")

    return app
