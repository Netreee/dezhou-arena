from collections.abc import Callable
from dataclasses import dataclass, field, replace
from secrets import token_urlsafe
from threading import Lock
from shared_logging import fingerprint, get_logger
from poker.application.commands import JoinCommand
from poker.domain.types import TableId

from poker.gui.interfaces import CliGuiPort, GuiSessionManager
from poker.gui.models import (
    CliLifecycleUpdate, CommandAcceptance, GuiCommandConflict, GuiError,
    GuiSessionId, GuiSessionNotFound, GuiSnapshot, GuiStatus,
)


@dataclass
class _Session:
    port: CliGuiPort
    snapshot: GuiSnapshot
    pending_line: str | None
    lock: Lock = field(default_factory=Lock)


class LocalGuiSessionManager(GuiSessionManager):
    """Own only per-player CLI caches and pending acknowledgements, never game state."""

    def __init__(self, port_factory: Callable[[], CliGuiPort]) -> None:
        self._port_factory = port_factory
        self._sessions: dict[GuiSessionId, _Session] = {}
        self._lock = Lock()

    def create(self, name: str, table_id: str | None = None) -> GuiSnapshot:
        name = name.strip()
        if not name or len(name) > 64 or "\n" in name or "\r" in name:
            raise ValueError("玩家姓名应为 1–64 个字符，不含换行")
        target = TableId(table_id.strip()) if table_id and table_id.strip() else None
        JoinCommand(name, target)
        sid = GuiSessionId(token_urlsafe(32))
        line = f"join_table {target} {name}" if target is not None else f"join {name}"
        session = _Session(self._port_factory(), GuiSnapshot(sid, GuiStatus.OPENING, command_pending=True), line)
        with self._lock:
            self._sessions[sid] = session
        get_logger("gui.session").bind(session_ref=fingerprint(sid)).emit("INFO", "gui.session_opened", "GUI session opening")
        try:
            session.port.start(name, target)
        except Exception:
            with self._lock:
                del self._sessions[sid]
            session.port.close()
            raise
        return self.snapshot(sid)

    def snapshot(self, session_id: GuiSessionId) -> GuiSnapshot:
        session = self._session(session_id)
        with session.lock:
            self._consume(session)
            return session.snapshot

    def submit(self, session_id: GuiSessionId, line: str) -> CommandAcceptance:
        line = line.strip()
        if not line or len(line) > 256 or "\n" in line or "\r" in line or line == "quit":
            raise ValueError("使用单行 CLI 命令；退出请关闭会话")
        session = self._session(session_id)
        with session.lock:
            self._consume(session)
            if session.snapshot.status is not GuiStatus.READY or session.pending_line is not None:
                raise GuiCommandConflict("会话尚未就绪或正在等待当前命令确认")
            session.port.submit_line(line)
            session.pending_line = line
            session.snapshot = replace(session.snapshot, command_pending=True, error=None)
            get_logger("gui.session").bind(session_ref=fingerprint(session_id)).emit("INFO", "gui.command_queued", "GUI command queued", {"command": line.split()[0]})
        return CommandAcceptance(session_id)

    def close(self, session_id: GuiSessionId) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            raise GuiSessionNotFound("GUI 会话不存在或已关闭")
        with session.lock:
            session.port.close()
            session.pending_line = None
            session.snapshot = replace(session.snapshot, status=GuiStatus.CLOSED, command_pending=False)
            get_logger("gui.session").bind(session_ref=fingerprint(session_id)).emit("INFO", "gui.session_closed", "GUI session closed")

    def close_all(self) -> None:
        with self._lock:
            ids = list(self._sessions)
        for sid in ids:
            try:
                self.close(sid)
            except GuiSessionNotFound:
                continue

    def _session(self, sid: GuiSessionId) -> _Session:
        with self._lock:
            session = self._sessions.get(sid)
        if session is None:
            raise GuiSessionNotFound("GUI 会话不存在或已关闭")
        return session

    @staticmethod
    def _consume(session: _Session) -> None:
        while (update := session.port.next_update()) is not None:
            snap = session.snapshot
            if isinstance(update, CliLifecycleUpdate):
                if update.closed or update.source == session.pending_line:
                    session.pending_line = None
                session.snapshot = replace(
                    snap, status=GuiStatus.CLOSED if update.closed else snap.status,
                    command_pending=session.pending_line is not None, error=update.error or snap.error,
                )
                continue
            confirmed = update.source != "poll" and update.source == session.pending_line
            if update.response.view is not None:
                view = update.response.view
                if snap.view is not None and (snap.view.me.player_id != view.me.player_id or snap.view.table_id != view.table_id):
                    session.port.close()
                    session.pending_line = None
                    session.snapshot = replace(snap, status=GuiStatus.CLOSED, command_pending=False,
                                               error=GuiError("identity_changed", "CLI 牌桌或玩家身份改变"))
                    raise RuntimeError("CLI 响应的牌桌或玩家身份发生变化")
                if snap.status is GuiStatus.OPENING and not confirmed:
                    continue
                if confirmed:
                    session.pending_line = None
                    get_logger("gui.session").bind(session_ref=fingerprint(snap.session_id), player_id=view.me.player_id,
                                                  hand_id=view.hand_id).emit("INFO", "gui.command_completed", "GUI command confirmed")
                session.snapshot = replace(
                    snap, status=GuiStatus.READY, view=view,
                    command_pending=session.pending_line is not None, error=None if confirmed else snap.error,
                )
            else:
                error = update.response.error
                assert error is not None
                if confirmed:
                    session.pending_line = None
                    get_logger("gui.session").bind(session_ref=fingerprint(snap.session_id)).emit(
                        "WARNING", "gui.command_completed", "GUI command rejected", {"ok": False, "code": error.code.value},
                    )
                failed_join = snap.status is GuiStatus.OPENING and confirmed
                session.snapshot = replace(
                    snap, status=GuiStatus.CLOSED if failed_join else snap.status,
                    command_pending=session.pending_line is not None, error=GuiError(error.code.value, error.message),
                )
                if failed_join:
                    session.port.close()
