from poker.application.commands import (
    ActCommand, Command, JoinCommand, SessionContext, StartHandCommand, StateCommand,
)
from poker.application.interfaces import CommandHandler, TableRepository
from poker.application.views import CommandResponse, ErrorInfo, PlayerViewBuilder
from poker.domain.errors import RuleViolation
from poker.domain.types import ErrorCode, TableId
from poker.engine.interfaces import GameEngine
from shared_logging import context_scope, get_logger

_log = get_logger("server.command")


class TableService(CommandHandler):
    """Called by a single queue consumer. No retries or concurrency locks."""

    def __init__(
        self, table_id: TableId, repository: TableRepository,
        engine: GameEngine, views: PlayerViewBuilder,
    ) -> None:
        self._table_id = table_id
        self._repository = repository
        self._engine = engine
        self._views = views

    def handle(self, command: Command, session: SessionContext) -> CommandResponse:
        with context_scope(table_id=self._table_id, connection_id=session.connection_id, correlation_id=session.correlation_id,
                           player_id=session.player_id,
                           hand_id=command.hand_id if isinstance(command, ActCommand) else None):
            return self._handle(command, session)

    def _handle(self, command: Command, session: SessionContext) -> CommandResponse:
        level = "DEBUG" if isinstance(command, StateCommand) else "INFO"
        _log.emit(level, "command.received", "Game command received", {"command": command.kind.value})
        try:
            response = self._execute(command, session)
            assert response.view is not None
            _log.bind(player_id=session.player_id, hand_id=response.view.hand_id).emit(
                level, "command.applied", "Game command completed", {"command": command.kind.value, "revision": response.view.revision},
            )
            return response
        except RuleViolation as error:
            _log.emit("WARNING", "command.rejected", "Game command rejected", {"command": command.kind.value, "code": error.code.value})
            return CommandResponse(error=ErrorInfo(error.code, str(error)))

    def _execute(self, command: Command, session: SessionContext) -> CommandResponse:
        if session.table_id is not None and session.table_id != self._table_id:
            raise RuleViolation(ErrorCode.NOT_SEATED, "Connection belongs to another table")
        if isinstance(command, JoinCommand) and command.table_id is not None and command.table_id != self._table_id:
            raise RuleViolation(ErrorCode.TABLE_NOT_FOUND, "Requested table does not exist at this handler")
        table = self._repository.load(self._table_id)
        if table is None:
            raise RuntimeError("Composition root must create the table first")
        actor_id = session.player_id
        changed = False
        if isinstance(command, JoinCommand):
            if actor_id is not None:
                raise RuleViolation(ErrorCode.ALREADY_SEATED, "Connection already joined")
            actor_id = self._engine.join(table, command.name)
            changed = True
        else:
            if actor_id is None:
                raise RuleViolation(ErrorCode.NOT_SEATED, "Join before issuing commands")
            table.player(actor_id)
            if isinstance(command, StartHandCommand):
                self._engine.start_hand(table, actor_id)
                changed = True
            elif isinstance(command, ActCommand):
                if command.expected_revision is not None and command.expected_revision != table.revision:
                    raise RuleViolation(ErrorCode.STALE_STATE, "Action observation revision is no longer current")
                self._engine.act(table, actor_id, command.hand_id, command.action)
                changed = True
            elif not isinstance(command, StateCommand):
                raise TypeError("Unsupported command type")
        if changed:
            table.revision += 1
        options = self._engine.legal_actions(table, actor_id)
        view = self._views.build(table, actor_id, options)
        if changed:
            self._repository.save(table)
        session.player_id = actor_id
        session.table_id = self._table_id
        return CommandResponse(view=view)
