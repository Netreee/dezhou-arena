"""One listener routes connections to independent, single-consumer tables."""

from threading import Thread

from poker.application.commands import Command, JoinCommand, SessionContext
from poker.application.interfaces import CommandHandler, TableRepository
from poker.application.service import TableService
from poker.application.views import CommandResponse, ErrorInfo, PlayerViewBuilder
from poker.domain.models import Table
from poker.domain.types import ErrorCode, TableId
from poker.engine.interfaces import GameEngine
from poker.messaging.local import LocalCommandQueue, QueuedCommandHandler, QueueWorker
from shared_logging import get_logger


class TableRuntime:
    def __init__(self, table: Table, repository: TableRepository, engine: GameEngine) -> None:
        self.table_id = table.id
        self._initial_table = table
        self._repository = repository
        service = TableService(table.id, repository, engine, PlayerViewBuilder())
        queue = LocalCommandQueue()
        self.handler = QueuedCommandHandler(queue)
        self._worker = QueueWorker(queue, service)
        self._thread = Thread(target=self._worker.run, name=f"table-{table.id}", daemon=True)

    def start(self) -> None:
        if self._thread.ident is not None:
            raise RuntimeError("Table runtime can only start once")
        self._repository.save(self._initial_table)
        self._thread.start()

    def close(self) -> None:
        if self._thread.ident is not None:
            self._worker.stop()
            self._thread.join()


class TableRegistry(CommandHandler):
    """An immutable active-table directory, owned by one server composition root."""

    def __init__(self, tables: tuple[TableRuntime, ...]) -> None:
        if not tables or len({table.table_id for table in tables}) != len(tables):
            raise ValueError("At least one table and unique table IDs are required")
        self._tables = {table.table_id: table for table in tables}
        self.default_table_id = tables[0].table_id

    @property
    def table_ids(self) -> tuple[TableId, ...]:
        return tuple(self._tables)

    def start(self) -> None:
        for table in self._tables.values():
            table.start()

    def close(self) -> None:
        for table in self._tables.values():
            table.close()

    def handle(self, command: Command, session: SessionContext) -> CommandResponse:
        if isinstance(command, JoinCommand):
            if session.table_id is not None or session.player_id is not None:
                return self._reject(ErrorCode.ALREADY_SEATED, "Connection already joined", session)
            table_id = command.table_id or self.default_table_id
        else:
            if session.table_id is None:
                return self._reject(ErrorCode.NOT_SEATED, "Join a table before issuing commands", session)
            table_id = session.table_id
        table = self._tables.get(table_id)
        if table is None:
            return self._reject(ErrorCode.TABLE_NOT_FOUND, "Requested table is not active", session, table_id)
        return table.handler.handle(command, session)

    @staticmethod
    def _reject(code: ErrorCode, message: str, session: SessionContext, table_id: TableId | None = None) -> CommandResponse:
        get_logger("server.router").bind(table_id=table_id or session.table_id,
                                        connection_id=session.connection_id,
                                        correlation_id=session.correlation_id).emit(
            "WARNING", "command.rejected", message, {"code": code.value},
        )
        return CommandResponse(error=ErrorInfo(code, message))
