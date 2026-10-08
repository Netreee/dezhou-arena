from abc import ABC, abstractmethod

from poker.application.commands import Command, SessionContext
from poker.application.views import CommandResponse
from poker.domain.models import Table
from poker.domain.types import TableId


class TableRepository(ABC):
    @abstractmethod
    def load(self, table_id: TableId) -> Table | None:
        """Return a detached working object, not an alias of saved state."""
        raise NotImplementedError

    @abstractmethod
    def save(self, table: Table) -> None:
        """Replace the whole table snapshot in one commit."""
        raise NotImplementedError


class CommandHandler(ABC):
    @abstractmethod
    def handle(self, command: Command, session: SessionContext) -> CommandResponse:
        """Exactly one response; identity is supplied by the server."""
        raise NotImplementedError
