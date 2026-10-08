from abc import ABC, abstractmethod

from poker.gui.models import CliUpdate, CommandAcceptance, GuiSessionId, GuiSnapshot
from poker.domain.types import TableId


class CliGuiPort(ABC):
    """Own one CLI runtime, one command queue and one player connection."""

    @abstractmethod
    def start(self, name: str, table_id: TableId | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def submit_line(self, line: str) -> None:
        """Enqueue text for the existing CommandParser; no direct TCP calls."""
        raise NotImplementedError

    @abstractmethod
    def next_update(self) -> CliUpdate | None:
        """Nonblocking read of structured CLI output, never terminal text parsing."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class GuiSessionManager(ABC):
    @abstractmethod
    def create(self, name: str, table_id: str | None = None) -> GuiSnapshot:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self, session_id: GuiSessionId) -> GuiSnapshot:
        raise NotImplementedError

    @abstractmethod
    def submit(self, session_id: GuiSessionId, line: str) -> CommandAcceptance:
        raise NotImplementedError

    @abstractmethod
    def close(self, session_id: GuiSessionId) -> None:
        raise NotImplementedError

    @abstractmethod
    def close_all(self) -> None:
        """Release every owned CLI runtime when the HTTP application stops."""
