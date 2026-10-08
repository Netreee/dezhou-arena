from abc import ABC, abstractmethod

from poker.application.commands import Command
from poker.application.views import CommandResponse


class CommandClient(ABC):
    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def send(self, command: Command) -> CommandResponse:
        """One request followed by one response; no retry."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class CommandServer(ABC):
    @abstractmethod
    def serve_forever(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Stop a serving instance and close its listening socket."""
        raise NotImplementedError
