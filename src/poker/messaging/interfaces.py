from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from queue import Queue

from poker.application.commands import Command, SessionContext
from poker.application.views import CommandResponse


@dataclass(slots=True)
class CommandEnvelope:
    command: Command
    session: SessionContext
    reply: Queue[CommandResponse] = field(default_factory=Queue)


class CommandQueue(ABC):
    @abstractmethod
    def put(self, message: CommandEnvelope | None) -> None:
        """None is the normal worker-stop marker, not a poker command."""
        raise NotImplementedError

    @abstractmethod
    def get(self) -> CommandEnvelope | None:
        raise NotImplementedError
