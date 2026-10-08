from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from uuid import uuid4
from enum import StrEnum
import re
from typing import TypeAlias

from poker.domain.models import PlayerAction
from poker.domain.types import HandId, PlayerId, TableId


class CommandKind(StrEnum):
    JOIN = "join"
    STATE = "state"
    START_HAND = "start_hand"
    ACT = "act"


class TableCommand(ABC):
    @property
    @abstractmethod
    def kind(self) -> CommandKind:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class JoinCommand(TableCommand):
    name: str
    table_id: TableId | None = None

    def __post_init__(self) -> None:
        if self.table_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.table_id):
            raise ValueError("Table ID must contain 1–64 letters, digits, underscores or hyphens")

    @property
    def kind(self) -> CommandKind:
        return CommandKind.JOIN


@dataclass(frozen=True, slots=True)
class StateCommand(TableCommand):
    @property
    def kind(self) -> CommandKind:
        return CommandKind.STATE


@dataclass(frozen=True, slots=True)
class StartHandCommand(TableCommand):
    @property
    def kind(self) -> CommandKind:
        return CommandKind.START_HAND


@dataclass(frozen=True, slots=True)
class ActCommand(TableCommand):
    hand_id: HandId
    action: PlayerAction
    expected_revision: int | None = None

    def __post_init__(self) -> None:
        if self.expected_revision is not None and (
            type(self.expected_revision) is not int or self.expected_revision < 0
        ):
            raise ValueError("expected_revision must be a nonnegative integer")

    @property
    def kind(self) -> CommandKind:
        return CommandKind.ACT


Command: TypeAlias = JoinCommand | StateCommand | StartHandCommand | ActCommand


@dataclass(slots=True)
class SessionContext:
    """Server-owned identity for one connection; absent from request JSON."""

    player_id: PlayerId | None = None
    connection_id: str = field(default_factory=lambda: uuid4().hex)
    correlation_id: str | None = None
    table_id: TableId | None = None
