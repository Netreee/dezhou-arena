from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, NewType, TypeAlias

from poker.application.views import CommandResponse, PlayerView

GuiSessionId = NewType("GuiSessionId", str)


class GuiStatus(StrEnum):
    OPENING = "opening"
    READY = "ready"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class GuiError:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class CliResponseUpdate:
    """source is the reserved 'poll' tag or the submitted CLI command text."""

    source: str
    response: CommandResponse


@dataclass(frozen=True, slots=True)
class CliLifecycleUpdate:
    """Local input/network errors and exit; do not invent a server response."""

    error: GuiError | None = None
    closed: bool = True
    source: str | None = None


CliUpdate: TypeAlias = CliResponseUpdate | CliLifecycleUpdate


@dataclass(frozen=True, slots=True)
class GuiSnapshot:
    session_id: GuiSessionId
    status: GuiStatus
    view: PlayerView | None = None
    command_pending: bool = False
    error: GuiError | None = None


@dataclass(frozen=True, slots=True)
class CommandAcceptance:
    """Queued at the CLI boundary, not confirmed by the game server."""

    session_id: GuiSessionId
    stage: Literal["queued"] = "queued"


class GuiSessionNotFound(LookupError):
    pass


class GuiCommandConflict(ValueError):
    """An opening/closed session or a command already awaiting its own response."""
