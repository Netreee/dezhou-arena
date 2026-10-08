"""Version-bound CLI input for automated clients; no transport ownership."""

from dataclasses import dataclass

from poker.application.views import CommandResponse, PlayerView
from poker.domain.types import HandId, PlayerId, TableId


@dataclass(frozen=True, slots=True)
class ViewToken:
    table_id: TableId
    hand_id: HandId | None
    revision: int
    player_id: PlayerId

    @classmethod
    def from_view(cls, view: PlayerView) -> "ViewToken":
        return cls(view.table_id, view.hand_id, view.revision, view.me.player_id)

    def matches(self, view: PlayerView) -> bool:
        return self == self.from_view(view)


@dataclass(frozen=True, slots=True)
class GuardedInput:
    request_id: str
    line: str
    token: ViewToken


@dataclass(frozen=True, slots=True)
class GuardedResult:
    """A server response or a local failure code, never both."""

    request_id: str
    response: CommandResponse | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if (self.response is None) == (self.error is None):
            raise ValueError("A guarded result requires exactly one response or error")
