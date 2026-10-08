from poker.application.commands import ActCommand, Command, JoinCommand, StartHandCommand, StateCommand
from poker.domain.models import PlayerAction
from poker.domain.types import ActionKind, HandId, TableId


class CommandParser:
    """Only syntax and current-hand attachment; poker legality stays server-side."""

    def __init__(self, table_id: TableId | None = None) -> None:
        JoinCommand("validate", table_id)
        self._table_id = table_id

    def parse(self, line: str, hand_id: HandId | None = None) -> Command:
        words = line.strip().split()
        if not words:
            raise ValueError("Enter a command")
        verb = words[0]
        if verb == "join" and len(words) > 1:
            return JoinCommand(" ".join(words[1:]), self._table_id)
        if verb == "join_table" and len(words) >= 3:
            return JoinCommand(" ".join(words[2:]), TableId(words[1]))
        if verb == "state" and len(words) == 1:
            return StateCommand()
        if verb == "start" and len(words) == 1:
            return StartHandCommand()
        kind = ActionKind(verb)
        sized = kind in (ActionKind.BET_TO, ActionKind.RAISE_TO)
        if len(words) != (2 if sized else 1):
            raise ValueError("Unexpected number of arguments")
        if hand_id is None:
            raise ValueError("Read the current hand before submitting an action")
        return ActCommand(hand_id, PlayerAction(kind, int(words[1]) if sized else None))
