from copy import deepcopy
from random import Random

from poker.application.interfaces import TableRepository
from poker.domain.cards import Deck
from poker.domain.errors import RuleViolation
from poker.domain.models import (
    ActionOption, BettingRound, Hand, HandPlayer, Player, PlayerAction, Table,
)
from poker.domain.types import ActionKind, ErrorCode, HandId, HandPhase, PlayerId, TableId
from poker.engine.interfaces import GameEngine


def make_hand(table: Table) -> Hand:
    """Projection/storage fixture, deliberately not a dealt poker game."""
    deck = Deck.shuffled(Random(7))
    members = {player.id: HandPlayer(deck.draw(2), hand_commit=10)
               for player in table.players}
    deck.burn()
    return Hand(
        HandId("fixture-hand"), HandPhase.FLOP, 0, 1, 0, deck,
        sum(p.stack for p in table.players) + sum(m.hand_commit for m in members.values()),
        members, list(deck.draw(3)),
        BettingRound(table.players[0].id, 0, 10, False, list(members)),
    )


def make_table(with_hand: bool = False) -> Table:
    table = Table(TableId("fixture-table"))
    table.add_player(Player(PlayerId("p1"), "甲", 0, 990))
    table.add_player(Player(PlayerId("p2"), "乙", 1, 990))
    if with_hand:
        table.hand = make_hand(table)
    return table


class MemoryRepository(TableRepository):
    """Detached snapshots, matching the SQL repository ownership contract."""

    def __init__(self, table: Table) -> None:
        self._table = deepcopy(table)
        self.saves = 0

    def load(self, table_id: TableId) -> Table | None:
        return deepcopy(self._table) if table_id == self._table.id else None

    def save(self, table: Table) -> None:
        self._table = deepcopy(table)
        self.saves += 1


class FakeEngine(GameEngine):
    """Application/transport test double. Never use as a Hold'em engine."""

    def __init__(self) -> None:
        self.reject_after_mutation = False

    def join(self, table: Table, name: str) -> PlayerId:
        player_id = PlayerId(f"p{len(table.players) + 1}")
        table.add_player(Player(player_id, name, len(table.players), table.config.starting_stack))
        return player_id

    def start_hand(self, table: Table, actor_id: PlayerId) -> None:
        for player in table.players:
            player.stack -= 10
        table.hand = make_hand(table)

    def act(
        self, table: Table, actor_id: PlayerId, hand_id: HandId, action: PlayerAction,
    ) -> None:
        if self.reject_after_mutation:
            table.player(actor_id).stack -= 7
            raise RuleViolation(ErrorCode.INVALID_ACTION, "Test rejection after local mutation")
        if table.hand is None or table.hand.id != hand_id:
            raise RuleViolation(ErrorCode.HAND_MISMATCH, "Wrong hand")
        if table.hand.betting.actor_id != actor_id:
            raise RuleViolation(ErrorCode.NOT_YOUR_TURN, "Test actor rejection")

    def legal_actions(self, table: Table, actor_id: PlayerId) -> tuple[ActionOption, ...]:
        if table.hand is not None and table.hand.betting.actor_id == actor_id:
            return (ActionOption(ActionKind.CHECK),)
        return ()
