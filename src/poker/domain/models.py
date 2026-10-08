from dataclasses import dataclass, field

from poker.domain.cards import Card, Deck
from poker.domain.errors import RuleViolation
from poker.domain.types import (
    ActionKind, ErrorCode, HandCategory, HandId, HandPhase, PlayerId,
    PlayerStatus, TableId,
)


@dataclass(frozen=True, slots=True)
class TableConfig:
    small_blind: int = 5
    big_blind: int = 10
    starting_stack: int = 1000
    max_players: int = 6

    def __post_init__(self) -> None:
        if not 0 < self.small_blind <= self.big_blind:
            raise ValueError("Expected 0 < small_blind <= big_blind")
        if self.starting_stack < self.big_blind or not 2 <= self.max_players <= 6:
            raise ValueError("Invalid starting stack or table size")


@dataclass(frozen=True, slots=True)
class PlayerAction:
    kind: ActionKind
    to: int | None = None

    def __post_init__(self) -> None:
        sized = self.kind in (ActionKind.BET_TO, ActionKind.RAISE_TO)
        if sized and (self.to is None or self.to <= 0):
            raise ValueError("bet_to and raise_to require a positive total")
        if not sized and self.to is not None:
            raise ValueError("This action does not accept an amount")


@dataclass(frozen=True, slots=True)
class ActionOption:
    kind: ActionKind
    pay: int | None = None
    min_to: int | None = None
    max_to: int | None = None


@dataclass(slots=True)
class Player:
    id: PlayerId
    name: str
    seat: int
    stack: int


@dataclass(slots=True)
class HandPlayer:
    hole_cards: tuple[Card, ...] = ()
    status: PlayerStatus = PlayerStatus.ACTIVE
    street_commit: int = 0
    hand_commit: int = 0


@dataclass(slots=True)
class BettingRound:
    actor_id: PlayerId | None = None
    current_bet: int = 0
    last_full_raise: int = 0
    full_opening_established: bool = False
    pending: list[PlayerId] = field(default_factory=list)
    last_action_bet: dict[PlayerId, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, order=True)
class HandValue:
    category: HandCategory
    kickers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Pot:
    amount: int
    eligible_ids: tuple[PlayerId, ...]


@dataclass(frozen=True, slots=True)
class ChipShare:
    player_id: PlayerId
    amount: int


@dataclass(frozen=True, slots=True)
class PotAward:
    pot: Pot
    shares: tuple[ChipShare, ...]


@dataclass(frozen=True, slots=True)
class HandResult:
    hand_id: HandId
    awards: tuple[PotAward, ...]
    refunds: tuple[ChipShare, ...] = ()
    revealed_hands: dict[PlayerId, tuple[Card, ...]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PublicActionRecord:
    """One public payment/action, captured before refunds and street advancement.

    ``to`` is the resulting total commitment on this street, including for
    calls, checks and folds. ``stack`` is the remaining stack at that instant.
    Sequence numbers start at one and include the two forced blind payments.
    """

    sequence: int
    phase: HandPhase
    player_id: PlayerId
    kind: str
    pay: int
    to: int
    stack: int


@dataclass(slots=True)
class Hand:
    id: HandId
    phase: HandPhase
    button_seat: int
    small_blind_seat: int
    big_blind_seat: int
    deck: Deck
    chip_total_at_start: int
    players: dict[PlayerId, HandPlayer] = field(default_factory=dict)
    board: list[Card] = field(default_factory=list)
    betting: BettingRound = field(default_factory=BettingRound)
    refunds: list[ChipShare] = field(default_factory=list)
    action_history: list[PublicActionRecord] = field(default_factory=list)
    history_complete: bool = False


@dataclass(slots=True)
class Table:
    """Aggregate root. Player.stack is the only remaining-chip account."""

    id: TableId
    config: TableConfig = field(default_factory=TableConfig)
    players: list[Player] = field(default_factory=list)
    button_seat: int | None = None
    hand: Hand | None = None
    last_result: HandResult | None = None
    revision: int = 0

    @property
    def between_hands(self) -> bool:
        return self.hand is None or self.hand.phase is HandPhase.COMPLETE

    def player(self, player_id: PlayerId) -> Player:
        for player in self.players:
            if player.id == player_id:
                return player
        raise RuleViolation(ErrorCode.NOT_SEATED, "Player is not seated")

    def add_player(self, player: Player) -> None:
        if not self.between_hands:
            raise RuleViolation(ErrorCode.HAND_IN_PROGRESS, "Join between hands")
        if any(p.id == player.id or p.seat == player.seat for p in self.players):
            raise RuleViolation(ErrorCode.ALREADY_SEATED, "ID or seat is occupied")
        if len(self.players) >= self.config.max_players:
            raise RuleViolation(ErrorCode.TABLE_FULL, "Table is full")
        if not 0 <= player.seat < self.config.max_players or player.stack < 0:
            raise ValueError("Invalid seat or stack")
        self.players.append(player)
