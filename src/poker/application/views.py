from dataclasses import dataclass

from poker.domain.models import ActionOption, PublicActionRecord, Table, TableConfig
from poker.domain.types import ErrorCode, HandId, HandPhase, PlayerId, PlayerStatus, TableId


@dataclass(frozen=True, slots=True)
class PublicPlayerView:
    player_id: PlayerId
    name: str
    seat: int
    stack: int
    status: PlayerStatus | None
    street_commit: int
    hand_commit: int
    revealed_cards: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PrivatePlayerView:
    player_id: PlayerId
    hole_cards: tuple[str, ...]
    legal_actions: tuple[ActionOption, ...]


@dataclass(frozen=True, slots=True)
class ShareView:
    player_id: PlayerId
    amount: int


@dataclass(frozen=True, slots=True)
class AwardView:
    amount: int
    eligible_ids: tuple[PlayerId, ...]
    shares: tuple[ShareView, ...]


@dataclass(frozen=True, slots=True)
class ResultView:
    hand_id: HandId
    awards: tuple[AwardView, ...]
    refunds: tuple[ShareView, ...]


@dataclass(frozen=True, slots=True)
class PlayerView:
    table_id: TableId
    revision: int
    hand_id: HandId | None
    phase: HandPhase | None
    board: tuple[str, ...]
    button_seat: int | None
    actor_id: PlayerId | None
    current_bet: int
    pot_total: int
    players: tuple[PublicPlayerView, ...]
    me: PrivatePlayerView
    result: ResultView | None
    config: TableConfig | None = None
    action_history: tuple[PublicActionRecord, ...] = ()
    history_complete: bool = False


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    code: ErrorCode
    message: str


@dataclass(frozen=True, slots=True)
class CommandResponse:
    view: PlayerView | None = None
    error: ErrorInfo | None = None

    def __post_init__(self) -> None:
        if (self.view is None) == (self.error is None):
            raise ValueError("A response must contain exactly one of view or error")

    @property
    def ok(self) -> bool:
        return self.error is None


class PlayerViewBuilder:
    """Explicit projection. Never serialize Table to the network."""

    def build(
        self, table: Table, viewer_id: PlayerId, options: tuple[ActionOption, ...] = (),
    ) -> PlayerView:
        table.player(viewer_id)
        hand = table.hand
        result = table.last_result
        reveal = (
            result.revealed_hands
            if result is not None and hand is not None
            and result.hand_id == hand.id and hand.phase is HandPhase.COMPLETE
            else {}
        )
        public: list[PublicPlayerView] = []
        for player in sorted(table.players, key=lambda p: p.seat):
            member = hand.players.get(player.id) if hand is not None else None
            public.append(PublicPlayerView(
                player.id, player.name, player.seat, player.stack,
                member.status if member is not None else None,
                member.street_commit if member is not None else 0,
                member.hand_commit if member is not None else 0,
                tuple(card.code for card in reveal.get(player.id, ())),
            ))
        own = hand.players.get(viewer_id) if hand is not None else None
        visible_result = None if result is None else ResultView(
            result.hand_id,
            tuple(AwardView(
                award.pot.amount, award.pot.eligible_ids,
                tuple(ShareView(share.player_id, share.amount) for share in award.shares),
            ) for award in result.awards),
            tuple(ShareView(refund.player_id, refund.amount) for refund in result.refunds),
        )
        return PlayerView(
            table.id, table.revision,
            hand.id if hand is not None else None,
            hand.phase if hand is not None else None,
            tuple(card.code for card in hand.board) if hand is not None else (),
            hand.button_seat if hand is not None else table.button_seat,
            hand.betting.actor_id if hand is not None else None,
            hand.betting.current_bet if hand is not None else 0,
            sum(p.hand_commit for p in hand.players.values()) if hand is not None else 0,
            tuple(public),
            PrivatePlayerView(
                viewer_id,
                tuple(card.code for card in own.hole_cards) if own is not None else (),
                options if hand is not None and hand.betting.actor_id == viewer_id else (),
            ),
            visible_result,
            table.config,
            tuple(hand.action_history) if hand is not None else (),
            hand.history_complete if hand is not None else False,
        )
