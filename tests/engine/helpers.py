"""Valid card/chip snapshots for isolated rule tests, not a substitute engine."""

from random import Random

from poker.domain.cards import Card, Deck
from poker.domain.models import BettingRound, Hand, HandPlayer, Player, Table
from poker.domain.types import HandId, HandPhase, PlayerId, PlayerStatus, TableId


def betting_table(
    stacks: tuple[int, ...],
    commits: tuple[int, ...] | None = None,
    *,
    current_bet: int = 0,
    last_full_raise: int = 10,
    phase: HandPhase = HandPhase.FLOP,
) -> Table:
    contributions = commits if commits is not None else (0,) * len(stacks)
    table = Table(TableId("rules"))
    deck = Deck.shuffled(Random(37))
    members: dict[PlayerId, HandPlayer] = {}
    for seat, (stack, commit) in enumerate(zip(stacks, contributions, strict=True)):
        pid = PlayerId(f"p{seat}")
        table.add_player(Player(pid, f"P{seat}", seat, stack))
        members[pid] = HandPlayer(
            deck.draw(2), PlayerStatus.ACTIVE if stack else PlayerStatus.ALL_IN, commit, commit,
        )
    board: list[Card] = []
    if phase is not HandPhase.PREFLOP:
        deck.burn()
        board.extend(deck.draw(3))
        if phase in (HandPhase.TURN, HandPhase.RIVER):
            deck.burn()
            board.extend(deck.draw())
        if phase is HandPhase.RIVER:
            deck.burn()
            board.extend(deck.draw())
    pending = [pid for pid, member in members.items() if member.status is PlayerStatus.ACTIVE]
    table.button_seat = 0
    table.hand = Hand(
        HandId("rules-hand"), phase, 0, 0 if len(stacks) == 2 else 1, len(stacks) - 1,
        deck, sum(stacks) + sum(contributions), members, board,
        BettingRound(next(iter(pending), None), current_bet, last_full_raise,
                     current_bet >= table.config.big_blind, pending),
    )
    return table


def showdown_cards(table: Table, board: str, holes: tuple[str, ...]) -> None:
    hand = table.hand
    assert hand is not None
    hand.board = [Card.parse(code) for code in board.split()]
    for member, codes in zip(hand.players.values(), holes, strict=True):
        member.hole_cards = tuple(Card.parse(code) for code in codes.split())
    used = hand.board + [card for member in hand.players.values() for card in member.hole_cards]
    assert len(set(used)) == len(used)
    remaining = [card for card in Deck.shuffled(Random(41)).remaining if card not in used]
    hand.deck = Deck(remaining[3:], remaining[:3])
    hand.phase = HandPhase.SHOWDOWN
    hand.betting.actor_id = None
    hand.betting.pending.clear()
