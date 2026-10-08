from uuid import uuid4
from shared_logging import get_logger

from poker.domain.cards import Deck
from poker.domain.errors import RuleViolation
from poker.domain.models import (
    ActionOption, BettingRound, Hand, HandPlayer, Player, PlayerAction, PublicActionRecord, Table,
)
from poker.domain.types import ErrorCode, HandId, HandPhase, PlayerId, PlayerStatus
from poker.engine.interfaces import BettingRules, GameEngine, HandEvaluator, PotAllocator


class HoldemEngine(GameEngine):
    """Coordinates three rules collaborators; no SQL, sockets or queues."""

    def __init__(
        self, betting: BettingRules, evaluator: HandEvaluator, pots: PotAllocator,
    ) -> None:
        self._betting = betting
        self._evaluator = evaluator
        self._pots = pots

    def join(self, table: Table, name: str) -> PlayerId:
        if not name.strip():
            raise RuleViolation(ErrorCode.BAD_REQUEST, "Name cannot be empty")
        occupied = {p.seat for p in table.players}
        seat = next((s for s in range(table.config.max_players) if s not in occupied), None)
        if seat is None:
            raise RuleViolation(ErrorCode.TABLE_FULL, "Table is full")
        player_id = PlayerId(uuid4().hex)
        table.add_player(Player(player_id, name.strip(), seat, table.config.starting_stack))
        return player_id

    def start_hand(self, table: Table, actor_id: PlayerId) -> None:
        table.player(actor_id)
        if not table.between_hands:
            raise RuleViolation(ErrorCode.HAND_IN_PROGRESS, "Finish the current hand first")
        players = sorted((p for p in table.players if p.stack > 0), key=lambda p: p.seat)
        if len(players) < 2:
            raise RuleViolation(ErrorCode.NOT_ENOUGH_PLAYERS, "Need two players with chips")

        button, small_blind, big_blind = self._positions(table, players)
        hand = Hand(
            HandId(uuid4().hex), HandPhase.SETUP,
            button.seat, small_blind.seat, big_blind.seat,
            Deck.shuffled(), sum(p.stack for p in players),
            {p.id: HandPlayer() for p in players},
            history_complete=True,
        )
        for player, amount, kind in (
            (small_blind, table.config.small_blind, "small_blind"),
            (big_blind, table.config.big_blind, "big_blind"),
        ):
            paid = min(player.stack, amount)
            player.stack -= paid
            member = hand.players[player.id]
            member.street_commit = member.hand_commit = paid
            if player.stack == 0:
                member.status = PlayerStatus.ALL_IN
            hand.action_history.append(PublicActionRecord(
                len(hand.action_history) + 1, HandPhase.PREFLOP, player.id,
                kind, paid, member.street_commit, player.stack,
            ))

        deal_order = self._after_seat(players, button.seat)
        for _ in range(2):
            for player in deal_order:
                hand.players[player.id].hole_cards += hand.deck.draw()

        pending = [
            p.id for p in self._after_seat(players, big_blind.seat)
            if hand.players[p.id].status is PlayerStatus.ACTIVE
        ]
        hand.betting = BettingRound(
            current_bet=table.config.big_blind,
            last_full_raise=table.config.big_blind,
            full_opening_established=True,
            pending=pending,
        )
        hand.phase = HandPhase.PREFLOP
        table.button_seat = button.seat
        table.hand = hand
        get_logger("server.engine").bind(hand_id=hand.id).emit("INFO", "hand.started", "Hand started", {"players": len(players), "button_seat": button.seat})
        self._advance_until_waiting(table)
        self._assert_invariants(table)

    @staticmethod
    def _after_seat(players: list[Player], seat: int) -> list[Player]:
        """Clockwise order over an already sorted list, skipping empty seats."""
        return [p for p in players if p.seat > seat] + [p for p in players if p.seat <= seat]

    def _positions(self, table: Table, players: list[Player]) -> tuple[Player, Player, Player]:
        button = (
            players[0] if table.button_seat is None
            else self._after_seat(players, table.button_seat)[0]
        )
        after_button = self._after_seat(players, button.seat)
        if len(players) == 2:
            previous = table.hand
            if (
                previous is not None and len(previous.players) > 2
                and after_button[0].seat == previous.big_blind_seat
            ):
                # On a transition to heads-up, the previous BB becomes the button/SB.
                button = after_button[0]
                after_button = self._after_seat(players, button.seat)
            return button, button, after_button[0]
        return button, after_button[0], after_button[1]

    def act(
        self, table: Table, actor_id: PlayerId, hand_id: HandId, action: PlayerAction,
    ) -> None:
        table.player(actor_id)
        if table.hand is None or table.hand.id != hand_id:
            raise RuleViolation(ErrorCode.HAND_MISMATCH, "Action must target the current hand")
        self._assert_invariants(table)
        hand = table.hand
        previous_stack = table.player(actor_id).stack
        self._betting.apply(table, actor_id, action)
        hand.action_history.append(PublicActionRecord(
            len(hand.action_history) + 1, hand.phase, actor_id, action.kind.value,
            previous_stack - table.player(actor_id).stack,
            hand.players[actor_id].street_commit, table.player(actor_id).stack,
        ))
        self._advance_until_waiting(table)
        self._assert_invariants(table)

    def legal_actions(self, table: Table, actor_id: PlayerId) -> tuple[ActionOption, ...]:
        if table.hand is None or table.hand.betting.actor_id != actor_id:
            return ()
        return self._betting.legal_actions(table, actor_id)

    def _advance_until_waiting(self, table: Table) -> None:
        """Run automatic work in this command until a player must choose."""
        hand = table.hand
        assert hand is not None
        while hand.phase is not HandPhase.COMPLETE:
            live = [pid for pid, member in hand.players.items() if member.status is not PlayerStatus.FOLDED]
            active = [pid for pid in live if hand.players[pid].status is PlayerStatus.ACTIVE]
            if len(live) == 1:
                hand.betting.pending.clear()
                hand.betting.actor_id = None
                self._pots.refund_uncalled(table)
                hand.phase = HandPhase.SETTLEMENT
                self._pots.settle(table, self._evaluator)
                return
            if len(active) <= 1:
                if active:
                    actor_id = active[0]
                    target = max(hand.players[pid].street_commit for pid in live if pid != actor_id)
                    if hand.players[actor_id].street_commit < target:
                        hand.betting.pending = [actor_id]
                        hand.betting.actor_id = actor_id
                        return
                self._pots.refund_uncalled(table)
                while hand.phase in (HandPhase.PREFLOP, HandPhase.FLOP, HandPhase.TURN):
                    self._next_street(table)
                hand.betting.pending.clear()
                hand.betting.actor_id = None
                hand.phase = HandPhase.SHOWDOWN
                self._pots.settle(table, self._evaluator)
                return
            if hand.betting.pending:
                hand.betting.actor_id = hand.betting.pending[0]
                return
            assert self._betting.round_complete(table), "A street cannot end with unpaid calls"
            self._pots.refund_uncalled(table)
            if hand.phase is HandPhase.RIVER:
                hand.betting.actor_id = None
                hand.phase = HandPhase.SHOWDOWN
                self._pots.settle(table, self._evaluator)
                return
            self._next_street(table)

    def _next_street(self, table: Table) -> None:
        hand = table.hand
        assert hand is not None
        next_phase, count = {
            HandPhase.PREFLOP: (HandPhase.FLOP, 3),
            HandPhase.FLOP: (HandPhase.TURN, 1),
            HandPhase.TURN: (HandPhase.RIVER, 1),
        }[hand.phase]
        hand.deck.burn()
        hand.board.extend(hand.deck.draw(count))
        hand.phase = next_phase
        get_logger("server.engine").bind(hand_id=hand.id).emit("INFO", "street.advanced", "Street advanced", {"phase": next_phase.value})
        for member in hand.players.values():
            member.street_commit = 0
        players = sorted((p for p in table.players if p.id in hand.players), key=lambda p: p.seat)
        pending = [p.id for p in self._after_seat(players, hand.button_seat)
                   if hand.players[p.id].status is PlayerStatus.ACTIVE]
        hand.betting = BettingRound(
            actor_id=next(iter(pending), None),
            last_full_raise=table.config.big_blind,
            pending=pending,
        )

    @staticmethod
    def _assert_invariants(table: Table) -> None:
        hand = table.hand
        assert hand is not None
        assert len(hand.players) >= 2
        assert set(hand.players) <= {p.id for p in table.players}
        assert all(p.stack >= 0 for p in table.players), "Negative chip balance"
        assert all(0 <= m.street_commit <= m.hand_commit for m in hand.players.values())
        assert any(m.status is not PlayerStatus.FOLDED for m in hand.players.values())
        assert all(len(m.hole_cards) == 2 for m in hand.players.values())
        cards = hand.deck.remaining + hand.deck.burned + hand.board + [
            card for member in hand.players.values() for card in member.hole_cards
        ]
        assert len(cards) == len(set(cards)) == 52, "Cards must partition the standard deck"
        assert len(hand.board) in (0, 3, 4, 5)
        assert len(hand.deck.burned) == {0: 0, 3: 1, 4: 2, 5: 3}[len(hand.board)]
        stacks = sum(table.player(pid).stack for pid in hand.players)
        pending = hand.betting.pending
        assert len(pending) == len(set(pending))
        assert all(pid in hand.players and hand.players[pid].status is PlayerStatus.ACTIVE for pid in pending)
        if hand.phase is HandPhase.COMPLETE:
            assert not pending and hand.betting.actor_id is None
            assert stacks == hand.chip_total_at_start, "Paid hands count stacks only"
            result = table.last_result
            assert result is not None and result.hand_id == hand.id
            assert sum(a.pot.amount for a in result.awards) == sum(m.hand_commit for m in hand.players.values())
            assert all(
                sum(s.amount for s in award.shares) == award.pot.amount
                and all(s.amount >= 0 and s.player_id in award.pot.eligible_ids for s in award.shares)
                for award in result.awards
            )
            assert all(
                pid in hand.players and hand.players[pid].status is not PlayerStatus.FOLDED
                and shown == hand.players[pid].hole_cards for pid, shown in result.revealed_hands.items()
            )
        else:
            assert hand.phase in (HandPhase.PREFLOP, HandPhase.FLOP, HandPhase.TURN, HandPhase.RIVER)
            assert len(hand.board) == {
                HandPhase.PREFLOP: 0, HandPhase.FLOP: 3, HandPhase.TURN: 4, HandPhase.RIVER: 5,
            }[hand.phase]
            assert stacks + sum(m.hand_commit for m in hand.players.values()) == hand.chip_total_at_start
            assert all(
                (m.status is PlayerStatus.ALL_IN and table.player(pid).stack == 0)
                or (m.status is PlayerStatus.ACTIVE and table.player(pid).stack > 0)
                or m.status is PlayerStatus.FOLDED for pid, m in hand.players.items()
            )
            assert hand.betting.actor_id in pending, "Waiting hand must have an active actor"
