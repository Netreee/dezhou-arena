from collections import Counter
from shared_logging import get_logger

from poker.domain.cards import Card
from poker.domain.errors import RuleViolation
from poker.domain.models import (
    ActionOption, ChipShare, Hand, HandPlayer, HandResult, HandValue, Player, PlayerAction,
    Pot, PotAward, Table,
)
from poker.domain.types import ActionKind, ErrorCode, HandCategory, HandPhase, PlayerId, PlayerStatus
from poker.engine.interfaces import BettingRules, HandEvaluator, PotAllocator


class NoLimitBettingRules(BettingRules):
    def legal_actions(self, table: Table, actor_id: PlayerId) -> tuple[ActionOption, ...]:
        if table.hand is None or table.hand.betting.actor_id != actor_id:
            return ()
        hand, player, member = self._validate_actor(table, actor_id)
        owed = self._call_amount(table, actor_id)
        options = [ActionOption(ActionKind.FOLD)]
        options.append(
            ActionOption(ActionKind.CALL, pay=min(player.stack, owed))
            if owed else ActionOption(ActionKind.CHECK)
        )
        total = member.street_commit + player.stack
        can_raise = self._can_raise(table, actor_id)
        if can_raise and total > hand.betting.current_bet:
            kind = ActionKind.BET_TO if hand.betting.current_bet == 0 else ActionKind.RAISE_TO
            options.append(ActionOption(kind, min_to=min(self._minimum_total(hand), total), max_to=total))
        if (owed > 0 and player.stack <= owed) or (can_raise and total > hand.betting.current_bet):
            options.append(ActionOption(ActionKind.ALL_IN, pay=player.stack))
        return tuple(options)

    def apply(self, table: Table, actor_id: PlayerId, action: PlayerAction) -> None:
        hand, player, member = self._validate_actor(table, actor_id)
        betting = hand.betting
        owed = self._call_amount(table, actor_id)
        old_bet = betting.current_bet
        paid = 0
        total: int | None = None
        if action.to is not None and type(action.to) is not int:
            raise RuleViolation(ErrorCode.BAD_REQUEST, "Bet total must be an integer")
        if action.kind is ActionKind.FOLD:
            member.status = PlayerStatus.FOLDED
        elif action.kind is ActionKind.CHECK:
            if owed:
                raise RuleViolation(ErrorCode.INVALID_ACTION, "Cannot check while owing chips")
        elif action.kind is ActionKind.CALL:
            if not owed:
                raise RuleViolation(ErrorCode.INVALID_ACTION, "No bet to call")
            paid = min(player.stack, owed)
        elif action.kind is ActionKind.ALL_IN:
            proposed = member.street_commit + player.stack
            if proposed > betting.current_bet:
                total = proposed
            elif owed and player.stack <= owed:
                paid = player.stack
            else:
                raise RuleViolation(ErrorCode.INVALID_ACTION, "No bet to call or raise")
        elif action.kind in (ActionKind.BET_TO, ActionKind.RAISE_TO):
            if (action.kind is ActionKind.BET_TO) != (betting.current_bet == 0):
                raise RuleViolation(ErrorCode.INVALID_ACTION, "Use bet_to to open and raise_to to raise")
            assert action.to is not None
            total = action.to
        else:
            raise RuleViolation(ErrorCode.INVALID_ACTION, "Unknown poker action")

        if total is not None:
            if not self._can_raise(table, actor_id):
                raise RuleViolation(ErrorCode.RAISE_NOT_REOPENED, "Player cannot raise this bet")
            if not betting.current_bet < total <= member.street_commit + player.stack:
                raise RuleViolation(ErrorCode.INVALID_ACTION, "Raise must increase the bet within the stack")
            paid = total - member.street_commit
            if total < self._minimum_total(hand) and paid != player.stack:
                raise RuleViolation(ErrorCode.RAISE_TOO_SMALL, "Only an all-in may be smaller than a full raise")

        self._commit(player, member, paid)
        if total is not None:
            increment = total - betting.current_bet
            if increment >= betting.last_full_raise:
                betting.last_full_raise = increment
            betting.current_bet = total
            if total >= table.config.big_blind:
                betting.full_opening_established = True
        betting.last_action_bet[actor_id] = betting.current_bet
        self._update_pending(table, actor_id, betting.current_bet > old_bet)

    def round_complete(self, table: Table) -> bool:
        hand = table.hand
        if hand is None:
            return True
        return not hand.betting.pending and all(
            self._call_amount(table, player_id) == 0
            for player_id, member in hand.players.items() if member.status is PlayerStatus.ACTIVE
        )

    @staticmethod
    def _validate_actor(table: Table, actor_id: PlayerId) -> tuple[Hand, Player, HandPlayer]:
        player = table.player(actor_id)
        hand = table.hand
        if hand is None or hand.phase not in (
            HandPhase.PREFLOP, HandPhase.FLOP, HandPhase.TURN, HandPhase.RIVER,
        ):
            raise RuleViolation(ErrorCode.INVALID_ACTION, "No open betting round")
        member = hand.players.get(actor_id)
        if (
            member is None or member.status is not PlayerStatus.ACTIVE or player.stack <= 0
            or hand.betting.actor_id != actor_id or actor_id not in hand.betting.pending
        ):
            raise RuleViolation(ErrorCode.NOT_YOUR_TURN, "Player is not the current actor")
        return hand, player, member

    @staticmethod
    def _call_amount(table: Table, actor_id: PlayerId) -> int:
        hand = table.hand
        assert hand is not None
        target = hand.betting.current_bet
        if not any(
            pid != actor_id and member.status is PlayerStatus.ACTIVE
            for pid, member in hand.players.items()
        ):
            # A lone active player owes only the actual live all-in opposition,
            # rather than a nominal short big blind or an uncontestable excess.
            target = max(
                (m.street_commit for pid, m in hand.players.items()
                 if pid != actor_id and m.status is not PlayerStatus.FOLDED), default=0,
            )
        return max(0, target - hand.players[actor_id].street_commit)

    @staticmethod
    def _can_raise(table: Table, actor_id: PlayerId) -> bool:
        hand = table.hand
        assert hand is not None
        if not any(
            pid != actor_id and member.status is PlayerStatus.ACTIVE
            for pid, member in hand.players.items()
        ):
            return False
        last = hand.betting.last_action_bet.get(actor_id)
        if last is None:
            return True
        threshold = (
            hand.betting.last_full_raise if hand.betting.full_opening_established
            else table.config.big_blind
        )
        return hand.betting.current_bet - last >= threshold

    @staticmethod
    def _minimum_total(hand: Hand) -> int:
        return hand.betting.current_bet + hand.betting.last_full_raise

    @staticmethod
    def _commit(player: Player, member: HandPlayer, paid: int) -> None:
        player.stack -= paid
        member.street_commit += paid
        member.hand_commit += paid
        if player.stack == 0 and member.status is PlayerStatus.ACTIVE:
            member.status = PlayerStatus.ALL_IN

    @staticmethod
    def _update_pending(table: Table, actor_id: PlayerId, raised: bool) -> None:
        hand = table.hand
        assert hand is not None
        pending = set(hand.betting.pending) - {actor_id}
        if raised:
            pending.update(
                pid for pid, member in hand.players.items()
                if member.street_commit < hand.betting.current_bet
            )
        actor_seat = table.player(actor_id).seat
        ordered = sorted(table.players, key=lambda p: (p.seat - actor_seat - 1) % table.config.max_players)
        hand.betting.pending = [
            p.id for p in ordered if p.id in pending and p.id in hand.players
            and hand.players[p.id].status is PlayerStatus.ACTIVE
        ]
        hand.betting.actor_id = next(iter(hand.betting.pending), None)


class FiveCardHighEvaluator(HandEvaluator):
    def evaluate_five(self, cards: tuple[Card, ...]) -> HandValue:
        if len(cards) != 5 or len(set(cards)) != 5:
            raise ValueError("Expected exactly five unique cards")
        ranks = sorted((int(card.rank) for card in cards), reverse=True)
        groups = sorted(((count, rank) for rank, count in Counter(ranks).items()), reverse=True)
        flush = len({card.suit for card in cards}) == 1
        straight_high = 0
        if len(groups) == 5:
            if ranks[0] - ranks[-1] == 4:
                straight_high = ranks[0]
            elif ranks == [14, 5, 4, 3, 2]:
                straight_high = 5
        if flush and straight_high:
            return HandValue(HandCategory.STRAIGHT_FLUSH, (straight_high,))
        if groups[0][0] == 4:
            return HandValue(HandCategory.FOUR_OF_A_KIND, (groups[0][1], groups[1][1]))
        if [count for count, _ in groups] == [3, 2]:
            return HandValue(HandCategory.FULL_HOUSE, (groups[0][1], groups[1][1]))
        if flush:
            return HandValue(HandCategory.FLUSH, tuple(ranks))
        if straight_high:
            return HandValue(HandCategory.STRAIGHT, (straight_high,))
        if groups[0][0] == 3:
            return HandValue(HandCategory.THREE_OF_A_KIND, tuple(rank for _, rank in groups))
        if [count for count, _ in groups[:2]] == [2, 2]:
            return HandValue(HandCategory.TWO_PAIR, tuple(rank for _, rank in groups))
        if groups[0][0] == 2:
            return HandValue(HandCategory.PAIR, tuple(rank for _, rank in groups))
        return HandValue(HandCategory.HIGH_CARD, tuple(ranks))


class SidePotAllocator(PotAllocator):
    def refund_uncalled(self, table: Table) -> None:
        hand = table.hand
        assert hand is not None
        contributions = sorted(
            ((member.street_commit, pid) for pid, member in hand.players.items()), reverse=True,
        )
        highest, player_id = contributions[0]
        second = contributions[1][0] if len(contributions) > 1 else 0
        amount = highest - second
        if amount == 0:
            return
        member = hand.players[player_id]
        player = table.player(player_id)
        player.stack += amount
        member.street_commit -= amount
        member.hand_commit -= amount
        if member.status is PlayerStatus.ALL_IN:
            member.status = PlayerStatus.ACTIVE
        hand.refunds.append(ChipShare(player_id, amount))

    def build_pots(self, table: Table) -> tuple[Pot, ...]:
        hand = table.hand
        assert hand is not None
        players = sorted((p for p in table.players if p.id in hand.players), key=lambda p: p.seat)
        levels = sorted({member.hand_commit for member in hand.players.values() if member.hand_commit > 0})
        pots = []
        previous = 0
        for level in levels:
            contributors = [p.id for p in players if hand.players[p.id].hand_commit >= level]
            eligible = tuple(pid for pid in contributors if hand.players[pid].status is not PlayerStatus.FOLDED)
            assert eligible, "A contribution layer must have a live eligible player"
            pots.append(Pot((level - previous) * len(contributors), eligible))
            previous = level
        return tuple(pots)

    def settle(self, table: Table, evaluator: HandEvaluator) -> HandResult:
        hand = table.hand
        assert hand is not None
        if hand.phase is HandPhase.COMPLETE:
            raise RuleViolation(ErrorCode.INVALID_ACTION, "Hand has already been paid")
        self.refund_uncalled(table)
        pots = self.build_pots(table)
        contested_ids = {pid for pot in pots if len(pot.eligible_ids) > 1 for pid in pot.eligible_ids}
        values = {
            pid: evaluator.evaluate_best(hand.players[pid].hole_cards + tuple(hand.board))
            for pid in contested_ids
        }
        awards = []
        for pot in pots:
            if len(pot.eligible_ids) == 1:
                winners = list(pot.eligible_ids)
            else:
                best = max(values[pid] for pid in pot.eligible_ids)
                winners = [pid for pid in pot.eligible_ids if values[pid] == best]
            winners.sort(key=lambda pid: (table.player(pid).seat - hand.button_seat - 1) % table.config.max_players)
            amount, remainder = divmod(pot.amount, len(winners))
            shares = tuple(ChipShare(pid, amount + (1 if index < remainder else 0))
                           for index, pid in enumerate(winners))
            awards.append(PotAward(pot, shares))
        revealed = {
            pid: member.hole_cards for pid, member in hand.players.items()
            if member.status is not PlayerStatus.FOLDED
        } if contested_ids else {}
        result = HandResult(hand.id, tuple(awards), tuple(hand.refunds), revealed)
        for award in awards:
            for share in award.shares:
                table.player(share.player_id).stack += share.amount
            get_logger("server.engine").bind(hand_id=hand.id).emit("INFO", "pot.awarded", "Pot awarded", {"amount": award.pot.amount})
        table.last_result = result
        hand.betting.actor_id = None
        hand.betting.pending.clear()
        hand.phase = HandPhase.COMPLETE
        get_logger("server.engine").bind(hand_id=hand.id).emit("INFO", "hand.completed", "Hand completed", {"pot_count": len(awards), "refund_count": len(hand.refunds)})
        return result
