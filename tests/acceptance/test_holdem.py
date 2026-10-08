"""Real E01-E06 poker acceptance specifications."""

import unittest

from poker.domain.cards import Card
from poker.domain.models import Player, PlayerAction, Table
from poker.domain.types import ActionKind, HandCategory, HandPhase, PlayerId, TableId
from poker.engine.holdem import HoldemEngine
from poker.engine.policies import FiveCardHighEvaluator, NoLimitBettingRules, SidePotAllocator


class HoldemAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = HoldemEngine(NoLimitBettingRules(), FiveCardHighEvaluator(), SidePotAllocator())
        self.table = Table(TableId("acceptance"))
        for index, stack in enumerate((1000, 1000, 1000)):
            self.table.add_player(Player(PlayerId(f"p{index}"), f"P{index}", index, stack))

    def _act_current(self, kind: ActionKind) -> None:
        hand = self.table.hand
        assert hand is not None and hand.betting.actor_id is not None
        self.engine.act(self.table, hand.betting.actor_id, hand.id, PlayerAction(kind))

    def test_limped_preflop_preserves_big_blind_option(self) -> None:
        self.engine.start_hand(self.table, PlayerId("p0"))
        self._act_current(ActionKind.CALL)
        self._act_current(ActionKind.CALL)
        hand = self.table.hand
        assert hand is not None and hand.betting.actor_id is not None
        self.assertEqual(hand.phase, HandPhase.PREFLOP)
        self.assertEqual(self.table.player(hand.betting.actor_id).seat, hand.big_blind_seat)
        options = self.engine.legal_actions(self.table, hand.betting.actor_id)
        self.assertIn(ActionKind.CHECK, [option.kind for option in options])

    def test_uncontested_win_ends_hand_without_revealing_hole_cards(self) -> None:
        self.engine.start_hand(self.table, PlayerId("p0"))
        self._act_current(ActionKind.FOLD)
        self._act_current(ActionKind.FOLD)
        assert self.table.hand is not None and self.table.last_result is not None
        self.assertEqual(self.table.hand.phase, HandPhase.COMPLETE)
        self.assertEqual(self.table.hand.board, [])
        self.assertEqual(self.table.last_result.revealed_hands, {})
        self.assertEqual(sum(p.stack for p in self.table.players), 3000)

    def test_three_all_ins_create_two_pots_and_refund_200(self) -> None:
        for player, stack in zip(self.table.players, (500, 100, 300), strict=True):
            player.stack = stack
        self.engine.start_hand(self.table, PlayerId("p0"))
        self._act_current(ActionKind.ALL_IN)
        self._act_current(ActionKind.ALL_IN)
        self._act_current(ActionKind.ALL_IN)
        assert self.table.last_result is not None and self.table.hand is not None
        self.assertEqual(self.table.hand.phase, HandPhase.COMPLETE)
        self.assertEqual([a.pot.amount for a in self.table.last_result.awards], [300, 400])
        self.assertEqual([(r.player_id, r.amount) for r in self.table.last_result.refunds], [(PlayerId("p0"), 200)])
        self.assertEqual(sum(p.stack for p in self.table.players), 900)

    def test_wheel_straight_uses_five_as_high_card(self) -> None:
        cards = tuple(Card.parse(code) for code in ("As", "2d", "3h", "4c", "5s"))
        value = FiveCardHighEvaluator().evaluate_five(cards)
        self.assertEqual((value.category, value.kickers), (HandCategory.STRAIGHT, (5,)))
