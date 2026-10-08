import unittest

from poker.domain.errors import RuleViolation
from poker.domain.models import PlayerAction, Table
from poker.domain.types import ActionKind, ErrorCode, HandPhase, PlayerId
from tests.engine.test_start_hand import hand_of, opening_engine, opening_table


class AdvancementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = opening_engine()

    def act_current(self, table: Table, kind: ActionKind, to: int | None = None) -> None:
        hand = hand_of(table)
        assert hand.betting.actor_id is not None
        self.engine.act(table, hand.betting.actor_id, hand.id, PlayerAction(kind, to))

    def limp_to_flop(self, table: Table) -> None:
        self.engine.start_hand(table, PlayerId("p0"))
        for _ in range(len(table.players) - 1):
            self.act_current(table, ActionKind.CALL)
        self.act_current(table, ActionKind.CHECK)

    def test_limpers_do_not_skip_big_blind_and_check_advances_exactly_one_street(self) -> None:
        table = opening_table()
        self.engine.start_hand(table, PlayerId("p0"))
        self.act_current(table, ActionKind.CALL)
        self.act_current(table, ActionKind.CALL)
        hand = hand_of(table)
        self.assertEqual(hand.phase, HandPhase.PREFLOP)
        self.assertEqual(hand.betting.actor_id, PlayerId("p2"))
        self.assertIn(ActionKind.CHECK, [o.kind for o in self.engine.legal_actions(table, PlayerId("p2"))])
        self.assertIn(ActionKind.RAISE_TO, [o.kind for o in self.engine.legal_actions(table, PlayerId("p2"))])
        self.act_current(table, ActionKind.CHECK)
        self.assertEqual(hand.phase, HandPhase.FLOP)
        self.assertEqual((len(hand.board), len(hand.deck.burned)), (3, 1))
        self.assertEqual(hand.betting.actor_id, PlayerId("p1"))
        self.assertEqual(hand.betting.pending, [PlayerId("p1"), PlayerId("p2"), PlayerId("p0")])
        self.assertEqual([m.street_commit for m in hand.players.values()], [0, 0, 0])
        self.assertEqual([m.hand_commit for m in hand.players.values()], [10, 10, 10])
        self.assertEqual((hand.betting.current_bet, hand.betting.last_full_raise,
                          hand.betting.full_opening_established, hand.betting.last_action_bet), (0, 10, False, {}))

    def test_checks_advance_flop_and_turn_one_street_at_a_time(self) -> None:
        table = opening_table()
        self.limp_to_flop(table)
        hand = hand_of(table)
        for expected, cards, burns in ((HandPhase.TURN, 4, 2), (HandPhase.RIVER, 5, 3)):
            for _ in range(3):
                self.act_current(table, ActionKind.CHECK)
            self.assertEqual((hand.phase, len(hand.board), len(hand.deck.burned)), (expected, cards, burns))
            self.assertEqual(hand.betting.actor_id, PlayerId("p1"))
            self.assertEqual(hand.betting.last_action_bet, {})
        self.assertEqual(len(hand.deck.remaining), 38)

    def test_heads_up_big_blind_acts_first_after_flop(self) -> None:
        table = opening_table((1000, 1000))
        self.limp_to_flop(table)
        self.assertEqual(hand_of(table).betting.pending, [PlayerId("p1"), PlayerId("p0")])
        self.assertEqual(hand_of(table).betting.actor_id, PlayerId("p1"))

    def test_short_big_blind_keeps_nominal_calls_while_two_players_can_bet(self) -> None:
        table = opening_table((1000, 1000, 3))
        self.engine.start_hand(table, PlayerId("p0"))
        self.assertEqual(next(o.pay for o in self.engine.legal_actions(table, PlayerId("p0"))
                              if o.kind is ActionKind.CALL), 10)
        self.act_current(table, ActionKind.CALL)
        self.act_current(table, ActionKind.CALL)
        hand = hand_of(table)
        self.assertEqual(hand.phase, HandPhase.FLOP)
        self.assertEqual([m.hand_commit for m in hand.players.values()], [10, 10, 3])
        self.assertEqual(hand.betting.pending, [PlayerId("p1"), PlayerId("p0")])

    def test_folds_finish_without_runout_or_public_hole_cards(self) -> None:
        table = opening_table()
        self.engine.start_hand(table, PlayerId("p0"))
        self.act_current(table, ActionKind.FOLD)
        self.act_current(table, ActionKind.FOLD)
        hand = hand_of(table)
        assert table.last_result is not None
        self.assertEqual(hand.phase, HandPhase.COMPLETE)
        self.assertEqual(hand.board, [])
        self.assertEqual(hand.deck.burned, [])
        self.assertEqual(table.last_result.revealed_hands, {})
        self.assertEqual([(r.player_id, r.amount) for r in hand.refunds], [(PlayerId("p2"), 5)])
        self.assertEqual([p.stack for p in table.players], [1000, 995, 1005])
        self.assertEqual(hand.betting.pending, [])
        self.assertIsNone(hand.betting.actor_id)

    def test_lone_active_player_must_choose_before_short_call_runs_out(self) -> None:
        table = opening_table((6, 10))
        self.engine.start_hand(table, PlayerId("p0"))
        self.assertEqual(hand_of(table).phase, HandPhase.PREFLOP)
        self.assertEqual(hand_of(table).board, [])
        self.assertEqual([o.kind for o in self.engine.legal_actions(table, PlayerId("p0"))],
                         [ActionKind.FOLD, ActionKind.CALL, ActionKind.ALL_IN])
        self.act_current(table, ActionKind.CALL)
        hand = hand_of(table)
        self.assertEqual(hand.phase, HandPhase.COMPLETE)
        self.assertEqual((len(hand.board), len(hand.deck.burned)), (5, 3))
        self.assertEqual([(r.player_id, r.amount) for r in hand.refunds], [(PlayerId("p1"), 4)])
        self.assertEqual(sum(p.stack for p in table.players), 16)

    def test_three_all_ins_run_out_in_last_action_and_refund_excess(self) -> None:
        table = opening_table((500, 100, 300))
        self.engine.start_hand(table, PlayerId("p0"))
        for _ in range(3):
            self.act_current(table, ActionKind.ALL_IN)
        hand = hand_of(table)
        assert table.last_result is not None
        self.assertEqual(hand.phase, HandPhase.COMPLETE)
        self.assertEqual((len(hand.board), len(hand.deck.burned)), (5, 3))
        self.assertEqual([a.pot.amount for a in table.last_result.awards], [300, 400])
        self.assertEqual([(r.player_id, r.amount) for r in hand.refunds], [(PlayerId("p0"), 200)])
        self.assertEqual(sum(p.stack for p in table.players), 900)

    def test_complete_rejects_additional_actions(self) -> None:
        table = opening_table()
        self.engine.start_hand(table, PlayerId("p0"))
        self.act_current(table, ActionKind.FOLD)
        self.act_current(table, ActionKind.FOLD)
        hand = hand_of(table)
        for player in table.players:
            self.assertEqual(self.engine.legal_actions(table, player.id), ())
            with self.assertRaises(RuleViolation) as caught:
                self.engine.act(table, player.id, hand.id, PlayerAction(ActionKind.ALL_IN))
            self.assertEqual(caught.exception.code, ErrorCode.INVALID_ACTION)
