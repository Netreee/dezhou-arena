import unittest
from copy import deepcopy

from poker.domain.errors import RuleViolation
from poker.domain.models import PlayerAction, Table
from poker.domain.types import ActionKind, ErrorCode, HandId, HandPhase, PlayerId, PlayerStatus
from poker.engine.policies import NoLimitBettingRules
from tests.engine.helpers import betting_table
from tests.engine.test_start_hand import hand_of, opening_engine, opening_table


class BettingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = NoLimitBettingRules()

    def apply_current(self, table: Table, kind: ActionKind, to: int | None = None) -> None:
        hand = hand_of(table)
        assert hand.betting.actor_id is not None
        self.rules.apply(table, hand.betting.actor_id, PlayerAction(kind, to))

    def assert_rejected(self, table: Table, action: PlayerAction, code: ErrorCode) -> None:
        before = deepcopy(table)
        hand = hand_of(table)
        assert hand.betting.actor_id is not None
        with self.assertRaises(RuleViolation) as caught:
            self.rules.apply(table, hand.betting.actor_id, action)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(table, before)

    def test_call_pays_difference_and_updates_both_accounts(self) -> None:
        table = betting_table((100, 100), (20, 60), current_bet=60, last_full_raise=40)
        self.apply_current(table, ActionKind.CALL)
        self.assertEqual(table.players[0].stack, 60)
        member = hand_of(table).players[PlayerId("p0")]
        self.assertEqual((member.street_commit, member.hand_commit), (60, 60))
        self.assertEqual(hand_of(table).betting.last_action_bet[PlayerId("p0")], 60)

    def test_short_call_marks_all_in_and_does_not_prevent_completion(self) -> None:
        table = betting_table((25, 100), (20, 60), current_bet=60, last_full_raise=40)
        self.apply_current(table, ActionKind.CALL)
        member = hand_of(table).players[PlayerId("p0")]
        self.assertEqual((table.players[0].stack, member.street_commit, member.hand_commit), (0, 45, 45))
        self.assertEqual(member.status, PlayerStatus.ALL_IN)
        self.assertNotIn(PlayerId("p0"), hand_of(table).betting.pending)
        self.apply_current(table, ActionKind.CHECK)
        self.assertTrue(self.rules.round_complete(table))

    def test_raise_to_is_a_total_not_an_additional_payment(self) -> None:
        table = betting_table((100, 100), (20, 60), current_bet=60, last_full_raise=40)
        self.apply_current(table, ActionKind.RAISE_TO, 100)
        member = hand_of(table).players[PlayerId("p0")]
        self.assertEqual((table.players[0].stack, member.street_commit, member.hand_commit), (20, 100, 100))
        self.assertEqual((hand_of(table).betting.current_bet, hand_of(table).betting.last_full_raise), (100, 40))

    def test_underraise_requires_committing_the_entire_stack(self) -> None:
        table = betting_table((100, 100), (20, 60), current_bet=60, last_full_raise=40)
        self.assert_rejected(table, PlayerAction(ActionKind.RAISE_TO, 90), ErrorCode.RAISE_TOO_SMALL)
        table.players[0].stack = 70
        self.apply_current(table, ActionKind.RAISE_TO, 90)
        self.assertEqual(hand_of(table).betting.last_full_raise, 40)
        self.assertEqual(hand_of(table).players[PlayerId("p0")].status, PlayerStatus.ALL_IN)

    def test_short_all_in_requires_response_without_reopening_original_bettor(self) -> None:
        table = betting_table((1000, 150, 1000))
        self.apply_current(table, ActionKind.BET_TO, 100)
        self.apply_current(table, ActionKind.ALL_IN)
        self.apply_current(table, ActionKind.CALL)
        hand = hand_of(table)
        self.assertEqual(hand.betting.actor_id, PlayerId("p0"))
        self.assertEqual(hand.betting.pending, [PlayerId("p0")])
        self.assertEqual(hand.betting.last_full_raise, 100)
        options = self.rules.legal_actions(table, PlayerId("p0"))
        self.assertEqual([o.kind for o in options], [ActionKind.FOLD, ActionKind.CALL])
        self.assertEqual(options[1].pay, 50)
        for action in (PlayerAction(ActionKind.RAISE_TO, 250), PlayerAction(ActionKind.ALL_IN)):
            self.assert_rejected(table, action, ErrorCode.RAISE_NOT_REOPENED)
        self.apply_current(table, ActionKind.CALL)
        self.assertTrue(self.rules.round_complete(table))

    def test_cumulative_short_all_ins_reopen_with_last_full_increment(self) -> None:
        table = betting_table((1000, 150, 200, 1000))
        self.apply_current(table, ActionKind.BET_TO, 100)
        self.apply_current(table, ActionKind.ALL_IN)
        self.apply_current(table, ActionKind.ALL_IN)
        self.apply_current(table, ActionKind.CALL)
        hand = hand_of(table)
        self.assertEqual(hand.betting.actor_id, PlayerId("p0"))
        self.assertEqual(hand.betting.last_full_raise, 100)
        raise_option = next(o for o in self.rules.legal_actions(table, PlayerId("p0"))
                            if o.kind is ActionKind.RAISE_TO)
        self.assertEqual(raise_option.min_to, 300)
        self.apply_current(table, ActionKind.RAISE_TO, 300)
        self.assertEqual(hand.betting.pending, [PlayerId("p3")])

    def test_reopening_is_computed_separately_for_each_player(self) -> None:
        table = betting_table((1000, 125, 1000, 200, 1000))
        for kind, amount in ((ActionKind.BET_TO, 100), (ActionKind.ALL_IN, None),
                             (ActionKind.CALL, None), (ActionKind.ALL_IN, None),
                             (ActionKind.CALL, None), (ActionKind.CALL, None)):
            self.apply_current(table, kind, amount)
        hand = hand_of(table)
        self.assertEqual(hand.betting.actor_id, PlayerId("p2"))
        self.assert_rejected(table, PlayerAction(ActionKind.RAISE_TO, 300), ErrorCode.RAISE_NOT_REOPENED)

    def test_unacted_big_blind_retains_raise_option_after_a_short_increase(self) -> None:
        table = opening_table((15, 1000, 1000))
        engine = opening_engine()
        engine.start_hand(table, PlayerId("p0"))
        self.apply_current(table, ActionKind.ALL_IN)
        self.apply_current(table, ActionKind.CALL)
        hand = hand_of(table)
        self.assertEqual(hand.betting.actor_id, PlayerId("p2"))
        self.assertNotIn(PlayerId("p2"), hand.betting.last_action_bet)
        option = next(o for o in self.rules.legal_actions(table, PlayerId("p2"))
                      if o.kind is ActionKind.RAISE_TO)
        self.assertEqual(option.min_to, 25)

    def test_short_postflop_opening_preserves_minimum_and_checked_players_rights(self) -> None:
        table = betting_table((1000, 5, 1000))
        self.apply_current(table, ActionKind.CHECK)
        self.apply_current(table, ActionKind.ALL_IN)
        hand = hand_of(table)
        self.assertFalse(hand.betting.full_opening_established)
        self.assertEqual(hand.betting.last_full_raise, 10)
        option = next(o for o in self.rules.legal_actions(table, PlayerId("p2"))
                      if o.kind is ActionKind.RAISE_TO)
        self.assertEqual(option.min_to, 15)
        self.apply_current(table, ActionKind.CALL)
        self.assert_rejected(table, PlayerAction(ActionKind.RAISE_TO, 15), ErrorCode.RAISE_NOT_REOPENED)

    def test_full_raise_over_short_opening_reopens_a_prior_check(self) -> None:
        table = betting_table((1000, 5, 1000))
        self.apply_current(table, ActionKind.CHECK)
        self.apply_current(table, ActionKind.ALL_IN)
        self.apply_current(table, ActionKind.RAISE_TO, 15)
        hand = hand_of(table)
        self.assertTrue(hand.betting.full_opening_established)
        self.assertEqual(hand.betting.last_full_raise, 10)
        option = next(o for o in self.rules.legal_actions(table, PlayerId("p0"))
                      if o.kind is ActionKind.RAISE_TO)
        self.assertEqual(option.min_to, 25)

    def test_cumulative_short_opening_reopens_check_when_it_reaches_big_blind(self) -> None:
        table = betting_table((1000, 5, 10, 1000))
        for kind in (ActionKind.CHECK, ActionKind.ALL_IN, ActionKind.ALL_IN, ActionKind.CALL):
            self.apply_current(table, kind)
        hand = hand_of(table)
        self.assertEqual(hand.betting.actor_id, PlayerId("p0"))
        self.assertTrue(hand.betting.full_opening_established)
        self.assertEqual(hand.betting.last_full_raise, 10)
        option = next(o for o in self.rules.legal_actions(table, PlayerId("p0"))
                      if o.kind is ActionKind.RAISE_TO)
        self.assertEqual(option.min_to, 20)

    def test_prior_call_of_short_opening_is_not_reopened_by_a_smaller_increase(self) -> None:
        table = betting_table((5, 1000, 10, 1000))
        for kind in (ActionKind.ALL_IN, ActionKind.CALL, ActionKind.ALL_IN, ActionKind.CALL):
            self.apply_current(table, kind)
        hand = hand_of(table)
        self.assertEqual(hand.betting.actor_id, PlayerId("p1"))
        self.assertEqual(hand.betting.last_full_raise, 10)
        self.assert_rejected(table, PlayerAction(ActionKind.RAISE_TO, 20), ErrorCode.RAISE_NOT_REOPENED)

    def test_legal_actions_are_read_only_and_include_exact_short_all_in(self) -> None:
        table = betting_table((70, 100), (20, 60), current_bet=60, last_full_raise=40)
        before = deepcopy(table)
        options = self.rules.legal_actions(table, PlayerId("p0"))
        option = next(o for o in options if o.kind is ActionKind.RAISE_TO)
        self.assertEqual((option.min_to, option.max_to), (90, 90))
        self.assertEqual(table, before)
        self.assertEqual(self.rules.legal_actions(table, PlayerId("p1")), ())

    def test_check_call_and_amount_errors_do_not_mutate(self) -> None:
        table = betting_table((100, 100), (20, 60), current_bet=60, last_full_raise=40)
        for action in (PlayerAction(ActionKind.CHECK), PlayerAction(ActionKind.BET_TO, 100),
                       PlayerAction(ActionKind.RAISE_TO, 60), PlayerAction(ActionKind.RAISE_TO, 121)):
            self.assert_rejected(table, action, ErrorCode.INVALID_ACTION)
        table = betting_table((100, 100))
        self.assert_rejected(table, PlayerAction(ActionKind.CALL), ErrorCode.INVALID_ACTION)
        self.assert_rejected(table, PlayerAction(ActionKind.RAISE_TO, 10), ErrorCode.INVALID_ACTION)
        self.assert_rejected(table, PlayerAction(ActionKind.BET_TO, True), ErrorCode.BAD_REQUEST)
        self.apply_current(table, ActionKind.CHECK)
        self.assertNotIn(PlayerId("p0"), hand_of(table).betting.pending)

    def test_fold_keeps_contributions_and_pending_follows_clockwise_seats(self) -> None:
        table = betting_table((100, 100, 100), (20, 60, 60), current_bet=60)
        self.apply_current(table, ActionKind.FOLD)
        hand = hand_of(table)
        self.assertEqual(hand.players[PlayerId("p0")].status, PlayerStatus.FOLDED)
        self.assertEqual(hand.players[PlayerId("p0")].hand_commit, 20)
        self.assertEqual(hand.betting.pending, [PlayerId("p1"), PlayerId("p2")])
        self.assertEqual(hand.betting.actor_id, PlayerId("p1"))

    def test_sole_active_player_only_calls_actual_short_blind_and_cannot_overbet(self) -> None:
        table = opening_table((1000, 5, 3))
        opening_engine().start_hand(table, PlayerId("p0"))
        options = self.rules.legal_actions(table, PlayerId("p0"))
        self.assertEqual([o.kind for o in options], [ActionKind.FOLD, ActionKind.CALL])
        self.assertEqual(options[1].pay, 5)
        self.assert_rejected(table, PlayerAction(ActionKind.ALL_IN), ErrorCode.RAISE_NOT_REOPENED)
        self.apply_current(table, ActionKind.CALL)
        self.assertEqual(table.players[0].stack, 995)
        self.assertTrue(self.rules.round_complete(table))
        table = betting_table((8, 0), (0, 3), current_bet=10, phase=HandPhase.PREFLOP)
        self.assert_rejected(table, PlayerAction(ActionKind.ALL_IN), ErrorCode.INVALID_ACTION)

    def test_wrong_actor_and_stale_hand_are_rejected_before_mutation(self) -> None:
        table = opening_table()
        engine = opening_engine()
        engine.start_hand(table, PlayerId("p0"))
        hand = hand_of(table)
        before = deepcopy(table)
        for actor, hid, code in ((PlayerId("p1"), hand.id, ErrorCode.NOT_YOUR_TURN),
                                 (PlayerId("p0"), HandId("stale"), ErrorCode.HAND_MISMATCH)):
            with self.assertRaises(RuleViolation) as caught:
                engine.act(table, actor, hid, PlayerAction(ActionKind.CALL))
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(table, before)
