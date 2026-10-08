import unittest
from copy import deepcopy
from random import Random
from unittest.mock import patch

from poker.application.commands import ActCommand, SessionContext, StartHandCommand, StateCommand
from poker.application.service import TableService
from poker.application.views import PlayerViewBuilder
from poker.domain.cards import Deck
from poker.domain.errors import RuleViolation
from poker.domain.models import PlayerAction
from poker.domain.types import ActionKind, ErrorCode, HandPhase, PlayerId, PlayerStatus
from poker.engine.policies import FiveCardHighEvaluator, SidePotAllocator
from tests.engine.helpers import betting_table, showdown_cards
from tests.engine.test_start_hand import hand_of, opening_engine, opening_table
from tests.fixtures import MemoryRepository


class SettlementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pots = SidePotAllocator()
        self.evaluator = FiveCardHighEvaluator()

    def test_single_showdown_winner_is_paid_once_and_all_live_hands_are_public(self) -> None:
        table = betting_table((100, 100, 100), (50, 50, 50))
        showdown_cards(table, "2c 3d 7h 9s Jc", ("As Ad", "Kh Kd", "Qh Qd"))
        result = self.pots.settle(table, self.evaluator)
        self.assertEqual([p.stack for p in table.players], [250, 100, 100])
        self.assertEqual([(s.player_id, s.amount) for s in result.awards[0].shares], [(PlayerId("p0"), 150)])
        self.assertEqual(set(result.revealed_hands), {p.id for p in table.players})
        self.assertEqual(hand_of(table).phase, HandPhase.COMPLETE)
        before = deepcopy(table)
        with self.assertRaises(RuleViolation) as caught:
            self.pots.settle(table, self.evaluator)
        self.assertEqual(caught.exception.code, ErrorCode.INVALID_ACTION)
        self.assertEqual(table, before)
        opening_engine()._assert_invariants(table)

    def test_main_and_side_pots_have_different_winners(self) -> None:
        table = betting_table((0, 0, 0), (100, 300, 300))
        showdown_cards(table, "2c 3d 7h 9s Jc", ("As Ad", "Kh Kd", "Qh Qd"))
        result = self.pots.settle(table, self.evaluator)
        self.assertEqual([p.stack for p in table.players], [300, 400, 0])
        self.assertEqual([a.pot.amount for a in result.awards], [300, 400])
        self.assertEqual([[s.player_id for s in a.shares] for a in result.awards],
                         [[PlayerId("p0")], [PlayerId("p1")]])
        opening_engine()._assert_invariants(table)

    def test_two_tied_winners_split_even_pot_and_folded_cards_stay_private(self) -> None:
        table = betting_table((0, 0, 0), (10, 10, 10))
        showdown_cards(table, "Ac Ad 9h 7s 2c", ("Kh Qh", "Kd Qs", "Jh Tc"))
        hand_of(table).players[PlayerId("p2")].status = PlayerStatus.FOLDED
        result = self.pots.settle(table, self.evaluator)
        self.assertEqual([p.stack for p in table.players], [15, 15, 0])
        self.assertNotIn(PlayerId("p2"), result.revealed_hands)
        self.assertEqual(set(result.revealed_hands), {PlayerId("p0"), PlayerId("p1")})

    def test_odd_chip_goes_clockwise_after_button_independently_of_list_order(self) -> None:
        for button, extra in ((0, 1), (1, 0), (2, 0)):
            with self.subTest(button=button):
                table = betting_table((0, 0, 0), (1, 1, 1))
                showdown_cards(table, "Ac Ad 9h 7s 2c", ("Kh Qh", "Kd Qs", "Jh Tc"))
                hand_of(table).button_seat = button
                table.players.reverse()
                self.pots.settle(table, self.evaluator)
                self.assertEqual(table.player(PlayerId(f"p{extra}")).stack, 2)
                self.assertEqual(table.player(PlayerId(f"p{1 - extra}")).stack, 1)
                self.assertEqual(table.player(PlayerId("p2")).stack, 0)

    def test_side_pot_can_tie_while_short_stack_wins_main_pot(self) -> None:
        table = betting_table((0, 0, 0), (100, 300, 300))
        showdown_cards(table, "2c 3d 7h 9s Jc", ("As Ad", "Kh Kd", "Ks Kc"))
        result = self.pots.settle(table, self.evaluator)
        self.assertEqual([p.stack for p in table.players], [300, 200, 200])
        self.assertEqual(len(result.awards[1].shares), 2)

    def test_uncontested_payment_needs_no_board_or_evaluation_and_does_not_reveal(self) -> None:
        table = betting_table((950, 990), (50, 10), current_bet=50, phase=HandPhase.PREFLOP)
        hand = hand_of(table)
        hand.players[PlayerId("p1")].status = PlayerStatus.FOLDED
        hand.phase = HandPhase.SETTLEMENT
        result = self.pots.settle(table, self.evaluator)
        self.assertEqual([p.stack for p in table.players], [1010, 990])
        self.assertEqual(result.revealed_hands, {})
        self.assertEqual([(r.player_id, r.amount) for r in result.refunds], [(PlayerId("p0"), 40)])
        self.assertEqual(result.awards[0].pot.amount, 20)
        opening_engine()._assert_invariants(table)

    def test_result_copies_refund_history_and_pays_net_pots_only(self) -> None:
        table = betting_table((0, 0, 0), (100, 300, 500), current_bet=500)
        showdown_cards(table, "2c 3d 7h 9s Jc", ("As Ad", "Kh Kd", "Qh Qd"))
        result = self.pots.settle(table, self.evaluator)
        self.assertEqual([p.stack for p in table.players], [300, 400, 200])
        self.assertEqual(result.refunds, tuple(hand_of(table).refunds))
        self.assertEqual(sum(a.pot.amount for a in result.awards), 700)
        self.assertEqual(sum(p.stack for p in table.players), 900)
        opening_engine()._assert_invariants(table)


class EngineRegressionTests(unittest.TestCase):
    def test_complete_state_reads_do_not_pay_again_and_next_hand_keeps_chips(self) -> None:
        table = opening_table()
        engine = opening_engine()
        engine.start_hand(table, PlayerId("p0"))
        repository = MemoryRepository(table)
        service = TableService(table.id, repository, engine, PlayerViewBuilder())
        old_id = hand_of(table).id
        for actor in (PlayerId("p0"), PlayerId("p1")):
            self.assertTrue(service.handle(ActCommand(old_id, PlayerAction(ActionKind.FOLD)),
                                           SessionContext(actor)).ok)
        saved = repository.load(table.id)
        assert saved is not None
        for _ in range(5):
            for player in table.players:
                response = service.handle(StateCommand(), SessionContext(player.id))
                assert response.view is not None and response.view.result is not None
                self.assertEqual(response.view.phase, HandPhase.COMPLETE)
                self.assertEqual(response.view.hand_id, old_id)
                self.assertEqual(response.view.revision, 2)
                self.assertTrue(all(not p.revealed_cards for p in response.view.players))
        self.assertEqual(repository.saves, 2)
        self.assertEqual(repository.load(table.id), saved)
        rejected = service.handle(ActCommand(old_id, PlayerAction(ActionKind.ALL_IN)), SessionContext(PlayerId("p2")))
        assert rejected.error is not None
        self.assertEqual(rejected.error.code, ErrorCode.INVALID_ACTION)
        self.assertEqual(repository.load(table.id), saved)
        response = service.handle(StartHandCommand(), SessionContext(PlayerId("p1")))
        assert response.view is not None and response.view.result is not None
        self.assertNotEqual(response.view.hand_id, old_id)
        self.assertEqual(response.view.result.hand_id, old_id)
        self.assertEqual([p.stack for p in response.view.players], [990, 995, 1000])
        self.assertTrue(all(not p.revealed_cards for p in response.view.players))
        self.assertEqual(response.view.pot_total, 15)
        self.assertEqual(repository.saves, 3)

    def test_real_illegal_actions_do_not_save_or_change_revision(self) -> None:
        table = opening_table()
        engine = opening_engine()
        engine.start_hand(table, PlayerId("p0"))
        repository = MemoryRepository(table)
        service = TableService(table.id, repository, engine, PlayerViewBuilder())
        for action, code in ((PlayerAction(ActionKind.CHECK), ErrorCode.INVALID_ACTION),
                             (PlayerAction(ActionKind.RAISE_TO, 19), ErrorCode.RAISE_TOO_SMALL),
                             (PlayerAction(ActionKind.RAISE_TO, 1001), ErrorCode.INVALID_ACTION)):
            response = service.handle(ActCommand(hand_of(table).id, action), SessionContext(PlayerId("p0")))
            assert response.error is not None
            self.assertEqual(response.error.code, code)
            self.assertEqual(repository.saves, 0)
            self.assertEqual(repository.load(table.id), table)

    def test_invariants_reject_missing_card_created_chips_and_invalid_actor(self) -> None:
        for corruption in ("card", "chips", "actor"):
            with self.subTest(corruption=corruption):
                table = opening_table()
                engine = opening_engine()
                engine.start_hand(table, PlayerId("p0"))
                hand = hand_of(table)
                if corruption == "card":
                    hand.deck.remaining.pop()
                elif corruption == "chips":
                    table.players[0].stack += 1
                else:
                    hand.betting.actor_id = PlayerId("missing")
                before = deepcopy(table)
                with self.assertRaises(AssertionError):
                    engine.act(table, PlayerId("p0"), hand.id, PlayerAction(ActionKind.CALL))
                self.assertEqual(table, before)

    def test_150_seeded_real_hands_terminate_and_conserve_cards_and_chips(self) -> None:
        rng = Random(20261007)
        shuffle = Deck.shuffled
        engine = opening_engine()
        with patch("poker.engine.holdem.Deck.shuffled", side_effect=lambda: shuffle(rng)):
            for count in range(2, 7):
                for sample in range(30):
                    with self.subTest(players=count, sample=sample):
                        stacks = tuple(rng.randint(2, 200) for _ in range(count))
                        table = opening_table(stacks)
                        rng.shuffle(table.players)
                        engine.start_hand(table, PlayerId("p0"))
                        hand = hand_of(table)
                        steps = 0
                        while hand.phase is not HandPhase.COMPLETE:
                            assert hand.betting.actor_id is not None
                            actor = hand.betting.actor_id
                            options = engine.legal_actions(table, actor)
                            option = rng.choice(options)
                            total = None
                            if option.kind in (ActionKind.BET_TO, ActionKind.RAISE_TO):
                                assert option.min_to is not None and option.max_to is not None
                                total = rng.randint(option.min_to, option.max_to)
                            engine.act(table, actor, hand.id, PlayerAction(option.kind, total))
                            steps += 1
                            self.assertLess(steps, 200, "Hand did not terminate")
                        self.assertEqual(sum(p.stack for p in table.players), sum(stacks))
                        all_cards = hand.deck.remaining + hand.deck.burned + hand.board + [
                            c for m in hand.players.values() for c in m.hole_cards
                        ]
                        self.assertEqual(len(all_cards), 52)
                        self.assertEqual(len(set(all_cards)), 52)
                        self.assertEqual(hand.betting.pending, [])
                        self.assertIsNone(hand.betting.actor_id)
