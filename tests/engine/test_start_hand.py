"""E01 opening tests using the real engine, without betting or settlement doubles."""

import unittest
from copy import deepcopy
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from unittest.mock import patch

from poker.application.commands import SessionContext, StartHandCommand, StateCommand
from poker.application.service import TableService
from poker.application.views import PlayerViewBuilder
from poker.domain.cards import Deck
from poker.domain.errors import RuleViolation
from poker.domain.models import Hand, HandResult, Player, Table
from poker.domain.types import ErrorCode, HandPhase, PlayerId, PlayerStatus, TableId
from poker.engine.holdem import HoldemEngine
from poker.engine.policies import FiveCardHighEvaluator, NoLimitBettingRules, SidePotAllocator
from poker.persistence.sqlite import SqliteTableRepository
from tests.fixtures import MemoryRepository


def opening_table(
    stacks: tuple[int, ...] = (1000, 1000, 1000),
    seats: tuple[int, ...] | None = None,
) -> Table:
    table = Table(TableId("opening"))
    for seat, stack in zip(seats if seats is not None else range(len(stacks)), stacks, strict=True):
        table.add_player(Player(PlayerId(f"p{seat}"), f"P{seat}", seat, stack))
    return table


def opening_engine() -> HoldemEngine:
    return HoldemEngine(NoLimitBettingRules(), FiveCardHighEvaluator(), SidePotAllocator())


def hand_of(table: Table) -> Hand:
    assert table.hand is not None
    return table.hand


def complete_input_fixture(table: Table) -> None:
    """Provide a prior-hand input for rotation tests; no settlement is exercised."""
    hand = hand_of(table)
    hand.phase = HandPhase.COMPLETE
    hand.betting.actor_id = None
    hand.betting.pending.clear()
    table.last_result = HandResult(hand.id, ())


class StartHandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = opening_engine()

    def test_three_player_opening_posts_blinds_and_preserves_big_blind_turn(self) -> None:
        table = opening_table()
        self.engine.start_hand(table, PlayerId("p1"))
        hand = hand_of(table)
        self.assertEqual(hand.phase, HandPhase.PREFLOP)
        self.assertEqual((table.button_seat, hand.button_seat, hand.small_blind_seat, hand.big_blind_seat),
                         (0, 0, 1, 2))
        self.assertEqual(hand.betting.actor_id, PlayerId("p0"))
        self.assertEqual(hand.betting.pending, [PlayerId("p0"), PlayerId("p1"), PlayerId("p2")])
        self.assertEqual(hand.betting.last_action_bet, {})
        self.assertEqual((hand.betting.current_bet, hand.betting.last_full_raise,
                          hand.betting.full_opening_established), (10, 10, True))
        self.assertEqual([p.stack for p in table.players], [1000, 995, 990])
        self.assertEqual([m.street_commit for m in hand.players.values()], [0, 5, 10])
        self.assertEqual([m.hand_commit for m in hand.players.values()], [0, 5, 10])
        self.assertEqual(sum(m.hand_commit for m in hand.players.values()), 15)
        self.assertEqual(hand.chip_total_at_start, 3000)
        self.assertEqual(sum(p.stack for p in table.players) + 15, 3000)
        self.assertEqual(hand.board, [])
        self.assertEqual(hand.deck.burned, [])
        self.assertTrue(all(m.status is PlayerStatus.ACTIVE for m in hand.players.values()))

    def test_two_dealing_rounds_start_after_button(self) -> None:
        table = opening_table()
        deck = Deck.shuffled(Random(23))
        original = tuple(deck.remaining)
        # Fix only the shuffled card order; all setup and dealing logic is real.
        with patch("poker.engine.holdem.Deck.shuffled", return_value=deck):
            self.engine.start_hand(table, PlayerId("p0"))
        hand = hand_of(table)
        for seat, offset in ((1, 0), (2, 1), (0, 2)):
            self.assertEqual(hand.players[PlayerId(f"p{seat}")].hole_cards,
                             (original[offset], original[offset + 3]))
        self.assertEqual(tuple(hand.deck.remaining), original[6:])

    def test_card_partition_is_52_unique_cards_for_two_through_six_players(self) -> None:
        for count in range(2, 7):
            with self.subTest(players=count):
                table = opening_table((1000,) * count)
                self.engine.start_hand(table, PlayerId("p0"))
                hand = hand_of(table)
                holes = [card for member in hand.players.values() for card in member.hole_cards]
                self.assertTrue(all(len(m.hole_cards) == 2 for m in hand.players.values()))
                self.assertEqual(len(hand.deck.remaining), 52 - 2 * count)
                cards = holes + hand.board + hand.deck.burned + hand.deck.remaining
                self.assertEqual(len(cards), 52)
                self.assertEqual(len(set(cards)), 52)

    def test_unordered_sparse_seats_skip_busted_players(self) -> None:
        table = opening_table((1000, 1000, 1000, 0), (5, 0, 3, 2))
        self.engine.start_hand(table, PlayerId("p2"))
        hand = hand_of(table)
        self.assertEqual((hand.button_seat, hand.small_blind_seat, hand.big_blind_seat), (0, 3, 5))
        self.assertEqual(list(hand.players), [PlayerId("p0"), PlayerId("p3"), PlayerId("p5")])
        self.assertEqual(hand.betting.pending, [PlayerId("p0"), PlayerId("p3"), PlayerId("p5")])
        self.assertNotIn(PlayerId("p2"), hand.players)
        self.assertEqual(table.player(PlayerId("p2")).stack, 0)
        self.assertEqual(hand.chip_total_at_start, 3000)

    def test_heads_up_button_posts_small_blind_and_acts_first(self) -> None:
        table = opening_table((1000, 1000), (1, 4))
        self.engine.start_hand(table, PlayerId("p4"))
        hand = hand_of(table)
        self.assertEqual((hand.button_seat, hand.small_blind_seat, hand.big_blind_seat), (1, 1, 4))
        self.assertEqual(hand.betting.actor_id, PlayerId("p1"))
        self.assertEqual(hand.betting.pending, [PlayerId("p1"), PlayerId("p4")])
        self.assertEqual([p.stack for p in table.players], [995, 990])

    def test_short_blinds_pay_actual_stacks_and_keep_nominal_big_blind(self) -> None:
        table = opening_table((1000, 5, 3))
        self.engine.start_hand(table, PlayerId("p0"))
        hand = hand_of(table)
        self.assertEqual([p.stack for p in table.players], [1000, 0, 0])
        self.assertEqual([m.hand_commit for m in hand.players.values()], [0, 5, 3])
        self.assertEqual([m.street_commit for m in hand.players.values()], [0, 5, 3])
        self.assertEqual(hand.players[PlayerId("p1")].status, PlayerStatus.ALL_IN)
        self.assertEqual(hand.players[PlayerId("p2")].status, PlayerStatus.ALL_IN)
        self.assertEqual(hand.betting.pending, [PlayerId("p0")])
        self.assertEqual(hand.betting.actor_id, PlayerId("p0"))
        self.assertEqual((hand.betting.current_bet, hand.betting.last_full_raise,
                          hand.betting.full_opening_established), (10, 10, True))
        self.assertEqual(sum(p.stack for p in table.players) + 8, hand.chip_total_at_start)

    def test_both_blinds_can_be_shorter_than_configured_amounts(self) -> None:
        table = opening_table((1000, 2, 3))
        self.engine.start_hand(table, PlayerId("p0"))
        hand = hand_of(table)
        self.assertEqual([m.hand_commit for m in hand.players.values()], [0, 2, 3])
        self.assertEqual(hand.betting.actor_id, PlayerId("p0"))
        self.assertEqual(hand.betting.current_bet, 10)
        self.assertEqual(sum(p.stack for p in table.players) + 5, hand.chip_total_at_start)

    def test_short_big_blind_is_excluded_from_pending_with_two_active_players(self) -> None:
        table = opening_table((1000, 1000, 3))
        self.engine.start_hand(table, PlayerId("p0"))
        hand = hand_of(table)
        self.assertEqual(hand.betting.pending, [PlayerId("p0"), PlayerId("p1")])
        self.assertEqual(hand.players[PlayerId("p2")].status, PlayerStatus.ALL_IN)
        self.assertEqual(hand.players[PlayerId("p2")].street_commit, 3)
        self.assertEqual(hand.betting.current_bet, 10)

    def test_sole_active_player_with_call_obligation_still_gets_a_turn(self) -> None:
        table = opening_table((6, 10))
        self.engine.start_hand(table, PlayerId("p1"))
        hand = hand_of(table)
        self.assertEqual([p.stack for p in table.players], [1, 0])
        self.assertEqual(hand.betting.pending, [PlayerId("p0")])
        self.assertEqual(hand.betting.actor_id, PlayerId("p0"))
        self.assertEqual(hand.phase, HandPhase.PREFLOP)
        self.assertEqual(hand.board, [])

    def test_not_enough_funded_players_rejects_before_mutation(self) -> None:
        for stacks in ((1000,), (1000, 0), (0, 0)):
            with self.subTest(stacks=stacks):
                table = opening_table(stacks)
                before = deepcopy(table)
                with self.assertRaises(RuleViolation) as caught:
                    self.engine.start_hand(table, PlayerId("p0"))
                self.assertEqual(caught.exception.code, ErrorCode.NOT_ENOUGH_PLAYERS)
                self.assertEqual(table, before)

    def test_unseated_requester_rejects_before_mutation(self) -> None:
        table = opening_table()
        before = deepcopy(table)
        with self.assertRaises(RuleViolation) as caught:
            self.engine.start_hand(table, PlayerId("missing"))
        self.assertEqual(caught.exception.code, ErrorCode.NOT_SEATED)
        self.assertEqual(table, before)

    def test_hand_in_progress_is_not_overwritten(self) -> None:
        table = opening_table()
        self.engine.start_hand(table, PlayerId("p0"))
        old_hand = hand_of(table)
        before = deepcopy(table)
        with self.assertRaises(RuleViolation) as caught:
            self.engine.start_hand(table, PlayerId("p1"))
        self.assertEqual(caught.exception.code, ErrorCode.HAND_IN_PROGRESS)
        self.assertIs(table.hand, old_hand)
        self.assertEqual(table, before)

    def test_next_hand_rotates_button_and_resets_hand_state(self) -> None:
        table = opening_table()
        self.engine.start_hand(table, PlayerId("p0"))
        previous = hand_of(table)
        complete_input_fixture(table)
        result = table.last_result
        chip_total = sum(p.stack for p in table.players)
        self.engine.start_hand(table, PlayerId("p2"))
        hand = hand_of(table)
        self.assertNotEqual(hand.id, previous.id)
        self.assertIsNot(hand.deck, previous.deck)
        self.assertIs(table.last_result, result)
        self.assertEqual((hand.button_seat, hand.small_blind_seat, hand.big_blind_seat), (1, 2, 0))
        self.assertEqual(hand.betting.pending, [PlayerId("p1"), PlayerId("p2"), PlayerId("p0")])
        self.assertEqual(hand.betting.actor_id, PlayerId("p1"))
        self.assertEqual(hand.board, [])
        self.assertEqual(hand.refunds, [])
        self.assertEqual(hand.betting.last_action_bet, {})
        self.assertEqual(hand.chip_total_at_start, chip_total)
        self.assertEqual(sum(p.stack for p in table.players) + 15, chip_total)
        self.assertEqual([m.hand_commit for m in hand.players.values()], [10, 0, 5])

    def test_button_rotation_skips_busted_button_and_wraps_around(self) -> None:
        table = opening_table((1000,) * 4)
        self.engine.start_hand(table, PlayerId("p0"))
        complete_input_fixture(table)
        table.player(PlayerId("p0")).stack = 0
        self.engine.start_hand(table, PlayerId("p1"))
        hand = hand_of(table)
        self.assertEqual((hand.button_seat, hand.small_blind_seat, hand.big_blind_seat), (1, 2, 3))
        for expected_button in (2, 3, 1):
            complete_input_fixture(table)
            self.engine.start_hand(table, PlayerId("p1"))
            self.assertEqual(hand_of(table).button_seat, expected_button)

    def test_transition_to_heads_up_avoids_repeating_big_blind(self) -> None:
        for busted, expected in ((0, (2, 2, 1)), (1, (2, 2, 0)), (2, (1, 1, 0))):
            with self.subTest(busted_seat=busted):
                table = opening_table()
                self.engine.start_hand(table, PlayerId("p0"))
                previous = hand_of(table)
                complete_input_fixture(table)
                table.player(PlayerId(f"p{busted}")).stack = 0
                requester = next(p.id for p in table.players if p.stack > 0)
                self.engine.start_hand(table, requester)
                hand = hand_of(table)
                self.assertEqual((hand.button_seat, hand.small_blind_seat, hand.big_blind_seat), expected)
                self.assertNotEqual(hand.big_blind_seat, previous.big_blind_seat)
                self.assertEqual(hand.betting.actor_id, table.players[expected[0]].id)

    def test_subsequent_heads_up_hands_alternate_blinds(self) -> None:
        table = opening_table((1000, 1000))
        for expected_button in (0, 1, 0):
            self.engine.start_hand(table, PlayerId("p0"))
            hand = hand_of(table)
            self.assertEqual((hand.button_seat, hand.small_blind_seat, hand.big_blind_seat),
                             (expected_button, expected_button, 1 - expected_button))
            complete_input_fixture(table)


class StartHandApplicationTests(unittest.TestCase):
    def test_not_enough_players_does_not_save_a_snapshot(self) -> None:
        table = opening_table((1000, 0))
        repository = MemoryRepository(table)
        service = TableService(table.id, repository, opening_engine(), PlayerViewBuilder())
        response = service.handle(StartHandCommand(), SessionContext(PlayerId("p0")))
        assert response.error is not None
        self.assertEqual(response.error.code, ErrorCode.NOT_ENOUGH_PLAYERS)
        self.assertEqual(repository.saves, 0)
        self.assertEqual(repository.load(table.id), table)

    def test_reopening_hand_in_progress_does_not_save_a_snapshot(self) -> None:
        table = opening_table()
        engine = opening_engine()
        engine.start_hand(table, PlayerId("p0"))
        repository = MemoryRepository(table)
        service = TableService(table.id, repository, engine, PlayerViewBuilder())
        response = service.handle(StartHandCommand(), SessionContext(PlayerId("p1")))
        assert response.error is not None
        self.assertEqual(response.error.code, ErrorCode.HAND_IN_PROGRESS)
        self.assertEqual(repository.saves, 0)
        self.assertEqual(repository.load(table.id), table)

    def test_opening_requiring_automatic_runout_completes_and_saves(self) -> None:
        # Backtracking E01 after E03-E06: these openings now finish in the same call.
        for stacks in ((2, 3), (5, 1000), (1000, 3)):
            with self.subTest(stacks=stacks):
                table = opening_table(stacks)
                repository = MemoryRepository(table)
                service = TableService(table.id, repository, opening_engine(), PlayerViewBuilder())
                response = service.handle(StartHandCommand(), SessionContext(PlayerId("p0")))
                assert response.view is not None and response.view.result is not None
                self.assertEqual(response.view.phase, HandPhase.COMPLETE)
                self.assertEqual(len(response.view.board), 5)
                self.assertIsNone(response.view.actor_id)
                self.assertEqual(sum(p.stack for p in response.view.players), sum(stacks))
                self.assertEqual(repository.saves, 1)
                saved = repository.load(table.id)
                assert saved is not None and saved.hand is not None
                self.assertEqual(saved.hand.phase, HandPhase.COMPLETE)
                self.assertEqual(saved.revision, 1)

    def test_real_opening_and_private_cards_round_trip_through_sqlite(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "opening.sqlite3"
            repository = SqliteTableRepository(path)
            table = opening_table()
            repository.save(table)
            service = TableService(table.id, repository, opening_engine(), PlayerViewBuilder())
            response = service.handle(StartHandCommand(), SessionContext(PlayerId("p0")))
            assert response.view is not None
            self.assertEqual(response.view.phase, HandPhase.PREFLOP)
            self.assertEqual(response.view.actor_id, PlayerId("p0"))
            self.assertEqual(response.view.revision, 1)
            self.assertEqual(response.view.pot_total, 15)
            self.assertIn("call", [o.kind.value for o in response.view.me.legal_actions])
            self.assertEqual(len(response.view.me.hole_cards), 2)
            self.assertTrue(all(not p.revealed_cards for p in response.view.players))
            saved = SqliteTableRepository(path).load(table.id)
            assert saved is not None
            hand = hand_of(saved)
            self.assertEqual(hand.id, response.view.hand_id)
            self.assertEqual(saved.revision, 1)
            self.assertEqual([p.stack for p in saved.players], [1000, 995, 990])
            self.assertEqual(sum(p.stack for p in saved.players) + 15, hand.chip_total_at_start)
            observer = service.handle(StateCommand(), SessionContext(PlayerId("p2")))
            assert observer.view is not None
            self.assertEqual(observer.view.players, response.view.players)
            self.assertNotEqual(observer.view.me.hole_cards, response.view.me.hole_cards)
            self.assertEqual(SqliteTableRepository(path).load(table.id), saved)
