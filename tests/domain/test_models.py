import unittest
from random import Random

from poker.domain.cards import Card, Deck
from poker.domain.errors import RuleViolation
from poker.domain.models import Player, PlayerAction, TableConfig
from poker.domain.types import ActionKind, HandPhase, PlayerId
from tests.fixtures import make_table


class DomainTests(unittest.TestCase):
    def test_standard_deck_is_unique_and_draw_burn_partition_is_preserved(self) -> None:
        deck = Deck.shuffled(Random(1))
        drawn = deck.draw(4)
        deck.burn()
        cards = list(drawn) + deck.burned + deck.remaining
        self.assertEqual((len(cards), len(set(cards))), (52, 52))

    def test_seeded_deck_is_reproducible(self) -> None:
        self.assertEqual(Deck.shuffled(Random(4)), Deck.shuffled(Random(4)))

    def test_all_card_codes_round_trip(self) -> None:
        for card in Deck.shuffled(Random(1)).remaining:
            self.assertEqual(Card.parse(card.code), card)

    def test_duplicate_card_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Deck([Card.parse("As"), Card.parse("As")])

    def test_sized_actions_require_a_total(self) -> None:
        with self.assertRaises(ValueError):
            PlayerAction(ActionKind.RAISE_TO)
        with self.assertRaises(ValueError):
            PlayerAction(ActionKind.CALL, 10)

    def test_invalid_table_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TableConfig(small_blind=20, big_blind=10)

    def test_duplicate_seat_and_join_during_hand_are_rejected(self) -> None:
        table = make_table()
        with self.assertRaises(RuleViolation):
            table.add_player(Player(PlayerId("p3"), "丙", 0, 1000))
        table = make_table(with_hand=True)
        with self.assertRaises(RuleViolation):
            table.add_player(Player(PlayerId("p3"), "丙", 2, 1000))
        assert table.hand is not None
        table.hand.phase = HandPhase.COMPLETE
        table.add_player(Player(PlayerId("p3"), "丙", 2, 1000))
        self.assertEqual(len(table.players), 3)
