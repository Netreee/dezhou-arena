import unittest

from poker.domain.cards import Card
from poker.domain.models import HandValue
from poker.domain.types import HandCategory
from poker.engine.holdem import HoldemEngine
from poker.engine.interfaces import HandEvaluator
from poker.engine.policies import FiveCardHighEvaluator, NoLimitBettingRules, SidePotAllocator
from tests.fixtures import make_table


class RecordingEvaluator(HandEvaluator):
    """Synthetic ordering exercises the shared algorithm, not poker ranking."""

    def __init__(self) -> None:
        self.calls: list[tuple[Card, ...]] = []

    def evaluate_five(self, cards: tuple[Card, ...]) -> HandValue:
        self.calls.append(cards)
        return HandValue(HandCategory.HIGH_CARD, tuple(sorted((int(c.rank) for c in cards), reverse=True)))


class EngineContractTests(unittest.TestCase):
    def test_parent_evaluator_enumerates_21_combinations_and_selects_maximum(self) -> None:
        evaluator = RecordingEvaluator()
        cards = tuple(Card.parse(code) for code in ("2c", "3d", "4h", "5s", "6c", "7d", "8h"))
        score = evaluator.evaluate_best(cards)
        self.assertEqual(len(evaluator.calls), 21)
        self.assertEqual(score.kickers, (8, 7, 6, 5, 4))

    def test_parent_evaluator_rejects_duplicate_input(self) -> None:
        with self.assertRaises(ValueError):
            RecordingEvaluator().evaluate_best((Card.parse("As"),) * 5)

    def test_join_uses_first_free_seat_and_configured_stack(self) -> None:
        engine = HoldemEngine(NoLimitBettingRules(), FiveCardHighEvaluator(), SidePotAllocator())
        table = make_table()
        player_id = engine.join(table, " 丙 ")
        self.assertEqual((table.player(player_id).seat, table.player(player_id).stack), (2, 1000))
        self.assertEqual(table.player(player_id).name, "丙")
