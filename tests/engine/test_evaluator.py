import unittest

from poker.domain.cards import Card
from poker.domain.models import HandValue
from poker.domain.types import HandCategory
from poker.engine.policies import FiveCardHighEvaluator


def cards(codes: str) -> tuple[Card, ...]:
    return tuple(Card.parse(code) for code in codes.split())


class EvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = FiveCardHighEvaluator()

    def test_all_nine_categories_have_complete_comparison_keys(self) -> None:
        examples = (
            ("As Kd 9h 5c 3s", HandCategory.HIGH_CARD, (14, 13, 9, 5, 3)),
            ("As Ad Kc 9h 3s", HandCategory.PAIR, (14, 13, 9, 3)),
            ("As Ad Kc Kh 3s", HandCategory.TWO_PAIR, (14, 13, 3)),
            ("As Ad Ac Kh 3s", HandCategory.THREE_OF_A_KIND, (14, 13, 3)),
            ("2c 3d 4h 5s 6c", HandCategory.STRAIGHT, (6,)),
            ("As Js 9s 5s 3s", HandCategory.FLUSH, (14, 11, 9, 5, 3)),
            ("As Ad Ac Kh Ks", HandCategory.FULL_HOUSE, (14, 13)),
            ("As Ad Ac Ah Ks", HandCategory.FOUR_OF_A_KIND, (14, 13)),
            ("Ts Js Qs Ks As", HandCategory.STRAIGHT_FLUSH, (14,)),
        )
        values = []
        for codes, category, kickers in examples:
            with self.subTest(category=category):
                value = self.evaluator.evaluate_five(cards(codes))
                self.assertEqual(value, HandValue(category, kickers))
                values.append(value)
        self.assertEqual(sorted(values), values)

    def test_wheel_and_wheel_straight_flush_use_five_high(self) -> None:
        self.assertEqual(self.evaluator.evaluate_five(cards("As 2d 3h 4c 5s")),
                         HandValue(HandCategory.STRAIGHT, (5,)))
        self.assertEqual(self.evaluator.evaluate_five(cards("As 2s 3s 4s 5s")),
                         HandValue(HandCategory.STRAIGHT_FLUSH, (5,)))

    def test_kickers_break_ties_at_every_required_position(self) -> None:
        comparisons = (
            ("As Ad Kc 9h 3s", "Ah Ac Qc Jh Ts"),
            ("As Ad Kc 9h 3s", "Ah Ac Kh 8d 7s"),
            ("As Ad Kc 9h 3s", "Ah Ac Kh 9d 2s"),
            ("As Ad Kc Kh 3s", "Ah Ac Qh Qd Ks"),
            ("As Ad Kc Kh 3s", "Ah Ac Kd Ks 2s"),
            ("As Ad Ac Kh Ks", "Kh Kd Kc Ah As"),
            ("As Ad Ac Ah Ks", "As Ad Ac Ah Qs"),
            ("As Js 9s 5s 3s", "Ah Jh 9h 5h 2h"),
        )
        for high, low in comparisons:
            with self.subTest(high=high, low=low):
                self.assertGreater(self.evaluator.evaluate_five(cards(high)),
                                   self.evaluator.evaluate_five(cards(low)))

    def test_equal_ranks_tie_without_suit_tiebreak(self) -> None:
        self.assertEqual(self.evaluator.evaluate_five(cards("As Kd 9h 5c 3s")),
                         self.evaluator.evaluate_five(cards("Ah Kc 9d 5s 3h")))
        self.assertEqual(self.evaluator.evaluate_five(cards("As Ks Qs Js Ts")),
                         self.evaluator.evaluate_five(cards("Ah Kh Qh Jh Th")))

    def test_best_seven_can_use_two_one_or_zero_hole_cards(self) -> None:
        examples = (
            ("As Ad", "2s 4d 6h 8c Ts", HandValue(HandCategory.PAIR, (14, 10, 8, 6))),
            ("Kc 2h", "Ks Kd 5c 6h 9d", HandValue(HandCategory.THREE_OF_A_KIND, (13, 9, 6))),
            ("2c 3d", "Ts Js Qs Ks As", HandValue(HandCategory.STRAIGHT_FLUSH, (14,))),
        )
        for hole, board, expected in examples:
            with self.subTest(hole=hole):
                self.assertEqual(self.evaluator.evaluate_best(cards(hole) + cards(board)), expected)

    def test_duplicate_cards_and_wrong_lengths_are_rejected(self) -> None:
        for codes in ("As As 3h 4c 5s", "As 2d 3h 4c", "As 2d 3h 4c 5s 6d"):
            with self.subTest(cards=codes), self.assertRaises(ValueError):
                self.evaluator.evaluate_five(cards(codes))
        with self.assertRaises(ValueError):
            self.evaluator.evaluate_best(cards("As 2d 3h 4c 5s 6d As"))
