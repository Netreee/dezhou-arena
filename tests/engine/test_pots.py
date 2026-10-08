import unittest
from copy import deepcopy

from poker.domain.types import PlayerId, PlayerStatus
from poker.engine.policies import SidePotAllocator
from tests.engine.helpers import betting_table
from tests.engine.test_start_hand import hand_of


class PotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pots = SidePotAllocator()

    def test_100_300_500_refunds_200_and_builds_300_400(self) -> None:
        table = betting_table((0, 0, 0), (100, 300, 500), current_bet=500)
        hand = hand_of(table)
        self.pots.refund_uncalled(table)
        self.assertEqual([(r.player_id, r.amount) for r in hand.refunds], [(PlayerId("p2"), 200)])
        self.assertEqual(table.players[2].stack, 200)
        self.assertEqual((hand.players[PlayerId("p2")].street_commit,
                          hand.players[PlayerId("p2")].hand_commit), (300, 300))
        pots = self.pots.build_pots(table)
        self.assertEqual([p.amount for p in pots], [300, 400])
        self.assertEqual(pots[0].eligible_ids, (PlayerId("p0"), PlayerId("p1"), PlayerId("p2")))
        self.assertEqual(pots[1].eligible_ids, (PlayerId("p1"), PlayerId("p2")))
        self.assertEqual(sum(p.stack for p in table.players) + sum(p.amount for p in pots), 900)

    def test_folded_contributions_stay_in_pots_but_lose_eligibility(self) -> None:
        table = betting_table((0, 0, 0), (100, 300, 300), current_bet=300)
        hand_of(table).players[PlayerId("p0")].status = PlayerStatus.FOLDED
        before = deepcopy(table)
        pots = self.pots.build_pots(table)
        self.assertEqual([p.amount for p in pots], [300, 400])
        self.assertEqual([p.eligible_ids for p in pots],
                         [(PlayerId("p1"), PlayerId("p2")), (PlayerId("p1"), PlayerId("p2"))])
        self.assertEqual(table, before)

    def test_uncalled_bet_refund_keeps_only_matched_money_for_uncontested_win(self) -> None:
        table = betting_table((950, 990), (50, 10), current_bet=50)
        hand = hand_of(table)
        hand.players[PlayerId("p1")].status = PlayerStatus.FOLDED
        self.pots.refund_uncalled(table)
        self.assertEqual([(r.player_id, r.amount) for r in hand.refunds], [(PlayerId("p0"), 40)])
        self.assertEqual([p.stack for p in table.players], [990, 990])
        pots = self.pots.build_pots(table)
        self.assertEqual([(p.amount, p.eligible_ids) for p in pots], [(20, (PlayerId("p0"),))])

    def test_refunds_on_multiple_streets_keep_complete_history(self) -> None:
        table = betting_table((900, 900), (100, 60), current_bet=100)
        hand = hand_of(table)
        self.pots.refund_uncalled(table)
        for player, amount in zip(table.players, (20, 10), strict=True):
            member = hand.players[player.id]
            member.street_commit = amount
            member.hand_commit += amount
            player.stack -= amount
        self.pots.refund_uncalled(table)
        self.assertEqual([(r.player_id, r.amount) for r in hand.refunds],
                         [(PlayerId("p0"), 40), (PlayerId("p0"), 10)])
        self.assertEqual([m.hand_commit for m in hand.players.values()], [70, 70])
        self.assertEqual([m.street_commit for m in hand.players.values()], [10, 10])
        self.assertEqual(sum(p.stack for p in table.players) + 140, hand.chip_total_at_start)

    def test_refund_uses_this_street_not_cumulative_hand_difference(self) -> None:
        table = betting_table((100, 100), (500, 300), current_bet=40)
        hand = hand_of(table)
        for member in hand.players.values():
            member.street_commit = 40
        before = deepcopy(table)
        self.pots.refund_uncalled(table)
        self.assertEqual(table, before)

    def test_refund_restores_all_in_status_without_adding_an_action(self) -> None:
        table = betting_table((0, 0), (100, 50), current_bet=100)
        hand = hand_of(table)
        self.pots.refund_uncalled(table)
        self.assertEqual(hand.players[PlayerId("p0")].status, PlayerStatus.ACTIVE)
        self.assertEqual(table.players[0].stack, 50)
        self.assertEqual(hand.betting.pending, [])
        self.assertIsNone(hand.betting.actor_id)
        self.pots.refund_uncalled(table)
        self.assertEqual(len(hand.refunds), 1)

    def test_layers_use_commitments_and_not_current_balances(self) -> None:
        table = betting_table((900, 100, 400), (50, 100, 200), current_bet=200)
        pots = self.pots.build_pots(table)
        self.assertEqual([p.amount for p in pots], [150, 100, 100])
        self.assertEqual(pots[-1].eligible_ids, (PlayerId("p2"),))
        self.assertEqual(sum(p.amount for p in pots), 350)
