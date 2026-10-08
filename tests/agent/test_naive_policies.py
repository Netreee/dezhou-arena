from dataclasses import replace
import json
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from typing import cast
import unittest
from unittest.mock import patch

from shared_logging import JsonObject, LoggingConfig, configure_logging, shutdown_logging

from poker.agent.context import DecisionContext, DecisionControl
from poker.agent.models import (
    ActionDistribution, ActionFeedback, DecisionCancelled, DecisionRequest,
    HandCompleted, Observation, ObservationChanged, select_action,
)
from poker.agent.naive_policies import (
    LookupPolicy, MixedPolicy, RulesPolicy, SolverPolicy, StatefulPolicy,
    approximate_payoff, estimate_equity, lookup, mixed, rules, solver, stateful,
)
from poker.agent.tools import ToolRegistry
from poker.application.views import PlayerView, ResultView
from poker.domain.cards import Card
from poker.domain.models import ActionOption, HandValue, PlayerAction, PublicActionRecord
from poker.domain.types import ActionKind, HandId, HandPhase, PlayerId
from poker.engine.policies import FiveCardHighEvaluator
from tests.agent.test_contracts import Clock, make_observation


def invocation(view: PlayerView | None = None, seed: int = 7) -> tuple[DecisionRequest, DecisionContext]:
    observation = make_observation() if view is None else Observation(view, 10.0)
    control = DecisionControl(20.0, clock=Clock())
    tools = ToolRegistry(()).bind(observation, control, max_calls=0, decision_id="naive-test")
    return DecisionRequest("naive-test", observation, 20.0), DecisionContext(tools, Random(seed), control)


def river_view(hero: tuple[str, str], board: tuple[str, ...], revealed: tuple[str, ...] = ()) -> PlayerView:
    view = make_observation().view
    players = tuple(replace(player, revealed_cards=revealed) if player.player_id != view.me.player_id else player
                    for player in view.players)
    return replace(view, phase=HandPhase.RIVER, board=board, players=players, pot_total=100,
                   me=replace(view.me, hole_cards=hero, legal_actions=(ActionOption(ActionKind.FOLD),
                                                                      ActionOption(ActionKind.CALL, pay=50))))


class NaivePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.log_path = configure_logging(LoggingConfig(Path(self.temp.name), "naive-policy-test", "test"))

    def tearDown(self) -> None:
        shutdown_logging()
        self.temp.cleanup()

    def test_rules_use_price_threshold_and_distinct_street_branch(self) -> None:
        view = make_observation().view
        expensive = replace(view, me=replace(view.me, legal_actions=(ActionOption(ActionKind.FOLD),
                                                                   ActionOption(ActionKind.CALL, pay=500))))
        request, context = invocation(expensive)
        self.assertEqual(RulesPolicy().decide(request, context).choice, PlayerAction(ActionKind.FOLD))
        self.assertEqual(RulesPolicy(verification_continuation=True).decide(request, context).choice,
                         PlayerAction(ActionKind.CALL))
        request, context = invocation(replace(view, phase=HandPhase.TURN))
        result = RulesPolicy(verification_continuation=True).decide(request, context)
        self.assertEqual(result.choice, PlayerAction(ActionKind.BET_TO, 10))
        self.assertEqual(result.annotations["matched_rule"], 2)

    def test_mixed_returns_real_probabilities_and_runtime_samples_both_actions(self) -> None:
        request, context = invocation()
        decision = MixedPolicy(verification_continuation=True).decide(request, context)
        self.assertIsInstance(decision.choice, ActionDistribution)
        distribution = cast(ActionDistribution, decision.choice)
        self.assertEqual(len(distribution.candidates), 2)
        self.assertTrue(all(0 < item.probability < 1 for item in distribution.candidates))
        self.assertAlmostEqual(sum(item.probability for item in distribution.candidates), 1)
        rng = Random(73)
        sampled = [select_action(decision, request.observation.view, rng).kind for _ in range(400)]
        self.assertEqual(set(sampled), {ActionKind.CHECK, ActionKind.BET_TO})
        self.assertTrue(0.7 < sampled.count(ActionKind.CHECK) / len(sampled) < 0.95)
        self.assertFalse(decision.annotations["degenerate"])

    def test_continuation_does_not_disguise_a_forced_single_candidate_as_a_mixture(self) -> None:
        view = make_observation().view
        history = (PublicActionRecord(1, view.phase or HandPhase.FLOP, PlayerId("p2"), "bet_to", 10, 10, 980),)
        view = replace(view, action_history=history, me=replace(view.me, legal_actions=(
            ActionOption(ActionKind.FOLD), ActionOption(ActionKind.CALL, pay=10),
            ActionOption(ActionKind.RAISE_TO, min_to=20, max_to=990), ActionOption(ActionKind.ALL_IN, pay=990))))
        request, context = invocation(view)
        decision = MixedPolicy(verification_continuation=True).decide(request, context)
        self.assertEqual(select_action(decision, view, Random(1)), PlayerAction(ActionKind.CALL))
        self.assertEqual(decision.annotations["positive_candidates"], 1)
        self.assertTrue(decision.annotations["degenerate"])

    def test_lookup_table_data_changes_the_decision_and_cache_records_real_hits(self) -> None:
        request, context = invocation()
        conservative = LookupPolicy({"flop:free": (ActionKind.CHECK, ActionKind.BET_TO)})
        active = LookupPolicy({"flop:free": (ActionKind.BET_TO, ActionKind.CHECK)})
        self.assertEqual(conservative.decide(request, context).choice, PlayerAction(ActionKind.CHECK))
        first = active.decide(request, context)
        second = active.decide(request, context)
        self.assertEqual(first.choice, PlayerAction(ActionKind.BET_TO, 10))
        self.assertTrue(first.annotations["table_hit"])
        self.assertFalse(first.annotations["cache_hit"])
        self.assertTrue(second.annotations["cache_hit"])
        self.assertEqual(second.annotations["cache_count"], 1)
        self.assertEqual(second.annotations["lookup_hits"], 2)
        miss = LookupPolicy({}).decide(request, context)
        self.assertFalse(miss.annotations["table_hit"])
        self.assertEqual(miss.annotations["cache_count"], 0)

    def test_state_changes_only_from_real_distinct_feedback_and_affects_selection(self) -> None:
        request, context = invocation()
        policy = StatefulPolicy(verification_continuation=True)
        policy.observe(ObservationChanged(request.observation))
        self.assertEqual(policy.decide(request, context).choice, PlayerAction(ActionKind.CHECK))
        first = ActionFeedback("d1", PlayerAction(ActionKind.CHECK), True)
        policy.observe(first)
        policy.observe(first)
        policy.observe(HandCompleted(ResultView(HandId("h0"), (), ())))
        policy.observe(HandCompleted(ResultView(HandId("h0"), (), ())))
        changed = policy.decide(request, context)
        self.assertEqual(changed.choice, PlayerAction(ActionKind.BET_TO, 10))
        self.assertEqual(changed.annotations["confirmed_memory"], 1)
        self.assertEqual(changed.annotations["completed_memory"], 1)
        self.assertEqual(changed.annotations["memory_mode"], 2)
        policy.observe(ActionFeedback("rejected", PlayerAction(ActionKind.CHECK), False, "stale_observation"))
        cautious = policy.decide(request, context)
        self.assertEqual(cautious.choice, PlayerAction(ActionKind.CHECK))
        self.assertEqual(cautious.annotations["rejected_memory"], 1)

    def test_equity_handles_forced_win_loss_and_board_tie(self) -> None:
        cases = (
            (river_view(("As", "Ks"), ("Qs", "Js", "Ts", "2d", "3c")), 1.0),
            (river_view(("Ah", "Kh"), ("9h", "9s", "2c", "3d", "4h"), ("9c", "9d")), 0.0),
            (river_view(("2d", "3d"), ("Tc", "Jc", "Qc", "Kc", "Ac")), 0.5),
        )
        for view, expected in cases:
            with self.subTest(expected=expected):
                request, context = invocation(view)
                result = estimate_equity(request.observation.view, context, 8)
                self.assertEqual(result.equity, expected)
                self.assertEqual(result.rollouts, 8)
                self.assertEqual(result.opponent_count, 1)
                self.assertEqual(result.evaluated_hands, 16)

    def test_solver_payoff_search_changes_between_known_winner_and_known_loser(self) -> None:
        winner = river_view(("As", "Ks"), ("Qs", "Js", "Ts", "2d", "3c"))
        loser = river_view(("Ah", "Kh"), ("9h", "9s", "2c", "3d", "4h"), ("9c", "9d"))
        for view, expected in ((winner, ActionKind.CALL), (loser, ActionKind.FOLD)):
            with self.subTest(expected=expected):
                request, context = invocation(view)
                decision = SolverPolicy(rollouts=8).decide(request, context)
                self.assertEqual(decision.choice, PlayerAction(expected))
                self.assertEqual(decision.annotations["candidate_count"], 2)
                self.assertEqual(decision.annotations["rollouts"], 8)
        request, context = invocation(loser)
        constrained = SolverPolicy(rollouts=8, verification_continuation=True).decide(request, context)
        self.assertEqual(constrained.choice, PlayerAction(ActionKind.CALL))
        self.assertLess(cast(float, constrained.annotations["selected_ev"]), 0)
        self.assertEqual(approximate_payoff(winner, PlayerAction(ActionKind.CALL), 1.0), 100)
        self.assertEqual(approximate_payoff(loser, PlayerAction(ActionKind.CALL), 0.0), -50)

    def test_unknown_card_samples_respect_visible_cards_and_share_one_board(self) -> None:
        view = make_observation().view
        captured: list[tuple[Card, ...]] = []

        class RecordingEvaluator(FiveCardHighEvaluator):
            def evaluate_best(self, cards: tuple[Card, ...]) -> HandValue:
                captured.append(cards)
                return super().evaluate_best(cards)

        request, context = invocation(view)
        with patch("poker.agent.naive_policies.FiveCardHighEvaluator", RecordingEvaluator):
            result = estimate_equity(view, context, 4)
        self.assertEqual(len(captured), result.evaluated_hands)
        original = {Card.parse(code) for code in (*view.me.hole_cards, *view.board)}
        for index in range(0, len(captured), 2):
            hero, opponent = captured[index:index + 2]
            self.assertEqual(len(set(hero)), 7)
            self.assertEqual(len(set(opponent)), 7)
            self.assertEqual(hero[2:], opponent[2:])
            self.assertFalse(set(opponent[:2]) & original)
            self.assertFalse(set(hero[2 + len(view.board):]) & original)
            self.assertFalse(set(hero[:2]) & set(opponent[:2]))

    def test_solver_sampling_is_seeded_and_obeys_cancellation(self) -> None:
        request, first = invocation(seed=31)
        _, second = invocation(seed=31)
        self.assertEqual(estimate_equity(request.observation.view, first, 12),
                         estimate_equity(request.observation.view, second, 12))
        first.control.cancel()
        with self.assertRaises(DecisionCancelled):
            estimate_equity(request.observation.view, first, 12)

    def test_factories_and_mechanism_logs_have_distinct_evidence_without_card_values(self) -> None:
        config: JsonObject = {"verification_continuation": True, "solver_rollouts": 4}
        factories = (rules, mixed, lookup, stateful, solver)
        request, context = invocation()
        names = []
        for factory in factories:
            policy = factory(config)
            names.append(policy.identity.name)
            decision = policy.decide(request, context)
            select_action(decision, request.observation.view, Random(1))
        self.assertEqual(len(set(names)), 5)
        rows = [json.loads(line) for line in self.log_path.read_text(encoding="utf-8").splitlines()]
        evidence = [row["data"] for row in rows if row["event"] == "naive.policy.decided"]
        self.assertEqual({row["representation"] for row in evidence}, {"rules", "mixed", "lookup", "stateful", "solver"})
        for row in evidence:
            self.assertEqual(row["decision_id"], request.decision_id)
            self.assertEqual(row["revision"], request.observation.view.revision)
            self.assertGreaterEqual(row["candidate_count"], 1)
        raw = self.log_path.read_text(encoding="utf-8")
        for card in (*request.observation.view.me.hole_cards, *request.observation.view.board):
            self.assertNotIn('"' + card + '"', raw)
        self.assertNotIn("hole_cards", raw)

    def test_factory_configuration_rejects_malformed_controls(self) -> None:
        for factory in (rules, mixed, lookup, stateful, solver):
            with self.subTest(factory=factory), self.assertRaises(ValueError):
                factory({"verification_continuation": "true"})
        for count in (0, 257, True, 3.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                solver({"solver_rollouts": count})
        with self.assertRaises(ValueError):
            lookup({"lookup_table": {"flop:free": []}})
        configured = lookup({"lookup_table": {"flop:free": ["bet_to", "check"]}})
        request, context = invocation()
        self.assertEqual(configured.decide(request, context).choice, PlayerAction(ActionKind.BET_TO, 10))
