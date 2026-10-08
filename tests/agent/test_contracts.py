from dataclasses import FrozenInstanceError, replace
from random import Random
import unittest
from typing import cast

from poker.agent.context import DecisionContext, DecisionControl
from poker.agent.models import (
    ActionDistribution, Decision, DecisionCancelled, DecisionDeadlineExceeded,
    DecisionRequest, InvalidDecision, Observation, ObservationChanged,
    PolicyIdentity, PolicySession, StopReason, WeightedAction, select_action,
)
from poker.agent.policy import Policy
from poker.agent.tools import ToolRegistry
from poker.application.views import PlayerViewBuilder
from poker.domain.models import ActionOption, PlayerAction
from poker.domain.types import ActionKind, PlayerId
from tests.fixtures import make_table


class Clock:
    def __init__(self, now: float = 10.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def make_observation() -> Observation:
    table = make_table(with_hand=True)
    options = (
        ActionOption(ActionKind.CHECK), ActionOption(ActionKind.FOLD),
        ActionOption(ActionKind.BET_TO, min_to=10, max_to=990),
    )
    return Observation(PlayerViewBuilder().build(table, PlayerId("p1"), options), 10.0)


class AgentContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observation = make_observation()
        self.view = self.observation.view

    def test_single_action_is_returned_without_resampling_or_mutation(self) -> None:
        action = PlayerAction(ActionKind.BET_TO, 375)
        random = Random(42)
        before = random.getstate()
        self.assertIs(select_action(Decision(action), self.view, random), action)
        self.assertEqual(random.getstate(), before)

    def test_sized_action_accepts_both_inclusive_server_bounds(self) -> None:
        for amount in (10, 990):
            with self.subTest(amount=amount):
                action = PlayerAction(ActionKind.BET_TO, amount)
                self.assertEqual(select_action(Decision(action), self.view, Random(1)), action)

    def test_unsupported_action_and_nonactor_are_rejected(self) -> None:
        with self.assertRaises(InvalidDecision):
            select_action(Decision(PlayerAction(ActionKind.CALL)), self.view, Random(1))
        waiting = replace(self.view, actor_id=PlayerId("p2"))
        with self.assertRaises(InvalidDecision):
            select_action(Decision(PlayerAction(ActionKind.CHECK)), waiting, Random(1))

    def test_amount_is_never_clamped_or_coerced(self) -> None:
        for amount in (1, 9, 991, True, 25.5, float("inf"), float("nan")):
            with self.subTest(amount=amount), self.assertRaises(InvalidDecision):
                action = PlayerAction(ActionKind.BET_TO, cast(int, amount))
                select_action(Decision(action), self.view, Random(1))

    def test_unknown_return_type_or_kind_is_not_accepted(self) -> None:
        malformed = (
            cast(Decision, PlayerAction(ActionKind.CHECK)),
            Decision(cast(PlayerAction, "check")),
            Decision(PlayerAction(cast(ActionKind, "check"))),
        )
        for decision in malformed:
            with self.subTest(decision=decision), self.assertRaises(InvalidDecision):
                select_action(decision, self.view, Random(1))

    def test_distribution_uses_seeded_sampling_and_preserves_requested_actions(self) -> None:
        check = PlayerAction(ActionKind.CHECK)
        bet = PlayerAction(ActionKind.BET_TO, 123)
        decision = Decision(ActionDistribution((WeightedAction(check, 0.25), WeightedAction(bet, 0.75))))
        left, right = Random(73), Random(73)
        first = [select_action(decision, self.view, left) for _ in range(100)]
        second = [select_action(decision, self.view, right) for _ in range(100)]
        self.assertEqual(first, second)
        self.assertEqual(set(first), {check, bet})
        self.assertEqual(select_action(decision, self.view, Random(1)), check)
        self.assertEqual(select_action(decision, self.view, Random(2)), bet)

    def test_zero_probability_is_valid_and_never_selected(self) -> None:
        check = PlayerAction(ActionKind.CHECK)
        fold = PlayerAction(ActionKind.FOLD)
        decision = Decision(ActionDistribution((WeightedAction(fold, 0.0), WeightedAction(check, 1.0))))
        for seed in range(25):
            self.assertEqual(select_action(decision, self.view, Random(seed)), check)

    def test_bad_probabilities_are_rejected_before_randomness_is_consumed(self) -> None:
        check, fold = PlayerAction(ActionKind.CHECK), PlayerAction(ActionKind.FOLD)
        cases = (
            (float("nan"), 1.0), (float("inf"), 0.0), (-0.1, 1.1),
            (True, 0.0), (0.2, 0.7), (0.2, 0.9), (0.0, 0.0),
            (cast(float, "0.2"), 0.8),
        )
        for first, second in cases:
            with self.subTest(probabilities=(first, second)):
                random = Random(4)
                before = random.getstate()
                decision = Decision(ActionDistribution((WeightedAction(check, first), WeightedAction(fold, second))))
                with self.assertRaises(InvalidDecision):
                    select_action(decision, self.view, random)
                self.assertEqual(random.getstate(), before)

    def test_empty_duplicate_and_malformed_distributions_are_rejected(self) -> None:
        check = PlayerAction(ActionKind.CHECK)
        distributions = (
            ActionDistribution(()),
            ActionDistribution((WeightedAction(check, 0.5), WeightedAction(check, 0.5))),
            ActionDistribution((cast(WeightedAction, check),)),
        )
        for distribution in distributions:
            with self.subTest(distribution=distribution), self.assertRaises(InvalidDecision):
                select_action(Decision(distribution), self.view, Random(1))

    def test_illegal_zero_weight_candidate_is_still_rejected(self) -> None:
        decision = Decision(ActionDistribution((
            WeightedAction(PlayerAction(ActionKind.CHECK), 1.0),
            WeightedAction(PlayerAction(ActionKind.BET_TO, 999), 0.0),
        )))
        with self.assertRaises(InvalidDecision):
            select_action(decision, self.view, Random(1))

    def test_multiple_sizes_of_one_action_are_distinct_candidates(self) -> None:
        small, large = PlayerAction(ActionKind.BET_TO, 10), PlayerAction(ActionKind.BET_TO, 20)
        decision = Decision(ActionDistribution((WeightedAction(small, 0.5), WeightedAction(large, 0.5))))
        self.assertEqual(select_action(decision, self.view, Random(1)), small)
        self.assertEqual(select_action(decision, self.view, Random(2)), large)

    def test_observation_and_nested_player_values_are_frozen(self) -> None:
        for target, name, value in (
            (self.observation, "received_at", 20.0),
            (self.view, "revision", 99),
            (self.view.me, "hole_cards", ("As", "Ah")),
            (self.view.players[0], "stack", 100000),
            (self.view.me.legal_actions[0], "kind", ActionKind.FOLD),
        ):
            with self.subTest(name=name), self.assertRaises(FrozenInstanceError):
                setattr(target, name, value)

    def test_policy_only_requires_decide_and_can_own_arbitrary_state(self) -> None:
        class StatefulPolicy(Policy):
            def __init__(self) -> None:
                self.calls = 0

            def decide(self, request: DecisionRequest, context: DecisionContext) -> Decision:
                context.check_cancelled()
                self.calls += 1
                return Decision(PlayerAction(request.observation.view.me.legal_actions[0].kind))

        clock = Clock()
        control = DecisionControl(20, clock=clock)
        tools = ToolRegistry(()).bind(self.observation, control, max_calls=0, decision_id="d1")
        context = DecisionContext(tools, Random(4), control)
        request = DecisionRequest("d1", self.observation, 20)
        policy = StatefulPolicy()
        policy.open(PolicySession("a1", "Agent"))
        policy.observe(ObservationChanged(self.observation))
        self.assertIsInstance(policy.identity, PolicyIdentity)
        self.assertEqual(policy.identity.version, "unspecified")
        for _ in range(2):
            self.assertEqual(policy.decide(request, context).choice, PlayerAction(ActionKind.CHECK))
        self.assertEqual(policy.calls, 2)
        policy.close(StopReason.REQUESTED)


class DecisionControlTests(unittest.TestCase):
    def test_deadline_uses_injected_monotonic_clock_and_expires_at_boundary(self) -> None:
        clock = Clock(10)
        control = DecisionControl(15, clock=clock)
        self.assertEqual(control.remaining_seconds(), 5)
        control.check()
        clock.now = 15
        self.assertEqual(control.remaining_seconds(), 0)
        with self.assertRaises(DecisionDeadlineExceeded):
            control.check()
        clock.now = 30
        self.assertEqual(control.remaining_seconds(), 0)

    def test_cancel_is_idempotent_and_takes_precedence_over_expiry(self) -> None:
        clock = Clock(10)
        control = DecisionControl(15, clock=clock)
        self.assertFalse(control.cancelled)
        control.cancel()
        control.cancel()
        self.assertTrue(control.cancelled)
        clock.now = 20
        with self.assertRaises(DecisionCancelled):
            control.check()

    def test_nonfinite_deadlines_are_rejected(self) -> None:
        for deadline in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(deadline=deadline), self.assertRaises(ValueError):
                DecisionControl(deadline)

    def test_context_propagates_remaining_time_and_cancellation(self) -> None:
        clock = Clock(3)
        control = DecisionControl(8, clock=clock)
        tools = ToolRegistry(()).bind(make_observation(), control, max_calls=0, decision_id="d1")
        context = DecisionContext(tools, Random(1), control)
        self.assertEqual(context.remaining_seconds(), 5)
        context.check_cancelled()
        control.cancel()
        with self.assertRaises(DecisionCancelled):
            context.check_cancelled()
