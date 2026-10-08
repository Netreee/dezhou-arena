from collections.abc import Callable
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
import unittest
from typing import cast

from shared_logging import JsonObject, LoggingConfig, configure_logging, shutdown_logging

from poker.agent.context import DecisionContext, DecisionControl
from poker.agent.language import (
    LanguageModel, ModelRequest, ModelToolCall, ModelTurn, PromptPolicy,
)
from poker.agent.models import (
    Decision, DecisionCancelled, DecisionDeadlineExceeded, DecisionRequest,
    InvalidDecision, PolicyError, ToolBudgetExceeded, select_action,
)
from poker.agent.tools import ToolRegistry, standard_tools
from poker.domain.models import PlayerAction
from poker.domain.types import ActionKind
from tests.agent.test_contracts import Clock, make_observation


class ScriptedModel(LanguageModel):
    def __init__(self, *steps: ModelTurn | Callable[[ModelRequest], ModelTurn]) -> None:
        self.steps = list(steps)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        if not self.steps:
            raise AssertionError("Unexpected extra model request")
        step = self.steps.pop(0)
        return step(request) if callable(step) else step


class LanguagePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.path = configure_logging(LoggingConfig(Path(self.temp.name), "language-policy-test", "test"))
        self.observation = make_observation()
        self.clock = Clock()
        self.control = DecisionControl(20, clock=self.clock)
        self.tools = ToolRegistry(standard_tools()).bind(
            self.observation, self.control, max_calls=5, decision_id="d1",
        )
        self.context = DecisionContext(self.tools, Random(4), self.control)
        self.request = DecisionRequest("d1", self.observation, 20)
        self.decision = Decision(PlayerAction(ActionKind.CHECK))

    def tearDown(self) -> None:
        shutdown_logging()
        self.temp.cleanup()

    def test_final_action_is_a_proposal_with_no_tool_or_arena_side_effect(self) -> None:
        model = ScriptedModel(ModelTurn(decision=self.decision))
        instructions = "# Policy\n\n保留原样的自然语言策略。"
        policy = PromptPolicy(instructions, model)
        decision = policy.decide(self.request, self.context)
        self.assertIs(decision, self.decision)
        self.assertEqual(self.tools.calls, 0)
        self.assertEqual(len(model.requests), 1)
        sent = model.requests[0]
        self.assertEqual(sent.instructions, instructions)
        self.assertEqual(sent.timeout_seconds, 10)
        self.assertEqual(sent.exchanges, ())
        self.assertEqual({spec.name for spec in sent.tools}, {"observation", "legal_actions", "public_history"})
        self.assertEqual(cast(JsonObject, sent.observation["me"])["hole_cards"], list(self.observation.view.me.hole_cards))
        self.assertNotIn("deck", sent.observation)
        self.assertNotIn("command", sent.observation)

    def test_runtime_retains_responsibility_for_final_action_legality(self) -> None:
        illegal = Decision(PlayerAction(ActionKind.CALL))
        model = ScriptedModel(ModelTurn(decision=illegal))
        returned = PromptPolicy("policy", model).decide(self.request, self.context)
        self.assertIs(returned, illegal)
        self.assertEqual(self.tools.calls, 0)
        with self.assertRaises(InvalidDecision):
            select_action(returned, self.observation.view, self.context.random)

    def test_tool_results_are_correlated_and_returned_to_model_before_final_decision(self) -> None:
        calls = (
            ModelToolCall("c1", "legal_actions", {}),
            ModelToolCall("c2", "public_history", {}),
        )

        def inspect_tools(request: ModelRequest) -> ModelTurn:
            self.assertEqual([exchange.call.call_id for exchange in request.exchanges], ["c1", "c2"])
            self.assertTrue(all(exchange.result.ok for exchange in request.exchanges))
            actions = cast(list[JsonObject], request.exchanges[0].result.value)
            self.assertEqual(actions[0]["kind"], "check")
            history = cast(JsonObject, request.exchanges[1].result.value)
            self.assertFalse(history["complete"])
            return ModelTurn(decision=self.decision)

        model = ScriptedModel(ModelTurn(tool_calls=calls), inspect_tools)
        self.assertEqual(PromptPolicy("policy", model).decide(self.request, self.context), self.decision)
        self.assertEqual(self.tools.calls, 2)
        self.assertEqual(len(model.requests), 2)

    def test_tool_errors_are_visible_to_model_and_never_become_fake_success(self) -> None:
        model = ScriptedModel(
            ModelTurn(tool_calls=(ModelToolCall("c1", "not_allowlisted", {}),)),
            ModelTurn(tool_calls=(ModelToolCall("c2", "observation", {"unexpected": 1}),)),
            ModelTurn(decision=self.decision),
        )
        PromptPolicy("policy", model).decide(self.request, self.context)
        exchanges = model.requests[-1].exchanges
        self.assertEqual(len(exchanges), 2)
        self.assertEqual([item.result.error.code if item.result.error else None for item in exchanges],
                         ["unknown_tool", "invalid_arguments"])
        self.assertTrue(all(item.result.value is None for item in exchanges))

    def test_model_round_budget_is_enforced_without_inventing_action(self) -> None:
        model = ScriptedModel(
            ModelTurn(tool_calls=(ModelToolCall("c1", "observation", {}),)),
            ModelTurn(tool_calls=(ModelToolCall("c2", "legal_actions", {}),)),
        )
        with self.assertRaisesRegex(PolicyError, "round budget"):
            PromptPolicy("policy", model, max_rounds=2).decide(self.request, self.context)
        self.assertEqual(len(model.requests), 2)

    def test_tool_budget_stops_model_loop(self) -> None:
        tools = ToolRegistry(standard_tools()).bind(
            self.observation, self.control, max_calls=1, decision_id="limited",
        )
        context = DecisionContext(tools, Random(1), self.control)
        model = ScriptedModel(ModelTurn(tool_calls=(
            ModelToolCall("c1", "observation", {}), ModelToolCall("c2", "legal_actions", {}),
        )))
        with self.assertRaises(ToolBudgetExceeded):
            PromptPolicy("policy", model).decide(self.request, context)
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(tools.calls, 1)

    def test_empty_or_duplicate_call_ids_are_rejected(self) -> None:
        scenarios = (
            (ModelTurn(tool_calls=(ModelToolCall("", "observation", {}),)),),
            (ModelTurn(tool_calls=(ModelToolCall("c1", "observation", {}),
                                   ModelToolCall("c1", "legal_actions", {}))),),
            (ModelTurn(tool_calls=(ModelToolCall("c1", "observation", {}),)),
             ModelTurn(tool_calls=(ModelToolCall("c1", "legal_actions", {}),))),
        )
        for steps in scenarios:
            with self.subTest(steps=steps), self.assertRaisesRegex(PolicyError, "IDs"):
                PromptPolicy("policy", ScriptedModel(*steps)).decide(self.request, self.context)

    def test_call_ids_and_exchanges_reset_for_each_decision(self) -> None:
        model = ScriptedModel(
            ModelTurn(tool_calls=(ModelToolCall("c1", "legal_actions", {}),)),
            ModelTurn(decision=self.decision),
            ModelTurn(tool_calls=(ModelToolCall("c1", "legal_actions", {}),)),
            ModelTurn(decision=self.decision),
        )
        policy = PromptPolicy("policy", model)
        policy.decide(self.request, self.context)
        second_tools = ToolRegistry(standard_tools()).bind(
            self.observation, self.control, max_calls=5, decision_id="d2",
        )
        second_context = DecisionContext(second_tools, Random(1), self.control)
        policy.decide(DecisionRequest("d2", self.observation, 20), second_context)
        self.assertEqual(model.requests[2].exchanges, ())
        self.assertEqual(len(model.requests[3].exchanges), 1)

    def test_empty_and_ambiguous_model_turns_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ModelTurn()
        with self.assertRaises(ValueError):
            ModelTurn(decision=self.decision, tool_calls=(ModelToolCall("c1", "observation", {}),))

    def test_blank_instructions_and_invalid_round_budgets_fail_at_construction(self) -> None:
        for instructions in ("", " \n\t"):
            with self.subTest(instructions=instructions), self.assertRaises(ValueError):
                PromptPolicy(instructions, ScriptedModel())
        for budget in (0, -1, True, 1.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                PromptPolicy("policy", ScriptedModel(), max_rounds=cast(int, budget))

    def test_prompt_identity_is_stable_and_sensitive_to_instruction_changes(self) -> None:
        first = PromptPolicy("# v1", ScriptedModel()).identity
        again = PromptPolicy("# v1", ScriptedModel()).identity
        changed = PromptPolicy("# v2", ScriptedModel()).identity
        self.assertEqual(first, again)
        self.assertNotEqual(first, changed)
        self.assertEqual(first.name, "prompt")
        self.assertNotIn("# v1", first.version)

    def test_expired_or_cancelled_decision_never_calls_model(self) -> None:
        model = ScriptedModel()
        policy = PromptPolicy("policy", model)
        self.clock.now = 20
        with self.assertRaises(DecisionDeadlineExceeded):
            policy.decide(self.request, self.context)
        self.clock.now = 10
        self.control.cancel()
        with self.assertRaises(DecisionCancelled):
            policy.decide(self.request, self.context)
        self.assertEqual(model.requests, [])

    def test_timeouts_shrink_across_model_calls(self) -> None:
        def first(request: ModelRequest) -> ModelTurn:
            self.clock.now = 14
            return ModelTurn(tool_calls=(ModelToolCall("c1", "legal_actions", {}),))

        model = ScriptedModel(first, ModelTurn(decision=self.decision))
        PromptPolicy("policy", model).decide(self.request, self.context)
        self.assertEqual([request.timeout_seconds for request in model.requests], [10, 6])

    def test_late_model_decision_is_discarded(self) -> None:
        def late(request: ModelRequest) -> ModelTurn:
            self.clock.now = 20
            return ModelTurn(decision=self.decision)

        model = ScriptedModel(late)
        with self.assertRaises(DecisionDeadlineExceeded):
            PromptPolicy("policy", model).decide(self.request, self.context)
        self.assertEqual(self.tools.calls, 0)

    def test_cancellation_during_model_request_prevents_tool_execution(self) -> None:
        def cancel(request: ModelRequest) -> ModelTurn:
            self.control.cancel()
            return ModelTurn(tool_calls=(ModelToolCall("c1", "legal_actions", {}),))

        model = ScriptedModel(cancel)
        with self.assertRaises(DecisionCancelled):
            PromptPolicy("policy", model).decide(self.request, self.context)
        self.assertEqual(self.tools.calls, 0)
