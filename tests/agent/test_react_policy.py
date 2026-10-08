from dataclasses import FrozenInstanceError, replace
from typing import cast
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator
from shared_logging import JsonObject, JsonValue

from poker.agent.context import DecisionControl
from poker.agent.models import (
    ActionFeedback, DecisionCancelled, DecisionDeadlineExceeded, DecisionRequest,
    HandCompleted, Observation, ObservationChanged, PolicyIdentity,
)
from poker.agent.react.analysis_tools import EquityTool, LookupTool
from poker.agent.react.memory import BoundedEventMemory, NullMemory
from poker.agent.react.policy import (
    Guidance, Policy, PolicyRequest, StructuredConfig, StructuredPolicy, TextPolicy,
    ToolExchange, WorkflowPolicy, WorkflowStage,
)
from poker.agent.tools import ToolContext, ToolError, ToolRegistry, ToolResult
from poker.application.views import ResultView
from poker.domain.models import ActionOption, PlayerAction
from poker.domain.types import ActionKind, HandId
from tests.agent.test_contracts import Clock, make_observation
from tests.agent.test_naive_policies import river_view


def request(*, memory: JsonObject | None = None, exchanges: tuple[ToolExchange, ...] = (),
            observation: Observation | None = None, round_index: int = 0) -> PolicyRequest:
    decision = DecisionRequest("react-test", observation or make_observation(), 20.0)
    return PolicyRequest(decision, {} if memory is None else memory, exchanges, round_index)


def exchange(name: str, *, call_id: str = "call-1", ok: bool = True) -> ToolExchange:
    result = ToolResult({"evidence": [1]}) if ok else ToolResult(error=ToolError("tool_failed", "Failed"))
    return ToolExchange(call_id, name, {}, result)


class ReactPolicyTests(unittest.TestCase):
    def test_guidance_is_the_only_required_policy_hook_and_returns_no_action(self) -> None:
        class Minimal(Policy):
            def guidance(self, request: PolicyRequest) -> Guidance:
                return Guidance("Choose based on evidence.")

        policy = Minimal()
        self.assertEqual(policy.identity, PolicyIdentity("ReactPolicyTests.test_guidance_is_the_only_required_policy_hook_and_returns_no_action.<locals>.Minimal"))
        self.assertIsInstance(policy.guidance(request()), Guidance)
        self.assertFalse(hasattr(policy, "decide"))
        self.assertFalse(hasattr(policy, "model"))

    def test_text_policy_instructions_and_context_are_detached(self) -> None:
        original: JsonObject = {"style": {"aggression": 0.1}}
        policy = TextPolicy("Avoid marginal calls.", context=original, allowed_tools=("equity",), required_tools=("equity",))
        cast(JsonObject, original["style"])["aggression"] = 0.9
        first = policy.guidance(request())
        cast(JsonObject, first.context["style"])["aggression"] = 0.8
        second = policy.guidance(request())
        self.assertEqual(second.context, {"style": {"aggression": 0.1}})
        self.assertEqual(second.required_tools, ("equity",))
        self.assertNotEqual(second.instructions, TextPolicy("Explore small value bets.").guidance(request()).instructions)

    def test_request_owns_copies_of_memory_and_tool_results(self) -> None:
        memory: JsonObject = {"records": [{"confirmed": True}]}
        result: JsonObject = {"evidence": [1]}
        arguments: JsonObject = {"query": [1]}
        tool_exchange = ToolExchange("x", "lookup", arguments, ToolResult(result))
        incoming = request(memory=memory, exchanges=(tool_exchange,))
        cast(list[JsonValue], incoming.memory["records"]).clear()
        cast(list[JsonValue], cast(JsonObject, incoming.exchanges[0].result.value)["evidence"]).append(2)
        cast(list[JsonValue], incoming.exchanges[0].arguments["query"]).append(2)
        self.assertEqual(memory, {"records": [{"confirmed": True}]})
        self.assertEqual(result, {"evidence": [1]})
        self.assertEqual(arguments, {"query": [1]})
        self.assertEqual(tool_exchange.result.value, {"evidence": [1]})
        with self.assertRaises(FrozenInstanceError):
            setattr(incoming, "round_index", 9)

    def test_guidance_rejects_invalid_types_without_coercion(self) -> None:
        invalid = (
            lambda: Guidance(cast(str, 7)),
            lambda: Guidance("x", allowed_tools=cast(tuple[str, ...], ["equity"])),
            lambda: Guidance("x", required_tools=cast(tuple[str, ...], "equity")),
            lambda: Guidance("x", required_tools=("",)),
            lambda: Guidance("x", allowed_tools=("equity", "equity")),
            lambda: Guidance("x", allowed_tools=("equity",), required_tools=("lookup",)),
            lambda: Guidance("x", context={"number": float("nan")}),
            lambda: Guidance("x", context=cast(JsonObject, {"nested": {1: "coerced"}})),
            lambda: Guidance("x", context=cast(JsonObject, {"tuple": (1, 2)})),
            lambda: Guidance("x", required_after=True),
            lambda: Guidance("x", required_after=-1),
            lambda: request(round_index=True),
        )
        for construct in invalid:
            with self.subTest(construct=construct), self.assertRaises(ValueError):
                construct()
        cyclic: JsonObject = {}
        cyclic["self"] = cyclic
        with self.assertRaises(ValueError):
            Guidance("x", context=cyclic)

    def test_same_observation_can_have_different_structured_guidance(self) -> None:
        observation = make_observation()
        priced = replace(observation, view=replace(observation.view, pot_total=90, me=replace(
            observation.view.me, legal_actions=(ActionOption(ActionKind.FOLD), ActionOption(ActionKind.CALL, pay=10)))))
        incoming = request(observation=priced, memory={"records": [{"confirmed": True}]})
        cautious = StructuredPolicy(StructuredConfig("Use conservative ranges.", {"risk": 0.1},
                                    require_equity_when_facing_bet=True, require_history=True))
        exploratory = StructuredPolicy(StructuredConfig("Explore thin value.", {"risk": 0.8}, require_lookup=True))
        first, second = cautious.guidance(incoming), exploratory.guidance(incoming)
        self.assertEqual(first.required_tools, ("equity", "public_history"))
        self.assertEqual(second.required_tools, ("lookup",))
        self.assertEqual(first.context["situation"], {"phase": "flop", "call_price": 10, "pot_total": 90,
                                                      "pot_odds": 0.1, "history_complete": False})
        self.assertNotEqual(first.instructions, second.instructions)
        self.assertNotEqual(first.context["risk"], second.context["risk"])
        cast(JsonObject, first.context["memory"])["records"] = []
        self.assertNotEqual(first.context["memory"], incoming.memory)
        self.assertEqual(cautious.guidance(request()).required_tools, ("public_history",))

    def test_structured_configuration_rejects_impossible_tool_requirements(self) -> None:
        with self.assertRaises(ValueError):
            StructuredConfig(allowed_tools=("lookup",), require_equity_when_facing_bet=True)
        with self.assertRaises(ValueError):
            StructuredConfig(require_history=cast(bool, 1))

    def test_workflow_advances_only_after_successful_stage_evidence(self) -> None:
        policy = WorkflowPolicy((WorkflowStage("history", "Inspect observed betting.", ("public_history",)),
                                 WorkflowStage("equity", "Compare rough estimates.", ("equity",)),
                                 WorkflowStage("final", "Choose using both pieces of evidence.")),
                                instructions="Preserve a conservative risk preference.")
        self.assertEqual(policy.guidance(request()).required_tools, ("public_history",))
        failed = exchange("public_history", ok=False)
        self.assertEqual(policy.guidance(request(exchanges=(failed,))).required_tools, ("public_history",))
        history = exchange("public_history", call_id="history")
        after_history = policy.guidance(request(exchanges=(failed, history)))
        self.assertEqual(after_history.required_tools, ("equity",))
        self.assertEqual(after_history.required_after, 2)
        self.assertEqual(cast(JsonObject, after_history.context["workflow"])["stage_name"], "equity")
        final = policy.guidance(request(exchanges=(failed, history, exchange("equity"))))
        self.assertEqual(final.required_tools, ())
        self.assertEqual(cast(JsonObject, final.context["workflow"])["stage_name"], "final")
        self.assertIn("Preserve a conservative", final.instructions)

    def test_workflow_requires_ordered_evidence_and_fresh_repeated_tools(self) -> None:
        policy = WorkflowPolicy((WorkflowStage("first", "First sample.", ("equity",)),
                                 WorkflowStage("second", "Sample again.", ("equity",))))
        first = exchange("equity", call_id="first")
        second_stage = policy.guidance(request(exchanges=(first,)))
        self.assertEqual(second_stage.required_tools, ("equity",))
        self.assertEqual(second_stage.required_after, 1)
        self.assertEqual(policy.guidance(request(exchanges=(first, exchange("equity", call_id="second")))).required_tools, ())
        ordered = WorkflowPolicy((WorkflowStage("a", "Get history.", ("public_history",)),
                                  WorkflowStage("b", "Then estimate.", ("equity",))))
        premature = ordered.guidance(request(exchanges=(first, exchange("public_history"))))
        self.assertEqual(premature.required_tools, ("equity",))
        self.assertEqual(premature.required_after, 2)

    def test_workflow_rejects_unreachable_stages_and_disallowed_tools(self) -> None:
        with self.assertRaises(ValueError):
            WorkflowPolicy((WorkflowStage("a", "Finalize."), WorkflowStage("b", "Impossible.", ("lookup",))))
        with self.assertRaises(ValueError):
            WorkflowPolicy((WorkflowStage("a", "Lookup.", ("lookup",)),), allowed_tools=("equity",))


class ReactMemoryTests(unittest.TestCase):
    def test_only_actual_feedback_is_stored_with_deduplication(self) -> None:
        memory = BoundedEventMemory(4)
        observation = make_observation()
        result = ResultView(HandId("completed"), (), ())
        memory.observe(ObservationChanged(replace(observation, view=replace(observation.view, result=result))))
        self.assertEqual(memory.read(observation)["records"], [])
        confirmed = ActionFeedback("d1", PlayerAction(ActionKind.CHECK), True)
        rejected = ActionFeedback("d2", PlayerAction(ActionKind.CHECK), False, "stale_observation")
        for event in (confirmed, confirmed, rejected, HandCompleted(result), HandCompleted(result)):
            memory.observe(event)
        records = cast(list[JsonObject], memory.read(observation)["records"])
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["confirmed"], True)
        self.assertEqual(records[1]["confirmed"], False)
        self.assertEqual(records[1]["error_code"], "stale_observation")
        self.assertEqual(records[2], {"type": "hand_completed", "result": {"hand_id": "completed", "awards": [], "refunds": []}})
        self.assertFalse(any("reward" in item or "equity" in item for item in records))

    def test_memory_is_bounded_and_reads_are_detached(self) -> None:
        memory = BoundedEventMemory(2)
        observation = make_observation()
        for number in range(5):
            memory.observe(ActionFeedback(str(number), PlayerAction(ActionKind.CHECK), True))
        first = memory.read(observation)
        records = cast(list[JsonObject], first["records"])
        self.assertEqual([item["decision_id"] for item in records], ["3", "4"])
        records[0]["confirmed"] = False
        records.clear()
        self.assertEqual(len(cast(list[JsonValue], memory.read(observation)["records"])), 2)
        self.assertEqual(first["deduplication_scope"], "retained_records")
        self.assertEqual(len(memory._records), 2)

    def test_null_memory_ignores_events_and_invalid_capacities_are_rejected(self) -> None:
        memory = NullMemory()
        memory.observe(ActionFeedback("x", PlayerAction(ActionKind.CHECK), True))
        self.assertEqual(memory.read(make_observation()), {})
        memory.close()
        for capacity in (0, -1, True, 2.5):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                BoundedEventMemory(cast(int, capacity))


class ReactAnalysisToolTests(unittest.TestCase):
    def test_input_schema_objects_are_closed_and_require_all_declared_properties(self) -> None:
        def check(value: JsonValue) -> None:
            if isinstance(value, dict):
                if value.get("type") == "object":
                    properties = cast(JsonObject, value.get("properties", {}))
                    self.assertEqual(set(cast(list[str], value["required"])), set(properties))
                    self.assertIs(value.get("additionalProperties"), False)
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)

        for tool in (EquityTool(), LookupTool()):
            with self.subTest(tool=tool.spec.name):
                check(tool.spec.input_schema)
                Draft202012Validator.check_schema(tool.spec.input_schema)
                self.assertTrue(Draft202012Validator(tool.spec.input_schema).is_valid({}))
        validator = Draft202012Validator(EquityTool().spec.input_schema)
        accepted: tuple[JsonObject, ...] = (
            {"candidates": [{"action": "check"}]},
            {"candidates": [{"action": "check", "to": None}]},
            {"candidates": [{"action": "bet_to", "to": 100}]},
            {"candidates": [{"action": "bet_to"}]},
        )
        for arguments in accepted:
            with self.subTest(arguments=arguments):
                # Legality is still checked by invoke; the schema preserves the
                # original optional-to syntax, including its error cases.
                self.assertTrue(validator.is_valid(arguments))
        rejected: tuple[JsonObject, ...] = (
            {"candidates": []}, {"candidates": [{"action": "check", "extra": 1}]},
            {"candidates": [{"action": "bet_to", "to": True}]}, {"extra": 1},
        )
        for invalid_arguments in rejected:
            with self.subTest(arguments=invalid_arguments):
                self.assertFalse(validator.is_valid(invalid_arguments))

    def test_equity_provides_all_candidate_values_without_selecting_an_action(self) -> None:
        observation = Observation(river_view(("As", "Ks"), ("Qs", "Js", "Ts", "2d", "3c")), 10.0)
        control = DecisionControl(20.0, clock=Clock())
        tools = ToolRegistry((EquityTool(rollouts=4),)).bind(observation, control, max_calls=1, decision_id="test")
        with patch("poker.agent.naive_policies.SolverPolicy.decide", side_effect=AssertionError("Must not select")):
            result = tools.call("equity", {})
        self.assertTrue(result.ok)
        value = cast(JsonObject, result.value)
        self.assertEqual(value["equity_estimate"], 1.0)
        scores = cast(list[JsonObject], value["candidate_evs"])
        self.assertEqual({item["action"]: item["estimated_ev"] for item in scores}, {"fold": 0.0, "call": 100.0})
        self.assertNotIn("selected_action", value)
        self.assertNotIn("choice", value)
        self.assertNotIn("decision", value)
        self.assertEqual(value["evaluated_hands"], 8)
        self.assertTrue(value["limitations"])

    def test_equity_repeatability_and_candidate_bounds(self) -> None:
        observation = make_observation()
        control = DecisionControl(20.0, clock=Clock())
        context = ToolContext(observation, control)
        tool = EquityTool(rollouts=4, seed=71)
        arguments: JsonObject = {"candidates": [{"action": "bet_to", "to": 100}, {"action": "check"}]}
        self.assertEqual(tool.invoke(arguments, context), tool.invoke(arguments, context))
        tools = ToolRegistry((tool,)).bind(observation, control, max_calls=4, decision_id="test")
        self.assertFalse(tools.call("equity", {"candidates": [{"action": "bet_to", "to": True}]}).ok)
        self.assertFalse(tools.call("equity", {"candidates": [{"action": "bet_to", "to": 10000}]}).ok)
        self.assertFalse(tools.call("equity", {"candidates": [{"action": "check"}, {"action": "check"}]}).ok)
        self.assertFalse(tools.call("equity", {"deck_seed": 7}).ok)

    def test_all_in_does_not_receive_a_false_zero_payment_ev(self) -> None:
        original = make_observation()
        observation = replace(original, view=replace(original.view, me=replace(original.view.me,
            legal_actions=(ActionOption(ActionKind.ALL_IN, pay=990),))))
        context = ToolContext(observation, DecisionControl(20.0, clock=Clock()))
        value = cast(JsonObject, EquityTool(rollouts=2).invoke({}, context))
        self.assertEqual(value["candidate_evs"], [{"action": "all_in", "to": None, "estimated_ev": None,
                                                  "unsupported_reason": "all_in_side_pot_payoff"}])

    def test_analysis_tools_obey_cancellation_and_deadline(self) -> None:
        clock = Clock()
        control = DecisionControl(20.0, clock=clock)
        context = ToolContext(make_observation(), control)
        control.cancel()
        for tool in (EquityTool(rollouts=2), LookupTool()):
            with self.subTest(tool=tool), self.assertRaises(DecisionCancelled):
                tool.invoke({}, context)
        clock.now = 30.0
        context = ToolContext(make_observation(), DecisionControl(20.0, clock=clock))
        with self.assertRaises(DecisionDeadlineExceeded):
            EquityTool().invoke({}, context)

    def test_lookup_returns_configured_advice_and_records_misses_without_picking(self) -> None:
        context = ToolContext(make_observation(), DecisionControl(20.0, clock=Clock()))
        conservative = LookupTool({"flop:free": (ActionKind.CHECK, ActionKind.BET_TO)})
        active = LookupTool({"flop:free": (ActionKind.BET_TO, ActionKind.CHECK)})
        left = cast(JsonObject, conservative.invoke({}, context))
        right = cast(JsonObject, active.invoke({}, context))
        self.assertNotEqual(left["suggested_priorities"], right["suggested_priorities"])
        self.assertEqual(right["offered_suggestions"], ["bet_to", "check"])
        self.assertNotIn("selected_action", right)
        missing = cast(JsonObject, LookupTool({}).invoke({}, context))
        self.assertFalse(missing["table_hit"])
        self.assertEqual(missing["suggested_priorities"], [])

    def test_tool_construction_rejects_malformed_sampling_or_lookup_configuration(self) -> None:
        for count in (0, 257, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                EquityTool(rollouts=count)
        with self.assertRaises(ValueError):
            EquityTool(seed=True)
        with self.assertRaises(ValueError):
            LookupTool(cast(dict[str, tuple[ActionKind, ...]], {"flop:free": ("check",)}))


if __name__ == "__main__":
    unittest.main()
