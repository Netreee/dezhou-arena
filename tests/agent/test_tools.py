from collections.abc import Callable
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from typing import cast

from jsonschema import SchemaError
from shared_logging import (
    JsonObject, JsonValue, LoggingConfig, configure_logging, shutdown_logging,
)

from poker.agent.context import DecisionControl
from poker.agent.models import DecisionCancelled, DecisionDeadlineExceeded, Observation, ToolBudgetExceeded
from poker.agent.tools import (
    BoundTools, Tool, ToolContext, ToolRegistry, ToolSpec, standard_tools,
)
from poker.application.views import PlayerViewBuilder
from poker.domain.models import ActionOption, PublicActionRecord
from poker.domain.types import ActionKind, HandPhase, PlayerId
from tests.agent.test_contracts import Clock, make_observation
from tests.fixtures import make_table


class FunctionTool(Tool):
    def __init__(self, spec: ToolSpec, function: Callable[[JsonObject, ToolContext], JsonValue]) -> None:
        self._spec = spec
        self.function = function
        self.invocations = 0

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def invoke(self, arguments: JsonObject, context: ToolContext) -> JsonValue:
        self.invocations += 1
        return self.function(arguments, context)


def echo_spec() -> ToolSpec:
    return ToolSpec("echo", "Return a validated integer value.", {
        "type": "object", "properties": {"value": {"type": "integer"}},
        "required": ["value"], "additionalProperties": False,
    }, {"type": "object", "required": ["value"]})


class AgentToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.path = configure_logging(LoggingConfig(Path(self.temp.name), "agent-tools-test", "test"))
        self.observation = make_observation()
        self.clock = Clock()
        self.control = DecisionControl(20, clock=self.clock)

    def tearDown(self) -> None:
        shutdown_logging()
        self.temp.cleanup()

    def bind(self, *tools: Tool, max_calls: int = 5) -> BoundTools:
        return ToolRegistry(tools).bind(self.observation, self.control, max_calls=max_calls, decision_id="d1")

    def test_registry_rejects_duplicate_empty_names_and_invalid_schemas(self) -> None:
        good = FunctionTool(echo_spec(), lambda args, ctx: args)
        with self.assertRaises(ValueError):
            ToolRegistry((good, good))
        with self.assertRaises(ValueError):
            ToolRegistry((FunctionTool(ToolSpec("", "bad", {}), lambda args, ctx: args),))
        for spec in (
            ToolSpec("bad", "bad", {"type": "not-a-json-type"}),
            ToolSpec("bad", "bad", {}, {"type": "not-a-json-type"}),
        ):
            with self.subTest(spec=spec), self.assertRaises(SchemaError):
                ToolRegistry((FunctionTool(spec, lambda args, ctx: args),))

    def test_descriptions_are_detached_from_tool_and_registry_schemas(self) -> None:
        spec = echo_spec()
        tool = FunctionTool(spec, lambda args, ctx: args)
        bound = self.bind(tool)
        spec.input_schema["required"] = []
        described = bound.describe()
        described[0].input_schema["required"] = []
        self.assertEqual(bound.describe()[0].input_schema["required"], ["value"])
        result = bound.call("echo", {})
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code if result.error else None, "invalid_arguments")
        self.assertEqual(tool.invocations, 0)

    def test_invalid_arguments_never_reach_tool_or_get_silently_repaired(self) -> None:
        tool = FunctionTool(echo_spec(), lambda args, ctx: args)
        bound = self.bind(tool, max_calls=10)
        invalid: tuple[JsonObject, ...] = ({}, {"value": "3"}, {"value": True}, {"value": 3, "extra": 1})
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                result = bound.call("echo", arguments)
                self.assertFalse(result.ok)
                self.assertEqual(result.error.code if result.error else None, "invalid_arguments")
        self.assertEqual(tool.invocations, 0)
        self.assertEqual(bound.calls, len(invalid))
        result = bound.call("echo", {"value": 3})
        self.assertTrue(result.ok)
        self.assertEqual(result.value, {"value": 3})
        self.assertEqual(tool.invocations, 1)

    def test_unknown_tools_are_unavailable_and_consume_attempt_budget(self) -> None:
        tool = FunctionTool(echo_spec(), lambda args, ctx: args)
        bound = self.bind(tool, max_calls=1)
        self.assertEqual([spec.name for spec in bound.describe()], ["echo"])
        result = bound.call("arena_private_state", {})
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code if result.error else None, "unknown_tool")
        self.assertEqual(tool.invocations, 0)
        with self.assertRaises(ToolBudgetExceeded):
            bound.call("echo", {"value": 3})
        self.assertEqual(bound.calls, 1)

    def test_budget_is_per_decision_and_zero_disables_calls(self) -> None:
        tool = FunctionTool(echo_spec(), lambda args, ctx: args)
        registry = ToolRegistry((tool,))
        first = registry.bind(self.observation, self.control, max_calls=1, decision_id="a")
        second = registry.bind(self.observation, self.control, max_calls=1, decision_id="b")
        self.assertTrue(first.call("echo", {"value": 1}).ok)
        with self.assertRaises(ToolBudgetExceeded):
            first.call("echo", {"value": 1})
        self.assertTrue(second.call("echo", {"value": 2}).ok)
        blocked = registry.bind(self.observation, self.control, max_calls=0, decision_id="c")
        self.assertEqual(len(blocked.describe()), 1)
        with self.assertRaises(ToolBudgetExceeded):
            blocked.call("echo", {"value": 3})

    def test_budget_rejects_negative_boolean_and_noninteger_values(self) -> None:
        registry = ToolRegistry(())
        for budget in (-1, True, 1.5, float("nan")):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                registry.bind(self.observation, self.control, max_calls=cast(int, budget), decision_id="d1")

    def test_mutating_tool_cannot_change_callers_input_or_returned_result(self) -> None:
        retained: JsonObject = {}

        def mutate(arguments: JsonObject, context: ToolContext) -> JsonValue:
            arguments["value"] = 99
            retained["nested"] = arguments
            return retained

        bound = self.bind(FunctionTool(ToolSpec("mutate", "test", {"type": "object"}), mutate))
        original: JsonObject = {"value": 1}
        result = bound.call("mutate", original)
        self.assertEqual(original, {"value": 1})
        self.assertEqual(result.value, {"nested": {"value": 99}})
        retained.clear()
        self.assertEqual(result.value, {"nested": {"value": 99}})

    def test_output_schema_failure_is_explicit_and_hides_invalid_payload(self) -> None:
        bound = self.bind(FunctionTool(echo_spec(), lambda args, ctx: "sensitive-invalid-output"))
        result = bound.call("echo", {"value": 2})
        self.assertFalse(result.ok)
        self.assertIsNone(result.value)
        self.assertEqual(result.error.code if result.error else None, "invalid_output")
        self.assertNotIn("sensitive-invalid-output", self.path.read_text(encoding="utf-8"))

    def test_nonfinite_json_arguments_and_results_are_rejected(self) -> None:
        tool = FunctionTool(ToolSpec("number", "test", {"type": "object"}), lambda args, ctx: float("nan"))
        bound = self.bind(tool)
        bad_input = bound.call("number", {"value": float("inf")})
        self.assertFalse(bad_input.ok)
        self.assertEqual(tool.invocations, 0)
        bad_output = bound.call("number", {})
        self.assertFalse(bad_output.ok)
        self.assertIsNone(bad_output.value)
        self.assertEqual(tool.invocations, 1)

    def test_tool_exception_text_arguments_and_result_are_not_logged(self) -> None:
        secret = "secret-value-not-in-any-diagnostic"

        def explode(arguments: JsonObject, context: ToolContext) -> JsonValue:
            raise RuntimeError(secret)

        bound = self.bind(FunctionTool(ToolSpec("explode", "test", {"type": "object"}), explode))
        result = bound.call("explode", {"prompt": secret})
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code if result.error else None, "tool_failed")
        self.assertNotIn(secret, repr(result))
        raw = self.path.read_text(encoding="utf-8")
        self.assertNotIn(secret, raw)
        rows = [json.loads(line) for line in raw.splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event"], "agent.tool.completed")
        self.assertEqual(rows[0]["data"]["error_code"], "tool_failed")

    def test_standard_tools_expose_only_frozen_player_view_and_detached_json(self) -> None:
        table = make_table(with_hand=True)
        view = PlayerViewBuilder().build(table, PlayerId("p1"), (ActionOption(ActionKind.CHECK),))
        observation = Observation(view, 10)
        bound = ToolRegistry(standard_tools()).bind(observation, self.control, max_calls=5, decision_id="d1")
        table.player(PlayerId("p1")).stack = 1
        state = bound.call("observation", {})
        self.assertTrue(state.ok)
        value = cast(JsonObject, state.value)
        self.assertNotIn("deck", value)
        self.assertEqual(cast(JsonObject, value["me"])["hole_cards"], list(view.me.hole_cards))
        players = cast(list[JsonObject], value["players"])
        self.assertEqual(players[0]["stack"], 990)
        self.assertNotIn("hole_cards", players[1])
        self.assertEqual(players[1]["revealed_cards"], [])
        players[0]["stack"] = 999999
        again = cast(JsonObject, bound.call("observation", {}).value)
        self.assertEqual(cast(list[JsonObject], again["players"])[0]["stack"], 990)
        legal = bound.call("legal_actions", {})
        self.assertEqual(legal.value, [{"kind": "check", "pay": None, "min_to": None, "max_to": None}])
        history = cast(JsonObject, bound.call("public_history", {}).value)
        self.assertEqual(history["hand_id"], view.hand_id)
        self.assertFalse(history["complete"])
        self.assertEqual(history["actions"], [])

    def test_cancelled_calls_and_descriptions_never_invoke_tools_or_consume_budget(self) -> None:
        tool = FunctionTool(echo_spec(), lambda args, ctx: args)
        bound = self.bind(tool)
        self.control.cancel()
        with self.assertRaises(DecisionCancelled):
            bound.describe()
        with self.assertRaises(DecisionCancelled):
            bound.call("echo", {"value": 1})
        self.assertEqual(tool.invocations, 0)
        self.assertEqual(bound.calls, 0)

    def test_history_tool_preserves_server_sequence_amounts_and_completeness(self) -> None:
        records = (
            PublicActionRecord(1, HandPhase.PREFLOP, PlayerId("p1"), "small_blind", 5, 5, 995),
            PublicActionRecord(2, HandPhase.PREFLOP, PlayerId("p2"), "big_blind", 10, 10, 990),
            PublicActionRecord(3, HandPhase.PREFLOP, PlayerId("p1"), "raise_to", 25, 30, 970),
        )
        view = replace(self.observation.view, action_history=records, history_complete=True)
        observation = Observation(view, 10)
        bound = ToolRegistry(standard_tools()).bind(observation, self.control, max_calls=1, decision_id="d1")
        result = cast(JsonObject, bound.call("public_history", {}).value)
        self.assertTrue(result["complete"])
        actions = cast(list[JsonObject], result["actions"])
        self.assertEqual([item["sequence"] for item in actions], [1, 2, 3])
        self.assertEqual(actions[2], {
            "sequence": 3, "phase": "preflop", "player_id": "p1", "kind": "raise_to",
            "pay": 25, "to": 30, "stack": 970,
        })

    def test_expired_calls_do_not_invoke_tools(self) -> None:
        tool = FunctionTool(echo_spec(), lambda args, ctx: args)
        bound = self.bind(tool)
        self.clock.now = 20
        with self.assertRaises(DecisionDeadlineExceeded):
            bound.call("echo", {"value": 1})
        self.assertEqual(tool.invocations, 0)

    def test_late_and_cancelled_results_are_discarded_after_tool_returns(self) -> None:
        def late(arguments: JsonObject, context: ToolContext) -> JsonValue:
            self.clock.now = 20
            return arguments

        with self.assertRaises(DecisionDeadlineExceeded):
            self.bind(FunctionTool(echo_spec(), late)).call("echo", {"value": 1})
        self.clock.now = 10

        def cancel(arguments: JsonObject, context: ToolContext) -> JsonValue:
            context.control.cancel()
            return arguments

        with self.assertRaises(DecisionCancelled):
            self.bind(FunctionTool(echo_spec(), cancel)).call("echo", {"value": 1})

    def test_cooperative_control_errors_are_not_converted_to_tool_failure(self) -> None:
        def cancel(arguments: JsonObject, context: ToolContext) -> JsonValue:
            context.control.cancel()
            context.control.check()
            return arguments

        with self.assertRaises(DecisionCancelled):
            self.bind(FunctionTool(echo_spec(), cancel)).call("echo", {"value": 1})
