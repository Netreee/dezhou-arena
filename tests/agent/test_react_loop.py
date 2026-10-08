from copy import deepcopy
import json
from dataclasses import replace
from random import Random
from typing import cast
import unittest

from shared_logging import JsonObject
from poker.agent.context import DecisionContext, DecisionControl
from poker.agent.models import DecisionCancelled, DecisionRequest, PolicyError, PolicySession, StopReason
from poker.agent.react.backend import BackendIdentity, BackendRequest, BackendResponse, BackendUsage, ModelBackend
from poker.agent.react.loop import LoopConfig, ReActAgent
from poker.agent.react.memory import NullMemory
from poker.agent.react.policy import Guidance, Policy, PolicyRequest, TextPolicy, WorkflowPolicy, WorkflowStage
from poker.agent.tools import Tool, ToolContext, ToolRegistry, ToolSpec, standard_tools
from poker.domain.types import ActionKind
from tests.agent.test_contracts import Clock, make_observation


FINAL: JsonObject = {"kind": "final", "calls": [], "action": {"kind": "check", "to": None}}


def tool(name: str, ident: str = "call-1") -> JsonObject:
    return {"kind": "tool_calls", "calls": [{"id": ident, "name": name, "arguments": {}}], "action": None}


class QueueBackend(ModelBackend):
    def __init__(self, *outputs: JsonObject | Exception) -> None:
        self.outputs = list(outputs)
        self.requests: list[BackendRequest] = []
        self.closed = 0

    @property
    def identity(self) -> BackendIdentity:
        return BackendIdentity("test", "queue", False)

    def generate(self, request: BackendRequest, control: DecisionControl) -> BackendResponse:
        self.requests.append(deepcopy(request))
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return BackendResponse(deepcopy(value), BackendUsage(input_tokens=7, output_tokens=2))

    def close(self) -> None:
        self.closed += 1


class AdviceTool(Tool):
    def __init__(self, *, fails: bool = False) -> None:
        self.calls = 0
        self.fails = fails

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec("advice", "A computational recommendation, never a submitted action.",
                        {"type": "object", "properties": {}, "additionalProperties": False})

    def invoke(self, arguments: JsonObject, context: ToolContext) -> JsonObject:
        self.calls += 1
        if self.fails:
            raise RuntimeError("private diagnostic")
        return {"suggested_action": "bet_to", "to": 10}


class ReactLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observation = make_observation()
        self.control = DecisionControl(30, clock=Clock())
        self.request = DecisionRequest("d1", self.observation, 30)

    def context(self, *extra: Tool) -> DecisionContext:
        tools = ToolRegistry((*standard_tools(), *extra)).bind(self.observation, self.control,
                                                              max_calls=8, decision_id="d1")
        return DecisionContext(tools, Random(1), self.control)

    def test_text_policy_still_requires_a_backend_final(self) -> None:
        backend = QueueBackend(FINAL)
        agent = ReActAgent(backend, TextPolicy("Use sound judgement."))
        result = agent.decide(self.request, self.context())
        self.assertEqual(result.choice.kind, ActionKind.CHECK)  # type: ignore[union-attr]
        self.assertEqual(len(backend.requests), 1)
        self.assertEqual(agent.usage["model_attempts"], 1)
        self.assertEqual(agent.usage["known_input_tokens"], 7)

    def test_premature_final_is_never_submitted_and_tool_result_reaches_model(self) -> None:
        backend = QueueBackend(FINAL, tool("legal_actions"), FINAL)
        agent = ReActAgent(backend, TextPolicy("First analyse.", required_tools=("legal_actions",)))
        result = agent.decide(self.request, self.context())
        self.assertEqual(result.choice.kind, ActionKind.CHECK)  # type: ignore[union-attr]
        self.assertEqual(len(backend.requests), 3)
        self.assertTrue(backend.requests[1].input["feedback"])
        exchanges = cast(list[JsonObject], backend.requests[2].input["exchanges"])
        self.assertEqual(exchanges[0]["name"], "legal_actions")
        self.assertIsNone(cast(JsonObject, exchanges[0]["result"])["error"])

    def test_solver_recommendation_is_information_and_final_model_choice_wins(self) -> None:
        solver = AdviceTool()
        backend = QueueBackend(tool("advice"), FINAL)
        result = ReActAgent(backend, TextPolicy("Compare the recommendation.", required_tools=("advice",))).decide(
            self.request, self.context(solver))
        self.assertEqual(solver.calls, 1)
        self.assertEqual(result.choice.kind, ActionKind.CHECK)  # type: ignore[union-attr]
        exchange = cast(list[JsonObject], backend.requests[1].input["exchanges"])[0]
        self.assertEqual(cast(JsonObject, cast(JsonObject, exchange["result"])["value"])["suggested_action"], "bet_to")

    def test_failed_analysis_does_not_satisfy_required_tools(self) -> None:
        backend = QueueBackend(tool("advice"), FINAL, FINAL)
        agent = ReActAgent(backend, TextPolicy("Analyse.", required_tools=("advice",)),
                           config=LoopConfig(max_model_calls_per_decision=3))
        with self.assertRaisesRegex(PolicyError, "budget"):
            agent.decide(self.request, self.context(AdviceTool(fails=True)))
        self.assertEqual(len(backend.requests), 3)
        self.assertEqual(backend.requests[-1].input["pending_tools"], ["advice"])

    def test_removing_guidance_requirement_cannot_erase_unfulfilled_obligation(self) -> None:
        class WithdrawingPolicy(Policy):
            def guidance(self, request: PolicyRequest) -> Guidance:
                return Guidance("Analyse.", required_tools=("legal_actions",) if request.round_index == 0 else ())
        backend = QueueBackend(FINAL, FINAL)
        with self.assertRaisesRegex(PolicyError, "budget"):
            ReActAgent(backend, WithdrawingPolicy(), config=LoopConfig(max_model_calls_per_decision=2)).decide(
                self.request, self.context())
        self.assertEqual(backend.requests[1].input["pending_tools"], ["legal_actions"])

    def test_workflow_requires_fresh_result_after_previous_stage(self) -> None:
        policy = WorkflowPolicy((WorkflowStage("history", "Read history.", ("public_history",)),
                                 WorkflowStage("analysis", "Then read actions.", ("legal_actions",))))
        backend = QueueBackend(tool("legal_actions", "early"), tool("public_history", "history"),
                               FINAL, tool("legal_actions", "after-history"), FINAL)
        result = ReActAgent(backend, policy).decide(self.request, self.context())
        self.assertEqual(result.choice.kind, ActionKind.CHECK)  # type: ignore[union-attr]
        self.assertEqual(len(backend.requests), 5)
        self.assertEqual(backend.requests[2].input["pending_tools"], ["legal_actions"])

    def test_workflow_repeating_same_tool_requires_two_executions(self) -> None:
        policy = WorkflowPolicy((WorkflowStage("one", "First.", ("legal_actions",)),
                                 WorkflowStage("two", "Again.", ("legal_actions",))))
        backend = QueueBackend(tool("legal_actions", "one"), FINAL, tool("legal_actions", "two"), FINAL)
        ReActAgent(backend, policy).decide(self.request, self.context())
        self.assertEqual(len(backend.requests), 4)

    def test_unavailable_required_tool_fails_before_backend(self) -> None:
        backend = QueueBackend()
        with self.assertRaisesRegex(PolicyError, "unavailable"):
            ReActAgent(backend, TextPolicy("Need a tool.", required_tools=("missing",))).decide(
                self.request, self.context())
        self.assertEqual(backend.requests, [])

    def test_disallowed_tool_cannot_execute(self) -> None:
        advice = AdviceTool()
        backend = QueueBackend(tool("advice"))
        with self.assertRaisesRegex(PolicyError, "protocol"):
            ReActAgent(backend, TextPolicy("Only read.", allowed_tools=("legal_actions",))).decide(
                self.request, self.context(advice))
        self.assertEqual(advice.calls, 0)

    def test_duplicate_call_id_cannot_execute_twice(self) -> None:
        backend = QueueBackend(tool("legal_actions"), tool("legal_actions"))
        context = self.context()
        with self.assertRaisesRegex(PolicyError, "unique"):
            ReActAgent(backend, TextPolicy("Read.")).decide(self.request, context)
        self.assertEqual(context.tools.calls, 1)  # type: ignore[attr-defined]

    def test_backend_cannot_mutate_schema_to_enable_disallowed_tool(self) -> None:
        class MutatingBackend(QueueBackend):
            def generate(self, request: BackendRequest, control: DecisionControl) -> BackendResponse:
                request.output_schema.clear()
                request.output_schema.update({"type": "object"})
                return super().generate(request, control)
        advice = AdviceTool()
        backend = MutatingBackend(tool("advice"), FINAL)
        with self.assertRaisesRegex(PolicyError, "protocol"):
            ReActAgent(backend, TextPolicy("Only read.", allowed_tools=("legal_actions",))).decide(
                self.request, self.context(advice))
        self.assertEqual(advice.calls, 0)

    def test_cancel_after_completed_response_preserves_known_usage(self) -> None:
        class CancellingBackend(QueueBackend):
            def generate(self, request: BackendRequest, control: DecisionControl) -> BackendResponse:
                control.cancel()
                return BackendResponse(deepcopy(FINAL), BackendUsage(input_tokens=120, output_tokens=30))
        agent = ReActAgent(CancellingBackend(), TextPolicy("Act."))
        with self.assertRaises(DecisionCancelled):
            agent.decide(self.request, self.context())
        self.assertEqual(agent.usage["model_attempts"], 1)
        self.assertEqual(agent.usage["model_completions"], 1)
        self.assertEqual(agent.usage["known_input_tokens"], 120)
        self.assertEqual(agent.usage["known_output_tokens"], 30)

    def test_failed_backend_call_consumes_session_budget_without_retry(self) -> None:
        backend = QueueBackend(PolicyError("provider failed"))
        agent = ReActAgent(backend, TextPolicy("Act."), config=LoopConfig(max_model_calls_per_session=1))
        with self.assertRaisesRegex(PolicyError, "provider failed"):
            agent.decide(self.request, self.context())
        with self.assertRaisesRegex(PolicyError, "session"):
            agent.decide(self.request, self.context())
        self.assertEqual(len(backend.requests), 1)
        self.assertEqual(agent.usage["model_attempts"], 1)

    def test_context_limit_prevents_model_call(self) -> None:
        backend = QueueBackend()
        with self.assertRaisesRegex(PolicyError, "context"):
            ReActAgent(backend, TextPolicy("Act."), config=LoopConfig(max_context_bytes=5)).decide(
                self.request, self.context())
        self.assertEqual(backend.requests, [])

    def test_invalid_output_never_becomes_an_implicit_default_action(self) -> None:
        values: tuple[object, ...] = ({}, {**FINAL, "extra": "private"}, {**FINAL, "action": {"kind": "call", "to": None}},
                                      json.loads('{"kind":"final","calls":[],"action":{"kind":"bet_to","to":true}}'))
        for value in values:
            with self.subTest(value=value), self.assertRaises(PolicyError):
                ReActAgent(QueueBackend(cast(JsonObject, value)), TextPolicy("Act.")).decide(self.request, self.context())

    def test_cancelled_decision_does_not_call_model(self) -> None:
        backend = QueueBackend()
        self.control.cancel()
        with self.assertRaises(DecisionCancelled):
            ReActAgent(backend, TextPolicy("Act.")).decide(self.request, self.context())
        self.assertEqual(backend.requests, [])

    def test_resources_close_once_and_agent_cannot_reopen(self) -> None:
        class CountingMemory(NullMemory):
            def __init__(self) -> None:
                self.closed = 0
            def close(self) -> None:
                self.closed += 1
        backend, memory = QueueBackend(), CountingMemory()
        agent = ReActAgent(backend, TextPolicy("Act."), memory=memory)
        agent.close(StopReason.REQUESTED)
        agent.close(StopReason.REQUESTED)
        self.assertEqual((backend.closed, memory.closed), (1, 1))
        with self.assertRaises(PolicyError):
            agent.open(PolicySession("agent", "name"))

    def test_invalid_budgets_fail_before_running(self) -> None:
        for value in (0, -1, True, 1.2):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(LoopConfig(), max_model_calls_per_session=value)  # type: ignore[arg-type]
