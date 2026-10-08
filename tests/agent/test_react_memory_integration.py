"""Runner acknowledgements, rather than model proposals, populate ReAct memory."""

from dataclasses import replace
from typing import cast
import unittest

from shared_logging import JsonObject

from poker.agent.cli_bridge import CliActionResult
from poker.agent.context import DecisionControl
from poker.agent.models import ActionFeedback, Observation, StopReason
from poker.agent.react.backend import BackendIdentity, BackendRequest, BackendResponse, ModelBackend
from poker.agent.react.loop import ReActAgent
from poker.agent.react.memory import BoundedEventMemory
from poker.agent.react.policy import TextPolicy
from poker.agent.runtime import AgentRunner
from poker.application.views import AwardView, ResultView, ShareView
from poker.client.guarded import GuardedResult, ViewToken
from poker.domain.models import PlayerAction
from poker.domain.types import ActionKind, HandId, HandPhase
from tests.agent.test_runtime import FakePort, RunnerHarness, config, offered_view, update


class RecordingBackend(ModelBackend):
    def __init__(self) -> None:
        self.requests: list[BackendRequest] = []
        self.closed = 0

    @property
    def identity(self) -> BackendIdentity:
        return BackendIdentity("test", "memory-integration-fixture", False)

    def generate(self, request: BackendRequest, control: DecisionControl) -> BackendResponse:
        control.check()
        # Retain the actual request object so later memory updates would expose
        # accidental sharing between the loop and a backend input.
        self.requests.append(request)
        return BackendResponse({"kind": "final", "calls": [], "action": {"kind": "check", "to": None}})

    def close(self) -> None:
        self.closed += 1


class ClosingMemory(BoundedEventMemory):
    def __init__(self, max_events: int = 2) -> None:
        super().__init__(max_events)
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def input_records(request: BackendRequest) -> list[JsonObject]:
    return cast(list[JsonObject], cast(JsonObject, request.input["memory"])["records"])


class ReactMemoryIntegrationTests(unittest.TestCase):
    def test_model_choice_waiting_for_cli_confirmation_does_not_create_memory(self) -> None:
        view = offered_view()
        port = FakePort(update(view))
        backend, memory = RecordingBackend(), ClosingMemory()
        agent = ReActAgent(backend, TextPolicy("Consider factual memory."), memory=memory)
        runner = AgentRunner(port, agent, config(max_actions=1, submission_timeout=2, session_timeout=3))
        harness = RunnerHarness(runner)
        try:
            self.assertTrue(port.submission_seen.wait(1), "The model proposal must reach the CLI boundary")
            self.assertEqual(len(backend.requests), 1)
            self.assertEqual(input_records(backend.requests[0]), [])
            self.assertEqual(memory.read(Observation(view, 0))["records"], [])
            _, _, decision_id = port.submitted[0]
            port.confirm(decision_id, replace(view, revision=2, actor_id=None))
            result = harness.result()
        finally:
            runner.request_stop()
            harness.thread.join(timeout=2)
        self.assertTrue(result.ok)
        self.assertEqual(result.confirmed_actions, 1)
        self.assertEqual(memory.read(Observation(view, 0))["records"], [{
            "type": "action_feedback", "decision_id": decision_id,
            "action": {"kind": "check", "to": None}, "confirmed": True, "error_code": None,
        }])
        self.assertEqual((backend.closed, memory.closed), (1, 1))

    def test_actual_action_feedback_and_new_hand_result_reach_next_model_input(self) -> None:
        initial = replace(offered_view(), result=ResultView(HandId("historical"), (), ()))
        hero = initial.me.player_id
        result_view = ResultView(HandId("h1"), (AwardView(20, (hero,), (ShareView(hero, 20),)),), ())
        completed = replace(initial, revision=2, phase=HandPhase.COMPLETE, actor_id=None,
                            me=replace(initial.me, legal_actions=()), result=result_view)
        next_hand = replace(offered_view(revision=3, hand_id="h2"), result=result_view)
        port = FakePort(update(initial))
        backend, memory = RecordingBackend(), ClosingMemory(max_events=2)
        agent = ReActAgent(backend, TextPolicy("Consider factual memory."), memory=memory)

        def acknowledge(action: PlayerAction, token: ViewToken, decision_id: str) -> None:
            if len(port.submitted) == 1:
                port.confirm(decision_id, completed)
                # The old h1 result remains visible in the h2 projection. It
                # must not be counted again as a new completion.
                port.events.put(update(next_hand))
            else:
                port.confirm(decision_id, replace(next_hand, revision=4, actor_id=None))

        port.on_submit = acknowledge
        result = AgentRunner(port, agent, config(max_actions=2)).run()
        self.assertTrue(result.ok)
        self.assertEqual(result.stop_reason, StopReason.MAX_ACTIONS)
        self.assertEqual((result.confirmed_actions, result.observed_hands, result.decisions), (2, 1, 2))
        self.assertEqual(len(backend.requests), 2)
        self.assertEqual(input_records(backend.requests[0]), [], "The result visible at join is historical")
        first_id, second_id = (item[2] for item in port.submitted)
        second_records = input_records(backend.requests[1])
        self.assertEqual(second_records, [
            {"type": "hand_completed", "result": {"hand_id": "h1", "awards": [
                {"amount": 20, "eligible_ids": [hero], "shares": [{"player_id": hero, "amount": 20}]}], "refunds": []}},
            {"type": "action_feedback", "decision_id": first_id, "action": {"kind": "check", "to": None},
             "confirmed": True, "error_code": None},
        ])
        # Graceful max_actions shutdown drains the final acknowledgement before
        # closing memory. Its bounded insertion evicts the oldest hand result.
        retained = cast(list[JsonObject], memory.read(Observation(next_hand, 0))["records"])
        self.assertEqual([item["decision_id"] for item in retained], [first_id, second_id])
        self.assertTrue(all(item["confirmed"] is True for item in retained))
        self.assertEqual((backend.closed, memory.closed), (1, 1))
        before = memory.read(Observation(next_hand, 0))
        agent.observe(ActionFeedback("after-close", PlayerAction(ActionKind.CHECK), True))
        self.assertEqual(memory.read(Observation(next_hand, 0)), before)

    def test_rejected_proposal_reaches_next_input_as_rejection_never_confirmation(self) -> None:
        initial, refreshed = offered_view(), offered_view(revision=2)
        port = FakePort(update(initial))
        backend, memory = RecordingBackend(), ClosingMemory()
        agent = ReActAgent(backend, TextPolicy("Consider factual memory."), memory=memory)

        def acknowledge(action: PlayerAction, token: ViewToken, decision_id: str) -> None:
            if len(port.submitted) == 1:
                port.events.put(update(refreshed))
                port.events.put(CliActionResult(GuardedResult(decision_id, error="stale_observation")))
            else:
                port.confirm(decision_id, replace(refreshed, revision=3, actor_id=None))

        port.on_submit = acknowledge
        result = AgentRunner(port, agent, config(max_actions=1)).run()
        self.assertTrue(result.ok)
        self.assertEqual((result.decisions, result.confirmed_actions), (2, 1))
        self.assertEqual(len(backend.requests), 2)
        self.assertEqual(input_records(backend.requests[0]), [])
        self.assertEqual(input_records(backend.requests[1]), [{
            "type": "action_feedback", "decision_id": port.submitted[0][2],
            "action": {"kind": "check", "to": None}, "confirmed": False, "error_code": "stale_observation",
        }])
        retained = cast(list[JsonObject], memory.read(Observation(refreshed, 0))["records"])
        self.assertEqual([item["confirmed"] for item in retained], [False, True])
        self.assertFalse(any("reward" in item for item in retained))
        self.assertEqual((backend.closed, memory.closed), (1, 1))


if __name__ == "__main__":
    unittest.main()
