from collections.abc import Callable
from dataclasses import replace
from queue import Empty, Queue
from threading import Event, Thread, get_ident
from typing import cast
import unittest

from poker.agent.cli_bridge import AgentCliPort, CliActionResult, CliEvent, CliObservation, CliStopped
from poker.agent.context import DecisionContext
from poker.agent.models import (
    ActionFeedback, Decision, DecisionRequest, HandCompleted, ObservationChanged,
    PolicyEvent, PolicySession, StopReason,
)
from poker.agent.policy import Policy
from poker.agent.runtime import AgentConfig, AgentRunner, RunResult
from poker.application.views import CommandResponse, ErrorInfo, PlayerView, ResultView
from poker.client.guarded import GuardedResult, ViewToken
from poker.domain.models import ActionOption, PlayerAction
from poker.domain.types import ActionKind, ErrorCode, HandId, HandPhase, PlayerId, TableId
from tests.client.test_polling import server_view


def offered_view(revision: int = 1, hand_id: str = "h1") -> PlayerView:
    view = server_view(hand_id, revision)
    return replace(view, me=replace(view.me, legal_actions=(ActionOption(ActionKind.CHECK),)))


def update(view: PlayerView, source: str = "poll") -> CliObservation:
    return CliObservation(source, CommandResponse(view=view))


class FakePort(AgentCliPort):
    """Thread-safe CLI boundary; never executes Policy callbacks itself."""

    def __init__(self, *initial: CliEvent) -> None:
        self.events: Queue[CliEvent] = Queue()
        for event in initial:
            self.events.put(event)
        self.starts = 0
        self.closes = 0
        self.start_hands = 0
        self.consumed: list[CliEvent] = []
        self.submitted: list[tuple[PlayerAction, ViewToken, str]] = []
        self.submission_seen = Event()
        self.on_submit: Callable[[PlayerAction, ViewToken, str], None] | None = None
        self.on_receive: Callable[[CliEvent], None] | None = None
        self.close_result = True

    def start(self, name: str) -> None:
        self.starts += 1

    def next_event(self, timeout: float = 0.0) -> CliEvent | None:
        if self.starts == 0:
            Event().wait(timeout)
            return None
        try:
            event = self.events.get(timeout=timeout)
        except Empty:
            return None
        self.consumed.append(event)
        if self.on_receive is not None:
            self.on_receive(event)
        return event

    def submit(self, action: PlayerAction, token: ViewToken, request_id: str) -> None:
        self.submitted.append((action, token, request_id))
        self.submission_seen.set()
        if self.on_submit is not None:
            self.on_submit(action, token, request_id)

    def start_hand(self) -> None:
        self.start_hands += 1

    def close(self) -> bool:
        self.closes += 1
        return self.close_result

    def confirm(self, request_id: str, view: PlayerView) -> None:
        response = CommandResponse(view=view)
        self.events.put(CliObservation("check", response))
        self.events.put(CliActionResult(GuardedResult(request_id, response=response)))


class RecordingPolicy(Policy):
    def __init__(self) -> None:
        self.trace: list[str] = []
        self.events: list[PolicyEvent] = []
        self.requests: list[DecisionRequest] = []
        self.owners: set[int] = set()
        self.closed = Event()
        self.opened = Event()
        self.on_open: Callable[[], None] | None = None
        self.on_observe: Callable[[PolicyEvent], None] | None = None
        self.on_decide: Callable[[DecisionRequest, DecisionContext], Decision] | None = None
        self.on_close: Callable[[], None] | None = None

    def open(self, session: PolicySession) -> None:
        self.owners.add(get_ident())
        self.trace.append("open")
        self.opened.set()
        if self.on_open is not None:
            self.on_open()

    def observe(self, event: PolicyEvent) -> None:
        self.owners.add(get_ident())
        self.trace.append("observe:" + type(event).__name__)
        self.events.append(event)
        if self.on_observe is not None:
            self.on_observe(event)

    def decide(self, request: DecisionRequest, context: DecisionContext) -> Decision:
        self.owners.add(get_ident())
        self.trace.append("decide.begin")
        self.requests.append(request)
        try:
            if self.on_decide is not None:
                return self.on_decide(request, context)
            return Decision(PlayerAction(ActionKind.CHECK))
        finally:
            self.trace.append("decide.end")

    def close(self, reason: StopReason) -> None:
        self.owners.add(get_ident())
        self.trace.append("close.begin")
        try:
            if self.on_close is not None:
                self.on_close()
        finally:
            self.trace.append("close.end")
            self.closed.set()


def config(
    *, callback_timeout: float = 0.3, decision_timeout: float = 0.3,
    submission_timeout: float = 0.3, shutdown_timeout: float = 0.1,
    session_timeout: float | None = 1.5, max_actions: int | None = None,
    max_hands: int | None = None, max_tool_calls: int = 8, auto_start: bool = False,
) -> AgentConfig:
    return AgentConfig(
        callback_timeout=callback_timeout, decision_timeout=decision_timeout,
        submission_timeout=submission_timeout, shutdown_timeout=shutdown_timeout,
        session_timeout=session_timeout, max_actions=max_actions, max_hands=max_hands,
        max_tool_calls=max_tool_calls, auto_start=auto_start,
    )


class RunnerHarness:
    def __init__(self, runner: AgentRunner) -> None:
        self.runner = runner
        self.results: Queue[RunResult | BaseException] = Queue()
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            self.results.put(self.runner.run())
        except BaseException as error:
            self.results.put(error)

    def result(self) -> RunResult:
        result = self.results.get(timeout=3)
        self.thread.join(0.5)
        if isinstance(result, BaseException):
            raise result
        return result


class AgentRuntimeTests(unittest.TestCase):
    def test_confirmed_action_drains_feedback_and_hooks_are_serial(self) -> None:
        view = offered_view()
        port = FakePort(update(view), update(view), update(view))
        port.on_submit = lambda action, token, rid: port.confirm(rid, replace(view, revision=2, actor_id=None))
        policy = RecordingPolicy()
        runner = AgentRunner(port, policy, config(max_actions=1))
        result = runner.run()
        self.assertEqual(result.stop_reason, StopReason.MAX_ACTIONS)
        self.assertTrue(result.ok)
        self.assertEqual((result.confirmed_actions, result.decisions), (1, 1))
        observations = [event for event in policy.events if isinstance(event, ObservationChanged)]
        self.assertEqual([event.observation.view.revision for event in observations], [1, 2])
        feedback = [event for event in policy.events if isinstance(event, ActionFeedback)]
        self.assertEqual(len(feedback), 1)
        self.assertTrue(feedback[0].confirmed)
        self.assertEqual(policy.trace[0], "open")
        self.assertEqual(policy.trace[-2:], ["close.begin", "close.end"])
        self.assertEqual(len(policy.owners), 1)
        self.assertNotIn(get_ident(), policy.owners)
        self.assertEqual((port.starts, port.closes), (1, 1))
        with self.assertRaisesRegex(RuntimeError, "only once"):
            runner.run()
        self.assertEqual(port.starts, 1)

    def test_slow_policy_does_not_block_cli_and_obsolete_result_is_discarded(self) -> None:
        view = offered_view()
        port = FakePort(update(view))
        policy = RecordingPolicy()
        entered, release, received_new = Event(), Event(), Event()
        newer = replace(view, revision=2)

        def receive(event: CliEvent) -> None:
            if isinstance(event, CliObservation) and event.response.view == newer:
                received_new.set()

        def decide(request: DecisionRequest, context: DecisionContext) -> Decision:
            if request.observation.view.revision == 1:
                entered.set()
                release.wait(2)
            return Decision(PlayerAction(ActionKind.CHECK))

        policy.on_decide = decide
        port.on_receive = receive
        port.on_submit = lambda action, token, rid: port.confirm(rid, replace(newer, revision=3, actor_id=None))
        runner = AgentRunner(port, policy, config(max_actions=1, decision_timeout=1.0))
        harness = RunnerHarness(runner)
        try:
            self.assertTrue(entered.wait(1))
            port.events.put(update(newer))
            self.assertTrue(received_new.wait(0.5), "CLI events must be consumed while Policy.decide is blocked")
            self.assertFalse(release.is_set())
            self.assertEqual(port.submitted, [])
        finally:
            release.set()
        result = harness.result()
        self.assertTrue(result.ok)
        self.assertEqual([token.revision for _, token, _ in port.submitted], [2])
        self.assertEqual(result.decisions, 2)
        first_end = policy.trace.index("decide.end")
        self.assertNotIn("observe:ObservationChanged", policy.trace[policy.trace.index("decide.begin") + 1:first_end])

    def test_same_revision_does_not_redecide_while_waiting_for_acknowledgement(self) -> None:
        view = offered_view()
        port = FakePort(update(view))
        policy = RecordingPolicy()
        runner = AgentRunner(port, policy, config(max_actions=1))
        harness = RunnerHarness(runner)
        try:
            self.assertTrue(port.submission_seen.wait(1))
            for _ in range(5):
                port.events.put(update(view))
            request_id = port.submitted[0][2]
            port.confirm(request_id, replace(view, revision=2, actor_id=None))
            result = harness.result()
        finally:
            runner.request_stop()
        self.assertTrue(result.ok)
        self.assertEqual((len(port.submitted), len(policy.requests)), (1, 1))

    def test_max_actions_counts_acknowledgements_not_submissions(self) -> None:
        view = offered_view()
        port = FakePort(update(view))
        result = AgentRunner(port, RecordingPolicy(), config(max_actions=1, submission_timeout=0.06)).run()
        self.assertEqual(result.stop_reason, StopReason.CLI_FAILURE)
        self.assertEqual(result.confirmed_actions, 0)
        self.assertEqual(len(port.submitted), 1)
        self.assertIn("unknown", result.error or "")

    def test_unknown_acknowledgement_fails_without_replay(self) -> None:
        view = offered_view()
        port = FakePort(update(view))
        port.on_submit = lambda action, token, rid: port.events.put(
            CliActionResult(GuardedResult("wrong-id", response=CommandResponse(view=view))))
        result = AgentRunner(port, RecordingPolicy(), config(max_actions=1)).run()
        self.assertEqual(result.stop_reason, StopReason.CLI_FAILURE)
        self.assertEqual(result.confirmed_actions, 0)
        self.assertEqual(len(port.submitted), 1)
        self.assertIn("uncorrelated", result.error or "")

    def test_stale_rejection_waits_for_new_observation_before_redeciding(self) -> None:
        view = offered_view()
        port = FakePort(update(view))
        policy = RecordingPolicy()

        def submit(action: PlayerAction, token: ViewToken, request_id: str) -> None:
            if token.revision == 1:
                port.events.put(CliActionResult(GuardedResult(request_id, error="stale_observation")))
                port.events.put(update(view))
                port.events.put(update(replace(view, revision=2)))
            else:
                port.confirm(request_id, replace(view, revision=3, actor_id=None))

        port.on_submit = submit
        result = AgentRunner(port, policy, config(max_actions=1)).run()
        self.assertTrue(result.ok)
        self.assertEqual([token.revision for _, token, _ in port.submitted], [1, 2])
        self.assertEqual(result.confirmed_actions, 1)
        feedback = [event for event in policy.events if isinstance(event, ActionFeedback)]
        self.assertEqual([event.confirmed for event in feedback], [False, True])

    def test_invalid_output_stops_without_hidden_fallback(self) -> None:
        for decision in (Decision(PlayerAction(ActionKind.FOLD)), cast(Decision, None)):
            with self.subTest(decision=decision):
                port = FakePort(update(offered_view()))
                policy = RecordingPolicy()
                policy.on_decide = lambda request, context: decision
                result = AgentRunner(port, policy, config()).run()
                self.assertEqual(result.stop_reason, StopReason.INVALID_DECISION)
                self.assertEqual(port.submitted, [])
                self.assertFalse(result.ok)

    def test_tool_budget_is_a_specific_failure_with_no_submission(self) -> None:
        port = FakePort(update(offered_view()))
        policy = RecordingPolicy()

        def decide(request: DecisionRequest, context: DecisionContext) -> Decision:
            context.tools.call("legal_actions", {})
            context.tools.call("observation", {})
            return Decision(PlayerAction(ActionKind.CHECK))

        policy.on_decide = decide
        result = AgentRunner(port, policy, config(max_tool_calls=1)).run()
        self.assertEqual(result.stop_reason, StopReason.TOOL_BUDGET)
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(port.submitted, [])

    def test_observed_hand_limit_uses_result_hand_id_and_deduplicates(self) -> None:
        view = replace(offered_view(hand_id="h2"), actor_id=None,
                       result=ResultView(HandId("h1"), (), ()))
        complete = replace(view, revision=3, phase=HandPhase.COMPLETE,
                           result=ResultView(HandId("h2"), (), ()))
        port = FakePort(update(view), update(view), update(replace(view, revision=2)), update(complete))
        policy = RecordingPolicy()
        result = AgentRunner(port, policy, config(max_hands=1)).run()
        self.assertEqual(result.stop_reason, StopReason.MAX_HANDS)
        self.assertEqual(result.observed_hands, 1)
        finished = [event.result.hand_id for event in policy.events if isinstance(event, HandCompleted)]
        self.assertEqual(finished, ["h2"])
        observations = [event for event in policy.events if isinstance(event, ObservationChanged)]
        self.assertEqual(observations[0].observation.view.result, view.result)
        self.assertEqual(port.submitted, [])
        self.assertTrue(result.ok)

    def test_hand_change_without_result_does_not_invent_completion(self) -> None:
        old = replace(offered_view(hand_id="h1"), actor_id=None)
        new = replace(offered_view(2, "h3"), actor_id=None)
        port = FakePort(update(old), update(new))
        result = AgentRunner(port, RecordingPolicy(), config(max_hands=1, session_timeout=0.08)).run()
        self.assertEqual(result.stop_reason, StopReason.SESSION_LIMIT)
        self.assertEqual(result.observed_hands, 0)
        self.assertTrue(result.ok)

    def test_exceptions_in_open_observe_and_decide_are_policy_failures(self) -> None:
        def fail() -> None:
            raise RuntimeError("private details must not enter result")

        for hook in ("open", "observe", "decide"):
            with self.subTest(hook=hook):
                port = FakePort(update(offered_view()))
                policy = RecordingPolicy()
                if hook == "open":
                    policy.on_open = fail
                elif hook == "observe":
                    policy.on_observe = lambda event: fail()
                else:
                    def fail_decide(request: DecisionRequest, context: DecisionContext) -> Decision:
                        fail()
                        raise AssertionError("unreachable")
                    policy.on_decide = fail_decide
                result = AgentRunner(port, policy, config()).run()
                self.assertEqual(result.stop_reason, StopReason.POLICY_FAILURE)
                self.assertIn(hook, result.error or "")
                self.assertNotIn("private details", result.error or "")
                self.assertTrue(result.policy_closed)
                self.assertTrue(policy.closed.is_set())
                self.assertEqual(port.submitted, [])
                self.assertEqual(port.starts, 0 if hook == "open" else 1)

    def test_blocked_callbacks_have_watchdogs_and_close_runs_only_after_release(self) -> None:
        for hook in ("open", "observe", "decide"):
            with self.subTest(hook=hook):
                entered, release = Event(), Event()
                policy = RecordingPolicy()

                def block() -> None:
                    entered.set()
                    release.wait(2)

                if hook == "open":
                    policy.on_open = block
                elif hook == "observe":
                    policy.on_observe = lambda event: block()
                else:
                    def blocked_decide(request: DecisionRequest, context: DecisionContext) -> Decision:
                        block()
                        return Decision(PlayerAction(ActionKind.CHECK))
                    policy.on_decide = blocked_decide
                port = FakePort(update(offered_view()))
                runner = AgentRunner(port, policy, config(callback_timeout=0.06, decision_timeout=0.06,
                                                          shutdown_timeout=0.02))
                harness = RunnerHarness(runner)
                try:
                    self.assertTrue(entered.wait(1))
                    result = harness.result()
                    self.assertEqual(result.stop_reason, StopReason.DEADLINE)
                    self.assertFalse(result.policy_closed)
                    self.assertTrue(result.cli_closed)
                    self.assertFalse(policy.closed.is_set(), "close must not run concurrently with a blocked callback")
                    self.assertEqual(port.submitted, [])
                    self.assertEqual(port.starts, 0 if hook == "open" else 1)
                finally:
                    release.set()
                    self.assertTrue(policy.closed.wait(1))
                self.assertEqual(port.submitted, [], "late callback results must never submit an action")

    def test_cleanup_exception_and_blocked_cleanup_are_reported_honestly(self) -> None:
        for blocked in (False, True):
            with self.subTest(blocked=blocked):
                release = Event()
                view = offered_view()
                port = FakePort(update(view))
                port.on_submit = lambda action, token, rid: port.confirm(rid, replace(view, revision=2, actor_id=None))
                policy = RecordingPolicy()

                def cleanup() -> None:
                    if blocked:
                        release.wait(2)
                    else:
                        raise RuntimeError("close failed")

                policy.on_close = cleanup
                try:
                    result = AgentRunner(port, policy, config(max_actions=1, shutdown_timeout=0.02)).run()
                    self.assertFalse(result.ok)
                    self.assertEqual(result.stop_reason, StopReason.MAX_ACTIONS)
                    self.assertEqual(result.policy_closed, not blocked)
                    self.assertIsNotNone(result.error)
                finally:
                    release.set()
                    self.assertTrue(policy.closed.wait(1))

    def test_incomplete_cli_cleanup_makes_normal_stop_unsuccessful(self) -> None:
        port = FakePort(update(replace(offered_view(), actor_id=None)))
        port.close_result = False
        result = AgentRunner(port, RecordingPolicy(), config(session_timeout=0.06)).run()
        self.assertEqual(result.stop_reason, StopReason.SESSION_LIMIT)
        self.assertFalse(result.cli_closed)
        self.assertTrue(result.policy_closed)
        self.assertFalse(result.ok)

    def test_cli_terminal_event_is_not_misreported_as_policy_error(self) -> None:
        result = AgentRunner(FakePort(CliStopped("offline")), RecordingPolicy(), config()).run()
        self.assertEqual(result.stop_reason, StopReason.CLI_FAILURE)
        self.assertIn("offline", result.error or "")
        self.assertEqual(result.confirmed_actions, 0)

    def test_auto_start_attempt_is_deduplicated_and_is_not_a_policy_action(self) -> None:
        view = server_view()
        port = FakePort(update(view), update(view), update(view))
        policy = RecordingPolicy()
        result = AgentRunner(port, policy, config(auto_start=True, session_timeout=0.3)).run()
        self.assertEqual(port.start_hands, 1)
        self.assertEqual((result.decisions, result.confirmed_actions), (0, 0))
        self.assertEqual(policy.requests, [])

    def test_fast_completed_callback_is_not_timed_out_by_late_main_loop(self) -> None:
        policy = RecordingPolicy()
        observing, finish_callback = Event(), Event()

        def observe(event: PolicyEvent) -> None:
            if isinstance(event, ObservationChanged) and event.observation.view.revision == 1:
                observing.set()
                finish_callback.wait(1)

        policy.on_observe = observe

        class DelayedReadPort(FakePort):
            delayed = False

            def next_event(self, timeout: float = 0.0) -> CliEvent | None:
                if self.consumed and not self.delayed:
                    self.delayed = True
                    observing.wait(1)
                    finish_callback.set()
                    # The callback finishes now; the owner loop is deliberately
                    # kept from checking its Future until after its deadline.
                    Event().wait(0.2)
                return super().next_event(timeout)

        view = offered_view()
        port = DelayedReadPort(update(view))
        port.on_submit = lambda action, token, rid: port.confirm(rid, replace(view, revision=2, actor_id=None))
        result = AgentRunner(port, policy, config(max_actions=1, callback_timeout=0.15,
                                                 decision_timeout=0.15)).run()
        self.assertTrue(result.ok, result)
        self.assertEqual(result.confirmed_actions, 1)

    def test_foreign_identity_and_inconsistent_revisions_never_reach_policy(self) -> None:
        initial = replace(offered_view(5), actor_id=None)
        invalid = (
            replace(initial, revision=6, table_id=TableId("foreign")),
            replace(initial, revision=6, me=replace(initial.me, player_id=PlayerId("foreign"))),
            replace(initial, revision=4),
            replace(initial, current_bet=1),
        )
        for changed in invalid:
            with self.subTest(changed=changed):
                port = FakePort(update(initial), update(changed))
                policy = RecordingPolicy()
                result = AgentRunner(port, policy, config()).run()
                self.assertEqual(result.stop_reason, StopReason.CLI_FAILURE)
                observed = [event.observation.view for event in policy.events if isinstance(event, ObservationChanged)]
                self.assertNotIn(changed, observed)
                self.assertEqual(port.submitted, [])

    def test_fatal_action_rejection_is_delivered_to_policy_before_close(self) -> None:
        port = FakePort(update(offered_view()))
        policy = RecordingPolicy()

        def reject(action: PlayerAction, token: ViewToken, request_id: str) -> None:
            response = CommandResponse(error=ErrorInfo(ErrorCode.INVALID_ACTION, "rejected"))
            port.events.put(CliObservation("check", response))
            port.events.put(CliActionResult(GuardedResult(request_id, response=response)))

        port.on_submit = reject
        result = AgentRunner(port, policy, config()).run()
        self.assertEqual(result.stop_reason, StopReason.SERVER_REJECTED)
        feedback = [event for event in policy.events if isinstance(event, ActionFeedback)]
        self.assertEqual(len(feedback), 1)
        self.assertFalse(feedback[0].confirmed)
        self.assertEqual(feedback[0].error_code, ErrorCode.INVALID_ACTION.value)
        self.assertEqual(result.confirmed_actions, 0)
        self.assertLess(policy.trace.index("observe:ActionFeedback"), policy.trace.index("close.begin"))

    def test_hand_limit_does_not_hide_an_unknown_pending_action(self) -> None:
        view = offered_view()
        complete = replace(view, revision=2, actor_id=None, phase=HandPhase.COMPLETE,
                           result=ResultView(HandId("h1"), (), ()))
        port = FakePort(update(view))
        port.on_submit = lambda action, token, rid: port.events.put(update(complete, "check"))
        result = AgentRunner(port, RecordingPolicy(), config(max_hands=1, submission_timeout=0.08)).run()
        self.assertEqual(result.observed_hands, 1)
        self.assertEqual(result.confirmed_actions, 0)
        self.assertEqual(result.stop_reason, StopReason.CLI_FAILURE)
        self.assertFalse(result.ok)
        self.assertEqual(len(port.submitted), 1)

    def test_feedback_timeout_keeps_original_rejection_and_does_not_overlap_close(self) -> None:
        port = FakePort(update(offered_view()))
        policy = RecordingPolicy()
        entered, release = Event(), Event()

        def reject(action: PlayerAction, token: ViewToken, request_id: str) -> None:
            response = CommandResponse(error=ErrorInfo(ErrorCode.INVALID_ACTION, "rejected"))
            port.events.put(CliObservation("check", response))
            port.events.put(CliActionResult(GuardedResult(request_id, response=response)))

        def observe(event: PolicyEvent) -> None:
            if isinstance(event, ActionFeedback):
                entered.set()
                release.wait(2)

        port.on_submit = reject
        policy.on_observe = observe
        runner = AgentRunner(port, policy, config(callback_timeout=0.06, shutdown_timeout=0.02))
        harness = RunnerHarness(runner)
        try:
            self.assertTrue(entered.wait(1))
            result = harness.result()
            self.assertEqual(result.stop_reason, StopReason.SERVER_REJECTED)
            self.assertIn("invalid_action", result.error or "")
            self.assertFalse(result.policy_closed)
            self.assertFalse(policy.closed.is_set())
        finally:
            release.set()
            self.assertTrue(policy.closed.wait(1))

    def test_requested_stop_cancels_pending_work_and_never_submits_late_result(self) -> None:
        entered, release = Event(), Event()
        port = FakePort(update(offered_view()))
        policy = RecordingPolicy()

        def decide(request: DecisionRequest, context: DecisionContext) -> Decision:
            entered.set()
            release.wait(2)
            return Decision(PlayerAction(ActionKind.CHECK))

        policy.on_decide = decide
        runner = AgentRunner(port, policy, config(decision_timeout=1.0, shutdown_timeout=0.02))
        harness = RunnerHarness(runner)
        try:
            self.assertTrue(entered.wait(1))
            runner.request_stop()
            result = harness.result()
            self.assertEqual(result.stop_reason, StopReason.REQUESTED)
            self.assertFalse(result.policy_closed)
            self.assertTrue(result.cli_closed)
        finally:
            release.set()
            self.assertTrue(policy.closed.wait(1))
        self.assertEqual(port.submitted, [])

    def test_normal_stop_with_unacknowledged_submission_reports_unknown_execution(self) -> None:
        for requested in (False, True):
            with self.subTest(requested=requested):
                port = FakePort(update(offered_view()))
                policy = RecordingPolicy()
                runner = AgentRunner(port, policy, config(session_timeout=0.4, submission_timeout=1.0))
                harness = RunnerHarness(runner)
                try:
                    self.assertTrue(port.submission_seen.wait(1))
                    if requested:
                        runner.request_stop()
                    result = harness.result()
                finally:
                    runner.request_stop()
                self.assertEqual(result.stop_reason, StopReason.REQUESTED if requested else StopReason.SESSION_LIMIT)
                self.assertTrue(result.cli_closed)
                self.assertTrue(result.policy_closed)
                self.assertFalse(result.ok)
                self.assertIn("unknown", (result.error or "").lower())
                self.assertEqual(result.confirmed_actions, 0)
                self.assertEqual(len(port.submitted), 1)

    def test_joined_existing_result_is_visible_but_not_counted_as_session_completion(self) -> None:
        previous = ResultView(HandId("before-join"), (), ())
        view = replace(offered_view(), actor_id=None, result=previous)
        port = FakePort(update(view, "join Agent"), update(view), update(replace(view, revision=2)))
        policy = RecordingPolicy()
        result = AgentRunner(port, policy, config(max_hands=1, session_timeout=0.25)).run()
        self.assertEqual(result.stop_reason, StopReason.SESSION_LIMIT)
        self.assertEqual(result.observed_hands, 0)
        self.assertFalse(any(isinstance(event, HandCompleted) for event in policy.events))
        observations = [event for event in policy.events if isinstance(event, ObservationChanged)]
        self.assertGreaterEqual(len(observations), 1)
        self.assertEqual(observations[0].observation.view.result, previous)
        self.assertTrue(result.ok)
