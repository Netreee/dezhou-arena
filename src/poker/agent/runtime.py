"""A serial policy worker beside a continuously polling, single-owner CLI."""

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from math import isfinite
from queue import Queue
from random import Random
from threading import Event, Thread
from time import monotonic
from uuid import uuid4

from shared_logging import JsonObject, context_scope, get_logger
from poker.agent.cli_bridge import AgentCliPort, CliActionResult, CliInputError, CliObservation, CliStopped
from poker.agent.context import DecisionContext, DecisionControl
from poker.agent.models import (
    ActionFeedback, Decision, DecisionCancelled, DecisionDeadlineExceeded, DecisionRequest,
    HandCompleted, InvalidDecision, Observation, ObservationChanged, PolicyEvent, PolicySession,
    StopReason, ToolBudgetExceeded, select_action,
)
from poker.agent.policy import Policy
from poker.agent.tools import BoundTools, ToolRegistry, standard_tools
from poker.client.guarded import ViewToken
from poker.domain.models import PlayerAction
from poker.domain.types import ErrorCode, HandPhase


@dataclass(frozen=True, slots=True)
class AgentConfig:
    name: str = "Agent"
    decision_timeout: float = 30.0
    callback_timeout: float = 5.0
    submission_timeout: float = 10.0
    shutdown_timeout: float = 1.0
    session_timeout: float | None = None
    max_actions: int | None = None
    max_hands: int | None = None
    max_tool_calls: int = 8
    seed: int = 0
    auto_start: bool = False
    min_players: int = 2

    def __post_init__(self) -> None:
        if not self.name.strip() or "\n" in self.name or "\r" in self.name:
            raise ValueError("Player name must be a nonempty single line")
        for value in (self.decision_timeout, self.callback_timeout, self.submission_timeout, self.shutdown_timeout):
            if not isfinite(value) or value <= 0:
                raise ValueError("Timeouts must be positive and finite")
        if self.session_timeout is not None and (not isfinite(self.session_timeout) or self.session_timeout <= 0):
            raise ValueError("Session timeout must be positive and finite")
        for limit in (self.max_actions, self.max_hands):
            if limit is not None and (type(limit) is not int or limit < 1):
                raise ValueError("Action and hand limits must be positive integers")
        if type(self.max_tool_calls) is not int or self.max_tool_calls < 0:
            raise ValueError("Tool call budget must be a nonnegative integer")
        if type(self.min_players) is not int or not 2 <= self.min_players <= 6:
            raise ValueError("min_players must be between two and six")


@dataclass(frozen=True, slots=True)
class RunResult:
    stop_reason: StopReason
    confirmed_actions: int
    observed_hands: int
    decisions: int
    tool_calls: int
    cli_closed: bool
    policy_closed: bool
    error: str | None = None

    @property
    def ok(self) -> bool:
        return (self.stop_reason in (StopReason.REQUESTED, StopReason.MAX_ACTIONS,
                                     StopReason.MAX_HANDS, StopReason.SESSION_LIMIT)
                and self.cli_closed and self.policy_closed and self.error is None)


@dataclass(frozen=True)
class _Completion:
    value: Decision | None
    error: BaseException | None
    finished_at: float


@dataclass(frozen=True)
class _Work:
    operation: Callable[[], Decision | None]
    future: Future[_Completion]


class _PolicyWorker:
    """Daemon worker: never enter close concurrently with a running callback.

    Python cannot kill arbitrary synchronous user code. On a watchdog failure the
    runner stops the CLI, rejects late results and reports unfinished cleanup.
    A standalone agent process can then exit without waiting for this daemon.
    """

    def __init__(self, policy: Policy) -> None:
        self._policy = policy
        self._queue: Queue[_Work | None] = Queue()
        self._stop = Event()
        self._reason = StopReason.REQUESTED
        self.closed = Event()
        self.close_error: str | None = None
        self._thread = Thread(target=self._run, name="agent-policy", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, operation: Callable[[], Decision | None]) -> Future[_Completion]:
        future: Future[_Completion] = Future()
        self._queue.put(_Work(operation, future))
        return future

    def stop(self, reason: StopReason, timeout: float) -> bool:
        self._reason = reason
        self._stop.set()
        self._queue.put(None)
        self._thread.join(timeout)
        return self.closed.is_set() and not self._thread.is_alive()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                work = self._queue.get()
                if work is None or self._stop.is_set():
                    break
                try:
                    value = work.operation()
                except BaseException as error:
                    work.future.set_result(_Completion(None, error, monotonic()))
                else:
                    work.future.set_result(_Completion(value, None, monotonic()))
        finally:
            try:
                self._policy.close(self._reason)
            except BaseException as error:
                self.close_error = type(error).__name__
            finally:
                self.closed.set()


@dataclass
class _Task:
    kind: str
    future: Future[_Completion]
    deadline: float
    request: DecisionRequest | None = None
    control: DecisionControl | None = None
    tools: BoundTools | None = None


@dataclass(frozen=True)
class _Submission:
    decision_id: str
    action: PlayerAction
    deadline: float


class AgentRunner:
    def __init__(self, port: AgentCliPort, policy: Policy, config: AgentConfig,
                 *, tools: ToolRegistry | None = None, stop: Event | None = None) -> None:
        self._port = port
        self._policy = policy
        self._config = config
        self._tools = tools if tools is not None else ToolRegistry(standard_tools())
        self._stop = stop if stop is not None else Event()
        self._started = False
        self._log = get_logger("agent.runtime")
        seed_source = Random(config.seed)
        self._policy_random = Random(seed_source.getrandbits(128))
        self._action_random = Random(seed_source.getrandbits(128))
        self._events: deque[PolicyEvent] = deque()
        self._latest: Observation | None = None
        self._last_result: str | None = None
        self._last_decision: ViewToken | None = None
        self._start_attempt: tuple[str, str | None, int] | None = None
        self._submission: _Submission | None = None
        self._task: _Task | None = None
        self._ready = False
        self._reason: StopReason | None = None
        self._error: str | None = None
        self._drain = False
        self._actions = 0
        self._hands = 0
        self._decisions = 0
        self._tool_calls = 0

    def request_stop(self) -> None:
        self._stop.set()

    def _fail(self, reason: StopReason, message: str, *, drain: bool = False) -> None:
        # A cleanup failure must not replace the first operational failure.
        if self._error is None:
            self._reason, self._error = reason, message
        self._drain = drain

    def _finish(self, reason: StopReason) -> None:
        if self._reason is None:
            self._reason, self._drain = reason, True

    def run(self) -> RunResult:
        if self._started:
            raise RuntimeError("An AgentRunner and its Policy may run only once")
        self._started = True
        started = monotonic()
        worker = _PolicyWorker(self._policy)
        session = PolicySession(uuid4().hex, self._config.name)
        self._log.emit("INFO", "agent.started", "Agent runtime started", {
            "agent_id": session.agent_id, "policy": self._policy.identity.name,
            "policy_version": self._policy.identity.version, "seed": self._config.seed,
        })
        worker.start()
        cli_closed = False
        policy_closed = False
        cli_started = False
        try:
            self._task = _Task("open", worker.submit(lambda: self._policy.open(session)),
                               monotonic() + self._config.callback_timeout)
            while True:
                now = monotonic()
                if self._stop.is_set():
                    self._reason = self._reason or StopReason.REQUESTED
                    self._drain = False
                if self._reason is None and self._config.session_timeout is not None and now - started >= self._config.session_timeout:
                    self._reason, self._drain = StopReason.SESSION_LIMIT, False
                self._poll_task(now)
                if self._submission is not None and now >= self._submission.deadline:
                    self._fail(StopReason.CLI_FAILURE, "Action acknowledgement timed out; execution may be unknown")
                if self._reason is not None and (not self._drain or (self._task is None and not self._events and self._submission is None)):
                    break
                # Initialization must succeed before occupying an Arena seat.
                if not cli_started:
                    if not self._ready:
                        self._stop.wait(0.005)
                        continue
                    self._port.start(self._config.name)
                    cli_started = True
                event = self._port.next_event(timeout=0.02)
                if isinstance(event, CliObservation):
                    self._observe(event)
                elif isinstance(event, CliActionResult):
                    self._acknowledge(event)
                elif isinstance(event, CliStopped):
                    self._fail(StopReason.CLI_FAILURE, event.error or "CLI stopped before the agent requested shutdown", drain=self._drain)
                elif isinstance(event, CliInputError):
                    self._fail(StopReason.CLI_FAILURE, f"CLI input failed: {event.code}")
                self._schedule(worker)
        except KeyboardInterrupt:
            self._reason = StopReason.REQUESTED
        except Exception as error:
            self._fail(StopReason.CLI_FAILURE, f"Agent runtime failed: {type(error).__name__}")
        finally:
            if self._submission is not None:
                self._error = self._error or "Stopped with an unconfirmed action; execution may be unknown"
            if self._task is not None and self._task.control is not None:
                self._task.control.cancel()
            try:
                cli_closed = self._port.close()
            except Exception as error:
                self._error = self._error or f"CLI cleanup failed: {type(error).__name__}"
            policy_closed = worker.stop(self._reason or StopReason.REQUESTED, self._config.shutdown_timeout)
            if self._task is not None and self._task.tools is not None:
                self._tool_calls += self._task.tools.calls
            if worker.close_error is not None:
                self._error = self._error or f"Policy cleanup failed: {worker.close_error}"
            if not cli_closed:
                self._error = self._error or "CLI owner did not stop within its shutdown deadline"
            if not policy_closed:
                self._error = self._error or "Policy worker is still running; late results are discarded"
        result = RunResult(self._reason or StopReason.REQUESTED, self._actions, self._hands,
                           self._decisions, self._tool_calls, cli_closed, policy_closed, self._error)
        data: JsonObject = {
            "stop_reason": result.stop_reason.value, "confirmed_actions": result.confirmed_actions,
            "observed_hands": result.observed_hands, "decisions": result.decisions, "tool_calls": result.tool_calls,
            "cli_closed": result.cli_closed, "policy_closed": result.policy_closed, "ok": result.ok,
            "error": result.error,
        }
        self._log.emit("INFO" if result.ok else "ERROR", "agent.stopped", "Agent runtime stopped", data)
        return result

    def _observe(self, event: CliObservation) -> None:
        # Once finishing, freeze the finite callback queue. A pending action's
        # separate acknowledgement is still consumed before successful shutdown.
        if self._reason is not None:
            return
        response = event.response
        if response.error is not None:
            # A guarded action has a correlated acknowledgement immediately after
            # its observation. Let that acknowledgement determine retry/stopping.
            if self._submission is not None:
                return
            self._fail(StopReason.SERVER_REJECTED, f"CLI command rejected: {response.error.code.value}")
            return
        view = response.view
        if view is None or (self._latest is not None and view == self._latest.view):
            return
        first_observation = self._latest is None
        if self._latest is not None:
            previous = self._latest.view
            if (view.table_id, view.me.player_id) != (previous.table_id, previous.me.player_id):
                self._fail(StopReason.CLI_FAILURE, "CLI player or table identity changed")
                return
            if view.revision <= previous.revision:
                self._fail(StopReason.CLI_FAILURE, "CLI observation revision regressed or changed without a new revision")
                return
        observation = Observation(view, monotonic())
        self._latest = observation
        self._events.append(ObservationChanged(observation))
        if self._task is not None and self._task.request is not None and self._task.control is not None:
            old = self._task.request.observation.view
            if (old.table_id, old.hand_id, old.revision, old.actor_id) != (view.table_id, view.hand_id, view.revision, view.actor_id):
                self._task.control.cancel()
        if first_observation and view.result is not None:
            # A result already present at join is historical context, not a
            # completion witnessed in this session.
            self._last_result = view.result.hand_id
        elif view.result is not None and view.result.hand_id != self._last_result:
            self._last_result = view.result.hand_id
            self._hands += 1
            self._events.append(HandCompleted(view.result))
            self._log.emit("INFO", "agent.hand.observed", "Completed hand observed through CLI", {
                "result_hand_id": view.result.hand_id, "observed_hands": self._hands,
            })
            if self._config.max_hands is not None and self._hands >= self._config.max_hands:
                self._finish(StopReason.MAX_HANDS)

    def _acknowledge(self, event: CliActionResult) -> None:
        result = event.result
        pending = self._submission
        if pending is None or result.request_id != pending.decision_id:
            self._fail(StopReason.CLI_FAILURE, "Received an uncorrelated action acknowledgement")
            return
        self._submission = None
        error_code = result.error
        if result.response is not None and result.response.error is not None:
            error_code = result.response.error.code.value
        confirmed = result.response is not None and result.response.ok and error_code is None
        self._events.append(ActionFeedback(pending.decision_id, pending.action, confirmed, error_code))
        if confirmed:
            self._actions += 1
            self._log.emit("INFO", "agent.action.confirmed", "Arena confirmed the proposed action", {
                "decision_id": pending.decision_id, "confirmed_actions": self._actions,
            })
            if self._config.max_actions is not None and self._actions >= self._config.max_actions:
                self._finish(StopReason.MAX_ACTIONS)
        elif error_code in ("stale_observation", ErrorCode.STALE_STATE.value):
            # Reconsider only when the CLI delivers a different view. Never
            # automatically replay an action whose execution is uncertain.
            self._log.emit("INFO", "agent.action.stale", "Stale decision rejected without execution", {
                "decision_id": pending.decision_id, "error_code": error_code,
            })
        else:
            reason = StopReason.CLI_FAILURE if error_code in ("connection_failed", "refresh_failed") else StopReason.SERVER_REJECTED
            self._fail(reason, f"Action was not confirmed: {error_code or 'unknown'}", drain=True)

    def _poll_task(self, now: float) -> None:
        task = self._task
        if task is None:
            return
        if not task.future.done():
            if now < task.deadline:
                return
            if task.control is not None:
                task.control.cancel()
            self._fail(StopReason.DEADLINE, f"Policy {task.kind} exceeded its deadline")
            return
        completion = task.future.result()
        self._task = None
        if task.tools is not None:
            self._tool_calls += task.tools.calls
        if completion.finished_at >= task.deadline:
            self._fail(StopReason.DEADLINE, f"Policy {task.kind} exceeded its deadline")
            return
        try:
            if completion.error is not None:
                raise completion.error
            value = completion.value
        except DecisionCancelled:
            if task.control is None or not task.control.cancelled:
                self._fail(StopReason.POLICY_FAILURE, "Policy cancelled a live request without runtime cancellation")
            return
        except DecisionDeadlineExceeded:
            self._fail(StopReason.DEADLINE, "Policy decision exceeded its deadline")
            return
        except ToolBudgetExceeded:
            self._fail(StopReason.TOOL_BUDGET, "Policy exhausted its tool budget")
            return
        except BaseException as error:
            self._fail(StopReason.POLICY_FAILURE, f"Policy {task.kind} failed: {type(error).__name__}")
            return
        if task.kind == "open":
            self._ready = True
            return
        request = task.request
        if request is None or task.control is None or self._reason is not None:
            return
        if task.control.cancelled:
            return
        if self._latest is None or not ViewToken.from_view(request.observation.view).matches(self._latest.view):
            return
        try:
            if value is None:
                raise InvalidDecision("Policy returned no decision")
            action = select_action(value, request.observation.view, self._action_random)
        except InvalidDecision as error:
            self._fail(StopReason.INVALID_DECISION, str(error))
            return
        token = ViewToken.from_view(request.observation.view)
        self._submission = _Submission(request.decision_id, action, monotonic() + self._config.submission_timeout)
        self._port.submit(action, token, request.decision_id)
        self._log.emit("INFO", "agent.action.queued", "Decision queued for guarded CLI submission", {
            "decision_id": request.decision_id, "hand_id": token.hand_id, "revision": token.revision,
            "action": action.kind.value, "to": action.to,
        })

    def _schedule(self, worker: _PolicyWorker) -> None:
        if self._task is not None or not self._ready or (self._reason is not None and not self._drain):
            return
        if self._events:
            event = self._events.popleft()
            self._task = _Task("observe", worker.submit(lambda: self._policy.observe(event)),
                               monotonic() + self._config.callback_timeout)
            return
        if self._reason is not None or self._submission is not None or self._latest is None:
            return
        view = self._latest.view
        if self._config.auto_start and view.phase in (None, HandPhase.COMPLETE):
            enough = sum(player.stack > 0 for player in view.players) >= self._config.min_players
            key = (view.table_id, view.hand_id, view.revision)
            if enough and self._start_attempt != key:
                self._start_attempt = key
                self._port.start_hand()
            return
        if view.hand_id is None or view.actor_id != view.me.player_id or not view.me.legal_actions:
            return
        token = ViewToken.from_view(view)
        if token == self._last_decision:
            return
        self._last_decision = token
        self._decisions += 1
        deadline = monotonic() + self._config.decision_timeout
        request = DecisionRequest(uuid4().hex, self._latest, deadline)
        control = DecisionControl(deadline)
        tools = self._tools.bind(self._latest, control, max_calls=self._config.max_tool_calls,
                                 decision_id=request.decision_id)
        context = DecisionContext(tools, self._policy_random, control)

        def decide() -> Decision:
            with context_scope(player_id=token.player_id, hand_id=token.hand_id,
                               correlation_id=request.decision_id):
                context.check_cancelled()
                decision = self._policy.decide(request, context)
                context.check_cancelled()
                return decision

        self._task = _Task("decide", worker.submit(decide), deadline, request, control, tools)
        self._log.emit("INFO", "agent.decision.started", "Policy decision started", {
            "decision_id": request.decision_id, "hand_id": token.hand_id, "revision": token.revision,
        })
