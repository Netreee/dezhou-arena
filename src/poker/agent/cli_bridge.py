"""Agent port backed exclusively by the existing player's CLI runtime."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
from queue import Empty, Queue
from threading import Lock, Thread
from typing import TypeAlias

from shared_logging import get_logger

from poker.application.views import CommandResponse
from poker.client.cli import PokerCli
from poker.client.guarded import GuardedInput, GuardedResult, ViewToken
from poker.domain.models import PlayerAction
from poker.domain.types import ActionKind


@dataclass(frozen=True, slots=True)
class CliObservation:
    source: str
    response: CommandResponse


@dataclass(frozen=True, slots=True)
class CliActionResult:
    result: GuardedResult


@dataclass(frozen=True, slots=True)
class CliStopped:
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CliInputError:
    source: str | None
    code: str
    message: str | None


CliEvent: TypeAlias = CliObservation | CliActionResult | CliStopped | CliInputError


class AgentCliPort(ABC):
    @abstractmethod
    def start(self, name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def next_event(self, timeout: float = 0.0) -> CliEvent | None:
        raise NotImplementedError

    @abstractmethod
    def submit(self, action: PlayerAction, token: ViewToken, request_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def start_hand(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> bool:
        """Request shutdown; True confirms that the CLI owner has stopped."""
        raise NotImplementedError


class PokerCliAgentAdapter(AgentCliPort):
    """A single owner thread performs every CLI/socket operation."""

    def __init__(
        self, cli: PokerCli, *, poll_interval: float = 1.0, close_timeout: float = 6.0,
    ) -> None:
        if not isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be positive and finite")
        if not isfinite(close_timeout) or close_timeout < 0:
            raise ValueError("close_timeout must be nonnegative and finite")
        self._cli = cli
        self._poll_interval = poll_interval
        self._close_timeout = close_timeout
        self._input: Queue[str | None] = Queue()
        self._guarded: Queue[GuardedInput] = Queue()
        self._events: Queue[CliEvent] = Queue()
        self._lock = Lock()
        self._thread: Thread | None = None
        self._closed = False
        self._failure: str | None = None
        self._request_ids: set[str] = set()

    def start(self, name: str) -> None:
        if "\n" in name or "\r" in name:
            raise ValueError("Player name must be 1-64 characters without newlines")
        name = name.strip()
        if not name or len(name) > 64:
            raise ValueError("Player name must be 1-64 characters without newlines")
        with self._lock:
            if self._thread is not None or self._closed:
                raise RuntimeError("CLI adapter can only start once")
            self._input.put(f"join {name}")
            self._thread = Thread(target=self._run, name="agent-cli", daemon=True)
            self._thread.start()

    def next_event(self, timeout: float = 0.0) -> CliEvent | None:
        if not isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be nonnegative and finite")
        try:
            return self._events.get(timeout=timeout)
        except Empty:
            return None

    def submit(self, action: PlayerAction, token: ViewToken, request_id: str) -> None:
        if not request_id or len(request_id) > 128:
            raise ValueError("request_id must contain 1-128 characters")
        if not isinstance(action.kind, ActionKind):
            raise ValueError("Unknown player action")
        sized = action.kind in (ActionKind.BET_TO, ActionKind.RAISE_TO)
        if sized and (type(action.to) is not int or action.to <= 0):
            raise ValueError("Sized actions require a positive integer total")
        if not sized and action.to is not None:
            raise ValueError("This action does not accept an amount")
        line = action.kind.value + (f" {action.to}" if sized else "")
        with self._lock:
            self._require_running()
            if request_id in self._request_ids:
                raise ValueError("request_id was already submitted")
            self._request_ids.add(request_id)
            self._guarded.put(GuardedInput(request_id, line, token))

    def start_hand(self) -> None:
        with self._lock:
            self._require_running()
            self._input.put("start")

    def close(self) -> bool:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._input.put("quit")
            thread = self._thread
        if thread is not None:
            thread.join(timeout=self._close_timeout)
            if thread.is_alive():
                get_logger("agent.cli").emit("ERROR", "agent.cli_close_timeout", "CLI owner did not stop in time")
                return False
        return True

    def _require_running(self) -> None:
        if self._thread is None or not self._thread.is_alive() or self._closed:
            raise RuntimeError("CLI adapter is not running")

    def _response(self, source: str, response: CommandResponse) -> None:
        self._events.put(CliObservation(source, response))

    def _action_result(self, result: GuardedResult) -> None:
        self._events.put(CliActionResult(result))

    def _lifecycle(self, code: str, message: str | None, source: str | None) -> None:
        if code == "input_error":
            self._events.put(CliInputError(source, code, message))
        elif code != "closed":
            self._failure = f"{code}: {message}" if message else code

    def _run(self) -> None:
        try:
            self._cli.run(
                input_queue=self._input, guarded_queue=self._guarded,
                poll_interval=self._poll_interval, write=lambda _: None,
                on_response=self._response, on_lifecycle=self._lifecycle,
                on_guarded_result=self._action_result,
            )
        except Exception as error:
            self._failure = f"cli_runtime_failed: {error}"
            get_logger("agent.cli").exception("agent.cli_failed", "Agent CLI runtime failed")
        finally:
            self._events.put(CliStopped(self._failure))
