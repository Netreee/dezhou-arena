"""Provider-neutral API and subprocess observations, usable outside this project."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Generic, TypeVar
from uuid import uuid4
import os
import math
import subprocess

from shared_logging import EventLogger, JsonObject, context_scope, get_logger
from shared_logging.runtime import logging_environment

T = TypeVar("T")


@dataclass(frozen=True)
class ApiUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None
    cost_source: str | None = None

    def __post_init__(self) -> None:
        if any(v is not None and (type(v) is not int or v < 0) for v in (self.input_tokens, self.output_tokens)):
            raise ValueError("Token usage must be a nonnegative integer or null")
        if self.cost is not None and (not math.isfinite(self.cost) or self.cost < 0 or not self.cost_source):
            raise ValueError("Known cost requires a finite nonnegative value and a source")


class ApiCallFailure(RuntimeError):
    def __init__(self, message: str, *, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status


@dataclass(frozen=True)
class ApiResult(Generic[T]):
    value: T
    usage: ApiUsage = ApiUsage()
    http_status: int | None = None
    stop_reason: str | None = None


def observe_api_call(provider: str, model: str, operation: Callable[[], ApiResult[T]], *, logger: EventLogger | None = None) -> T:
    log = (logger or get_logger("api")).bind(api_call_id=uuid4().hex)
    started = perf_counter()
    base: JsonObject = {"provider": provider, "model": model}
    log.emit("INFO", "api.call.started", "API call started", base)
    try:
        result = operation()
    except Exception as error:
        log.exception("api.call.failed", "API call failed", {
            **base, "duration_ms": (perf_counter() - started) * 1000,
            "http_status": error.http_status if isinstance(error, ApiCallFailure) else None,
            "input_tokens": None, "output_tokens": None, "cost": None, "cost_source": None,
        })
        raise
    log.emit("INFO", "api.call.completed", "API call completed", {
        **base, "duration_ms": (perf_counter() - started) * 1000, "http_status": result.http_status,
        "stop_reason": result.stop_reason, "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens, "cost": result.usage.cost,
        "cost_source": result.usage.cost_source,
    })
    return result.value


@contextmanager
def bot_decision(logger: EventLogger, *, player_id: str, hand_id: str | None) -> Iterator[None]:
    with context_scope(player_id=player_id, hand_id=hand_id, correlation_id=uuid4().hex):
        logger.emit("INFO", "bot.decision_requested", "Bot decision requested")
        try:
            yield
        except Exception:
            logger.exception("bot.failed", "Bot decision failed")
            raise
        else:
            logger.emit("INFO", "bot.decision_completed", "Bot decision completed")


def run_logged_process(command: list[str], *, role: str = "bot", environment: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    log = get_logger("process.child")
    env = {**os.environ, **logging_environment(), **(environment or {}), "APP_LOG_ROLE": role}
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", env=env,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    log.emit("INFO", "bot.started" if role == "bot" else "process.started", "Child started", {"program": os.path.basename(command[0]), "child_pid": process.pid})
    stdout, stderr = process.communicate()
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if result.stdout:
        log.emit("DEBUG", "process.stdout", "Child stdout", {"text": result.stdout})
    if result.stderr:
        log.emit("WARNING", "process.stderr", "Child stderr", {"text": result.stderr})
    log.emit("INFO" if result.returncode == 0 else "ERROR", ("bot.stopped" if role == "bot" else "process.stopped") if result.returncode == 0 else "process.failed",
             "Child exited", {"exit_code": result.returncode, "child_pid": process.pid})
    return result
