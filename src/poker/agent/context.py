from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from random import Random
from threading import Event
from time import monotonic
from typing import TYPE_CHECKING

from poker.agent.models import DecisionCancelled, DecisionDeadlineExceeded

if TYPE_CHECKING:
    from poker.agent.tools import ToolAccess


class DecisionControl:
    """Cooperative cancellation. The runtime also rejects all late results."""

    def __init__(self, deadline: float, *, clock: Callable[[], float] = monotonic) -> None:
        if isinstance(deadline, bool) or not isfinite(deadline):
            raise ValueError("Deadline must be finite")
        self.deadline = deadline
        self._clock = clock
        self._cancelled = Event()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        self._cancelled.set()

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - self._clock())

    def check(self) -> None:
        if self.cancelled:
            raise DecisionCancelled("This decision was cancelled")
        if self.remaining_seconds() <= 0:
            raise DecisionDeadlineExceeded("The decision deadline has expired")


@dataclass(frozen=True, slots=True)
class DecisionContext:
    tools: ToolAccess
    random: Random
    control: DecisionControl

    def remaining_seconds(self) -> float:
        return self.control.remaining_seconds()

    def check_cancelled(self) -> None:
        self.control.check()
