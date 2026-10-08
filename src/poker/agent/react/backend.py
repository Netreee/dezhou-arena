"""The inference boundary: JSON in, JSON out, with no application tools."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite

from poker.agent.context import DecisionControl
from poker.agent.models import PolicyError
from shared_logging import JsonObject


class BackendError(PolicyError):
    """A sanitized provider failure. The caller decides whether to stop."""


@dataclass(frozen=True, slots=True)
class BackendIdentity:
    provider: str
    model: str
    is_live: bool


@dataclass(frozen=True, slots=True)
class BackendRequest:
    instructions: str
    input: JsonObject
    output_schema: JsonObject
    max_output_tokens: int = 512

    def __post_init__(self) -> None:
        if type(self.max_output_tokens) is not int or self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be a positive integer")


@dataclass(frozen=True, slots=True)
class BackendUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    cost: float | None = None

    def __post_init__(self) -> None:
        for value in (self.input_tokens, self.output_tokens, self.cached_input_tokens):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("Token usage must be a nonnegative integer or null")
        if self.cost is not None and (isinstance(self.cost, bool) or not isfinite(self.cost) or self.cost < 0):
            raise ValueError("Cost must be finite and nonnegative or null")


@dataclass(frozen=True, slots=True)
class BackendResponse:
    output: JsonObject
    usage: BackendUsage = BackendUsage()
    request_id: str | None = None
    finish_reason: str | None = None


class ModelBackend(ABC):
    """One inference attempt per generate; lifecycle and tools belong to Agent.

    Implementations must cooperate with control, return a JSON mapping, and
    must not execute the tool requests described in that mapping. Implementations
    report unavailable usage as null. Token limits are only hard provider request
    limits when supports_output_token_limit is true; this is not a dollar cap.
    """

    @property
    @abstractmethod
    def identity(self) -> BackendIdentity:
        raise NotImplementedError

    @property
    def supports_output_token_limit(self) -> bool:
        return False

    @abstractmethod
    def generate(self, request: BackendRequest, control: DecisionControl) -> BackendResponse:
        raise NotImplementedError

    def close(self) -> None:
        """Release owned resources. The default stateless backend has none."""
