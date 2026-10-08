"""Optional language-policy adapter; the core Policy contract stays model-free."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from hashlib import sha256

from shared_logging import JsonObject
from poker.agent.context import DecisionContext
from poker.agent.models import Decision, DecisionRequest, PolicyError, PolicyIdentity
from poker.agent.policy import Policy
from poker.agent.tools import ToolResult, ToolSpec, view_json


@dataclass(frozen=True, slots=True)
class ModelToolCall:
    call_id: str
    name: str
    arguments: JsonObject


@dataclass(frozen=True, slots=True)
class ModelToolExchange:
    call: ModelToolCall
    result: ToolResult


@dataclass(frozen=True, slots=True)
class ModelRequest:
    instructions: str
    observation: JsonObject
    tools: tuple[ToolSpec, ...]
    exchanges: tuple[ModelToolExchange, ...]
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class ModelTurn:
    decision: Decision | None = None
    tool_calls: tuple[ModelToolCall, ...] = ()

    def __post_init__(self) -> None:
        if (self.decision is None) == (not self.tool_calls):
            raise ValueError("A model turn must contain either a decision or analysis tool calls")


class LanguageModel(ABC):
    @abstractmethod
    def complete(self, request: ModelRequest) -> ModelTurn:
        """Translate provider messages/tool calls; honor timeout_seconds for I/O.

        A final action tool call must become ModelTurn(decision=...), not an
        immediate Arena side effect. Actual API usage can be recorded through
        shared_logging.observations.observe_api_call in the provider adapter.
        """
        raise NotImplementedError


class PromptPolicy(Policy):
    def __init__(self, instructions: str, model: LanguageModel, *, max_rounds: int = 8) -> None:
        if not instructions.strip() or type(max_rounds) is not int or max_rounds < 1:
            raise ValueError("Prompt policy needs instructions and a positive round limit")
        self._instructions = instructions
        self._model = model
        self._max_rounds = max_rounds

    @property
    def identity(self) -> PolicyIdentity:
        return PolicyIdentity("prompt", sha256(self._instructions.encode("utf-8")).hexdigest()[:16])

    def decide(self, request: DecisionRequest, context: DecisionContext) -> Decision:
        exchanges: list[ModelToolExchange] = []
        seen_calls: set[str] = set()
        for _ in range(self._max_rounds):
            context.check_cancelled()
            turn = self._model.complete(ModelRequest(
                self._instructions, view_json(request.observation), context.tools.describe(),
                tuple(exchanges), context.remaining_seconds(),
            ))
            context.check_cancelled()
            if turn.decision is not None:
                return turn.decision
            for call in turn.tool_calls:
                if not call.call_id or call.call_id in seen_calls:
                    raise PolicyError("Model tool call IDs must be nonempty and unique within a decision")
                seen_calls.add(call.call_id)
                result = context.tools.call(call.name, call.arguments)
                exchanges.append(ModelToolExchange(call, result))
        raise PolicyError("The language policy exhausted its model round budget")
