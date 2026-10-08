"""Guidance policies for an LM-driven loop; no model or action execution here."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
import json
from math import isfinite
from typing import cast

from shared_logging import JsonObject

from poker.agent.models import DecisionRequest, PolicyIdentity
from poker.agent.tools import ToolResult
from poker.domain.types import ActionKind


def _validate_json(value: object, ancestors: set[int]) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("JSON values must be finite")
        return
    if not isinstance(value, (list, dict)):
        raise ValueError("Expected JSON scalars, arrays and objects")
    if id(value) in ancestors:
        raise ValueError("JSON values cannot contain cycles")
    ancestors.add(id(value))
    children: Iterable[object]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        children = value.values()
    else:
        children = value
    for child in children:
        _validate_json(child, ancestors)
    ancestors.remove(id(value))


def _object_copy(value: JsonObject) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("Context and memory must be JSON objects")
    _validate_json(value, set())
    try:
        return cast(JsonObject, json.loads(json.dumps(value, allow_nan=False)))
    except (TypeError, ValueError) as error:
        raise ValueError("Context and memory must contain finite JSON values") from error


def _names(value: tuple[str, ...] | None, label: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, tuple) or any(not isinstance(name, str) or not name.strip() or name != name.strip()
                                           for name in value):
        raise ValueError(f"{label} must be a tuple of nonempty tool names")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} cannot contain duplicate tool names")


@dataclass(frozen=True, slots=True)
class ToolExchange:
    call_id: str
    name: str
    arguments: JsonObject
    result: ToolResult

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id.strip():
            raise ValueError("A tool exchange requires a call ID")
        _names((self.name,), "Tool name")
        if not isinstance(self.result, ToolResult):
            raise ValueError("A tool exchange requires a ToolResult")
        _validate_json(self.result.value, set())
        object.__setattr__(self, "arguments", _object_copy(self.arguments))
        object.__setattr__(self, "result", deepcopy(self.result))


@dataclass(frozen=True, slots=True)
class PolicyRequest:
    decision: DecisionRequest
    memory: JsonObject
    exchanges: tuple[ToolExchange, ...]
    round_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.decision, DecisionRequest):
            raise ValueError("Policy guidance requires a DecisionRequest")
        if type(self.round_index) is not int or self.round_index < 0:
            raise ValueError("round_index must be a nonnegative integer")
        if not isinstance(self.exchanges, tuple) or any(not isinstance(item, ToolExchange) for item in self.exchanges):
            raise ValueError("exchanges must be a tuple of ToolExchange values")
        object.__setattr__(self, "memory", _object_copy(self.memory))
        object.__setattr__(self, "exchanges", deepcopy(self.exchanges))


@dataclass(frozen=True, slots=True)
class Guidance:
    instructions: str
    context: JsonObject = field(default_factory=dict)
    allowed_tools: tuple[str, ...] | None = None
    required_tools: tuple[str, ...] = ()
    # Successful exchanges before this index do not satisfy this requirement.
    required_after: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.instructions, str):
            raise ValueError("Guidance instructions must be text")
        _names(self.allowed_tools, "allowed_tools", optional=True)
        _names(self.required_tools, "required_tools")
        if type(self.required_after) is not int or self.required_after < 0:
            raise ValueError("required_after must be a nonnegative exchange index")
        if self.allowed_tools is not None and not set(self.required_tools) <= set(self.allowed_tools):
            raise ValueError("Required tools must be allowed in this guidance round")
        object.__setattr__(self, "context", _object_copy(self.context))


class Policy(ABC):
    """One required guidance hook; the loop alone invokes models and tools.

    Every invocation receives detached JSON memory and exchange values. Internal
    implementation state is unrestricted. Required tools are enforced by the loop
    and need successful results; a policy cannot return a final poker action.
    """

    @property
    def identity(self) -> PolicyIdentity:
        return PolicyIdentity(type(self).__qualname__)

    @abstractmethod
    def guidance(self, request: PolicyRequest) -> Guidance:
        raise NotImplementedError


class TextPolicy(Policy):
    def __init__(self, instructions: str, *, context: JsonObject | None = None,
                 allowed_tools: tuple[str, ...] | None = None, required_tools: tuple[str, ...] = ()) -> None:
        self._guidance = Guidance(instructions, {} if context is None else context, allowed_tools, required_tools)

    def guidance(self, request: PolicyRequest) -> Guidance:
        return deepcopy(self._guidance)


@dataclass(frozen=True, slots=True)
class StructuredConfig:
    instructions: str = "Compare the available evidence and choose a legal poker action."
    context: JsonObject = field(default_factory=dict)
    allowed_tools: tuple[str, ...] | None = None
    required_tools: tuple[str, ...] = ()
    require_equity_when_facing_bet: bool = False
    require_history: bool = False
    require_lookup: bool = False

    def __post_init__(self) -> None:
        Guidance(self.instructions, self.context, self.allowed_tools, self.required_tools)
        for value in (self.require_equity_when_facing_bet, self.require_history, self.require_lookup):
            if type(value) is not bool:
                raise ValueError("Structured tool conditions must be booleans")
        conditional = (("equity", self.require_equity_when_facing_bet),
                       ("public_history", self.require_history), ("lookup", self.require_lookup))
        if self.allowed_tools is not None and any(enabled and name not in self.allowed_tools
                                                 for name, enabled in conditional):
            raise ValueError("Conditionally required tools must be allowed")
        object.__setattr__(self, "context", _object_copy(self.context))


class StructuredPolicy(Policy):
    def __init__(self, config: StructuredConfig) -> None:
        if not isinstance(config, StructuredConfig):
            raise ValueError("StructuredPolicy requires StructuredConfig")
        self.config = deepcopy(config)

    def guidance(self, request: PolicyRequest) -> Guidance:
        view = request.decision.observation.view
        call_price = next((option.pay or 0 for option in view.me.legal_actions
                           if option.kind is ActionKind.CALL), 0)
        required = list(self.config.required_tools)
        conditions = (("equity", self.config.require_equity_when_facing_bet and call_price > 0),
                      ("public_history", self.config.require_history), ("lookup", self.config.require_lookup))
        for name, needed in conditions:
            if needed and name not in required:
                required.append(name)
        context = _object_copy(self.config.context)
        context["situation"] = {
            "phase": view.phase.value if view.phase else None,
            "call_price": call_price,
            "pot_total": view.pot_total,
            "pot_odds": call_price / (view.pot_total + call_price) if call_price > 0 else 0.0,
            "history_complete": view.history_complete,
        }
        context["memory"] = _object_copy(request.memory)
        return Guidance(self.config.instructions, context, self.config.allowed_tools, tuple(required))


@dataclass(frozen=True, slots=True)
class WorkflowStage:
    name: str
    instructions: str
    required_tools: tuple[str, ...] = ()
    context: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Workflow stages require a name")
        Guidance(self.instructions, self.context, required_tools=self.required_tools)
        object.__setattr__(self, "context", _object_copy(self.context))


class WorkflowPolicy(Policy):
    """Advance stages only after their ordered successful tool exchanges.

    A no-tool stage is terminal. A later stage cannot be satisfied by a result
    that preceded completion of the earlier stage; repeated tool names across
    stages therefore need fresh successful exchanges.
    """

    def __init__(self, stages: tuple[WorkflowStage, ...], *, instructions: str = "",
                 context: JsonObject | None = None, allowed_tools: tuple[str, ...] | None = None) -> None:
        if not isinstance(stages, tuple) or not stages or any(not isinstance(stage, WorkflowStage) for stage in stages):
            raise ValueError("WorkflowPolicy requires a nonempty tuple of stages")
        if len({stage.name for stage in stages}) != len(stages):
            raise ValueError("Workflow stage names must be unique")
        if any(not stage.required_tools for stage in stages[:-1]):
            raise ValueError("Only the final workflow stage may require no tools")
        required = tuple(dict.fromkeys(name for stage in stages for name in stage.required_tools))
        self._base = Guidance(instructions, {} if context is None else context, allowed_tools, required)
        self.stages = deepcopy(stages)

    def guidance(self, request: PolicyRequest) -> Guidance:
        cursor = 0
        instructions = [self._base.instructions]
        context = _object_copy(self._base.context)
        completed: list[str] = []
        for index, stage in enumerate(self.stages):
            stage_start_cursor = cursor
            instructions.append(stage.instructions)
            context.update(_object_copy(stage.context))
            missing = set(stage.required_tools)
            while missing and cursor < len(request.exchanges):
                exchange = request.exchanges[cursor]
                cursor += 1
                if exchange.result.ok:
                    missing.discard(exchange.name)
            if missing or not stage.required_tools:
                context["workflow"] = {"stage_index": index, "stage_name": stage.name,
                                       "completed_stages": list(completed)}
                return Guidance("\n".join(text for text in instructions if text), context,
                                self._base.allowed_tools, tuple(name for name in stage.required_tools if name in missing),
                                required_after=stage_start_cursor)
            completed.append(stage.name)
        context["workflow"] = {"stage_index": len(self.stages), "stage_name": "complete",
                               "completed_stages": list(completed)}
        return Guidance("\n".join(text for text in instructions if text), context, self._base.allowed_tools)
