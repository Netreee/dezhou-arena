"""Allowlisted, schema-checked analysis tools bound to one frozen observation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from time import monotonic
from typing import cast

from jsonschema import Draft202012Validator
from shared_logging import JsonObject, JsonValue, get_logger

from poker.agent.context import DecisionControl
from poker.agent.models import Observation, PolicyError, ToolBudgetExceeded


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: JsonObject
    output_schema: JsonObject | None = None


@dataclass(frozen=True, slots=True)
class ToolError:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    value: JsonValue = None
    error: ToolError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True, slots=True)
class ToolContext:
    observation: Observation
    control: DecisionControl


class Tool(ABC):
    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        raise NotImplementedError

    @abstractmethod
    def invoke(self, arguments: JsonObject, context: ToolContext) -> JsonValue:
        raise NotImplementedError


class ToolAccess(ABC):
    @abstractmethod
    def describe(self) -> tuple[ToolSpec, ...]:
        raise NotImplementedError

    @abstractmethod
    def call(self, name: str, arguments: JsonObject) -> ToolResult:
        raise NotImplementedError


class ToolRegistry:
    def __init__(self, tools: Sequence[Tool]) -> None:
        self._tools: dict[str, tuple[Tool, ToolSpec]] = {}
        for tool in tools:
            spec = deepcopy(tool.spec)
            if not spec.name or spec.name in self._tools:
                raise ValueError("Tool names must be nonempty and unique")
            Draft202012Validator.check_schema(spec.input_schema)
            if spec.output_schema is not None:
                Draft202012Validator.check_schema(spec.output_schema)
            self._tools[spec.name] = tool, spec

    def bind(self, observation: Observation, control: DecisionControl, *, max_calls: int, decision_id: str) -> BoundTools:
        if type(max_calls) is not int or max_calls < 0:
            raise ValueError("Tool budget cannot be negative")
        return BoundTools(self._tools, observation, control, max_calls, decision_id)


class BoundTools(ToolAccess):
    def __init__(self, tools: dict[str, tuple[Tool, ToolSpec]], observation: Observation,
                 control: DecisionControl, max_calls: int, decision_id: str) -> None:
        self._tools = tools
        self._context = ToolContext(observation, control)
        self._max_calls = max_calls
        self.calls = 0
        self._log = get_logger("agent.tools").bind(correlation_id=decision_id,
                                                  player_id=observation.view.me.player_id,
                                                  hand_id=observation.view.hand_id)

    def describe(self) -> tuple[ToolSpec, ...]:
        self._context.control.check()
        return tuple(deepcopy(spec) for _, spec in self._tools.values())

    def call(self, name: str, arguments: JsonObject) -> ToolResult:
        self._context.control.check()
        if self.calls >= self._max_calls:
            raise ToolBudgetExceeded("The per-decision tool budget is exhausted")
        self.calls += 1
        started = monotonic()
        result: ToolResult
        entry = self._tools.get(name)
        if entry is None:
            result = ToolResult(error=ToolError("unknown_tool", "The requested tool is not available"))
        else:
            tool, spec = entry
            try:
                json.dumps(arguments, allow_nan=False)
                argument_error = next(Draft202012Validator(spec.input_schema).iter_errors(arguments), None)
                if argument_error is not None:
                    result = ToolResult(error=ToolError("invalid_arguments", "Tool arguments do not match the schema"))
                else:
                    value = tool.invoke(deepcopy(arguments), self._context)
                    json.dumps(value, allow_nan=False)
                    output_error = (next(Draft202012Validator(spec.output_schema).iter_errors(value), None)
                                    if spec.output_schema is not None else None)
                    if output_error is not None:
                        result = ToolResult(error=ToolError("invalid_output", "Tool output does not match its schema"))
                    else:
                        result = ToolResult(value=deepcopy(value))
            except PolicyError:
                raise
            except Exception:
                # Exception text may contain cards, prompts or credentials.
                result = ToolResult(error=ToolError("tool_failed", "Tool execution failed"))
        self._context.control.check()
        self._log.emit("INFO" if result.ok else "WARNING", "agent.tool.completed", "Analysis tool completed", {
            "tool": name, "ok": result.ok, "error_code": result.error.code if result.error else None,
            "duration_ms": (monotonic() - started) * 1000, "call_number": self.calls,
        })
        return result


def view_json(observation: Observation) -> JsonObject:
    """Serialize only the safe CLI DTO. Never accept an Arena Table here."""
    return cast(JsonObject, json.loads(json.dumps(asdict(observation.view))))


class ObservationTool(Tool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec("observation", "Return this decision's frozen player-visible CLI observation.",
                        {"type": "object", "properties": {}, "additionalProperties": False}, {"type": "object"})

    def invoke(self, arguments: JsonObject, context: ToolContext) -> JsonValue:
        return view_json(context.observation)


class LegalActionsTool(Tool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec("legal_actions", "Return server-offered actions. min_to/max_to are total street contributions.",
                        {"type": "object", "properties": {}, "additionalProperties": False}, {"type": "array"})

    def invoke(self, arguments: JsonObject, context: ToolContext) -> JsonValue:
        return cast(JsonValue, json.loads(json.dumps([asdict(a) for a in context.observation.view.me.legal_actions])))


class PublicHistoryTool(Tool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec("public_history", "Return the current hand's public action history and completeness flag.",
                        {"type": "object", "properties": {}, "additionalProperties": False}, {"type": "object"})

    def invoke(self, arguments: JsonObject, context: ToolContext) -> JsonValue:
        view = context.observation.view
        return {"hand_id": view.hand_id, "complete": view.history_complete,
                "actions": cast(JsonValue, json.loads(json.dumps([asdict(a) for a in view.action_history])))}


def standard_tools() -> tuple[Tool, ...]:
    return ObservationTool(), LegalActionsTool(), PublicHistoryTool()
