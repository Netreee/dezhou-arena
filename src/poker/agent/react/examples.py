"""Composition examples and an explicit offline model fixture.

FixtureBackend is deterministic test code, never a language model. The six
guidance examples also work with an explicitly configured live ModelBackend.
"""

import hashlib
import json
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from shared_logging import JsonObject, JsonValue, get_logger

from poker.agent.context import DecisionControl
from poker.agent.models import PolicyError
from poker.agent.tools import Tool
from poker.agent.react.analysis_tools import EquityTool, LookupTool
from poker.agent.react.backend import BackendIdentity, BackendRequest, BackendResponse, ModelBackend
from poker.agent.react.backends import ChatCompletionsBackend, CodexBackend
from poker.agent.react.memory import BoundedEventMemory, Memory
from poker.agent.react.policy import Policy, StructuredConfig, StructuredPolicy, TextPolicy, WorkflowPolicy, WorkflowStage


class FixtureBackend(ModelBackend):
    """Exercise real request/tool exchanges without any external inference.

    pending_tools is authoritative: a workflow may require a fresh tool result
    even when the same tool already succeeded in an earlier workflow stage.
    """

    @property
    def identity(self) -> BackendIdentity:
        return BackendIdentity("offline-fixture", "deterministic-tool-loop-v1", False)

    @property
    def supports_output_token_limit(self) -> bool:
        return False

    def generate(self, request: BackendRequest, control: DecisionControl) -> BackendResponse:
        control.check()
        payload = request.input
        pending = payload.get("pending_tools")
        if not isinstance(pending, list) or any(not isinstance(name, str) for name in pending):
            raise PolicyError("Fixture requires the loop's authoritative pending_tools array")
        round_index = payload.get("round_index")
        if type(round_index) is not int:
            raise PolicyError("Fixture requires an integer round_index")
        output: JsonObject
        if pending:
            output = {"kind": "tool_calls", "calls": [
                {"id": f"fixture-{round_index}-{index}", "name": name, "arguments": {}}
                for index, name in enumerate(pending)], "action": None}
        else:
            exchanges = payload.get("exchanges")
            if not isinstance(exchanges, list):
                raise PolicyError("Fixture requires tool exchanges")
            legal: list[JsonValue] | None = None
            for exchange in exchanges:
                if isinstance(exchange, dict) and exchange.get("name") == "legal_actions":
                    result = exchange.get("result")
                    if isinstance(result, dict) and result.get("error") is None and isinstance(result.get("value"), list):
                        legal = cast(list[JsonValue], result["value"])
            if legal is None:
                raise PolicyError("Fixture final action requires a successful legal_actions exchange")
            # A conservative deterministic action keeps the fixture's six
            # players funded. No competitive poker claim follows from this.
            chosen = next((item for kind in ("check", "call") for item in legal
                           if isinstance(item, dict) and item.get("kind") == kind), None)
            if not isinstance(chosen, dict):
                raise PolicyError("Fixture expected a check or call; it will not invent a legal move")
            output = {"kind": "final", "calls": [], "action": {"kind": chosen["kind"], "to": None}}
        memory_value = payload.get("memory", {})
        records = memory_value.get("records", []) if isinstance(memory_value, dict) else []
        get_logger("react.fixture").emit("INFO", "react.fixture.generated", "Offline fixture generated a protocol response", {
            "is_live": False, "round_index": round_index, "output_kind": output["kind"],
            "pending_tools": pending, "memory_records": len(records) if isinstance(records, list) else 0,
            "memory_sha256": hashlib.sha256(json.dumps(memory_value, sort_keys=True).encode()).hexdigest(),
            "input_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        })
        control.check()
        return BackendResponse(output=output, finish_reason="offline_fixture")


def fixture_backend(config: JsonObject) -> ModelBackend:
    return FixtureBackend()


class LocalHttpFixtureBackend(ChatCompletionsBackend):
    """Use the production HTTP transport against a loopback test responder."""

    def __init__(self, base_url: str) -> None:
        url = urlsplit(base_url)
        if url.scheme != "http" or url.hostname != "127.0.0.1" or url.username or url.password:
            raise ValueError("HTTP fixture must use explicit http://127.0.0.1 loopback")
        super().__init__(model="local-http-fixture-v1", base_url=base_url, api_key_env="POKER_REACT_FIXTURE_KEY")

    @property
    def identity(self) -> BackendIdentity:
        return BackendIdentity("offline-http-fixture", "local-http-fixture-v1", False)


def http_fixture_backend(config: JsonObject) -> ModelBackend:
    return LocalHttpFixtureBackend(_required_text(config, "base_url"))


def _required_text(config: JsonObject, key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Explicit nonempty '{key}' is required in factory configuration")
    return value


def codex_backend(config: JsonObject) -> ModelBackend:
    executable = config.get("executable")
    if executable is not None and (not isinstance(executable, str) or not executable.strip()):
        raise ValueError("executable must be an explicit nonempty path")
    return CodexBackend(model=_required_text(config, "model"), executable=Path(executable) if isinstance(executable, str) else None)


def chat_completions_backend(config: JsonObject) -> ModelBackend:
    return ChatCompletionsBackend(model=_required_text(config, "model"), base_url=_required_text(config, "base_url"),
                                  api_key_env=_required_text(config, "api_key_env"))


_INSTRUCTIONS = "Use the required analysis results and choose one action offered by the CLI."


def text(config: JsonObject) -> Policy:
    instructions = config.get("instructions", _INSTRUCTIONS)
    if not isinstance(instructions, str):
        raise ValueError("instructions must be text")
    return TextPolicy(instructions, context={"representation": "text"},
                      allowed_tools=("legal_actions",), required_tools=("legal_actions",))


def rules(config: JsonObject) -> Policy:
    return StructuredPolicy(StructuredConfig(instructions=_INSTRUCTIONS, context={"representation": "rules"},
                            allowed_tools=("legal_actions", "public_history"), required_tools=("legal_actions",),
                            require_history=True))


def lookup(config: JsonObject) -> Policy:
    return StructuredPolicy(StructuredConfig(instructions=_INSTRUCTIONS + " Compare the lookup evidence.",
                            context={"representation": "lookup"}, allowed_tools=("legal_actions", "lookup"),
                            required_tools=("legal_actions",), require_lookup=True))


def memory(config: JsonObject) -> Policy:
    return TextPolicy(_INSTRUCTIONS + " Consider factual confirmed actions and completed hands in memory.",
                      context={"representation": "memory"}, allowed_tools=("legal_actions", "public_history"),
                      required_tools=("legal_actions", "public_history"))


def solver(config: JsonObject) -> Policy:
    return TextPolicy(_INSTRUCTIONS + " Compare sampled equity and the offered call price.",
                      context={"representation": "solver"}, allowed_tools=("legal_actions", "equity"),
                      required_tools=("legal_actions", "equity"))


def combined(config: JsonObject) -> Policy:
    return WorkflowPolicy((
        WorkflowStage("observe", "Read legal actions and public action history.", ("legal_actions", "public_history")),
        WorkflowStage("analyse", "Compare lookup and sampled equity after reading the history.", ("lookup", "equity")),
        WorkflowStage("decide", "Use the collected evidence and factual memory to choose one legal action."),
    ), instructions=_INSTRUCTIONS, context={"representation": "combined"},
       allowed_tools=("legal_actions", "public_history", "lookup", "equity"))


def event_memory(config: JsonObject) -> Memory:
    capacity = config.get("memory_capacity", 64)
    if type(capacity) is not int:
        raise ValueError("memory_capacity must be an integer")
    return BoundedEventMemory(max_events=capacity)


def analysis_tools(config: JsonObject) -> list[Tool]:
    rollouts = config.get("equity_rollouts", 24)
    if type(rollouts) is not int:
        raise ValueError("equity_rollouts must be an integer")
    return [LookupTool(), EquityTool(rollouts=rollouts, seed=0)]
