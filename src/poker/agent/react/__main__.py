"""Compose a ReAct model, guidance policy and memory behind the ordinary CLI."""

import argparse
from collections.abc import Sequence
from dataclasses import asdict
import json
from pathlib import Path
from typing import cast

from shared_logging import JsonObject, LoggingConfig, get_logger, process_logging

from poker.agent.__main__ import _factory
from poker.agent.cli_bridge import PokerCliAgentAdapter
from poker.agent.react.backend import ModelBackend
from poker.agent.react.loop import LoopConfig, ReActAgent
from poker.agent.react.memory import Memory
from poker.agent.react.policy import Policy
from poker.agent.runtime import AgentConfig, AgentRunner
from poker.agent.tools import Tool, ToolRegistry, standard_tools
from poker.client.cli import PokerCli
from poker.client.parser import CommandParser
from poker.transport.local_tcp import LocalTcpClient


def build_agent(*, backend_factory: str, policy_factory: str, config: JsonObject,
                memory_factory: str | None = None, tool_factories: Sequence[str] = (),
                loop_config: LoopConfig | None = None) -> tuple[ReActAgent, ToolRegistry]:
    """Factories are trusted local Python; factory configuration is never logged."""
    backend = _factory(backend_factory)(config)
    if not isinstance(backend, ModelBackend):
        raise ValueError("Backend factory must return a ModelBackend instance")
    memory: Memory | None = None
    try:
        policy = _factory(policy_factory)(config)
        if not isinstance(policy, Policy):
            raise ValueError("Policy factory must return a react.policy.Policy instance")
        candidate = _factory(memory_factory)(config) if memory_factory else None
        if candidate is not None and not isinstance(candidate, Memory):
            raise ValueError("Memory factory must return a Memory instance")
        memory = candidate
        tools = list(standard_tools())
        for reference in tool_factories:
            extra = _factory(reference)(config)
            if not isinstance(extra, (list, tuple)) or not all(isinstance(tool, Tool) for tool in extra):
                raise ValueError("Tool factory must return a list or tuple of Tool instances")
            tools.extend(cast(Sequence[Tool], extra))
        registry = ToolRegistry(tools)
        return ReActAgent(backend, policy, memory=memory, config=loop_config), registry
    except BaseException:
        # Preserve the original factory/validation failure, even if cleanup
        # raises. Exception messages may contain factory secrets, so omit them.
        for resource in (memory, backend):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as cleanup_error:
                try:
                    get_logger("react.process").emit("WARNING", "react.setup.cleanup_failed", "Component cleanup failed after setup rejection", {
                        "component": type(resource).__qualname__, "error_type": type(cleanup_error).__name__,
                    })
                except BaseException:
                    pass
        raise


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ReAct Agent: explicit model backend, guidance Policy and ordinary Poker CLI")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--name", default="ReActAgent")
    parser.add_argument("--backend", required=True, help="Trusted ModelBackend factory, module:callable; no implicit provider/model")
    parser.add_argument("--policy", required=True, help="Trusted guidance Policy factory, module:callable")
    parser.add_argument("--memory-factory", help="Optional trusted Memory factory, module:callable")
    parser.add_argument("--tool-factory", action="append", default=[], help="Additional analysis tools factory, module:callable")
    parser.add_argument("--config", type=Path, help="Explicit JSON object passed to all selected factories")
    parser.add_argument("--max-model-calls-per-decision", type=int, default=6)
    parser.add_argument("--max-model-calls-per-session", type=int, default=32)
    parser.add_argument("--max-context-bytes", type=int, default=65536)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--max-response-bytes", type=int, default=65536)
    parser.add_argument("--poll-interval", type=float, default=0.2)
    parser.add_argument("--decision-timeout", type=float, default=30.0)
    parser.add_argument("--callback-timeout", type=float, default=5.0)
    parser.add_argument("--submission-timeout", type=float, default=10.0)
    parser.add_argument("--shutdown-timeout", type=float, default=1.0)
    parser.add_argument("--session-timeout", type=float)
    parser.add_argument("--max-actions", type=int)
    parser.add_argument("--max-hands", type=int)
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--min-players", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = argument_parser()
    args = parser.parse_args(argv)
    with process_logging(LoggingConfig.from_environment("agent"), component="react.process"):
        try:
            raw: object = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
            if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
                raise ValueError("Configuration must be a JSON object")
            runtime_config = AgentConfig(
                name=args.name, decision_timeout=args.decision_timeout, callback_timeout=args.callback_timeout,
                submission_timeout=args.submission_timeout, shutdown_timeout=args.shutdown_timeout,
                session_timeout=args.session_timeout, max_actions=args.max_actions, max_hands=args.max_hands,
                max_tool_calls=args.max_tool_calls, seed=args.seed, auto_start=args.auto_start, min_players=args.min_players,
            )
            port = PokerCliAgentAdapter(PokerCli(LocalTcpClient(args.port), CommandParser()), poll_interval=args.poll_interval)
            policy, tools = build_agent(
                backend_factory=args.backend, policy_factory=args.policy, config=cast(JsonObject, raw),
                memory_factory=args.memory_factory, tool_factories=args.tool_factory,
                loop_config=LoopConfig(max_model_calls_per_decision=args.max_model_calls_per_decision,
                                       max_model_calls_per_session=args.max_model_calls_per_session,
                                       max_context_bytes=args.max_context_bytes, max_output_tokens=args.max_output_tokens,
                                       max_response_bytes=args.max_response_bytes),
            )
            get_logger("react.process").emit("INFO", "react.process.configured", "Explicit ReAct components loaded", {
                "backend_factory": args.backend, "policy_factory": args.policy, "memory_factory": args.memory_factory,
                "tool_factories": list(args.tool_factory), "agent_class": type(policy).__qualname__,
            })
            result = AgentRunner(port, policy, runtime_config, tools=tools).run()
        except (ValueError, ImportError, AttributeError, OSError) as error:
            parser.error(str(error))
        print(json.dumps({**asdict(result), "ok": result.ok}, ensure_ascii=False))
        raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
