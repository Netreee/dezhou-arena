"""Standalone agent process using the same CLI as human players."""

import argparse
from collections.abc import Callable, Sequence
from dataclasses import asdict
import importlib
import json
from pathlib import Path
from typing import cast

from shared_logging import JsonObject, LoggingConfig, process_logging
from poker.agent.cli_bridge import PokerCliAgentAdapter
from poker.agent.policy import Policy
from poker.agent.runtime import AgentConfig, AgentRunner
from poker.agent.tools import Tool, ToolRegistry, standard_tools
from poker.client.cli import PokerCli
from poker.client.parser import CommandParser
from poker.domain.types import TableId
from poker.transport.local_tcp import LocalTcpClient


def _factory(reference: str) -> Callable[[JsonObject], object]:
    module, separator, attribute = reference.partition(":")
    if not separator or not module or not attribute or "." in attribute:
        raise ValueError("Factory must use module:callable syntax")
    value: object = getattr(importlib.import_module(module), attribute)
    if not callable(value):
        raise ValueError("Factory is not callable")
    return cast(Callable[[JsonObject], object], value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy-neutral agent connected exclusively through the ordinary Poker CLI")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--table", help="Table to join; omitted selects the server's default table")
    parser.add_argument("--name", default="Agent")
    parser.add_argument("--policy", required=True, help="Trusted Python factory, module:callable, accepting a JSON object")
    parser.add_argument("--config", type=Path, help="Policy/tool factory JSON configuration")
    parser.add_argument("--tool-factory", action="append", default=[], help="Additional tools factory, module:callable")
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
    parser.add_argument("--auto-start", action="store_true", help="Let this one participant request new hands; independent of Policy")
    parser.add_argument("--min-players", type=int, default=2)
    args = parser.parse_args()
    with process_logging(LoggingConfig.from_environment("agent"), component="agent.process"):
        try:
            raw: object = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
            if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
                raise ValueError("Configuration must be a JSON object")
            config = cast(JsonObject, raw)
            policy = _factory(args.policy)(config)
            if not isinstance(policy, Policy):
                raise ValueError("Policy factory must return a Policy instance")
            tools = list(standard_tools())
            for reference in args.tool_factory:
                extra = _factory(reference)(config)
                if not isinstance(extra, (list, tuple)) or not all(isinstance(tool, Tool) for tool in extra):
                    raise ValueError("Tool factory must return a list or tuple of Tool instances")
                tools.extend(cast(Sequence[Tool], extra))
            runtime_config = AgentConfig(
                name=args.name, decision_timeout=args.decision_timeout, callback_timeout=args.callback_timeout,
                submission_timeout=args.submission_timeout, shutdown_timeout=args.shutdown_timeout,
                session_timeout=args.session_timeout, max_actions=args.max_actions, max_hands=args.max_hands,
                max_tool_calls=args.max_tool_calls, seed=args.seed, auto_start=args.auto_start, min_players=args.min_players,
            )
            port = PokerCliAgentAdapter(PokerCli(LocalTcpClient(args.port), CommandParser(TableId(args.table) if args.table else None)), poll_interval=args.poll_interval)
            result = AgentRunner(port, policy, runtime_config, tools=ToolRegistry(tools)).run()
        except (ValueError, ImportError, AttributeError, OSError) as error:
            parser.error(str(error))
        print(json.dumps({**asdict(result), "ok": result.ok}, ensure_ascii=False))
        raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
