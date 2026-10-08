from collections.abc import Callable
from contextlib import nullcontext, redirect_stderr
from dataclasses import asdict
import io
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic
from typing import cast
import unittest
from unittest.mock import patch

from poker.agent.context import DecisionControl
from poker.agent.react.__main__ import argument_parser, build_agent, main
from poker.agent.react.backend import BackendRequest
from poker.agent.react.examples import FixtureBackend, LocalHttpFixtureBackend, codex_backend, chat_completions_backend
from poker.agent.react.loop import LoopConfig, ReActAgent
from poker.agent.react.memory import BoundedEventMemory
from poker.agent.react.policy import TextPolicy
from poker.agent.tools import standard_tools
from poker.agent.models import PolicyError
from shared_logging import JsonObject, JsonValue


class ReActCompositionTests(unittest.TestCase):
    def request(self, *, pending: list[str], exchanges: list[JsonObject] | None = None) -> BackendRequest:
        return BackendRequest("Fixture test", {"pending_tools": cast(list[JsonValue], pending), "required_tools": ["legal_actions"],
                              "exchanges": cast(list[JsonValue], exchanges or []), "round_index": 2, "memory": {}}, {})

    def test_fixture_is_explicitly_offline_and_obeys_current_pending_tools(self) -> None:
        backend = FixtureBackend()
        self.assertEqual(asdict(backend.identity), {"provider": "offline-fixture", "model": "deterministic-tool-loop-v1", "is_live": False})
        old: JsonObject = {"name": "legal_actions", "result": {"error": None, "value": [{"kind": "check"}]}}
        response = backend.generate(self.request(pending=["legal_actions"], exchanges=[old]), DecisionControl(monotonic() + 10))
        self.assertEqual(response.output, {"kind": "tool_calls", "calls": [
            {"id": "fixture-2-0", "name": "legal_actions", "arguments": {}}], "action": None})

    def test_fixture_final_requires_a_successful_actual_tool_exchange(self) -> None:
        backend = FixtureBackend()
        with self.assertRaises(PolicyError):
            backend.generate(self.request(pending=[]), DecisionControl(monotonic() + 10))
        failed: JsonObject = {"name": "legal_actions", "result": {"error": {"code": "failure"}, "value": [{"kind": "check"}]}}
        with self.assertRaises(PolicyError):
            backend.generate(self.request(pending=[], exchanges=[failed]), DecisionControl(monotonic() + 10))
        success: JsonObject = {"name": "legal_actions", "result": {"error": None, "value": [{"kind": "fold"}, {"kind": "call", "pay": 10}]}}
        output = backend.generate(self.request(pending=[], exchanges=[success]), DecisionControl(monotonic() + 10)).output
        self.assertEqual(output, {"kind": "final", "calls": [], "action": {"kind": "call", "to": None}})

    def test_fixture_does_not_fall_back_to_unoffered_actions(self) -> None:
        exchange: JsonObject = {"name": "legal_actions", "result": {"error": None, "value": [{"kind": "fold"}]}}
        with self.assertRaises(PolicyError):
            FixtureBackend().generate(self.request(pending=[], exchanges=[exchange]), DecisionControl(monotonic() + 10))

    def test_all_six_factories_compose_the_same_react_agent(self) -> None:
        for name in ("text", "rules", "lookup", "memory", "solver", "combined"):
            with self.subTest(name=name):
                agent, registry = build_agent(backend_factory="poker.agent.react.examples:fixture_backend",
                                             policy_factory=f"poker.agent.react.examples:{name}", config={},
                                             memory_factory="poker.agent.react.examples:event_memory",
                                             tool_factories=("poker.agent.react.examples:analysis_tools",))
                self.assertIsInstance(agent, ReActAgent)
                self.assertIsNotNone(registry)

    def test_invalid_factory_types_are_rejected_before_connecting(self) -> None:
        with patch("poker.agent.react.__main__._factory", return_value=lambda _: object()):
            with self.assertRaisesRegex(ValueError, "ModelBackend"):
                build_agent(backend_factory="example:bad", policy_factory="example:unused", config={})
        with self.assertRaisesRegex(ValueError, "react.policy.Policy"):
            build_agent(backend_factory="poker.agent.react.examples:fixture_backend",
                        policy_factory="poker.agent.examples:uniform", config={})
        with self.assertRaisesRegex(ValueError, "Memory"):
            build_agent(backend_factory="poker.agent.react.examples:fixture_backend",
                        policy_factory="poker.agent.react.examples:text",
                        memory_factory="poker.agent.react.examples:text", config={})

    def test_live_backend_factories_require_explicit_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "model"):
            codex_backend({})
        with self.assertRaisesRegex(ValueError, "model"):
            chat_completions_backend({})
        with self.assertRaisesRegex(ValueError, "base_url"):
            chat_completions_backend({"model": "explicit-model"})
        with self.assertRaisesRegex(ValueError, "api_key_env"):
            chat_completions_backend({"model": "explicit-model", "base_url": "http://127.0.0.1:1/v1"})

    def test_http_fixture_refuses_external_endpoints_and_reports_no_live_inference(self) -> None:
        for address in ("https://example.com/v1", "http://example.com/v1", "http://user:pass@127.0.0.1/v1"):
            with self.subTest(address=address), self.assertRaises(ValueError):
                LocalHttpFixtureBackend(address)
        backend = LocalHttpFixtureBackend("http://127.0.0.1:1/v1")
        self.assertFalse(backend.identity.is_live)
        self.assertEqual(backend.identity.provider, "offline-http-fixture")

    def test_cli_has_no_implicit_backend_and_exposes_loop_limits(self) -> None:
        args = argument_parser().parse_args(["--backend", "poker.agent.react.examples:fixture_backend",
                                            "--policy", "poker.agent.react.examples:text"])
        self.assertEqual((args.max_model_calls_per_decision, args.max_model_calls_per_session,
                          args.max_context_bytes, args.max_output_tokens, args.max_response_bytes), (6, 32, 65536, 512, 65536))
        custom = argument_parser().parse_args(["--backend", "x:y", "--policy", "x:z", "--max-response-bytes", "1234"])
        agent, _ = build_agent(backend_factory="poker.agent.react.examples:fixture_backend",
                               policy_factory="poker.agent.react.examples:text", config={},
                               loop_config=LoopConfig(max_response_bytes=custom.max_response_bytes))
        self.assertEqual(agent.config.max_response_bytes, 1234)

    def test_failed_setup_closes_all_previously_constructed_resources(self) -> None:
        for stage in ("policy_type", "memory_type", "tool_type", "tool_registry"):
            with self.subTest(stage=stage):
                closed: list[str] = []

                class Backend(FixtureBackend):
                    def close(self) -> None:
                        closed.append("backend")

                class Memory(BoundedEventMemory):
                    def close(self) -> None:
                        closed.append("memory")

                factories: dict[str, Callable[[JsonObject], object]] = {
                    "backend": lambda _: Backend(),
                    "policy": lambda _: object() if stage == "policy_type" else TextPolicy("test"),
                    "memory": lambda _: object() if stage == "memory_type" else Memory(),
                    "tools": lambda _: object() if stage == "tool_type" else list(standard_tools()),
                }
                with patch("poker.agent.react.__main__._factory", side_effect=lambda reference: factories[reference]):
                    with self.assertRaises(ValueError):
                        build_agent(backend_factory="backend", policy_factory="policy", memory_factory="memory",
                                    tool_factories=("tools",), config={})
                self.assertEqual(closed, ["memory", "backend"] if stage.startswith("tool_") else ["backend"])

    def test_cleanup_errors_are_sanitized_and_cannot_mask_factory_failure(self) -> None:
        failure = RuntimeError("original factory failure")
        closed: list[str] = []

        class Backend(FixtureBackend):
            def close(self) -> None:
                closed.append("backend")
                raise ValueError("SECRET_FROM_BACKEND")

        class Memory(BoundedEventMemory):
            def close(self) -> None:
                closed.append("memory")
                raise ValueError("SECRET_FROM_MEMORY")

        def fail(config: JsonObject) -> object:
            raise failure

        factories: dict[str, Callable[[JsonObject], object]] = {
            "backend": lambda _: Backend(), "policy": lambda _: TextPolicy("test"),
            "memory": lambda _: Memory(), "tools": fail,
        }
        with patch("poker.agent.react.__main__._factory", side_effect=lambda reference: factories[reference]), \
                patch("poker.agent.react.__main__.get_logger") as logger:
            with self.assertRaises(RuntimeError) as raised:
                build_agent(backend_factory="backend", policy_factory="policy", memory_factory="memory",
                            tool_factories=("tools",), config={})
            self.assertIs(raised.exception, failure)
            self.assertNotIn("SECRET_", str(logger.return_value.emit.call_args_list))
            self.assertEqual(logger.return_value.emit.call_count, 2)
        self.assertEqual(closed, ["memory", "backend"])

    def test_invalid_runtime_or_transport_options_do_not_construct_components(self) -> None:
        for options in (["--max-actions", "0"], ["--poll-interval", "0"]):
            with self.subTest(options=options), patch("poker.agent.react.__main__.build_agent") as build, \
                    patch("poker.agent.react.__main__.process_logging", return_value=nullcontext()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(["--backend", "x:y", "--policy", "x:z", *options])
                self.assertEqual(raised.exception.code, 2)
                build.assert_not_called()

    def test_official_module_help_is_runnable(self) -> None:
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run([sys.executable, "-X", "utf8", "-m", "poker.agent.react", "--help"],
                                cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--backend", result.stdout)
        self.assertIn("--memory-factory", result.stdout)
        self.assertIn("--max-model-calls-per-session", result.stdout)
        self.assertNotIn("api_key", json.dumps(vars(argument_parser().parse_args([
            "--backend", "x:y", "--policy", "x:z"]))))


if __name__ == "__main__":
    unittest.main()
