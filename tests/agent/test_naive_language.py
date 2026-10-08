from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from queue import Queue
from random import Random
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import monotonic
from typing import cast
import unittest
from unittest.mock import patch

from jsonschema import ValidationError
from shared_logging import JsonObject, LoggingConfig, configure_logging, shutdown_logging

from poker.agent.context import DecisionContext, DecisionControl
from poker.agent.language import ModelRequest
from poker.agent.models import Decision, DecisionCancelled, DecisionRequest, PolicyError
from poker.agent.naive_language import (
    CodexLanguageModel, NaiveLanguagePolicy, OfflineLanguageFixture, _turn, natural_language,
)
from poker.agent.tools import ToolRegistry, standard_tools, view_json
from poker.domain.models import PlayerAction
from poker.domain.types import ActionKind
from tests.agent.test_contracts import Clock, make_observation


CHECK: JsonObject = {"mode": "decision", "tool": None, "action": "check", "to": None}
TOOL: JsonObject = {"mode": "tool", "tool": "legal_actions", "action": None, "to": None}
COMPLETED: JsonObject = {"type": "turn.completed", "usage": {}}
SECRET = "PRIVATE-PROMPT-OR-PROVIDER-DIAGNOSTIC-should-not-be-logged"


@dataclass
class Scenario:
    answer: object = None
    events: tuple[JsonObject, ...] = (COMPLETED,)
    exit_code: int = 0
    first_error: BaseException | None = None
    repeat_timeouts: bool = False


class FakeProcess:
    def __init__(self, command: list[str], options: dict[str, object], scenario: Scenario, pid: int) -> None:
        self.command = command
        self.options = options
        self.scenario = scenario
        self.pid = pid
        self.returncode: int | None = None
        self.communications: list[tuple[str | None, float | None]] = []
        self.kills = 0
        self.waits = 0

    def communicate(self, input: str | None = None, timeout: float | None = None) -> tuple[str, str]:
        self.communications.append((input, timeout))
        if self.scenario.repeat_timeouts and not self.kills:
            raise subprocess.TimeoutExpired("fixture-codex", timeout or 0)
        if len(self.communications) == 1 and self.scenario.first_error is not None:
            raise self.scenario.first_error
        if self.kills:
            self.returncode = -9
        else:
            self.returncode = self.scenario.exit_code
        if self.scenario.answer is not None:
            answer_index = self.command.index("--output-last-message") + 1
            Path(self.command[answer_index]).write_text(json.dumps(self.scenario.answer), encoding="utf-8")
        return "\n".join(json.dumps(event) for event in self.scenario.events), SECRET

    def kill(self) -> None:
        self.kills += 1

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.waits += 1
        self.returncode = -9 if self.kills else self.scenario.exit_code
        return self.returncode


class FakePopen:
    def __init__(self, *scenarios: Scenario) -> None:
        self.scenarios = list(scenarios)
        self.processes: list[FakeProcess] = []

    def __call__(self, command: list[str], **options: object) -> FakeProcess:
        if not self.scenarios:
            raise AssertionError("Unexpected provider process; retries/fallback are not authorized")
        process = FakeProcess(command, options, self.scenarios.pop(0), 1000 + len(self.processes))
        self.processes.append(process)
        return process


class NaiveLanguageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.log_path = configure_logging(LoggingConfig(self.root / "logs", "naive-language-test", "test"))
        self.observation = make_observation()
        self.control = DecisionControl(20, clock=Clock())
        self.tools = ToolRegistry(standard_tools()).bind(self.observation, self.control, max_calls=5, decision_id="d1")
        self.context = DecisionContext(self.tools, Random(4), self.control)
        self.request = DecisionRequest("d1", self.observation, 20)
        self.model_request = ModelRequest(SECRET, view_json(self.observation), self.tools.describe(), (), 3.0)

    def tearDown(self) -> None:
        shutdown_logging()
        self.temp.cleanup()

    def model(self, *, max_calls: int = 8) -> CodexLanguageModel:
        return CodexLanguageModel(model="fixture-model", max_calls=max_calls,
                                  executable=Path("fixture-codex.exe"), evidence_dir=self.root / "evidence")

    def evidence(self) -> list[JsonObject]:
        return [cast(JsonObject, json.loads(path.read_text(encoding="utf-8")))
                for path in sorted((self.root / "evidence").glob("*.json"))]

    def log_rows(self) -> list[JsonObject]:
        return [cast(JsonObject, json.loads(line)) for line in self.log_path.read_text(encoding="utf-8").splitlines()]

    def test_structured_turn_rejects_invalid_mode_combinations_and_amounts(self) -> None:
        invalid: tuple[object, ...] = (
            {**TOOL, "action": "check"}, {**TOOL, "to": 10}, {**TOOL, "tool": None},
            {**CHECK, "tool": "legal_actions"}, {**CHECK, "action": None},
            {**CHECK, "to": 1}, {**CHECK, "action": "raise_to", "to": None},
            {**CHECK, "action": "raise_to", "to": True},
            {**CHECK, "action": "raise_to", "to": 0},
            {**TOOL, "tool": "shell"}, {**CHECK, "unexpected": SECRET}, [], {},
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises((PolicyError, ValueError, ValidationError)):
                _turn(raw, "tool-call")
        tool = _turn(TOOL, "tool-call")
        self.assertEqual(tool.tool_calls[0].call_id, "tool-call")
        decision = _turn(CHECK, "decision").decision
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.choice, PlayerAction(ActionKind.CHECK))

    def test_provider_failure_consumes_budget_and_never_runs_a_fallback(self) -> None:
        provider = FakePopen(Scenario(answer=CHECK, exit_code=7))
        model = self.model(max_calls=1)
        with patch("poker.agent.naive_language.subprocess.Popen", provider):
            with self.assertRaisesRegex(PolicyError, "failed"):
                model.complete(self.model_request)
            with self.assertRaisesRegex(PolicyError, "budget"):
                model.complete(self.model_request)
        self.assertEqual(model.calls, 1)
        self.assertEqual(len(provider.processes), 1)
        evidence = self.evidence()
        self.assertEqual(len(evidence), 1)
        self.assertFalse(evidence[0]["success"])
        self.assertEqual(evidence[0]["exit_code"], 7)

    def test_process_creation_failure_also_consumes_attempt_budget(self) -> None:
        model = self.model(max_calls=1)
        with patch("poker.agent.naive_language.subprocess.Popen", side_effect=OSError("could not start")) as popen:
            with self.assertRaises((OSError, PolicyError)):
                model.complete(self.model_request)
            with self.assertRaisesRegex(PolicyError, "budget"):
                model.complete(self.model_request)
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(model.calls, 1)
        self.assertFalse(self.evidence()[0]["success"])
        self.assertEqual(self.evidence()[0]["status"], "failed")

    def test_attempt_receipt_is_durable_before_launch_and_updated_after_completion(self) -> None:
        test_case = self

        class EvidenceCheckingProcess(FakeProcess):
            def communicate(self, input: str | None = None, timeout: float | None = None) -> tuple[str, str]:
                receipt = test_case.evidence()[0]
                test_case.assertEqual(receipt["status"], "started")
                test_case.assertEqual(receipt["child_pid"], self.pid)
                test_case.assertFalse(receipt["success"])
                return super().communicate(input, timeout)

        def launch(command: list[str], **options: object) -> FakeProcess:
            receipts = self.evidence()
            self.assertEqual(len(receipts), 1)
            self.assertEqual(receipts[0]["status"], "started")
            self.assertEqual(receipts[0]["call_number"], 1)
            self.assertNotIn("child_pid", receipts[0])
            return EvidenceCheckingProcess(command, options, Scenario(answer=CHECK), 7654)

        with patch("poker.agent.naive_language.subprocess.Popen", side_effect=launch):
            self.model().complete(self.model_request)
        receipts = self.evidence()
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0]["status"], "completed")
        self.assertTrue(receipts[0]["success"])
        self.assertEqual(list((self.root / "evidence").glob("*.tmp")), [])

    def test_startup_error_item_is_a_diagnostic_when_a_completed_answer_exists(self) -> None:
        diagnostic: JsonObject = {"type": "item.completed", "item": {"type": "error", "message": SECRET}}
        provider = FakePopen(Scenario(answer=CHECK, events=(diagnostic, COMPLETED)))
        with patch("poker.agent.naive_language.subprocess.Popen", provider):
            turn = self.model().complete(self.model_request)
        self.assertIsNotNone(turn.decision)
        receipt = self.evidence()[0]
        self.assertTrue(receipt["success"])
        self.assertEqual(receipt["cli_diagnostic_items"], 1)
        self.assertEqual(receipt["environment_tool_items"], 0)

    def test_completed_turn_and_answer_file_are_both_required(self) -> None:
        diagnostic: JsonObject = {"type": "item.completed", "item": {"type": "error", "message": SECRET}}
        scenarios = (
            Scenario(answer=CHECK, events=(diagnostic,)),
            Scenario(answer=None, events=(COMPLETED,)),
            Scenario(answer=CHECK, events=({"type": "turn.failed"}, COMPLETED)),
        )
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                provider = FakePopen(scenario)
                with patch("poker.agent.naive_language.subprocess.Popen", provider):
                    with self.assertRaisesRegex(PolicyError, "completed"):
                        self.model().complete(self.model_request)
                self.assertEqual(len(provider.processes), 1)

    def test_environment_tool_items_are_rejected_including_progress_updates(self) -> None:
        for event_type in ("item.started", "item.updated", "item.completed"):
            for kind in ("command_execution", "mcp_tool_call", "web_search"):
                with self.subTest(event_type=event_type, kind=kind):
                    event: JsonObject = {"type": event_type, "item": {"type": kind, "command": SECRET}}
                    provider = FakePopen(Scenario(answer=CHECK, events=(event, COMPLETED)))
                    with patch("poker.agent.naive_language.subprocess.Popen", provider):
                        with self.assertRaisesRegex(PolicyError, "environment"):
                            self.model().complete(self.model_request)

    def test_timeout_kills_and_reaps_process_without_retry_or_fallback(self) -> None:
        provider = FakePopen(Scenario(repeat_timeouts=True))
        with patch("poker.agent.naive_language.subprocess.Popen", provider):
            with self.assertRaisesRegex(PolicyError, "deadline"):
                self.model().complete(replace(self.model_request, timeout_seconds=0.05))
        process = provider.processes[0]
        self.assertEqual(process.kills, 1)
        self.assertEqual(process.communications[-1], (None, 5))
        self.assertIsNotNone(process.returncode)
        self.assertEqual(len(provider.processes), 1)
        self.assertFalse(self.evidence()[0]["success"])

    def test_communicate_poll_timeout_keeps_one_call_and_does_not_resend_stdin(self) -> None:
        provider = FakePopen(Scenario(answer=CHECK, first_error=subprocess.TimeoutExpired("fixture-codex", 0.1)))
        model = self.model(max_calls=1)
        with patch("poker.agent.naive_language.subprocess.Popen", provider):
            result = model.complete(self.model_request)
        self.assertIsNotNone(result.decision)
        self.assertEqual(model.calls, 1)
        self.assertEqual(len(provider.processes), 1)
        process = provider.processes[0]
        self.assertEqual(process.kills, 0)
        self.assertEqual(len(process.communications), 2)
        self.assertIsNotNone(process.communications[0][0])
        self.assertIsNone(process.communications[1][0])
        self.assertTrue(all(timeout is not None and timeout <= 0.1 for _, timeout in process.communications))

    def test_external_decision_cancel_kills_child_and_restores_control_scope(self) -> None:
        entered, release = Event(), Event()
        processes: list[FakeProcess] = []
        results: Queue[Decision | BaseException] = Queue()

        class BlockingProcess(FakeProcess):
            def communicate(self, input: str | None = None, timeout: float | None = None) -> tuple[str, str]:
                if not self.communications and not self.kills:
                    self.communications.append((input, timeout))
                    entered.set()
                    release.wait(1)
                    raise subprocess.TimeoutExpired("fixture-codex", timeout or 0)
                return super().communicate(input, timeout)

        def launch(command: list[str], **options: object) -> FakeProcess:
            process = BlockingProcess(command, options, Scenario(), 8765)
            processes.append(process)
            return process

        model = self.model()
        policy = NaiveLanguagePolicy(model, continuation=True)

        def infer() -> None:
            try:
                results.put(policy.decide(self.request, self.context))
            except BaseException as error:
                results.put(error)

        with patch("poker.agent.naive_language.subprocess.Popen", side_effect=launch):
            thread = Thread(target=infer, daemon=True)
            thread.start()
            try:
                self.assertTrue(entered.wait(1))
                self.control.cancel()
            finally:
                release.set()
            result = results.get(timeout=2)
            thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(result, DecisionCancelled)
        self.assertEqual(processes[0].kills, 1)
        self.assertIsNotNone(processes[0].returncode)
        self.assertEqual((model.calls, self.tools.calls), (1, 0))
        self.assertEqual(self.evidence()[0]["status"], "failed")
        # The cancelled invocation's control must not poison later direct calls.
        next_provider = FakePopen(Scenario(answer=CHECK))
        with patch("poker.agent.naive_language.subprocess.Popen", next_provider):
            self.assertIsNotNone(model.complete(self.model_request).decision)
        self.assertEqual(model.calls, 2)

    def test_external_cancel_reaps_a_real_local_python_subprocess(self) -> None:
        """Exercise actual pipes/process termination without starting a model."""
        real_popen = subprocess.Popen
        processes: list[subprocess.Popen[str]] = []
        results: Queue[Decision | BaseException] = Queue()
        ready = self.root / "local-child-ready.txt"
        # The Windows venv redirector can create another interpreter process;
        # use its base executable so Popen owns the actual sleeper directly.
        executable = str(getattr(sys, "_base_executable", sys.executable))
        child_code = (
            "import pathlib,sys,time; sys.stdin.read(); "
            "pathlib.Path(sys.argv[1]).write_text('ready', encoding='utf-8'); "
            "time.sleep(30)"
        )

        def launch(command: list[str], **options: object) -> subprocess.Popen[str]:
            self.assertEqual(command[0], "fixture-codex.exe")
            process = real_popen(
                [executable, "-u", "-c", child_code, str(ready)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", cwd=cast(Path, options["cwd"]),
                creationflags=cast(int, options["creationflags"]),
            )
            processes.append(process)
            return process

        model = self.model()
        policy = NaiveLanguagePolicy(model, continuation=True)

        def infer() -> None:
            try:
                results.put(policy.decide(self.request, self.context))
            except BaseException as error:
                results.put(error)

        with patch("poker.agent.naive_language.subprocess.Popen", side_effect=launch):
            thread = Thread(target=infer, daemon=True)
            thread.start()
            try:
                deadline = monotonic() + 3
                while not ready.exists() and monotonic() < deadline:
                    Event().wait(0.01)
                self.assertTrue(ready.exists(), "Local Python child must actually read stdin before cancellation")
                self.assertEqual(ready.read_text(encoding="utf-8"), "ready")
                self.assertEqual(len(processes), 1)
                self.assertIsNone(processes[0].poll())
                self.control.cancel()
                result = results.get(timeout=5)
                thread.join(1)
                self.assertFalse(thread.is_alive())
                self.assertIsInstance(result, DecisionCancelled)
                self.assertIsNotNone(processes[0].poll(), "The actual owned child must be gone")
                self.assertEqual((len(processes), model.calls, self.tools.calls), (1, 1, 0))
                self.assertEqual(self.evidence()[0]["child_pid"], processes[0].pid)
                self.assertEqual(self.evidence()[0]["status"], "failed")
            finally:
                self.control.cancel()
                thread.join(2)
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=5)
                thread.join(1)

    def test_cancelled_control_scope_never_launches_or_spends_a_call(self) -> None:
        model = self.model()
        self.control.cancel()
        with patch("poker.agent.naive_language.subprocess.Popen", side_effect=AssertionError("must not launch")):
            with model.decision_control(self.control), self.assertRaises(DecisionCancelled):
                model.complete(self.model_request)
        self.assertEqual(model.calls, 0)
        self.assertEqual(self.evidence(), [])

    def test_pid_evidence_write_failure_still_kills_and_reaps_owned_process(self) -> None:
        provider = FakePopen(Scenario())
        model = self.model()
        write = model._write_evidence
        writes = 0

        def fail_pid_write(call_id: str, evidence: JsonObject) -> None:
            nonlocal writes
            writes += 1
            if writes == 2:
                raise OSError("simulated receipt write failure")
            write(call_id, evidence)

        with patch("poker.agent.naive_language.subprocess.Popen", provider), \
                patch.object(model, "_write_evidence", side_effect=fail_pid_write):
            with self.assertRaises(PolicyError):
                model.complete(self.model_request)
        process = provider.processes[0]
        self.assertEqual(process.kills, 1)
        self.assertIsNotNone(process.returncode)
        self.assertEqual(process.communications, [(None, 5)])
        self.assertEqual(model.calls, 1)
        self.assertEqual(self.evidence()[0]["status"], "failed")

    def test_cleanup_pipe_error_uses_wait_to_reap_terminated_child(self) -> None:
        processes: list[FakeProcess] = []

        class BrokenCleanupProcess(FakeProcess):
            def communicate(self, input: str | None = None, timeout: float | None = None) -> tuple[str, str]:
                if self.kills:
                    self.communications.append((input, timeout))
                    raise UnicodeError("unreadable cleanup output")
                return super().communicate(input, timeout)

        def launch(command: list[str], **options: object) -> FakeProcess:
            process = BrokenCleanupProcess(command, options, Scenario(first_error=OSError("pipe broke")), 8766)
            processes.append(process)
            return process

        with patch("poker.agent.naive_language.subprocess.Popen", side_effect=launch):
            with self.assertRaises(PolicyError):
                self.model().complete(self.model_request)
        self.assertEqual((processes[0].kills, processes[0].waits), (1, 1))
        self.assertIsNotNone(processes[0].returncode)

    def test_unexpected_communicate_error_also_kills_and_reaps_process(self) -> None:
        for error in (OSError("pipe broke"), UnicodeError("provider encoding broke"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                provider = FakePopen(Scenario(first_error=error))
                with patch("poker.agent.naive_language.subprocess.Popen", provider):
                    with self.assertRaises((PolicyError, OSError, UnicodeError, KeyboardInterrupt)):
                        self.model().complete(self.model_request)
                process = provider.processes[0]
                self.assertEqual(process.kills, 1)
                self.assertTrue(len(process.communications) >= 2 or process.waits >= 1)
                self.assertIsNotNone(process.returncode)

    def test_stdin_contains_exact_request_and_only_player_view_game_information(self) -> None:
        provider = FakePopen(Scenario(answer=CHECK))
        with patch("poker.agent.naive_language.subprocess.Popen", provider):
            self.model().complete(self.model_request)
        process = provider.processes[0]
        supplied, timeout = process.communications[0]
        self.assertIsNotNone(supplied)
        assert supplied is not None
        payload = json.loads(supplied.split("\n", 1)[1])
        self.assertEqual(payload, json.loads(json.dumps(asdict(self.model_request))))
        observation = payload["observation"]
        self.assertEqual(observation["me"]["hole_cards"], list(self.observation.view.me.hole_cards))
        self.assertTrue(all("hole_cards" not in player for player in observation["players"]))
        for forbidden in ("deck", "server_port", "sqlite", "database", "server_snapshot", "all_hole_cards"):
            self.assertNotIn(forbidden, observation)
        self.assertEqual(process.command[-1], "-")
        self.assertNotIn(SECRET, " ".join(process.command))
        self.assertEqual(process.options["stdin"], subprocess.PIPE)
        self.assertEqual(process.options["stderr"], subprocess.PIPE)
        self.assertIn("--ignore-user-config", process.command)
        self.assertIn("--ephemeral", process.command)
        self.assertIn("read-only", process.command)
        self.assertIn("shell_tool", process.command)
        self.assertIsNotNone(timeout)
        assert timeout is not None
        self.assertGreater(timeout, 0)
        self.assertLessEqual(timeout, self.model_request.timeout_seconds)

    def test_evidence_uses_hashes_and_null_unknown_usage_without_raw_private_text(self) -> None:
        diagnostic: JsonObject = {"type": "item.completed", "item": {"type": "error", "message": SECRET}}
        provider = FakePopen(Scenario(answer=CHECK, events=(diagnostic, COMPLETED)))
        with patch("poker.agent.naive_language.subprocess.Popen", provider):
            self.model().complete(self.model_request)
        receipt = self.evidence()[0]
        self.assertEqual(receipt["backend"], "codex")
        self.assertTrue(receipt["live"])
        for field in ("input_tokens", "output_tokens", "cost", "cost_source"):
            self.assertIsNone(receipt[field])
        for field in ("input_sha256", "instructions_sha256", "stream_sha256"):
            self.assertEqual(len(cast(str, receipt[field])), 64)
        text = "\n".join(path.read_text(encoding="utf-8") for path in (self.root / "evidence").glob("*.json"))
        self.assertNotIn(SECRET, text)
        for field in ("prompt", "instructions", "stderr", "stdout", "observation"):
            self.assertNotIn(field, receipt)
        self.assertNotIn(SECRET, self.log_path.read_text(encoding="utf-8"))

    def test_reported_usage_is_preserved_but_unavailable_cost_stays_null(self) -> None:
        completion: JsonObject = {"type": "turn.completed", "usage": {"input_tokens": 23, "output_tokens": 7}}
        provider = FakePopen(Scenario(answer=CHECK, events=(completion,)))
        with patch("poker.agent.naive_language.subprocess.Popen", provider):
            self.model().complete(self.model_request)
        receipt = self.evidence()[0]
        self.assertEqual((receipt["input_tokens"], receipt["output_tokens"]), (23, 7))
        self.assertIsNone(receipt["cost"])
        completed = [row for row in self.log_rows() if row["event"] == "api.call.completed"]
        data = cast(JsonObject, completed[0]["data"])
        self.assertEqual((data["input_tokens"], data["output_tokens"]), (23, 7))
        self.assertIsNone(data["cost"])

    def test_invalid_model_answer_cannot_leak_raw_answer_into_api_exception_log(self) -> None:
        provider = FakePopen(Scenario(answer={**CHECK, "prompt_echo": SECRET}))
        with patch("poker.agent.naive_language.subprocess.Popen", provider):
            with self.assertRaises((PolicyError, ValidationError, ValueError)):
                self.model().complete(self.model_request)
        self.assertNotIn(SECRET, self.log_path.read_text(encoding="utf-8"))
        receipt = self.evidence()[0]
        self.assertFalse(receipt["success"])
        self.assertNotIn("answer", receipt)

    def test_live_adapter_prompt_policy_performs_two_real_protocol_rounds(self) -> None:
        provider = FakePopen(Scenario(answer=TOOL), Scenario(answer=CHECK))
        model = self.model()
        policy = NaiveLanguagePolicy(model, continuation=True)
        with patch("poker.agent.naive_language.subprocess.Popen", provider):
            decision = policy.decide(self.request, self.context)
        self.assertEqual(decision.choice, PlayerAction(ActionKind.CHECK))
        self.assertEqual((model.calls, self.tools.calls), (2, 1))
        first_prompt = provider.processes[0].communications[0][0]
        second_prompt = provider.processes[1].communications[0][0]
        assert first_prompt is not None and second_prompt is not None
        first = json.loads(first_prompt.split("\n", 1)[1])
        second = json.loads(second_prompt.split("\n", 1)[1])
        self.assertEqual(first["exchanges"], [])
        self.assertEqual(len(second["exchanges"]), 1)
        exchange = second["exchanges"][0]
        self.assertEqual(exchange["call"]["name"], "legal_actions")
        self.assertEqual(exchange["result"]["value"][0]["kind"], "check")
        self.assertIsNone(exchange["result"]["error"])
        self.assertIn(".codex", policy.identity.name)
        logs = [cast(JsonObject, row["data"]) for row in self.log_rows() if row["event"] == "naive.policy.decided"]
        self.assertEqual(logs[-1]["backend"], "codex")
        self.assertEqual(logs[-1]["live_model_calls"], 2)

    def test_stub_backend_is_explicit_and_never_claims_live_model_calls(self) -> None:
        with patch("poker.agent.naive_language.subprocess.Popen", side_effect=AssertionError("No paid calls allowed")):
            policy = natural_language({"language_backend": "stub", "verification_continuation": True})
            decision = policy.decide(self.request, self.context)
        self.assertEqual(decision.choice, PlayerAction(ActionKind.CHECK))
        self.assertIsInstance(policy, NaiveLanguagePolicy)
        assert isinstance(policy, NaiveLanguagePolicy)
        self.assertIsInstance(policy.backend, OfflineLanguageFixture)
        self.assertIn("offline_fixture", policy.identity.name)
        self.assertEqual((policy.backend.calls, self.tools.calls), (2, 1))
        logs = [cast(JsonObject, row["data"]) for row in self.log_rows() if row["event"] == "naive.policy.decided"]
        self.assertEqual(logs[-1]["backend"], "stub")
        self.assertEqual(logs[-1]["live_model_calls"], 0)
        self.assertEqual(logs[-1]["rounds"], 2)

    def test_backend_selection_is_explicit_and_missing_codex_does_not_fall_back(self) -> None:
        with patch("poker.agent.naive_language._executable", side_effect=ValueError("unavailable")):
            with self.assertRaisesRegex(ValueError, "unavailable"):
                natural_language({"language_backend": "codex", "language_model": "fixture-model"})
        for config in (
            {"language_backend": "automatic"}, {"verification_continuation": "true"},
            {"language_backend": "codex", "language_max_calls": True},
            {"language_backend": "codex", "language_model": 10},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                natural_language(cast(JsonObject, config))
