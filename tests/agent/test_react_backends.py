"""Provider contract tests: local HTTP and local Python, never live inference."""

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from queue import Queue
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import monotonic
from typing import cast
import unittest
from unittest.mock import patch

from poker.agent.context import DecisionControl
from poker.agent.models import DecisionCancelled, DecisionDeadlineExceeded
from poker.agent.react.backend import BackendError, BackendRequest, BackendResponse, BackendUsage
from poker.agent.react.backends import ChatCompletionsBackend, CodexBackend
from poker.agent.react.analysis_tools import EquityTool, LookupTool
from poker.agent.react.loop import response_schema
from poker.agent.react.schema import validate_strict_object_schemas
from poker.agent.tools import standard_tools
from shared_logging import JsonObject, LoggingConfig, configure_logging, shutdown_logging


SECRET = "PRIVATE-input-output-key-diagnostic"
SCHEMA: JsonObject = {"type": "object", "properties": {"answer": {"type": "integer"}},
                      "additionalProperties": False, "required": ["answer"]}
OUTPUT: JsonObject = {"answer": 42}
REQUEST = BackendRequest(SECRET, {"observation": {"only_player_view": SECRET}}, SCHEMA, 123)
COMPLETE: JsonObject = {"type": "turn.completed", "usage": {}}


def control(seconds: float = 5) -> DecisionControl:
    return DecisionControl(monotonic() + seconds)


def completion(content: str = '{"answer":42}', *, usage: JsonObject | None = None,
               reason: str = "stop", extra: JsonObject | None = None) -> JsonObject:
    return {"id": "provider-request-1", "choices": [{"finish_reason": reason,
            "message": {"role": "assistant", "content": content, **(extra or {})}}], "usage": usage}


@dataclass
class HttpScenario:
    payload: JsonObject = field(default_factory=completion)
    status: int = 200
    wait_at: str | None = None
    raw: bytes | None = None
    requests: list[tuple[str, JsonObject, str | None]] = field(default_factory=list)
    started: Event = field(default_factory=Event)
    disconnected: Event = field(default_factory=Event)


@contextmanager
def local_http(scenario: HttpScenario) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_POST(self) -> None:
            payload = cast(JsonObject, json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            scenario.requests.append((self.path, payload, self.headers.get("Authorization")))
            raw = scenario.raw if scenario.raw is not None else json.dumps(scenario.payload).encode()
            if scenario.wait_at != "headers":
                self.send_response(scenario.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Location", f"http://127.0.0.1:{cast(ThreadingHTTPServer, self.server).server_port}/redirected")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
            scenario.started.set()
            if scenario.wait_at:
                self.connection.settimeout(4)
                try:
                    if self.connection.recv(1) == b"":
                        scenario.disconnected.set()
                except (OSError, TimeoutError):
                    pass
                return
            try:
                self.wfile.write(raw)
            except OSError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@dataclass
class ProcessScenario:
    answer: object = field(default_factory=lambda: dict(OUTPUT))
    events: tuple[JsonObject, ...] = (COMPLETE,)
    error: BaseException | None = None
    exit_code: int = 0
    timeout_once: bool = False


class FakeProcess:
    def __init__(self, command: list[str], kwargs: dict[str, object], scenario: ProcessScenario) -> None:
        self.command = command
        self.kwargs = kwargs
        self.scenario = scenario
        self.returncode: int | None = None
        self.inputs: list[str | None] = []
        self.kills = 0
        self.waits = 0
        self.pid = 9191
        self.schema = json.loads(Path(command[command.index("--output-schema") + 1]).read_text(encoding="utf-8"))

    def communicate(self, input: str | None = None, timeout: float | None = None) -> tuple[str, str]:
        self.inputs.append(input)
        if not self.kills and len(self.inputs) == 1:
            if self.scenario.error is not None:
                raise self.scenario.error
            if self.scenario.timeout_once:
                raise subprocess.TimeoutExpired("fixture", timeout or 0)
        self.returncode = -9 if self.kills else self.scenario.exit_code
        if self.scenario.answer is not None:
            path = Path(self.command[self.command.index("--output-last-message") + 1])
            path.write_text(json.dumps(self.scenario.answer), encoding="utf-8")
        return "\n".join(json.dumps(row) for row in self.scenario.events), SECRET

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.kills += 1

    def wait(self, timeout: float | None = None) -> int:
        self.waits += 1
        self.returncode = -9
        return self.returncode


class FakePopen:
    def __init__(self, scenario: ProcessScenario | None = None) -> None:
        self.scenario = scenario or ProcessScenario()
        self.processes: list[FakeProcess] = []

    def __call__(self, command: list[str], **kwargs: object) -> FakeProcess:
        if self.processes:
            raise AssertionError("Unexpected second inference process")
        process = FakeProcess(command, kwargs, self.scenario)
        self.processes.append(process)
        return process


class ReactBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.log_path = configure_logging(LoggingConfig(self.root / "logs", "react-backend-test", "test"))
        # Test-only variable: no real credential name is read or modified.
        self.environment = patch.dict(os.environ, {"REACT_TEST_FAKE_KEY": SECRET})
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()
        shutdown_logging()
        self.temp.cleanup()

    def http(self, url: str) -> ChatCompletionsBackend:
        return ChatCompletionsBackend(model="fixture-model", base_url=url, api_key_env="REACT_TEST_FAKE_KEY")

    def codex(self) -> CodexBackend:
        return CodexBackend(model="fixture-model", executable=Path("fixture-codex.exe"))

    def rows(self) -> list[JsonObject]:
        return [cast(JsonObject, json.loads(row)) for row in self.log_path.read_text(encoding="utf-8").splitlines()]

    def test_contract_unknown_usage_and_explicit_models(self) -> None:
        self.assertEqual(asdict(BackendUsage()), {"input_tokens": None, "output_tokens": None,
                                               "cached_input_tokens": None, "cost": None})
        self.assertEqual(BackendResponse(OUTPUT).usage, BackendUsage())
        for bad in (-1, True, 0):
            with self.subTest(tokens=bad), self.assertRaises(ValueError):
                replace(REQUEST, max_output_tokens=bad)
        with self.assertRaises(ValueError):
            BackendUsage(input_tokens=True)
        with self.assertRaises(ValueError):
            BackendUsage(cost=float("nan"))
        for name in ("", " ", "secret\ninvalid"):
            with self.subTest(model=name), self.assertRaises(ValueError):
                CodexBackend(model=name, executable=Path("fake"))
            with self.assertRaises(ValueError):
                ChatCompletionsBackend(model=name)
        self.assertFalse(self.codex().supports_output_token_limit)
        self.assertTrue(self.http("http://localhost:12345/v1").supports_output_token_limit)

    def test_codex_generic_schema_explicit_model_stdin_and_tool_restrictions(self) -> None:
        fake = FakePopen(ProcessScenario(events=({"type": "thread.started", "thread_id": "thread-1"},
                                               {"type": "item.updated", "item": {"type": "reasoning", "text": SECRET}},
                                               {"type": "item.completed", "item": {"type": "error", "message": SECRET}},
                                               {"type": "turn.completed", "usage": {"input_tokens": 11, "output_tokens": 7,
                                                                                       "cached_input_tokens": 6}})))
        with patch("poker.agent.react.backends.subprocess.Popen", fake):
            response = self.codex().generate(REQUEST, control())
        self.assertEqual(response.output, OUTPUT)
        self.assertEqual(response.usage, BackendUsage(11, 7, 6))
        self.assertEqual(response.request_id, "thread-1")
        process = fake.processes[0]
        command = process.command
        self.assertEqual(command[command.index("--model") + 1], "fixture-model")
        self.assertEqual(process.schema, SCHEMA)
        self.assertIn("--ignore-user-config", command)
        self.assertIn("--ephemeral", command)
        for feature in ("shell_tool", "unified_exec", "code_mode_host", "apps", "plugins", "hooks",
                        "memories", "multi_agent", "browser_use", "computer_use", "image_generation",
                        "skill_search", "workspace_dependencies"):
            self.assertIn(["--disable", feature], [command[index:index + 2] for index in range(len(command) - 1)])
        self.assertNotIn(SECRET, " ".join(command))
        self.assertEqual(json.loads(cast(str, process.inputs[0]).split("\n", 1)[1]), asdict(REQUEST))
        self.assertFalse(Path(cast(Path, process.kwargs["cwd"])).exists())
        self.assertNotIn(SECRET, self.log_path.read_text(encoding="utf-8"))
        row = self.rows()[-1]
        self.assertEqual(row["event"], "api.call.completed")
        self.assertIsNone(cast(JsonObject, row["data"])["cost"])

    def test_codex_timeout_slice_does_not_retry_or_resend_input(self) -> None:
        fake = FakePopen(ProcessScenario(timeout_once=True))
        with patch("poker.agent.react.backends.subprocess.Popen", fake):
            self.codex().generate(REQUEST, control())
        self.assertEqual(len(fake.processes), 1)
        self.assertEqual(len(fake.processes[0].inputs), 2)
        self.assertIsNone(fake.processes[0].inputs[1])
        self.assertEqual(fake.processes[0].kills, 0)

    def test_codex_invalid_output_fails_without_raw_payload_or_fallback(self) -> None:
        scenarios = [ProcessScenario(answer={"answer": SECRET}), ProcessScenario(answer=[]),
                     ProcessScenario(answer={"answer": 1, "secret": SECRET}),
                     ProcessScenario(exit_code=3), ProcessScenario(events=()),
                     ProcessScenario(events=({"type": "turn.failed", "error": SECRET}, COMPLETE)),
                     ProcessScenario(events=(COMPLETE, COMPLETE)), ProcessScenario(answer=None)]
        for scenario in scenarios:
            fake = FakePopen(scenario)
            with self.subTest(scenario=scenario), patch("poker.agent.react.backends.subprocess.Popen", fake):
                with self.assertRaises(BackendError) as raised:
                    self.codex().generate(REQUEST, control())
                self.assertNotIn(SECRET, str(raised.exception))
            self.assertEqual(len(fake.processes), 1)
        self.assertNotIn(SECRET, self.log_path.read_text(encoding="utf-8"))

    def test_codex_every_environment_item_phase_is_rejected(self) -> None:
        for phase in ("started", "updated", "completed"):
            for kind in ("command_execution", "mcp_tool_call", "web_search", "file_change", "unknown"):
                fake = FakePopen(ProcessScenario(events=({"type": "item." + phase,
                                                         "item": {"type": kind, "private": SECRET}}, COMPLETE)))
                with self.subTest(phase=phase, kind=kind), patch("poker.agent.react.backends.subprocess.Popen", fake):
                    with self.assertRaisesRegex(BackendError, "forbidden"):
                        self.codex().generate(REQUEST, control())
        self.assertNotIn(SECRET, self.log_path.read_text(encoding="utf-8"))

    def test_codex_communication_failure_kills_and_reaps_without_echo(self) -> None:
        for error in (OSError(SECRET), UnicodeError(SECRET), KeyboardInterrupt(SECRET)):
            fake = FakePopen(ProcessScenario(error=error))
            with self.subTest(error=type(error)), patch("poker.agent.react.backends.subprocess.Popen", fake):
                with self.assertRaises(KeyboardInterrupt if isinstance(error, KeyboardInterrupt) else BackendError):
                    self.codex().generate(REQUEST, control())
            self.assertEqual(fake.processes[0].kills, 1)
            self.assertEqual(fake.processes[0].returncode, -9)
        self.assertNotIn(SECRET, self.log_path.read_text(encoding="utf-8"))

    def test_precancel_and_closed_backend_do_not_launch(self) -> None:
        cancelled = control()
        cancelled.cancel()
        backend = self.codex()
        with patch("poker.agent.react.backends.subprocess.Popen") as popen:
            with self.assertRaises(DecisionCancelled):
                backend.generate(REQUEST, cancelled)
            backend.close()
            with self.assertRaisesRegex(BackendError, "closed"):
                backend.generate(REQUEST, control())
            popen.assert_not_called()

    def test_completed_response_usage_survives_cancellation_before_delivery(self) -> None:
        for stop in ("cancel", "deadline"):
            with self.subTest(stop=stop):
                clock = [0.0]
                deadline = DecisionControl(10, clock=lambda: clock[0])
                backend = self.codex()
                before = len(self.rows())

                def complete(request: BackendRequest, control: DecisionControl) -> BackendResponse:
                    if stop == "cancel":
                        control.cancel()
                    else:
                        clock[0] = 11
                    return BackendResponse(OUTPUT, BackendUsage(17, 9, 4), "completed-request", "stop")

                with patch.object(backend, "_generate", side_effect=complete) as inference:
                    with self.assertRaises(DecisionCancelled if stop == "cancel" else DecisionDeadlineExceeded):
                        backend.generate(REQUEST, deadline)
                    inference.assert_called_once()
                rows = self.rows()[before:]
                self.assertEqual([row["event"] for row in rows], [
                    "api.call.started", "api.call.completed", "api.response.discarded",
                ])
                completed = cast(JsonObject, rows[1]["data"])
                self.assertEqual(completed["input_tokens"], 17)
                self.assertEqual(completed["output_tokens"], 9)
                self.assertEqual(completed["cached_input_tokens"], 4)
                self.assertIsNone(completed["cost"])
                discarded = cast(JsonObject, rows[2]["data"])
                self.assertEqual(discarded["error_type"], "DecisionCancelled" if stop == "cancel" else "DecisionDeadlineExceeded")
                self.assertNotIn(SECRET, self.log_path.read_text(encoding="utf-8"))

    def test_codex_local_real_sleep_child_cancel_deadline_and_close_are_reaped(self) -> None:
        # The base executable avoids the Windows virtualenv redirector spawning
        # another child and confusing which concrete process we own and reap.
        real_popen = subprocess.Popen
        executable = getattr(sys, "_base_executable", sys.executable)
        for stop in ("cancel", "deadline", "close"):
            with self.subTest(stop=stop):
                ready = self.root / (stop + ".ready")
                processes: list[subprocess.Popen[str]] = []
                outcome: Queue[BackendResponse | BaseException] = Queue()
                deadline = control(0.5 if stop == "deadline" else 5)
                backend = self.codex()

                def launch(command: list[str], **kwargs: object) -> subprocess.Popen[str]:
                    script = "import pathlib,sys,time;sys.stdin.read();pathlib.Path(sys.argv[1]).write_text('ready');time.sleep(30)"
                    child = real_popen([executable, "-c", script, str(ready)], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    processes.append(child)
                    return child

                def invoke() -> None:
                    try:
                        outcome.put(backend.generate(REQUEST, deadline))
                    except BaseException as error:
                        outcome.put(error)

                thread = Thread(target=invoke, daemon=True)
                try:
                    with patch("poker.agent.react.backends.subprocess.Popen", side_effect=launch):
                        thread.start()
                        until = monotonic() + 3
                        while not ready.exists() and monotonic() < until and thread.is_alive():
                            Event().wait(0.01)
                        self.assertTrue(ready.exists())
                        if stop == "close":
                            backend.close()
                        elif stop == "cancel":
                            deadline.cancel()
                        thread.join(timeout=3)
                    self.assertFalse(thread.is_alive())
                    self.assertEqual(len(processes), 1)
                    self.assertIsNotNone(processes[0].poll())
                    self.assertIsInstance(outcome.get_nowait(), DecisionDeadlineExceeded if stop == "deadline" else DecisionCancelled)
                finally:
                    deadline.cancel()
                    for process in processes:
                        if process.poll() is None:
                            process.kill()
                        process.communicate(timeout=3)
                    thread.join(timeout=3)

    def test_http_exact_post_schema_input_usage_and_no_native_tools(self) -> None:
        scenario = HttpScenario(payload=completion(usage={"prompt_tokens": 91, "completion_tokens": 8,
                                                        "prompt_tokens_details": {"cached_tokens": 32}}))
        with local_http(scenario) as url:
            response = self.http(url).generate(REQUEST, control())
        self.assertEqual(response, BackendResponse(OUTPUT, BackendUsage(91, 8, 32), "provider-request-1", "stop"))
        self.assertEqual(len(scenario.requests), 1)
        path, body, authorization = scenario.requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(authorization, "Bearer " + SECRET)
        self.assertEqual(body["model"], "fixture-model")
        self.assertEqual(body["max_completion_tokens"], 123)
        self.assertEqual(body["response_format"], {"type": "json_schema", "json_schema": {
            "name": "agent_output", "strict": True, "schema": SCHEMA}})
        messages = cast(list[JsonObject], body["messages"])
        self.assertEqual(messages[0], {"role": "system", "content": SECRET})
        self.assertEqual(json.loads(cast(str, messages[1]["content"])), REQUEST.input)
        self.assertNotIn("tools", body)
        self.assertNotIn("functions", body)
        self.assertNotIn(SECRET, self.log_path.read_text(encoding="utf-8"))

    def test_http_unknown_usage_is_null_not_zero_and_invalid_counts_are_unknown(self) -> None:
        examples: tuple[JsonObject | None, ...] = (None, {}, {"prompt_tokens": True, "completion_tokens": -8,
                                                            "prompt_tokens_details": {"cached_tokens": "many"}})
        for usage in examples:
            scenario = HttpScenario(payload=completion(usage=usage))
            with self.subTest(usage=usage), local_http(scenario) as url:
                response = self.http(url).generate(REQUEST, control())
            self.assertEqual(response.usage, BackendUsage())

    def test_http_failures_never_retry_or_follow_redirect_or_log_body(self) -> None:
        for status in (302, 401, 429, 500):
            scenario = HttpScenario(status=status, payload={"private_error": SECRET})
            with self.subTest(status=status), local_http(scenario) as url:
                with self.assertRaisesRegex(BackendError, str(status)):
                    self.http(url).generate(REQUEST, control())
            self.assertEqual(len(scenario.requests), 1)
        self.assertNotIn(SECRET, self.log_path.read_text(encoding="utf-8"))

    def test_http_invalid_and_incomplete_outputs_do_not_return_decisions(self) -> None:
        payloads: list[JsonObject] = [completion("not-json " + SECRET), completion(json.dumps({"answer": SECRET})),
                    completion("[]"), completion(reason="length"), completion(reason="content_filter"),
                    completion(extra={"tool_calls": [{"function": {"name": "shell", "arguments": SECRET}}]}),
                    completion(extra={"function_call": {"name": "shell"}}), completion(extra={"refusal": SECRET}),
                    {"choices": []}, {"choices": [1]}, {"choices": [{"finish_reason": "stop", "message": {"content": []}}]}]
        for payload in payloads:
            scenario = HttpScenario(payload=payload)
            with self.subTest(payload=payload), local_http(scenario) as url:
                with self.assertRaises(BackendError) as raised:
                    self.http(url).generate(REQUEST, control())
                self.assertNotIn(SECRET, str(raised.exception))
            self.assertEqual(len(scenario.requests), 1)
        self.assertNotIn(SECRET, self.log_path.read_text(encoding="utf-8"))

    def test_http_cancel_deadline_and_close_interrupt_headers_and_body(self) -> None:
        for at in ("headers", "body"):
            for stop in ("cancel", "deadline", "close"):
                scenario = HttpScenario(wait_at=at)
                outcomes: Queue[BackendResponse | BaseException] = Queue()
                deadline = control(0.3 if stop == "deadline" else 4)
                with self.subTest(at=at, stop=stop), local_http(scenario) as url:
                    backend = self.http(url)

                    def invoke() -> None:
                        try:
                            outcomes.put(backend.generate(REQUEST, deadline))
                        except BaseException as error:
                            outcomes.put(error)

                    thread = Thread(target=invoke, daemon=True)
                    thread.start()
                    self.assertTrue(scenario.started.wait(2))
                    if stop == "close":
                        backend.close()
                    elif stop == "cancel":
                        deadline.cancel()
                    thread.join(timeout=2)
                    self.assertFalse(thread.is_alive())
                    self.assertIsInstance(outcomes.get_nowait(), DecisionDeadlineExceeded if stop == "deadline" else DecisionCancelled)
                    self.assertTrue(scenario.disconnected.wait(1))
                self.assertEqual(len(scenario.requests), 1)

    def test_http_cancel_during_dns_cannot_send_a_late_request(self) -> None:
        started, release = Event(), Event()
        original_resolve = socket.getaddrinfo
        outcomes: Queue[BackendResponse | BaseException] = Queue()
        deadline = control()

        def slow_dns(host: str, port: int, *, type: int) -> object:
            started.set()
            release.wait(3)
            return original_resolve(host, port, type=type)

        scenario = HttpScenario()
        with local_http(scenario) as url:
            backend = self.http(url)

            def invoke() -> None:
                try:
                    outcomes.put(backend.generate(REQUEST, deadline))
                except BaseException as error:
                    outcomes.put(error)

            thread = Thread(target=invoke, daemon=True)
            with patch("poker.agent.react.backends.socket.getaddrinfo", side_effect=slow_dns):
                thread.start()
                self.assertTrue(started.wait(1))
                deadline.cancel()
                thread.join(timeout=1)
                self.assertFalse(thread.is_alive())
                release.set()
            self.assertIsInstance(outcomes.get_nowait(), DecisionCancelled)
            self.assertFalse(scenario.started.wait(0.1))
            self.assertEqual(scenario.requests, [])

    def test_local_schema_refs_only_and_key_validation_happen_before_connect(self) -> None:
        for schema in ({"$ref": "https://example.invalid/secret"}, {"type": "invalid"}):
            with self.subTest(schema=schema), patch("poker.agent.react.backends._connect") as connect:
                with self.assertRaises(BackendError):
                    self.http("http://localhost:3/v1").generate(replace(REQUEST, output_schema=cast(JsonObject, schema)), control())
                connect.assert_not_called()
        with patch("poker.agent.react.backends._connect") as connect:
            backend = ChatCompletionsBackend(model="fixture", base_url="http://localhost:3/v1",
                                             api_key_env="REACT_TEST_KEY_THAT_DOES_NOT_EXIST")
            with self.assertRaisesRegex(BackendError, "environment"):
                backend.generate(REQUEST, control())
            connect.assert_not_called()
        for url in ("http://example.com/v1", "https://secret@example.com/v1", "https://example.com/v1?key=secret",
                    "ftp://example.com/v1", "https://example.com/#secret"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.http(url)

    def test_strict_schema_preflight_rejects_optional_or_open_nested_objects_before_inference(self) -> None:
        optional: JsonObject = {"type": "object", "properties": {SECRET: {"type": "integer"}},
                                "additionalProperties": False, "required": []}
        open_object: JsonObject = {"type": "object", "properties": {}, "required": []}
        examples: tuple[JsonObject, ...] = (
            optional, open_object, {"type": "object", "properties": {}, "additionalProperties": True},
            {"type": "object", "properties": {"items": {"type": "array", "items": optional}},
             "additionalProperties": False, "required": ["items"]},
            {"type": "object", "properties": {"variant": {"anyOf": [{"type": "null"}, optional]}},
             "additionalProperties": False, "required": ["variant"]},
            {"type": "object", "properties": {"child": {"$ref": "#/$defs/child"}},
             "additionalProperties": False, "required": ["child"], "$defs": {"child": optional}},
        )
        for schema in examples:
            with self.subTest(schema=schema):
                request = replace(REQUEST, output_schema=schema)
                http_backend, codex_backend = self.http("http://localhost:3/v1"), self.codex()
                with patch("poker.agent.react.backends._connect") as connect, \
                        patch("poker.agent.react.backends.subprocess.Popen") as popen, \
                        patch("poker.agent.react.backends.os.environ.get", side_effect=AssertionError("Key read before preflight")) as key:
                    for backend in (http_backend, codex_backend):
                        with self.assertRaisesRegex(BackendError, "Strict structured output") as raised:
                            backend.generate(request, control())
                        self.assertNotIn(SECRET, str(raised.exception))
                    connect.assert_not_called()
                    popen.assert_not_called()
                    key.assert_not_called()
        self.assertNotIn(SECRET, self.log_path.read_text(encoding="utf-8"))

    def test_strict_schema_preflight_preserves_nullable_and_anyof_semantics(self) -> None:
        empty: JsonObject = {"type": "object", "properties": {}, "additionalProperties": False}
        named: JsonObject = {"type": "object", "properties": {"name": {"type": ["string", "null"]}},
                            "additionalProperties": False, "required": ["name"]}
        schema: JsonObject = {"type": "object", "properties": {
            "variant": {"anyOf": [empty, named]},
            # Annotation values must not be interpreted as child schemas.
            "text": {"type": "string", "examples": [{"type": "object", "properties": {"optional": {}}}]},
        }, "additionalProperties": False, "required": ["variant", "text"]}
        before = deepcopy(schema)
        validate_strict_object_schemas(schema)
        self.assertEqual(schema, before)
        validate_strict_object_schemas({"type": ["object", "null"], "properties": {}, "additionalProperties": False})

    def test_all_production_analysis_tools_generate_a_strict_compatible_loop_schema(self) -> None:
        specs = tuple(tool.spec for tool in (*standard_tools(), EquityTool(), LookupTool()))
        self.assertEqual({spec.name for spec in specs}, {"observation", "legal_actions", "public_history", "equity", "lookup"})
        for selected in ((), specs):
            schema = response_schema(selected)
            original = deepcopy(schema)
            validate_strict_object_schemas(schema)
            self.assertEqual(schema, original)


if __name__ == "__main__":
    unittest.main()
