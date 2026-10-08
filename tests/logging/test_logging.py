import json
from collections.abc import Callable
import logging
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from typing import cast
from unittest.mock import patch

from jsonschema import Draft202012Validator, FormatChecker
from shared_logging import LoggingConfig, configure_logging, context_scope, get_logger, process_logging, shutdown_logging
from shared_logging.observations import ApiCallFailure, ApiResult, ApiUsage, bot_decision, observe_api_call, run_logged_process


class LoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.config = LoggingConfig(self.directory, "test-run", "test", level="DEBUG")
        self.path = configure_logging(self.config)
        schema = json.loads(Path("src/shared_logging/event.schema.json").read_text(encoding="utf-8"))
        self.validator = Draft202012Validator(schema, format_checker=FormatChecker())

    def tearDown(self) -> None:
        shutdown_logging()
        self.temp.cleanup()

    def rows(self) -> list[dict[str, object]]:
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            self.validator.validate(row)
        return rows

    def test_business_and_third_party_share_schema_unicode_and_levels(self) -> None:
        get_logger("test").emit("INFO", "test.started", "中文")
        logging.getLogger("third.party").warning("foreign")
        rows = self.rows()
        self.assertEqual([r["event"] for r in rows], ["test.started", "third_party.log"])
        self.assertEqual([r["level"] for r in rows], ["INFO", "WARNING"])
        self.assertEqual(rows[0]["message"], "中文")

    def test_exception_is_single_json_line_and_sensitive_values_are_redacted(self) -> None:
        try:
            raise ValueError("api_key='TOP-SECRET' https://name:password@host/path?token=hidden")
        except ValueError:
            get_logger("test").exception("test.failed", "Authorization: Bearer SECRET-TOKEN", {
                "nested": {"api_key": "KEY", "cookie": "COOKIE"}, "deck": ["As"],
                "hole_cards": ["Kd"], "session_id": "FULL-HANDLE",
            })
        rows = self.rows()
        raw = self.path.read_text(encoding="utf-8")
        for secret in ("TOP-SECRET", "SECRET-TOKEN", "KEY", "COOKIE", "FULL-HANDLE", "name:password", "As", "Kd"):
            self.assertNotIn(secret, raw)
        self.assertIsNotNone(rows[0]["error"])
        self.assertEqual(len(raw.splitlines()), 1)

    def test_context_is_cleared_and_isolated_between_threads(self) -> None:
        def work(identifier: str) -> None:
            with context_scope(correlation_id=identifier):
                get_logger("test").emit("INFO", "test.context", identifier)
        threads = [Thread(target=work, args=(identifier,)) for identifier in ("a", "b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        get_logger("test").emit("INFO", "test.context", "after")
        rows = self.rows()
        self.assertEqual({json.dumps(r["context"], sort_keys=True) for r in rows[:2]},
                         {json.dumps({"correlation_id": "a"}), json.dumps({"correlation_id": "b"})})
        self.assertEqual(rows[-1]["context"], {})

    def test_exception_context_resets_and_url_session_handle_is_not_logged(self) -> None:
        with self.assertRaises(ValueError), context_scope(correlation_id="temporary"):
            raise ValueError("failed")
        logging.getLogger("uvicorn.access").info("GET /api/gui/sessions/OPAQUE-HANDLE?token=hidden")
        row = self.rows()[0]
        self.assertEqual(row["context"], {})
        self.assertNotIn("OPAQUE-HANDLE", str(row["message"]))
        self.assertNotIn("hidden", str(row["message"]))

    def test_plain_token_text_and_relative_query_values_are_redacted(self) -> None:
        get_logger("test").emit("INFO", "test.redaction", "token=TOPSECRET GET /path?custom=QUERYSECRET")
        raw = self.path.read_text(encoding="utf-8")
        self.assertNotIn("TOPSECRET", raw)
        self.assertNotIn("QUERYSECRET", raw)

    def test_global_level_filters_foreign_loggers_with_their_own_debug_level(self) -> None:
        configure_logging(LoggingConfig(self.directory, "info-test", "test", level="INFO"))
        foreign = logging.getLogger("foreign.level-test")
        previous_level = foreign.level
        foreign.setLevel(logging.DEBUG)
        try:
            foreign.debug("filtered")
            foreign.warning("retained")
            logging.getLogger("uvicorn.access").info("routine poll")
            rows = [json.loads(line) for file in (self.directory / "info-test").glob("*.jsonl")
                    for line in file.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["message"] for row in rows], ["retained"])
        finally:
            foreign.setLevel(previous_level)

    def test_unhandled_process_hook_records_exception_and_calls_original(self) -> None:
        with patch("shared_logging.runtime._original_sys_hook") as original:
            error = RuntimeError("unhandled")
            sys.excepthook(RuntimeError, error, None)
            original.assert_called_once_with(RuntimeError, error, None)
        row = self.rows()[0]
        self.assertEqual(row["event"], "process.failed")
        self.assertEqual(cast(dict[str, object], row["error"])["type"], "RuntimeError")

    def test_runtime_failure_is_logged_before_sink_is_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "runtime failed"), process_logging(self.config, component="test"):
            raise RuntimeError("runtime failed")
        events = [r["event"] for r in self.rows()]
        self.assertEqual(events, ["process.started", "process.failed", "process.stopped"])

    def test_unhandled_thread_hook_records_exception(self) -> None:
        def failed() -> None:
            raise RuntimeError("thread failure")
        with patch("shared_logging.runtime._original_thread_hook") as original:
            thread = Thread(target=failed)
            thread.start()
            thread.join()
            original.assert_called_once()
        self.assertEqual(self.rows()[0]["event"], "process.failed")

    def test_repeated_configuration_does_not_duplicate_handlers_and_rotation_is_valid(self) -> None:
        self.assertEqual(configure_logging(self.config), self.path)
        get_logger("test").emit("INFO", "test.once", "one")
        self.assertEqual(len(self.rows()), 1)
        config = LoggingConfig(self.directory, "rotate", "test", max_bytes=700, backup_count=4)
        path = configure_logging(config)
        for _ in range(5):
            get_logger("test").emit("INFO", "test.rotate", "record")
        files = list(path.parent.glob("*.jsonl*"))
        self.assertGreater(len(files), 1)
        for file in files:
            for line in file.read_text(encoding="utf-8").splitlines():
                self.validator.validate(json.loads(line))

    def test_api_success_records_actual_usage_and_unknown_cost_stays_null(self) -> None:
        value = observe_api_call("fixture", "model", lambda: ApiResult(7, ApiUsage(12, 4), 200, "stop"))
        self.assertEqual(value, 7)
        rows = self.rows()
        self.assertEqual([r["event"] for r in rows], ["api.call.started", "api.call.completed"])
        data = cast(dict[str, object], rows[-1]["data"])
        self.assertEqual(data, {"provider": "fixture", "model": "model",
                         "duration_ms": data["duration_ms"], "http_status": 200, "stop_reason": "stop",
                         "input_tokens": 12, "output_tokens": 4, "cost": None, "cost_source": None})

    def test_api_failure_records_known_http_status_and_usage_validation(self) -> None:
        def unavailable() -> ApiResult[int]:
            raise ApiCallFailure("unavailable", http_status=503)
        with self.assertRaises(ApiCallFailure):
            observe_api_call("fixture", "model", unavailable)
        data = cast(dict[str, object], self.rows()[-1]["data"])
        self.assertEqual(data["http_status"], 503)
        self.assertIsNone(data["cost"])
        builders: tuple[Callable[[], ApiUsage], ...] = (
            lambda: ApiUsage(input_tokens=-1), lambda: ApiUsage(input_tokens=True),
            lambda: ApiUsage(cost=0.1), lambda: ApiUsage(cost=float("nan"), cost_source="receipt"),
        )
        for make in builders:
            with self.assertRaises(ValueError):
                make()
        self.assertEqual(ApiUsage(cost=0, cost_source="provider receipt").cost, 0)

    def test_api_failure_and_bot_decision_are_logged_without_raw_response(self) -> None:
        def failed() -> ApiResult[int]:
            raise RuntimeError("bad provider")
        with self.assertRaises(RuntimeError), bot_decision(get_logger("bot"), player_id="p1", hand_id="h1"):
            observe_api_call("fixture", "model", failed)
        events = [r["event"] for r in self.rows()]
        self.assertEqual(events, ["bot.decision_requested", "api.call.started", "api.call.failed", "bot.failed"])
        contexts = [cast(dict[str, object], r["context"]) for r in self.rows()]
        self.assertEqual(len({c["correlation_id"] for c in contexts}), 1)
        self.assertTrue(all(c["player_id"] == "p1" and c["hand_id"] == "h1" for c in contexts))

    def test_shared_library_runs_in_independent_non_poker_child_and_same_run(self) -> None:
        code = ("from shared_logging import *; "
                "configure_logging(LoggingConfig.from_environment('bot')); "
                "get_logger('other.project').emit('INFO','bot.started','Standalone library'); "
                "shutdown_logging()")
        result = run_logged_process([sys.executable, "-X", "utf8", "-c", code])
        self.assertEqual(result.returncode, 0, result.stderr)
        files = list((self.directory / "test-run").glob("bot-*.jsonl"))
        self.assertEqual(len(files), 1)
        child = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
        self.validator.validate(child)
        self.assertEqual(child["run_id"], "test-run")
        self.assertNotEqual(child["pid"], self.rows()[0]["pid"])
