from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.run_agent_overnight import (
    OvernightConfig, RoundPlan, RoundRecord, claim_output, count_attempts,
    driver_command, parse_utc, plan_round, run_supervisor, source_hashes, stop_owned_process, summarize,
)


_CHILD = r'''
import json, pathlib, sys, time
output = pathlib.Path(sys.argv[1])
plan = json.loads(sys.argv[2])
mode = sys.argv[3]
output.mkdir(parents=True)
print('fake child ready', flush=True)
if mode == 'sleep':
    time.sleep(60)
if mode == 'stop':
    (output.parent.parent / 'STOP').touch()
    time.sleep(60)
if plan['language_backend'] == 'codex':
    calls = output / 'model_calls'
    calls.mkdir()
    states = ('started', 'failed', 'completed') if mode == 'failed' else ('completed',)
    for i, status in enumerate(states):
        (calls / (str(i) + '.json')).write_text(json.dumps({
            'status': status, 'success': status == 'completed', 'live': True,
            'input_tokens': 3 if status == 'completed' else None,
            'output_tokens': 2 if status == 'completed' else None}), encoding='utf-8')
if mode == 'no-receipt':
    sys.exit(0)
ok = mode != 'failed'
data = dict(plan, ok=ok, server_completed_hands=['h1', 'h2'], wire={'actions': 25},
            error=None if ok else 'fake provider failure', checks={'fake_acceptance': True},
            processes=[{'runtime_pid': 100 + i, 'exit_code': 0} for i in range(7)])
if mode == 'wrong-backend':
    data['language_backend'] = 'stub'
if mode == 'no-hands':
    data['server_completed_hands'] = []
if mode == 'bad-check':
    data['checks']['fake_acceptance'] = False
if mode == 'duplicate-pids':
    data['processes'][1]['runtime_pid'] = 100
if mode == 'unfinished-child':
    data['processes'][1]['exit_code'] = None
(output / ('receipt.json' if ok else 'failure.json')).write_text(json.dumps(data), encoding='utf-8')
sys.exit(0 if ok else 1)
'''


class OvernightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = OvernightConfig(self.root / "run", datetime.now(UTC) + timedelta(seconds=60),
                                      rounds=1, interval_seconds=0, max_live_calls=4, round_timeout=10,
                                      poll_seconds=0.02)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def child(plan: RoundPlan, output: Path, allowance: int, timeout: float, mode: str = "ok") -> list[str]:
        data = {"language_backend": plan.language_backend, "deck_mode": plan.deck_mode,
                "seed": plan.seed, "seat_rotation": plan.seat_rotation, "verification_continuation": True}
        return [sys.executable, "-u", "-c", _CHILD, str(output), json.dumps(data), mode]

    def test_round_plan_rotates_seats_seeds_decks_and_distinguishes_real_calls(self) -> None:
        config = replace(self.config, rounds=18, live_every=9)
        plans = [plan_round(index, config) for index in range(config.rounds)]
        self.assertEqual([p.index for p in plans if p.language_backend == "codex"], [0, 9])
        self.assertEqual([p.deck_mode for p in plans[:4]], ["fixed", "random", "fixed", "random"])
        self.assertEqual([p.seat_rotation for p in plans[:8]], [0, 1, 2, 3, 4, 5, 0, 1])
        self.assertEqual([p.seed for p in plans[:3]], [7, 8, 9])
        self.assertTrue(all(p.verification_continuation for p in plans))

    def test_attempt_count_includes_failed_started_and_unreadable_files(self) -> None:
        directory = self.root / "calls"
        directory.mkdir()
        for index, state in enumerate(("started", "failed", "completed")):
            (directory / f"{index}.json").write_text(json.dumps({
                "status": state, "success": state == "completed",
                "input_tokens": 12 if state == "completed" else None,
                "output_tokens": 4 if state == "completed" else None}), encoding="utf-8")
        (directory / "corrupt.json").write_text("{", encoding="utf-8")
        stats = count_attempts(directory)
        self.assertEqual((stats.attempts, stats.completed, stats.failed, stats.pending, stats.unreadable), (4, 1, 1, 2, 1))
        self.assertEqual((stats.known_input_tokens, stats.known_output_tokens, stats.usage_known_calls), (12, 4, 1))
        self.assertIsNone(stats.cost)

    def test_failed_round_spends_budget_and_future_live_round_is_skipped_not_stub(self) -> None:
        def command(plan: RoundPlan, output: Path, allowance: int, timeout: float) -> list[str]:
            if plan.index == 0:
                self.assertEqual(allowance, 3)
            return self.child(plan, output, allowance, timeout, "failed" if plan.index == 0 else "ok")

        summary = run_supervisor(replace(self.config, rounds=3, live_every=2, max_live_calls=3), command_builder=command)
        self.assertEqual(summary["overall_status"], "finished_with_failures")
        self.assertEqual((summary["started"], summary["completed"], summary["failed"], summary["skipped"]), (2, 1, 1, 1))
        self.assertEqual(summary["live_model_attempts"], 3)
        self.assertEqual(summary["remaining_live_calls"], 0)
        skipped = json.loads((self.config.output_dir / "round_002/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(skipped["status"], "skipped_budget")
        self.assertEqual(skipped["language_backend"], "codex")
        self.assertFalse((self.config.output_dir / "round_002/evidence").exists())

    def test_success_requires_receipt_hands_and_real_model_evidence(self) -> None:
        summary = run_supervisor(self.config, command_builder=self.child)
        self.assertEqual(summary["overall_status"], "completed")
        self.assertEqual(summary["live_rounds_completed"], 1)
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["live_model_attempts"], 1)
        status = json.loads((self.config.output_dir / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["owner_pid"], os.getpid())
        self.assertIsNone(status["current_round"])
        report = (self.config.output_dir / "REPORT.md").read_text(encoding="utf-8")
        self.assertIn("费用未知", report)
        self.assertIn("stub 只验证协议", report)
        self.assertIn("fake child ready", (self.config.output_dir / "round_000/stdout.log").read_text(encoding="utf-8"))

    def test_exit_zero_without_valid_evidence_is_not_completion(self) -> None:
        for mode in ("no-receipt", "wrong-backend", "no-hands", "bad-check", "duplicate-pids", "unfinished-child"):
            with self.subTest(mode=mode):
                config = replace(self.config, output_dir=self.root / mode)
                summary = run_supervisor(config, command_builder=lambda p, d, a, t: self.child(p, d, a, t, mode))
                self.assertEqual(summary["completed"], 0)
                self.assertEqual(summary["failed"], 1)
                self.assertEqual(summary["overall_status"], "finished_with_failures")

    def test_three_consecutive_failures_open_circuit_and_keep_failure_receipts(self) -> None:
        config = replace(self.config, rounds=5, max_live_calls=30, live_every=10)
        summary = run_supervisor(config, command_builder=lambda p, d, a, t: self.child(p, d, a, t, "failed"))
        self.assertEqual((summary["started"], summary["failed"], summary["skipped"]), (3, 3, 2))
        self.assertEqual(summary["overall_status"], "finished_with_failures")
        for index in range(3):
            self.assertTrue((config.output_dir / f"round_{index:03d}/manifest.json").is_file())
            self.assertTrue((config.output_dir / f"round_{index:03d}/evidence/failure.json").is_file())

    def test_repeat_launch_is_rejected_and_preserves_original_status(self) -> None:
        directory = self.config.output_dir
        directory.mkdir()
        (directory / "PLAN.md").write_text("night plan", encoding="utf-8")
        claim_output(directory, {"owner_pid": 123, "overall_status": "running"})
        before = (directory / "status.json").read_bytes()
        with self.assertRaises(FileExistsError):
            run_supervisor(self.config, command_builder=self.child)
        self.assertEqual((directory / "status.json").read_bytes(), before)

    def test_expired_deadline_and_existing_stop_do_not_start_a_child(self) -> None:
        expired = replace(self.config, until_utc=datetime.now(UTC) - timedelta(seconds=1), rounds=2)
        with patch("scripts.run_agent_overnight.subprocess.Popen") as popen:
            summary = run_supervisor(expired, command_builder=self.child)
        popen.assert_not_called()
        self.assertEqual((summary["overall_status"], summary["started"], summary["skipped"]), ("deadline", 0, 2))
        stop = self.root / "STOP"
        stop.touch()
        config = replace(self.config, output_dir=self.root / "stopped", stop_file=stop)
        with patch("scripts.run_agent_overnight.subprocess.Popen") as popen:
            summary = run_supervisor(config, command_builder=self.child)
        popen.assert_not_called()
        self.assertEqual(summary["overall_status"], "stopped")
        self.assertEqual(summary["started"], 0)

    def test_timeout_cleans_up_a_real_owned_sleeping_child(self) -> None:
        config = replace(self.config, round_timeout=0.3)
        summary = run_supervisor(config, command_builder=lambda p, d, a, t: self.child(p, d, a, t, "sleep"))
        self.assertEqual(summary["failed"], 1)
        row = json.loads((config.output_dir / "round_000/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(row["status"], "timed_out")
        self.assertTrue(row["cleanup_confirmed"])
        self.assertIsNotNone(row["exit_code"])
        self.assertIsNotNone(row["child_pid"])

    def test_stop_during_child_cleans_up_and_marks_remaining_rounds_skipped(self) -> None:
        config = replace(self.config, rounds=2)
        summary = run_supervisor(config, command_builder=lambda p, d, a, t: self.child(p, d, a, t, "stop"))
        self.assertEqual((summary["overall_status"], summary["started"], summary["skipped"]), ("stopped", 1, 1))
        self.assertEqual(summary["incomplete"], 1)
        row = json.loads((config.output_dir / "round_000/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(row["status"], "stopped")
        self.assertTrue(row["cleanup_confirmed"])

    def test_deadline_caps_running_child_and_preserves_incomplete_evidence(self) -> None:
        config = replace(self.config, rounds=2, until_utc=datetime.now(UTC) + timedelta(seconds=0.3))
        summary = run_supervisor(config, command_builder=lambda p, d, a, t: self.child(p, d, a, t, "sleep"))
        self.assertEqual((summary["overall_status"], summary["started"], summary["failed"], summary["skipped"]),
                         ("deadline", 1, 1, 1))
        row = json.loads((config.output_dir / "round_000/manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(row["cleanup_confirmed"])
        self.assertLessEqual(row["timeout_seconds"], 0.3)

    def test_live_round_allowance_is_capped_at_32_and_start_end_sources_are_hashed(self) -> None:
        seen: list[int] = []

        def command(plan: RoundPlan, output: Path, allowance: int, timeout: float) -> list[str]:
            seen.append(allowance)
            return self.child(plan, output, allowance, timeout)

        summary = run_supervisor(replace(self.config, max_live_calls=64), command_builder=command)
        self.assertEqual(summary["overall_status"], "completed")
        self.assertEqual(seen, [32])
        row = json.loads((self.config.output_dir / "round_000/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(row["source_sha256_start"], row["source_sha256_end"])
        self.assertEqual(set(row["source_sha256_start"]), {"scripts/run_agent_overnight.py", "scripts/verify_six_agents.py"})
        self.assertTrue(all(len(value) == 64 for value in row["source_sha256_start"].values()))

    def test_cleanup_does_not_signal_a_process_that_already_exited(self) -> None:
        process = subprocess.Popen([sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                                   start_new_session=os.name != "nt")
        process.wait(timeout=10)
        with patch("scripts.run_agent_overnight.subprocess.run") as taskkill:
            self.assertTrue(stop_owned_process(process))
        taskkill.assert_not_called()

    def test_midround_driver_source_change_invalidates_success_without_losing_model_budget(self) -> None:
        original = source_hashes()
        changed = {**original, "scripts/verify_six_agents.py": "different-driver-version"}
        with patch("scripts.run_agent_overnight.source_hashes", side_effect=(original, changed)):
            summary = run_supervisor(self.config, command_builder=self.child)
        self.assertEqual((summary["overall_status"], summary["completed"], summary["failed"]),
                         ("finished_with_failures", 0, 1))
        self.assertEqual(summary["live_model_attempts"], 1)
        self.assertEqual(summary["source_drift_rounds"], 1)
        row = json.loads((self.config.output_dir / "round_000/manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(row["source_changed"])
        self.assertEqual(row["changed_source_files"], ["scripts/verify_six_agents.py"])
        self.assertEqual((row["completed_hands"], row["actions"]), (2, 25))
        self.assertTrue((self.config.output_dir / "round_000/evidence/receipt.json").is_file())

    def test_source_drift_preserves_original_round_failure(self) -> None:
        original = source_hashes()
        changed = {**original, "scripts/run_agent_overnight.py": "replacement-supervisor"}
        with patch("scripts.run_agent_overnight.source_hashes", side_effect=(original, changed)):
            summary = run_supervisor(self.config, command_builder=lambda p, d, a, t: self.child(p, d, a, t, "failed"))
        self.assertEqual(summary["live_model_attempts"], 3)
        row = json.loads((self.config.output_dir / "round_000/manifest.json").read_text(encoding="utf-8"))
        self.assertIn("fake provider failure", row["error"])
        self.assertIn("Source drift", row["error"])
        self.assertEqual(row["status"], "failed")

    def test_source_replacement_between_rounds_does_not_relabel_the_old_supervisor(self) -> None:
        original = source_hashes()
        changed = {**original, "scripts/run_agent_overnight.py": "replacement-supervisor"}
        config = replace(self.config, rounds=3)
        with patch("scripts.run_agent_overnight.source_hashes", side_effect=(original, original, changed)):
            summary = run_supervisor(config, command_builder=self.child)
        self.assertEqual((summary["overall_status"], summary["started"], summary["completed"], summary["skipped"]),
                         ("finished_with_failures", 1, 1, 2))
        second = json.loads((self.config.output_dir / "round_001/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(second["status"], "skipped_source_drift")
        self.assertTrue(second["source_changed"])
        self.assertNotEqual(second["loaded_supervisor_sha256"], second["source_sha256_start"]["scripts/run_agent_overnight.py"])
        self.assertFalse((self.config.output_dir / "round_001/evidence").exists())

    def test_summary_does_not_count_running_or_skipped_rounds_as_completed(self) -> None:
        records = [RoundRecord(plan_round(i, self.config)) for i in range(4)]
        records[0].status, records[0].started_utc = "completed", "start"
        records[1].status, records[1].started_utc = "running", "start"
        records[2].status = "skipped_budget"
        summary = summarize(records, self.config, "running")
        self.assertEqual((summary["started"], summary["completed"], summary["skipped"], summary["incomplete"]), (2, 1, 1, 2))

    def test_command_passes_remaining_budget_and_bounded_driver_timeout(self) -> None:
        command = driver_command(plan_round(9, self.config), self.root / "evidence", 17, 90, self.config)
        self.assertEqual(command[command.index("--language-max-calls") + 1], "17")
        self.assertEqual(command[command.index("--timeout") + 1], "90")
        self.assertEqual(command[command.index("--decision-timeout") + 1], "45.0")
        self.assertEqual(command[command.index("--language-backend") + 1], "codex")
        self.assertEqual(command[command.index("--seat-rotation") + 1], "3")

    def test_until_requires_timezone_and_configuration_rejects_unbounded_values(self) -> None:
        self.assertEqual(parse_utc("2026-10-09T08:00:00+02:00"), datetime(2026, 10, 9, 6, tzinfo=UTC))
        with self.assertRaises(ValueError):
            parse_utc("2026-10-09T08:00:00")
        makers: tuple[Callable[[], OvernightConfig], ...] = (
            lambda: replace(self.config, rounds=0), lambda: replace(self.config, max_live_calls=-1),
            lambda: replace(self.config, live_every=0), lambda: replace(self.config, interval_seconds=float("inf")),
            lambda: replace(self.config, round_timeout=float("nan")),
        )
        for make in makers:
            with self.assertRaises(ValueError):
                make()
