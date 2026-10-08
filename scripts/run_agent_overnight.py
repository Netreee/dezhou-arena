"""Run a finite, auditable sequence of independent six-agent experiments.

Live Codex rounds are explicitly distinguished from offline language fixtures.
The driver always runs as a child process and owns its Arena/Agent processes.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import monotonic, sleep
from typing import cast

from poker.agent.naive_language import DEFAULT_CODEX_MODEL


_ROOT = Path(__file__).resolve().parents[1]
_TERMINAL_FAILURES = {"failed", "timed_out"}
_SUPERVISOR_SOURCE = "scripts/run_agent_overnight.py"
# Capture the process's version once. Later reads must not relabel old loaded
# supervisor code with a replacement source file's hash between rounds.
_LOADED_SUPERVISOR_SHA256 = sha256(Path(__file__).read_bytes()).hexdigest()


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("until-utc must include a timezone offset or Z")
    return parsed.astimezone(UTC)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class OvernightConfig:
    output_dir: Path
    until_utc: datetime
    rounds: int = 18
    interval_seconds: float = 900.0
    max_live_calls: int = 64
    live_every: int = 9
    seed: int = 7
    hands: int = 2
    round_timeout: float = 900.0
    language_model: str | None = DEFAULT_CODEX_MODEL
    stop_file: Path | None = None
    poll_seconds: float = 5.0

    def __post_init__(self) -> None:
        for value in (self.rounds, self.live_every, self.hands):
            if type(value) is not int or value < 1:
                raise ValueError("rounds, live_every and hands must be positive integers")
        if type(self.max_live_calls) is not int or self.max_live_calls < 0:
            raise ValueError("max_live_calls must be a nonnegative integer")
        for timeout in (self.round_timeout, self.poll_seconds):
            if not isfinite(timeout) or timeout <= 0:
                raise ValueError("Timeout and polling intervals must be positive and finite")
        if not isfinite(self.interval_seconds) or self.interval_seconds < 0:
            raise ValueError("interval_seconds must be nonnegative and finite")
        if self.until_utc.tzinfo is None or self.until_utc.utcoffset() is None:
            raise ValueError("until_utc must be timezone-aware")


@dataclass(frozen=True)
class RoundPlan:
    index: int
    language_backend: str
    deck_mode: str
    seat_rotation: int
    seed: int
    verification_continuation: bool = True


def plan_round(index: int, config: OvernightConfig) -> RoundPlan:
    return RoundPlan(index, "codex" if index % config.live_every == 0 else "stub",
                     "fixed" if index % 2 == 0 else "random", index % 6, config.seed + index)


@dataclass
class AttemptStats:
    attempts: int = 0
    completed: int = 0
    failed: int = 0
    pending: int = 0
    unreadable: int = 0
    known_input_tokens: int = 0
    known_output_tokens: int = 0
    usage_known_calls: int = 0
    cost: None = None


def count_attempts(directory: Path) -> AttemptStats:
    """A persisted attempt consumes budget even if inference never completed."""
    result = AttemptStats()
    for path in sorted(directory.glob("*.json")):
        result.attempts += 1
        try:
            value: object = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Attempt must be an object")
            data = cast(dict[str, object], value)
        except (OSError, ValueError):
            result.unreadable += 1
            result.pending += 1
            continue
        if data.get("status") == "completed" and data.get("success") is True:
            result.completed += 1
        elif data.get("status") == "failed":
            result.failed += 1
        else:
            result.pending += 1
        input_tokens, output_tokens = data.get("input_tokens"), data.get("output_tokens")
        if type(input_tokens) is int and type(output_tokens) is int:
            result.known_input_tokens += input_tokens
            result.known_output_tokens += output_tokens
            result.usage_known_calls += 1
    return result


@dataclass
class RoundRecord:
    plan: RoundPlan
    status: str = "planned"
    started_utc: str | None = None
    finished_utc: str | None = None
    child_pid: int | None = None
    exit_code: int | None = None
    allowance: int = 0
    timeout_seconds: float | None = None
    actions: int | None = None
    completed_hands: int = 0
    error: str | None = None
    cleanup_confirmed: bool | None = None
    model_calls: AttemptStats = field(default_factory=AttemptStats)
    command: list[str] = field(default_factory=list)
    source_sha256_start: dict[str, str] = field(default_factory=dict)
    source_sha256_end: dict[str, str] = field(default_factory=dict)
    loaded_supervisor_sha256: str = _LOADED_SUPERVISOR_SHA256
    source_changed: bool = False
    changed_source_files: list[str] = field(default_factory=list)

    def as_json(self) -> dict[str, object]:
        result: dict[str, object] = asdict(self)
        result.pop("plan")
        result.update(asdict(self.plan))
        result["language_is_stub"] = self.plan.language_backend == "stub"
        return result


def summarize(records: list[RoundRecord], config: OvernightConfig, overall_status: str) -> dict[str, object]:
    live_calls = sum(row.model_calls.attempts for row in records)
    return {
        "overall_status": overall_status, "planned": len(records),
        "started": sum(row.started_utc is not None for row in records),
        "completed": sum(row.status == "completed" for row in records),
        "failed": sum(row.status in _TERMINAL_FAILURES for row in records),
        "skipped": sum(row.status.startswith("skipped_") for row in records),
        "incomplete": sum(row.status in ("planned", "running", "stopped") for row in records),
        "live_rounds_completed": sum(row.status == "completed" and row.plan.language_backend == "codex" for row in records),
        "stub_rounds_completed": sum(row.status == "completed" and row.plan.language_backend == "stub" for row in records),
        "live_model_attempts": live_calls, "max_live_calls": config.max_live_calls,
        "remaining_live_calls": max(0, config.max_live_calls - live_calls),
        "model_attempts_completed": sum(row.model_calls.completed for row in records),
        "model_attempts_failed": sum(row.model_calls.failed for row in records),
        "model_attempts_pending": sum(row.model_calls.pending for row in records),
        "known_input_tokens": sum(row.model_calls.known_input_tokens for row in records),
        "known_output_tokens": sum(row.model_calls.known_output_tokens for row in records),
        "usage_known_calls": sum(row.model_calls.usage_known_calls for row in records),
        "source_drift_rounds": sum(row.source_changed for row in records),
        "loaded_supervisor_sha256": _LOADED_SUPERVISOR_SHA256,
        "cost": None, "competitive_strength_claim": False,
        "rounds": [row.as_json() for row in records],
    }


def render_report(summary: dict[str, object], records: list[RoundRecord], until: datetime) -> str:
    lines = ["# 六种 Policy 夜间运行记录", "",
             f"状态：`{summary['overall_status']}`。截止时间：{until.astimezone(UTC).isoformat()}（UTC）。", "",
             f"计划 {summary['planned']} 轮，启动 {summary['started']} 轮，完成 {summary['completed']} 轮，"
             f"失败 {summary['failed']} 轮，跳过 {summary['skipped']} 轮，未完成 {summary['incomplete']} 轮。", "",
             f"真实模型完成轮数 {summary['live_rounds_completed']}；离线 stub 完成轮数 {summary['stub_rounds_completed']}。"
             "stub 只验证协议，不计作自然语言模型实测。", "",
             f"真实模型尝试 {summary['live_model_attempts']}/{summary['max_live_calls']} 次，失败及 started 状态均计入预算。"
             f"已知输入 token {summary['known_input_tokens']}、输出 token {summary['known_output_tokens']}，"
             f"具备完整 usage 的调用 {summary['usage_known_calls']} 次；费用未知（JSON 为 null）。", "",
             "所有轮次使用 verification_continuation=true，属于机制和工程验收；固定牌序和随机牌序分别记录，不比较竞技实力。", "",
             "| 轮次 | backend | 牌序 | 座位旋转 | seed | 状态 | 已确认动作 | 完成手数 | 模型尝试 | 错误 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for row in records:
        error = (row.error or "").replace("|", "/").replace("\n", " ")[:240]
        lines.append(f"| {row.plan.index} | {row.plan.language_backend} | {row.plan.deck_mode} | "
                     f"{row.plan.seat_rotation} | {row.plan.seed} | {row.status} | "
                     f"{row.actions if row.actions is not None else '—'} | {row.completed_hands} | "
                     f"{row.model_calls.attempts} | {error} |")
    lines.extend(["", "每轮 round_NNN/manifest.json 保留启动与终态，stdout.log 保留 driver 输出；"
                  "evidence/ 内保留六人验收回执或失败记录、牌局与模型尝试证据。启动不代表完成。", ""])
    return "\n".join(lines)


def claim_output(path: Path, initial_status: dict[str, object]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(item.name != "PLAN.md" for item in path.iterdir()):
        raise FileExistsError("Use a new output directory (an existing PLAN.md is allowed); do not restart an existing run")
    # Exclusive creation prevents two supervisors from claiming the same plan.
    with (path / "owner.lock").open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()) + "\n")
    atomic_json(path / "status.json", initial_status)


def source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in (_SUPERVISOR_SOURCE, "scripts/verify_six_agents.py"):
        try:
            result[name] = sha256((_ROOT / name).read_bytes()).hexdigest()
        except FileNotFoundError:
            result[name] = "missing"
    return result


def verify_round_sources(row: RoundRecord) -> None:
    changed = {name for name in row.source_sha256_start.keys() | row.source_sha256_end.keys()
               if row.source_sha256_start.get(name) != row.source_sha256_end.get(name)}
    if any(values.get(_SUPERVISOR_SOURCE) != row.loaded_supervisor_sha256
           for values in (row.source_sha256_start, row.source_sha256_end)):
        changed.add(_SUPERVISOR_SOURCE)
    row.changed_source_files = sorted(changed)
    row.source_changed = bool(changed)
    if changed:
        row.status = "failed"
        message = "Source drift invalidated this round: " + ", ".join(row.changed_source_files)
        row.error = f"{row.error}; {message}" if row.error else message


CommandBuilder = Callable[[RoundPlan, Path, int, float], list[str]]


def driver_command(plan: RoundPlan, evidence_dir: Path, allowance: int, timeout: float,
                   config: OvernightConfig) -> list[str]:
    command = [sys.executable, "-u", "-X", "utf8", str(_ROOT / "scripts/verify_six_agents.py"),
               "--output-dir", str(evidence_dir), "--hands", str(config.hands), "--timeout", str(timeout),
               "--decision-timeout", str(min(180.0, timeout / 2)), "--seed", str(plan.seed),
               "--deck-mode", plan.deck_mode, "--language-backend", plan.language_backend,
               "--seat-rotation", str(plan.seat_rotation), "--language-max-calls", str(max(1, allowance))]
    if config.language_model is not None:
        command.extend(["--language-model", config.language_model])
    return command


def stop_owned_process(process: subprocess.Popen[bytes]) -> bool:
    """Stop only the still-owned child tree, never search arbitrary process names."""
    if process.poll() is not None:
        return True
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                       creationflags=subprocess.CREATE_NO_WINDOW, timeout=15)
    else:
        # run_supervisor starts a new session, so this group belongs to this child.
        try:
            kill_group = cast(Callable[[int, int], None], getattr(os, "killpg"))
            kill_group(process.pid, cast(int, getattr(signal, "SIGKILL")))
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        return False
    return process.poll() is not None


def _read_result(evidence_dir: Path, row: RoundRecord, hands: int) -> None:
    path = evidence_dir / "receipt.json"
    if not path.exists():
        path = evidence_dir / "failure.json"
    if not path.exists():
        raise ValueError("Driver did not preserve receipt.json or failure.json")
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Driver receipt must be an object")
    data = cast(dict[str, object], raw)
    completed = data.get("server_completed_hands")
    row.completed_hands = len(completed) if isinstance(completed, list) else 0
    wire = data.get("wire")
    if isinstance(wire, dict) and type(wire.get("actions")) is int:
        row.actions = cast(int, wire["actions"])
    if data.get("ok") is not True:
        raise RuntimeError(str(data.get("error") or "Driver reported failure"))
    checks = data.get("checks")
    if not isinstance(checks, dict) or not checks or any(value is not True for value in checks.values()):
        raise ValueError("Driver receipt did not pass every declared acceptance check")
    processes = data.get("processes")
    if not isinstance(processes, list) or len(processes) != 7:
        raise ValueError("Driver receipt does not contain seven processes")
    pids: set[int] = set()
    for child in processes:
        if not isinstance(child, dict) or child.get("exit_code") != 0 or type(child.get("runtime_pid")) is not int:
            raise ValueError("Driver receipt has an unfinished or failed process")
        pids.add(child["runtime_pid"])
    if len(pids) != 7:
        raise ValueError("Driver receipt runtime process identities are not distinct")
    for key, expected in (("language_backend", row.plan.language_backend), ("deck_mode", row.plan.deck_mode),
                          ("seed", row.plan.seed), ("seat_rotation", row.plan.seat_rotation),
                          ("verification_continuation", True)):
        if data.get(key) != expected:
            raise ValueError(f"Driver receipt does not match planned {key}")
    if row.completed_hands < hands:
        raise ValueError("Successful process did not complete the target server hands")
    if row.exit_code != 0:
        raise RuntimeError(f"Driver exited {row.exit_code} despite a success receipt")
    if row.plan.language_backend == "codex" and row.model_calls.completed < 1:
        raise ValueError("Live round has no completed real model attempt")
    if row.plan.language_backend == "stub" and row.model_calls.attempts:
        raise ValueError("An offline stub round unexpectedly wrote live model attempts")
    if row.model_calls.attempts > row.allowance:
        raise ValueError("Provider attempts exceeded the assigned live budget")


def run_supervisor(config: OvernightConfig, *, command_builder: CommandBuilder | None = None) -> dict[str, object]:
    output = config.output_dir.resolve()
    stop_file = config.stop_file or output / "STOP"
    created = utc_now().isoformat()
    records = [RoundRecord(plan_round(index, config)) for index in range(config.rounds)]
    overall, current_index, last_progress = "running", None, created
    claim_output(output, {"owner_pid": os.getpid(), "overall_status": overall, "created_utc": created})
    consecutive_failures = 0
    next_start = monotonic()

    def checkpoint() -> dict[str, object]:
        summary = summarize(records, config, overall)
        summary.update({"owner_pid": os.getpid(), "created_utc": created, "updated_utc": utc_now().isoformat(),
                        "until_utc": config.until_utc.astimezone(UTC).isoformat(), "last_progress_utc": last_progress})
        atomic_json(output / "summary.json", summary)
        atomic_json(output / "status.json", {key: value for key, value in summary.items() if key != "rounds"}
                    | {"current_round": current_index, "stop_file": str(stop_file),
                       "consecutive_failures": consecutive_failures})
        report = output / "REPORT.md.tmp"
        report.write_text(render_report(summary, records, config.until_utc), encoding="utf-8")
        report.replace(output / "REPORT.md")
        return summary

    def skip_remaining(index: int, status: str) -> None:
        for row in records[index:]:
            if row.status == "planned":
                row.status, row.finished_utc = status, utc_now().isoformat()

    checkpoint()
    try:
        for index, row in enumerate(records):
            current_index = index
            while monotonic() < next_start:
                if stop_file.exists() or utc_now() >= config.until_utc:
                    break
                checkpoint()
                sleep(min(config.poll_seconds, max(0.0, next_start - monotonic()),
                          max(0.0, (config.until_utc - utc_now()).total_seconds())))
            if stop_file.exists():
                overall = "stopped"
                skip_remaining(index, "skipped_stop")
                break
            remaining_time = (config.until_utc - utc_now()).total_seconds()
            if remaining_time <= 0 or (command_builder is None and remaining_time <= 10):
                overall = "deadline"
                skip_remaining(index, "skipped_deadline")
                break
            if consecutive_failures >= 3:
                overall = "finished_with_failures"
                skip_remaining(index, "skipped_circuit")
                break
            round_dir = output / f"round_{index:03d}"
            round_dir.mkdir()
            evidence_dir = round_dir / "evidence"
            row.source_sha256_start = source_hashes()
            if row.source_sha256_start.get(_SUPERVISOR_SOURCE) != row.loaded_supervisor_sha256:
                row.source_sha256_end = dict(row.source_sha256_start)
                verify_round_sources(row)
                row.status, row.finished_utc = "skipped_source_drift", utc_now().isoformat()
                last_progress = row.finished_utc
                atomic_json(round_dir / "manifest.json", row.as_json())
                overall = "finished_with_failures"
                skip_remaining(index + 1, "skipped_source_drift")
                break
            remaining_calls = config.max_live_calls - sum(item.model_calls.attempts for item in records)
            row.allowance = min(32, max(0, remaining_calls)) if row.plan.language_backend == "codex" else 1
            if row.plan.language_backend == "codex" and remaining_calls < 2:
                row.status, row.finished_utc = "skipped_budget", utc_now().isoformat()
                row.error = "Fewer than two live calls remain; this round was not changed to stub"
                row.source_sha256_end = source_hashes()
                last_progress = row.finished_utc
                atomic_json(round_dir / "manifest.json", row.as_json())
                next_start = monotonic() + config.interval_seconds
                checkpoint()
                continue
            row.timeout_seconds = min(config.round_timeout, remaining_time)
            row.command = (command_builder(row.plan, evidence_dir, row.allowance, row.timeout_seconds)
                           if command_builder is not None else driver_command(row.plan, evidence_dir, row.allowance,
                                                                              row.timeout_seconds, config))
            row.status, row.started_utc = "running", utc_now().isoformat()
            last_progress = row.started_utc
            round_started = monotonic()
            next_start = round_started + config.interval_seconds
            atomic_json(round_dir / "manifest.json", row.as_json())
            checkpoint()
            process: subprocess.Popen[bytes] | None = None
            try:
                with (round_dir / "stdout.log").open("wb") as stdout:
                    process = subprocess.Popen(row.command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=subprocess.STDOUT,
                                               cwd=_ROOT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                                               start_new_session=os.name != "nt")
                    row.child_pid = process.pid
                    atomic_json(round_dir / "manifest.json", row.as_json())
                    while process.poll() is None:
                        previous_calls = row.model_calls.attempts
                        row.model_calls = count_attempts(evidence_dir / "model_calls")
                        if row.model_calls.attempts != previous_calls:
                            last_progress = utc_now().isoformat()
                        checkpoint()
                        if stop_file.exists():
                            overall, row.status, row.error = "stopped", "stopped", "Supervisor stop file requested shutdown"
                            break
                        if utc_now() >= config.until_utc:
                            overall, row.status, row.error = "deadline", "timed_out", "Overall UTC deadline reached"
                            break
                        if monotonic() - round_started >= row.timeout_seconds:
                            row.status, row.error = "timed_out", "Driver exceeded this round's wall-clock limit"
                            break
                        if row.plan.language_backend == "codex" and row.model_calls.attempts > row.allowance:
                            row.status, row.error = "failed", "Provider attempts exceeded the assigned live budget"
                            break
                        sleep(min(config.poll_seconds, max(0.001, row.timeout_seconds - (monotonic() - round_started)),
                                  max(0.001, (config.until_utc - utc_now()).total_seconds())))
                    if process.poll() is None:
                        row.cleanup_confirmed = stop_owned_process(process)
                    else:
                        row.cleanup_confirmed = True
                    row.exit_code = process.poll()
                row.model_calls = count_attempts(evidence_dir / "model_calls")
                if row.status == "running":
                    _read_result(evidence_dir, row, config.hands)
                    row.status = "completed"
            except KeyboardInterrupt:
                overall, row.status, row.error = "stopped", "stopped", "Supervisor interrupted"
                raise
            except Exception as error:
                row.status, row.error = "failed", f"{type(error).__name__}: {error}"
            finally:
                if process is not None:
                    try:
                        row.cleanup_confirmed = stop_owned_process(process)
                    except (OSError, subprocess.SubprocessError) as error:
                        row.cleanup_confirmed = False
                        row.error = (row.error or "") + f"; owned process cleanup failed: {type(error).__name__}"
                    row.exit_code = process.poll()
                    if not row.cleanup_confirmed:
                        row.status = "failed"
                        row.error = row.error or "Owned child did not exit during cleanup"
                row.model_calls = count_attempts(evidence_dir / "model_calls")
                row.finished_utc = utc_now().isoformat()
                last_progress = row.finished_utc
                row.source_sha256_end = source_hashes()
                verify_round_sources(row)
                atomic_json(round_dir / "manifest.json", row.as_json())
                consecutive_failures = consecutive_failures + 1 if row.status in _TERMINAL_FAILURES else 0
                checkpoint()
            if overall in ("stopped", "deadline"):
                skip_remaining(index + 1, "skipped_stop" if overall == "stopped" else "skipped_deadline")
                break
        if overall == "running":
            overall = "completed" if all(row.status == "completed" for row in records) else "finished_with_failures"
    except KeyboardInterrupt:
        overall = "stopped"
        skip_remaining(0, "skipped_stop")
    except Exception as error:
        overall = "finished_with_failures"
        atomic_json(output / "supervisor_error.json", {"type": type(error).__name__, "message": str(error),
                                                     "timestamp_utc": utc_now().isoformat()})
        skip_remaining(0, "skipped_supervisor_error")
    current_index = None
    return checkpoint()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--until-utc", type=parse_utc, required=True)
    parser.add_argument("--rounds", type=int, default=18)
    parser.add_argument("--interval-seconds", type=float, default=900.0)
    parser.add_argument("--max-live-calls", type=int, default=64)
    parser.add_argument("--live-every", type=int, default=9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hands", type=int, default=2)
    parser.add_argument("--round-timeout", type=float, default=900.0)
    parser.add_argument("--language-model", default=DEFAULT_CODEX_MODEL,
                        help=f"Explicit Codex model (project default: {DEFAULT_CODEX_MODEL}; reasoning: low)")
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    try:
        config = OvernightConfig(args.output_dir, args.until_utc, args.rounds, args.interval_seconds,
                                 args.max_live_calls, args.live_every, args.seed, args.hands,
                                 args.round_timeout, args.language_model, args.stop_file)
        summary = run_supervisor(config)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps({key: value for key, value in summary.items() if key != "rounds"}, ensure_ascii=False), flush=True)
    raise SystemExit(0 if summary["overall_status"] == "completed" else 1)


if __name__ == "__main__":
    main()
