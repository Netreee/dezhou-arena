"""Seven real Arena/CLI processes with six explicitly offline ReAct agents.

The deterministic backend tests orchestration, tools and acknowledgements. It
does not perform external language-model inference or measure poker strength.
"""

import argparse
from datetime import UTC, datetime
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import Event, Lock, Thread
from time import monotonic
from typing import cast
from uuid import uuid4

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from jsonschema import Draft202012Validator
from shared_logging import JsonObject

from scripts.verify_agent_runtime import CapturedProcess, RecordedRequest, read_requests
from scripts.verify_six_agents import _completed, _logs, _snapshots, fixture_server, verify_view
from poker.application.commands import ActCommand, JoinCommand, StartHandCommand, StateCommand
from poker.agent.context import DecisionControl
from poker.agent.react.backend import BackendRequest
from poker.agent.react.examples import FixtureBackend
from poker.domain.models import Table
from poker.domain.types import ActionKind, HandPhase, TableId
from poker.persistence.snapshot import TableSnapshotCodec
from poker.persistence.sqlite import SqliteTableRepository

REPRESENTATIONS = ("text", "rules", "lookup", "memory", "solver", "combined")
EXAMPLES = "poker.agent.react.examples"


class HttpFixture:
    """Loopback-only protocol responder, running inside the verification driver."""

    def __init__(self, path: Path) -> None:
        self.records: list[dict[str, object]] = []
        lock = Lock()
        records = self.records

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                try:
                    assert self.path == "/v1/chat/completions"
                    assert self.headers.get("Authorization") == "Bearer offline-fixture-token"
                    length = int(self.headers.get("Content-Length", "0"))
                    assert 0 < length < 300000
                    body = json.loads(self.rfile.read(length))
                    assert body["model"] == "local-http-fixture-v1" and body["stream"] is False and body["n"] == 1
                    assert body["max_completion_tokens"] == 512
                    messages = body["messages"]
                    assert len(messages) == 2 and messages[0]["role"] == "system" and messages[1]["role"] == "user"
                    payload = cast(JsonObject, json.loads(messages[1]["content"]))
                    schema = cast(JsonObject, body["response_format"]["json_schema"]["schema"])
                    request = BackendRequest(messages[0]["content"], payload, schema, body["max_completion_tokens"])
                    response = FixtureBackend().generate(request, DecisionControl(monotonic() + 5))
                    Draft202012Validator(schema).validate(response.output)
                    observation = cast(JsonObject, payload["observation"])
                    player = cast(JsonObject, observation["me"])
                    memory = payload.get("memory", {})
                    record: dict[str, object] = {
                        "is_live": False, "player_id": player["player_id"], "hand_id": observation["hand_id"],
                        "revision": observation["revision"], "round_index": payload["round_index"],
                        "pending_tools": payload["pending_tools"], "output_kind": response.output["kind"],
                        "input_sha256": hashlib.sha256(messages[1]["content"].encode()).hexdigest(),
                        "memory_sha256": hashlib.sha256(json.dumps(memory, sort_keys=True).encode()).hexdigest(),
                        "model": body["model"], "max_completion_tokens": body["max_completion_tokens"],
                    }
                    with lock:
                        records.append(record)
                        with path.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(record) + "\n")
                    envelope = {"id": f"offline-{uuid4().hex}", "choices": [{"finish_reason": "stop", "message": {
                        "role": "assistant", "content": json.dumps(response.output)}}]}
                    raw = json.dumps(envelope).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                except Exception:
                    self.send_error(500, "Offline fixture rejected the request")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, name="react-http-fixture", daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        assert not self.thread.is_alive()


def audit_wire(rows: list[RecordedRequest], snapshots: dict[int, Table], *, hands: int) -> dict[str, object]:
    assert rows and all(row.response.ok for row in rows), "A CLI request was rejected"
    joins = [row for row in rows if isinstance(row.command, JoinCommand)]
    assert len(joins) == len({row.player_id for row in joins}) == 6
    assert {cast(JoinCommand, row.command).name for row in joins} == set(REPRESENTATIONS)
    starter = next(row.player_id for row in joins if cast(JoinCommand, row.command).name == "rules")
    starts = [row for row in rows if isinstance(row.command, StartHandCommand)]
    assert len(starts) == hands and all(row.player_id == starter for row in starts)
    for row in rows:
        view = row.response.view
        assert view is not None
        verify_view(view, snapshots[row.revision], row.player_id)
        raw_view = cast(dict[str, object], json.loads(cast(str, row.raw["response_json"]))["view"])
        assert not {"deck", "pending", "last_action_bet"} & raw_view.keys()
        assert all("hole_cards" not in item for item in cast(list[dict[str, object]], raw_view["players"]))
        if isinstance(row.command, ActCommand):
            assert row.command.expected_revision == row.revision - 1
            assert row.command.hand_id == view.hand_id
            history, previous = view.action_history[-1], snapshots[row.revision - 1]
            assert previous.hand is not None
            assert (history.player_id, history.kind) == (row.player_id, row.command.action.kind.value)
            assert history.stack + history.pay == previous.player(row.player_id).stack
            assert history.to == previous.hand.players[row.player_id].street_commit + history.pay
            if row.command.action.kind in (ActionKind.BET_TO, ActionKind.RAISE_TO):
                assert history.to == row.command.action.to
            if row.command.action.kind in (ActionKind.CHECK, ActionKind.FOLD):
                assert history.pay == 0
    completed = _completed(snapshots)
    assert len(completed) == hands
    for hand_id, table in completed.items():
        assert table.hand is not None and len(table.hand.board) == 5
        assert len(table.hand.players) == 6
        actions = sum(isinstance(row.command, ActCommand) and row.command.hand_id == hand_id for row in rows)
        assert len(table.hand.action_history) == actions + 2
        assert {item.phase for item in table.hand.action_history[2:]} == {
            HandPhase.PREFLOP, HandPhase.FLOP, HandPhase.TURN, HandPhase.RIVER,
        }
    counts = {row.player_id: sum(isinstance(item.command, ActCommand) and item.player_id == row.player_id for item in rows) for row in joins}
    polls = {row.player_id: sum(isinstance(item.command, StateCommand) and item.player_id == row.player_id for item in rows) for row in joins}
    assert all(value >= 1 for value in counts.values()) and all(value >= 1 for value in polls.values())
    return {"requests": len(rows), "actions": sum(counts.values()), "actions_by_player": counts,
            "automatic_state_counts": polls, "completed_hands": list(completed)}


def audit_logs(output: Path, run_id: str, agents: list[CapturedProcess], rows: list[RecordedRequest],
               *, hands: int, backend: str, http_records: list[dict[str, object]]) -> dict[str, object]:
    validator = Draft202012Validator(json.loads((_ROOT / "src/shared_logging/event.schema.json").read_text(encoding="utf-8")))
    events = _logs(output, run_id)
    for event in events:
        validator.validate(event)
        assert event["run_id"] == run_id and event["level"] not in ("ERROR", "CRITICAL")
    summaries: list[dict[str, object]] = []
    for agent in agents:
        joined = next(row for row in rows if isinstance(row.command, JoinCommand) and row.command.name == agent.label)
        pids = {cast(int, event["pid"]) for event in events if event["event"] == "client.joined"
                and cast(dict[str, object], event["context"]).get("player_id") == joined.player_id}
        assert len(pids) == 1
        agent.runtime_pid = pids.pop()
        own = [event for event in events if event["pid"] == agent.runtime_pid]

        def data(name: str) -> list[dict[str, object]]:
            return [cast(dict[str, object], event["data"]) for event in own if event["event"] == name]

        configured = data("react.process.configured")
        assert len(configured) == 1 and configured[0]["agent_class"] == "ReActAgent"
        factory = "fixture_backend" if backend == "fixture" else "http_fixture_backend"
        assert configured[0]["backend_factory"] == f"{EXAMPLES}:{factory}"
        assert configured[0]["policy_factory"] == f"{EXAMPLES}:{agent.label}"
        queued = {item["decision_id"]: item for item in data("agent.action.queued")}
        confirmed = [item["decision_id"] for item in data("agent.action.confirmed")]
        assert len(queued) == len(data("agent.action.queued")) == len(confirmed) == len(set(confirmed)) >= 1
        assert set(queued) == set(confirmed)
        applied = [row for row in rows if row.player_id == joined.player_id and isinstance(row.command, ActCommand)]
        assert len(applied) == len(confirmed)
        expected = {(str(row.command.hand_id), row.command.expected_revision, row.command.action.kind.value, row.command.action.to)
                    for row in applied if isinstance(row.command, ActCommand)}
        actual = {(item["hand_id"], item["revision"], item["action"], item.get("to")) for item in queued.values()}
        assert expected == actual
        decisions = data("react.decision.completed")
        assert {item["decision_id"] for item in decisions} == set(confirmed)
        started, completed = data("react.model.started"), data("react.model.completed")
        generated = (data("react.fixture.generated") if backend == "fixture" else
                     [item for item in http_records if item["player_id"] == joined.player_id])
        assert len(started) == len(completed) == len(generated) >= 2 * len(confirmed)
        assert not data("react.model.failed") and not data("api.call.failed")
        if backend == "fixture":
            assert not data("api.call.started")
        else:
            assert len(data("api.call.started")) == len(data("api.call.completed")) == len(completed)
            assert all(item["is_live"] is False for item in data("api.call.completed"))
        assert all(item["is_live"] is False for item in (*started, *completed, *decisions))
        tools = data("react.tool.completed")
        standard_tools = data("agent.tool.completed")
        assert len(tools) == len(standard_tools) and all(item["ok"] is True for item in (*tools, *standard_tools))
        for decision in decisions:
            identifier = decision["decision_id"]
            successful = {item["tool"] for item in tools if item["decision_id"] == identifier and item["ok"] is True}
            required = cast(list[str], decision["required_tools"])
            assert required and set(required) <= successful == set(cast(list[str], decision["successful_tools"]))
            assert "legal_actions" in successful
            assert decision["model_calls"] == sum(item["decision_id"] == identifier for item in completed)
            if backend == "http-fixture":
                matching_http = [item for item in generated if item["revision"] == decision["revision"]
                                 and item["hand_id"] == decision["hand_id"]]
                assert len(matching_http) == decision["model_calls"]
                assert matching_http[-1]["output_kind"] == "final"
            assert decision["tool_calls"] == sum(item["decision_id"] == identifier for item in tools)
            assert (decision["hand_id"], decision["revision"]) == (queued[identifier]["hand_id"], queued[identifier]["revision"])
        tool_names = {item["tool"] for item in tools}
        if agent.label in ("lookup", "combined"):
            assert "lookup" in tool_names
        if agent.label in ("solver", "combined"):
            assert "equity" in tool_names
        if agent.label in ("rules", "memory", "combined"):
            assert "public_history" in tool_names
        if agent.label in ("memory", "combined"):
            assert configured[0]["memory_factory"] == f"{EXAMPLES}:event_memory"
            assert len({item["memory_sha256"] for item in generated}) >= 2, "Memory never changed between model calls"
        stop = data("agent.stopped")[-1]
        assert stop["stop_reason"] == "max_hands" and stop["observed_hands"] == hands
        assert stop["confirmed_actions"] == len(confirmed) and stop["ok"] is True and stop["error"] is None
        assert stop["cli_closed"] is True and stop["policy_closed"] is True
        summaries.append({"representation": agent.label, "player_id": joined.player_id, "runtime_pid": agent.runtime_pid,
                          "confirmed_actions": len(confirmed), "model_calls": len(completed), "tool_calls": len(tools),
                          "tools": sorted(cast(set[str], tool_names)), "is_live": False, "decisions": decisions, "stop": stop})
    server_pids = {cast(int, event["pid"]) for event in events if event["process_role"] == "server"}
    assert len(server_pids) == 1
    return {"events": len(events), "server_pid": server_pids.pop(), "agents": summaries,
            "model_calls": sum(cast(int, item["model_calls"]) for item in summaries)}


def source_hashes() -> dict[str, str]:
    files = list((_ROOT / "src/poker").rglob("*.py")) + list((_ROOT / "src/shared_logging").rglob("*.py"))
    files.extend(_ROOT / name for name in ("pyproject.toml", "scripts/__init__.py", "scripts/verify_react_agents.py",
                                          "scripts/verify_six_agents.py", "scripts/verify_agent_runtime.py",
                                          "src/shared_logging/event.schema.json", "src/poker/persistence/schema.sql"))
    return {path.relative_to(_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(set(files))}


def verify(output_dir: Path, *, hands: int = 2, timeout: float = 180.0, seed: int = 7,
           backend: str = "fixture") -> dict[str, object]:
    if type(hands) is not int or hands < 1 or not 10 < timeout <= 3600:
        raise ValueError("Require hands >= 1 and 10 < timeout <= 3600")
    if backend not in ("fixture", "http-fixture"):
        raise ValueError("Only the explicit offline fixture and http-fixture backends are supported")
    output = output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Choose a new or empty evidence directory")
    output.mkdir(parents=True, exist_ok=True)
    database, wire_path = output / "final.sqlite3", output / "server_commands.jsonl"
    snapshot_path, stop = output / "arena_snapshots.jsonl", output / "stop"
    config_path = output / "policy_config.json"
    factory_config: dict[str, object] = {"verification_continuation": True, "equity_rollouts": 24}
    run_id = f"react-offline-{uuid4().hex}"
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(_ROOT / "src"),
                   "APP_RUN_ID": run_id, "APP_LOG_DIR": str(output / "logs"), "LOG_LEVEL": "INFO",
                   "LOG_MAX_BYTES": "50000000", "LOG_BACKUP_COUNT": "3"}
    environment.pop("APP_LOG_ROLE", None)
    captures: list[CapturedProcess] = []
    agents: list[CapturedProcess] = []
    server: CapturedProcess | None = None
    http_fixture: HttpFixture | None = None
    error: str | None = None
    result: dict[str, object] = {}
    initial: dict[str, str] = {}
    final: dict[str, str] = {}
    started = monotonic()
    try:
        initial = source_hashes()
        if backend == "http-fixture":
            http_fixture = HttpFixture(output / "http_fixture_calls.jsonl")
            factory_config["base_url"] = http_fixture.base_url
            environment["POKER_REACT_FIXTURE_KEY"] = "offline-fixture-token"
        config_path.write_text(json.dumps(factory_config), encoding="utf-8")
        server = CapturedProcess("server", [sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()),
                                 "--fixture-server", "--database", str(database), "--wire-log", str(wire_path),
                                 "--snapshots", str(snapshot_path), "--stop-file", str(stop)], environment, _ROOT)
        captures.append(server)
        ready_deadline = monotonic() + 20
        while True:
            match = re.search(r"on \('127\.0\.0\.1', (\d+)\); table=([0-9a-f]+)", server.next_line(ready_deadline))
            if match:
                port, table_id = int(match.group(1)), TableId(match.group(2))
                break
        for index, representation in enumerate(REPRESENTATIONS):
            backend_factory = "fixture_backend" if backend == "fixture" else "http_fixture_backend"
            command = [sys.executable, "-u", "-X", "utf8", "-m", "poker.agent.react", "--port", str(port),
                       "--name", representation, "--backend", f"{EXAMPLES}:{backend_factory}", "--policy", f"{EXAMPLES}:{representation}",
                       "--tool-factory", f"{EXAMPLES}:analysis_tools", "--config", str(config_path), "--min-players", "6",
                       "--max-hands", str(hands), "--poll-interval", "0.2", "--decision-timeout", "15",
                       "--session-timeout", str(timeout - 5), "--max-model-calls-per-session", str(hands * 64), "--seed", str(seed + index)]
            if representation == "rules":
                command.append("--auto-start")
            if representation in ("memory", "combined"):
                command.extend(("--memory-factory", f"{EXAMPLES}:event_memory"))
            agent = CapturedProcess(representation, command, environment, _ROOT)
            captures.append(agent)
            agents.append(agent)
            deadline = min(started + timeout, monotonic() + 20)
            while True:
                snapshots = _snapshots(snapshot_path)
                table = snapshots[max(snapshots)] if snapshots else None
                if table is not None and any(player.name == representation for player in table.players):
                    assert next(player.seat for player in table.players if player.name == representation) == index
                    break
                if agent.process.poll() is not None:
                    raise RuntimeError(f"{representation} failed to join: {agent.lines[-6:]}")
                if monotonic() > deadline:
                    raise TimeoutError(f"{representation} join timed out")
                Event().wait(0.05)
        while any(agent.process.poll() is None for agent in agents):
            for agent in agents:
                if agent.process.poll() not in (None, 0):
                    raise RuntimeError(f"{agent.label} failed: {agent.lines[-6:]}")
            if server.process.poll() is not None:
                raise RuntimeError("Arena exited before the agents")
            if monotonic() > started + timeout:
                raise TimeoutError("Offline ReAct session exceeded its wall-clock limit")
            Event().wait(0.2)
        for agent in agents:
            agent.finish()
        stop.touch()
        server.finish()
        rows, snapshots = read_requests(wire_path), _snapshots(snapshot_path)
        wire = audit_wire(rows, snapshots, hands=hands)
        logging = audit_logs(output, run_id, agents, rows, hands=hands, backend=backend,
                             http_records=http_fixture.records if http_fixture else [])
        if http_fixture:
            assert len(http_fixture.records) == logging["model_calls"]
        server.runtime_pid = cast(int, logging["server_pid"])
        assert len({item.runtime_pid for item in captures}) == 7
        table = SqliteTableRepository(database).load(table_id)
        assert table is not None and table == snapshots[max(snapshots)]
        assert table.hand is not None and table.hand.phase is HandPhase.COMPLETE
        assert sum(player.stack for player in table.players) == 6000
        result = {"wire": wire, "logging": logging, "server_completed_hands": list(_completed(snapshots)),
                  "final_snapshot": json.loads(TableSnapshotCodec().encode(table)), "checks": {
                      "seven_distinct_runtime_processes": True, "all_six_use_react_agent": True,
                      "every_action_has_backend_tool_and_cli_confirmation": True, "required_tools_completed": True,
                      "memory_changes_visible_to_backend": True, "safe_views_match_sqlite_revision": True,
                      "chips_conserved_6000": True, "full_four_street_hands": True, "public_history_complete": True,
                      "all_processes_exit_zero": True, "sqlite_reopened": True, "log_schema_valid": True}}
    except (Exception, KeyboardInterrupt) as failure:
        error = f"{type(failure).__name__}: {failure}"
    finally:
        if server is not None and server.process.poll() is None:
            stop.touch()
        for capture in reversed(captures):
            try:
                if error is not None and capture.process.poll() is None and capture is not server and os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(capture.process.pid), "/T", "/F"], check=False,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW, timeout=15)
                if capture is server and capture.process.poll() is None:
                    capture.process.wait(timeout=5)
                capture.cleanup()
            except (OSError, subprocess.SubprocessError, AssertionError) as failure:
                error = error or f"Cleanup failed: {type(failure).__name__}"
            (output / f"{capture.label}_stdout.log").write_text("\n".join(capture.lines) + "\n", encoding="utf-8")
        if http_fixture:
            try:
                http_fixture.close()
            except Exception as failure:
                error = error or f"HTTP fixture cleanup failed: {type(failure).__name__}"
    try:
        final = source_hashes()
        changed = sorted(path for path in set(initial) | set(final) if initial.get(path) != final.get(path))
        if changed:
            error = error or "SourceChangedDuringRun: source hashes changed during verification"
        checks = cast(dict[str, object], result.setdefault("checks", {}))
        checks["source_unchanged_during_run"] = not changed
        result.update(source_changed=bool(changed), changed_source_files=changed)
    except (OSError, ValueError) as failure:
        error = error or f"Source manifest failed: {type(failure).__name__}"
    evidence = {path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(output.rglob("*")) if path.is_file() and path.name not in ("receipt.json", "failure.json")}
    summary: dict[str, object] = {"ok": error is None, "error": error, "scope": "Offline fixture model; real Arena, CLI, tools and processes",
                                "external_model_inference": False, "is_live": False, "competitive_strength_claim": False,
                                "backend": backend, "http_fixture_calls": len(http_fixture.records) if http_fixture else 0,
                                "http_fixture_host_process": os.getpid() if http_fixture else None,
                                "timestamp_utc": datetime.now(UTC).isoformat(), "run_id": run_id,
                                "elapsed_seconds": round(monotonic() - started, 3), "target_hands": hands, "seed": seed,
                                "processes": [{"label": item.label, "launcher_pid": item.process.pid, "runtime_pid": item.runtime_pid,
                                               "exit_code": item.process.poll(), "command": item.command} for item in captures],
                                "source_sha256_start": initial, "source_sha256": final, "evidence_sha256": evidence, **result}
    destination = output / ("receipt.json" if summary["ok"] else "failure.json")
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"ok": summary["ok"], "receipt": str(destination), "error": error,
            "actions": cast(dict[str, object], result.get("wire", {})).get("actions"),
            "model_calls": cast(dict[str, object], result.get("logging", {})).get("model_calls"),
            "external_model_inference": False, "process_exit_codes": [item.process.poll() for item in captures]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--hands", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--backend", choices=("fixture", "http-fixture"), default="fixture")
    parser.add_argument("--fixture-server", action="store_true", help=argparse.SUPPRESS)
    for option in ("database", "wire-log", "snapshots", "stop-file"):
        parser.add_argument(f"--{option}", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.fixture_server:
        fixture_server(args.database, args.wire_log, args.snapshots, args.stop_file, deck_mode="fixed", deck_seed=0)
        return
    if args.output_dir is None:
        parser.error("--output-dir is required")
    outcome = verify(args.output_dir, hands=args.hands, timeout=args.timeout, seed=args.seed, backend=args.backend)
    print(json.dumps(outcome, ensure_ascii=False))
    raise SystemExit(0 if outcome["ok"] else 1)


if __name__ == "__main__":
    main()
