"""Verify two independent agent processes through the production CLI and Arena.

Uses a fixed showdown board solely to keep both example players funded for two
hands. The production server, CLI, rules, queue, payment and SQLite paths run.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Queue
import re
import subprocess
import sys
from threading import Event, Thread
from time import monotonic
from typing import cast
from unittest.mock import patch
from uuid import uuid4

from jsonschema import Draft202012Validator

import poker
from poker.application.commands import ActCommand, Command, JoinCommand, SessionContext, StartHandCommand, StateCommand
from poker.application.service import TableService
from poker.application.views import CommandResponse, PlayerView
from poker.bootstrap import server_main
from poker.domain.cards import Card, Deck
from poker.domain.models import Table
from poker.domain.types import ActionKind, HandPhase, PlayerId, TableId
from poker.engine.holdem import HoldemEngine
from poker.engine.policies import FiveCardHighEvaluator, NoLimitBettingRules, SidePotAllocator
from poker.persistence.snapshot import TableSnapshotCodec
from poker.persistence.sqlite import SqliteTableRepository
from poker.transport.codec import JsonLineCodec
from poker.transport.local_tcp import LocalTcpServer


def fixture_deck() -> Deck:
    """Both live players share a royal flush at showdown; folded hands stay folded."""
    holes = tuple(Card.parse(code) for code in "2c 3d 4c 5d".split())
    board = tuple(Card.parse(code) for code in "Ts Js Qs Ks As".split())
    all_cards = Deck.shuffled()
    unused = [card for card in all_cards.remaining if card not in (*holes, *board)]
    # Sorting makes all 52 positions deterministic, including unobserved cards.
    unused.sort(key=lambda card: card.code)
    cards = [*holes, unused[0], *board[:3], unused[1], board[3], unused[2], board[4], *unused[3:]]
    assert len(cards) == len(set(cards)) == 52
    return Deck(cards)


def fixture_server(database: Path, wire_path: Path, snapshot_path: Path, stop_path: Path) -> None:
    """Invoke the production composition root with audit, card-order and stop hooks."""
    wire, snapshots = JsonLineCodec(), TableSnapshotCodec()
    original_handle = TableService.handle
    original_serve = LocalTcpServer.serve_forever
    # Build before patching Deck.shuffled; each hand gets an independent deck.
    fixed_cards = tuple(fixture_deck().remaining)
    saved_revisions: set[int] = set()

    def recorded_handle(service: TableService, command: Command, session: SessionContext) -> CommandResponse:
        engine = service._engine
        assert type(engine) is HoldemEngine
        assert isinstance(engine, HoldemEngine)
        assert type(engine._betting) is NoLimitBettingRules
        assert type(engine._evaluator) is FiveCardHighEvaluator
        assert type(engine._pots) is SidePotAllocator
        response = original_handle(service, command, session)
        table = service._repository.load(service._table_id)
        assert table is not None
        if table.revision not in saved_revisions:
            with snapshot_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(snapshots.encode(table) + "\n")
            saved_revisions.add(table.revision)
        row = {
            "player_id": session.player_id, "request_json": wire.encode_command(command),
            "response_json": wire.encode_response(response), "revision": table.revision,
            "implementations": [type(engine).__name__, type(engine._betting).__name__,
                                type(engine._evaluator).__name__, type(engine._pots).__name__],
        }
        with wire_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return response

    def serve_with_stop(server: LocalTcpServer) -> None:
        done = Event()

        def stop_when_requested() -> None:
            while not done.wait(0.02):
                if stop_path.exists():
                    server.close()
                    return

        stopper = Thread(target=stop_when_requested, name="agent-verification-stop", daemon=True)
        stopper.start()
        try:
            original_serve(server)
        finally:
            done.set()
            stopper.join(timeout=5)
            assert not stopper.is_alive()

    print(json.dumps({"loaded_package": poker.__file__, "composition_root": server_main.__module__}), flush=True)
    sys.argv = ["poker-server", "--port", "0", "--db", str(database)]
    with patch.object(Deck, "shuffled", side_effect=lambda: Deck(list(fixed_cards))), \
            patch.object(TableService, "handle", new=recorded_handle), \
            patch.object(LocalTcpServer, "serve_forever", new=serve_with_stop):
        server_main()


class CapturedProcess:
    def __init__(self, label: str, command: list[str], environment: dict[str, str], cwd: Path) -> None:
        self.label, self.command = label, command
        self.runtime_pid: int | None = None
        self.process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", env=environment, cwd=cwd,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self.lines: list[str] = []
        self.pending: Queue[str | None] = Queue()
        self.reader = Thread(target=self._capture, name=f"agent-audit-{self.process.pid}", daemon=True)
        self.reader.start()

    def _capture(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            clean = line.rstrip("\r\n")
            self.lines.append(clean)
            self.pending.put(clean)
        self.pending.put(None)

    def next_line(self, deadline: float) -> str:
        try:
            line = self.pending.get(timeout=max(0.0, deadline - monotonic()))
        except Empty as error:
            raise RuntimeError(f"{self.label} output timeout: {self.lines[-20:]}") from error
        if line is None:
            raise RuntimeError(f"{self.label} exited early: {self.lines[-20:]}")
        return line

    def finish(self) -> None:
        code = self.process.wait(timeout=8)
        self.reader.join(timeout=5)
        assert not self.reader.is_alive()
        assert code == 0, (self.label, code, self.lines[-20:])

    def cleanup(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=5)
        self.reader.join(timeout=5)
        if self.process.stdout is not None:
            self.process.stdout.close()


@dataclass(frozen=True)
class RecordedRequest:
    player_id: PlayerId
    command: Command
    response: CommandResponse
    revision: int
    raw: dict[str, object]


def read_requests(path: Path) -> list[RecordedRequest]:
    codec = JsonLineCodec()
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        data = cast(dict[str, object], json.loads(line))
        assert data["implementations"] == ["HoldemEngine", "NoLimitBettingRules", "FiveCardHighEvaluator", "SidePotAllocator"]
        records.append(RecordedRequest(
            PlayerId(cast(str, data["player_id"])), codec.decode_command(cast(str, data["request_json"])),
            codec.decode_response(cast(str, data["response_json"])), cast(int, data["revision"]), data,
        ))
    return records


def verify_view(view: PlayerView, table: Table, player_id: PlayerId) -> None:
    """Check the actual wire projection against the separately recorded SQL state."""
    hand = table.hand
    assert (view.table_id, view.revision, view.me.player_id) == (table.id, table.revision, player_id)
    assert view.config == table.config
    assert view.hand_id == (hand.id if hand else None)
    assert view.phase == (hand.phase if hand else None)
    assert view.board == (tuple(card.code for card in hand.board) if hand else ())
    assert view.actor_id == (hand.betting.actor_id if hand else None)
    member = hand.players.get(player_id) if hand else None
    assert view.me.hole_cards == (tuple(card.code for card in member.hole_cards) if member else ())
    assert len(view.players) == len(table.players)
    reveal = (table.last_result.revealed_hands if hand and hand.phase is HandPhase.COMPLETE
              and table.last_result and table.last_result.hand_id == hand.id else {})
    for public in view.players:
        player = table.player(public.player_id)
        current = hand.players.get(public.player_id) if hand else None
        assert (public.name, public.seat, public.stack) == (player.name, player.seat, player.stack)
        assert public.status == (current.status if current else None)
        assert public.street_commit == (current.street_commit if current else 0)
        assert public.hand_commit == (current.hand_commit if current else 0)
        assert public.revealed_cards == tuple(card.code for card in reveal.get(public.player_id, ()))
    if view.actor_id != player_id:
        assert not view.me.legal_actions
    if hand is None:
        assert not view.action_history and not view.history_complete
        return
    assert view.history_complete and hand.history_complete
    assert view.action_history == tuple(hand.action_history)
    assert [event.sequence for event in view.action_history] == list(range(1, len(view.action_history) + 1))
    assert [event.kind for event in view.action_history[:2]] == ["small_blind", "big_blind"]
    assert view.pot_total == sum(member.hand_commit for member in hand.players.values())
    cards = hand.deck.remaining + hand.deck.burned + hand.board + [card for member in hand.players.values() for card in member.hole_cards]
    assert len(cards) == len(set(cards)) == 52
    assets = sum(table.player(pid).stack for pid in hand.players)
    if hand.phase is not HandPhase.COMPLETE:
        assets += sum(member.hand_commit for member in hand.players.values())
    assert assets == hand.chip_total_at_start == 2000


def audit_logs(output_dir: Path, run_id: str, agents: list[CapturedProcess], rows: list[RecordedRequest]) -> dict[str, object]:
    schema = json.loads((Path(__file__).resolve().parents[1] / "src/shared_logging/event.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    paths = sorted((output_dir / "logs" / run_id).glob("*.jsonl*"))
    events: list[dict[str, object]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            event = cast(dict[str, object], json.loads(line))
            validator.validate(event)
            assert event["run_id"] == run_id
            assert event["level"] not in ("ERROR", "CRITICAL"), event
            events.append(event)
    summaries: list[dict[str, object]] = []
    for agent in agents:
        joined = next(row for row in rows if isinstance(row.command, JoinCommand) and row.command.name == agent.label)
        # Windows venv python.exe can be a redirector, so Popen.pid is not
        # necessarily os.getpid() in the interpreter. Bind through the actual
        # CLI join's server-issued player identity, never infer from filenames.
        runtime_pids = {cast(int, event["pid"]) for event in events
                        if event["event"] == "client.joined"
                        and cast(dict[str, object], event["context"]).get("player_id") == joined.player_id}
        assert len(runtime_pids) == 1, (agent.label, runtime_pids)
        agent.runtime_pid = runtime_pids.pop()
        own = [event for event in events if event["pid"] == agent.runtime_pid]
        names = [event["event"] for event in own]
        for required in ("agent.started", "agent.decision.started", "agent.action.queued", "agent.action.confirmed", "agent.stopped"):
            assert required in names, (agent.label, required, names)
        accepted = sum(isinstance(row.command, ActCommand) and row.player_id == joined.player_id and row.response.ok for row in rows)
        confirmed = names.count("agent.action.confirmed")
        assert confirmed == accepted, (agent.label, confirmed, accepted)
        assert names.count("agent.action.queued") == confirmed
        stopped = next(event for event in reversed(own) if event["event"] == "agent.stopped")
        data = cast(dict[str, object], stopped["data"])
        assert data["stop_reason"] == "max_hands", data
        assert data["confirmed_actions"] == confirmed
        assert data["observed_hands"] == 2
        summaries.append({"name": agent.label, "launcher_pid": agent.process.pid,
                          "runtime_pid": agent.runtime_pid, "player_id": joined.player_id,
                          "confirmed_actions": confirmed, "decisions": names.count("agent.decision.started"),
                          "tools": names.count("agent.tool.completed"), "stop": data})
    assert any(cast(int, summary["tools"]) > 0 for summary in summaries), "The tool-using policy did not execute tools"
    server_pids = {cast(int, event["pid"]) for event in events if event["process_role"] == "server"}
    assert len(server_pids) == 1
    return {"files": len(paths), "events": len(events), "agents": summaries, "server_pid": server_pids.pop()}


def verify(output_dir: Path, *, timeout: float = 60.0, exercise_streets: bool = False) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    assert poker.__file__ is not None and Path(poker.__file__).resolve() == root / "src/poker/__init__.py"
    output_dir = output_dir.resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Choose an empty or new --output-dir to preserve prior agent evidence")
    output_dir.mkdir(parents=True, exist_ok=True)
    database, wire_path = output_dir / "final.sqlite3", output_dir / "server_commands.jsonl"
    snapshot_path, stop_path = output_dir / "arena_snapshots.jsonl", output_dir / "stop"
    run_id = f"agent-{uuid4().hex}"
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                   "APP_RUN_ID": run_id, "APP_LOG_DIR": str(output_dir / "logs"), "LOG_LEVEL": "DEBUG"}
    environment.pop("APP_LOG_ROLE", None)
    captures: list[CapturedProcess] = []
    agents: list[CapturedProcess] = []
    started = monotonic()
    server = CapturedProcess("server", [sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()),
                             "--fixture-server", "--database", str(database), "--wire-log", str(wire_path),
                             "--snapshots", str(snapshot_path), "--stop-file", str(stop_path)], environment, root)
    captures.append(server)
    try:
        deadline = monotonic() + 10
        while True:
            match = re.search(r"on \('127\.0\.0\.1', (\d+)\); table=([0-9a-f]+)", server.next_line(deadline))
            if match:
                port, table_id = int(match.group(1)), TableId(match.group(2))
                break
        participants = (("AgentTool", "scripts.agent_fixture_policy:tool_check_call", True),
                        ("AgentDistribution", "scripts.agent_fixture_policy:distribution_check_call", False)) if exercise_streets else (
                            ("AgentTool", "poker.agent.examples:tool_first", True),
                            ("AgentUniform", "poker.agent.examples:uniform", False))
        for name, policy, auto_start in participants:
            command = [sys.executable, "-u", "-X", "utf8", "-m", "poker.agent", "--port", str(port),
                       "--name", name, "--policy", policy, "--min-players", "2", "--max-hands", "2",
                       "--poll-interval", "0.05", "--decision-timeout", "5", "--seed", "7"]
            if auto_start:
                command.append("--auto-start")
            agent = CapturedProcess(name, command, environment, root)
            captures.append(agent)
            agents.append(agent)
        assert len({capture.process.pid for capture in captures}) == 3
        deadline = monotonic() + timeout
        while any(agent.process.poll() is None for agent in agents):
            for agent in agents:
                code = agent.process.poll()
                assert code in (None, 0), (agent.label, code, agent.lines[-20:])
            assert server.process.poll() is None, server.lines[-20:]
            if monotonic() >= deadline:
                raise TimeoutError(f"Agents did not finish two hands: {[(agent.label, agent.lines[-20:]) for agent in agents]}")
            Event().wait(0.02)
        for agent in agents:
            agent.finish()
        stop_path.touch()
        server.finish()
        rows = read_requests(wire_path)
        snapshots = {table.revision: table for table in (
            TableSnapshotCodec().decode(line) for line in snapshot_path.read_text(encoding="utf-8").splitlines()
        )}
        assert rows and all(row.response.ok for row in rows)
        joins = [row for row in rows if isinstance(row.command, JoinCommand)]
        assert len(joins) == 2 and len({row.player_id for row in joins}) == 2
        starts = [row for row in rows if isinstance(row.command, StartHandCommand)]
        assert len(starts) == 2
        starter = next(row.player_id for row in joins if isinstance(row.command, JoinCommand) and row.command.name == "AgentTool")
        assert all(row.player_id == starter for row in starts)
        completed: dict[str, Table] = {}
        for row in rows:
            view = row.response.view
            assert view is not None
            saved = snapshots[row.revision]
            verify_view(view, saved, row.player_id)
            raw_view = cast(dict[str, object], json.loads(cast(str, row.raw["response_json"]))["view"])
            assert not {"deck", "pending", "last_action_bet"} & raw_view.keys()
            assert all("hole_cards" not in player for player in cast(list[dict[str, object]], raw_view["players"]))
            if isinstance(row.command, ActCommand):
                assert row.command.expected_revision == row.revision - 1
                assert row.command.hand_id == view.hand_id
                assert view.action_history[-1].player_id == row.player_id
                assert view.action_history[-1].kind == row.command.action.kind.value
                previous = snapshots[row.revision - 1]
                assert previous.hand is not None
                record = view.action_history[-1]
                assert record.stack + record.pay == previous.player(row.player_id).stack
                assert record.to == previous.hand.players[row.player_id].street_commit + record.pay
                if row.command.action.kind in (ActionKind.BET_TO, ActionKind.RAISE_TO):
                    assert record.to == row.command.action.to
                if row.command.action.kind in (ActionKind.CHECK, ActionKind.FOLD):
                    assert record.pay == 0
            if saved.hand and saved.hand.phase is HandPhase.COMPLETE:
                completed[str(saved.hand.id)] = saved
        assert len(completed) == 2
        for hand_id, table in completed.items():
            assert table.hand is not None
            count = sum(isinstance(row.command, ActCommand) and row.command.hand_id == hand_id for row in rows)
            assert len(table.hand.action_history) == count + 2
            if exercise_streets:
                assert len(table.hand.board) == 5
                assert {event.phase for event in table.hand.action_history[2:]} == {
                    HandPhase.PREFLOP, HandPhase.FLOP, HandPhase.TURN, HandPhase.RIVER,
                }
        polls = {pid: sum(isinstance(row.command, StateCommand) and row.player_id == pid for row in rows) for pid in {row.player_id for row in joins}}
        assert all(count > 0 for count in polls.values())
        final = SqliteTableRepository(database).load(table_id)
        assert final is not None and final.hand is not None and final.hand.phase is HandPhase.COMPLETE
        assert final == snapshots[max(snapshots)]
        assert sum(player.stack for player in final.players) == 2000
        logging = audit_logs(output_dir, run_id, agents, rows)
        server.runtime_pid = cast(int, logging["server_pid"])
        assert len({capture.runtime_pid for capture in captures}) == 3
        for capture in captures:
            (output_dir / f"{capture.label}_stdout.log").write_text("\n".join(capture.lines) + "\n", encoding="utf-8")
        # Freeze the complete local execution chain, including protocol guards,
        # dependency metadata, queue/storage implementations and logging schema.
        # Enumerating packages also captures newly added runtime modules.
        source_packages = ("agent", "client", "application", "domain", "engine", "transport", "messaging", "persistence")
        source_paths = [path for package in source_packages
                        for path in (root / "src/poker" / package).rglob("*.py")]
        source_paths.extend((root / "src/shared_logging").rglob("*.py"))
        source_paths.extend(root / path for path in (
            "pyproject.toml", "scripts/verify_agent_runtime.py", "src/poker/__init__.py", "src/poker/bootstrap.py",
            "src/poker/persistence/schema.sql", "src/shared_logging/event.schema.json",
        ))
        if exercise_streets:
            source_paths.append(root / "scripts/agent_fixture_policy.py")
        source_paths = sorted(set(source_paths))
        receipt = {
            "scope": "Two independent production agent entrypoints, each using PokerCli, and an independent production server_main process",
            "timestamp_utc": datetime.now(UTC).isoformat(), "run_id": run_id,
            "elapsed_seconds": round(monotonic() - started, 3), "table_id": table_id,
            "controls": ["fixed 52-card order with royal-flush board to keep both players funded", "request and SQL snapshot audit", "graceful server stop watcher"],
            "policy_strength_claim": False, "provider_calls": False,
            "policy_scenario": "verification-only check/call tools and distribution" if exercise_streets else "shipped tool_first and uniform examples",
            "processes": [{"label": capture.label, "launcher_pid": capture.process.pid,
                           "runtime_pid": capture.runtime_pid, "command": capture.command,
                           "exit_code": capture.process.returncode} for capture in captures],
            "completed_hands": [{"hand_id": hand_id, "revision": table.revision,
                                  "action_count": len(table.hand.action_history) - 2 if table.hand else 0,
                                  "stacks": {player.name: player.stack for player in table.players}}
                                 for hand_id, table in completed.items()],
            "request_count": len(rows), "automatic_state_counts": polls, "logging": logging,
            "final_snapshot": json.loads(TableSnapshotCodec().encode(final)),
            "source_sha256": {str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
            "evidence_sha256": {str(path.relative_to(output_dir)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted(output_dir.rglob("*")) if path.is_file()},
            "checks": {"three_distinct_processes": True, "two_distinct_players": True,
                       "all_views_match_sqlite_revision": True, "private_cards_isolated": True,
                       "complete_history_in_all_hand_views": True, "two_hands_completed": True,
                       "four_streets_each_hand": exercise_streets,
                       "guarded_actions_confirmed_once": True, "analysis_tools_executed": True,
                       "single_auto_start_owner": True, "chips_conserved_2000": True,
                       "cards_partition_52": True, "sqlite_reopened": True,
                       "structured_logs_validated": True, "all_processes_exit_zero": True},
        }
        (output_dir / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"receipt": str(output_dir / "receipt.json"), "completed_hands": 2, "requests": len(rows),
                "confirmed_actions": sum(isinstance(row.command, ActCommand) for row in rows),
                "process_exit_codes": [capture.process.returncode for capture in captures], "logging": logging}
    finally:
        if server.process.poll() is None:
            stop_path.touch()
            try:
                server.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        for capture in reversed(captures):
            capture.cleanup()
            (output_dir / f"{capture.label}_stdout.log").write_text("\n".join(capture.lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".data/verification/agent-runtime"))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--exercise-streets", action="store_true", help="Use verification-only check/call plugins to cover all four streets")
    parser.add_argument("--fixture-server", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--database", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--wire-log", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--snapshots", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--stop-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.fixture_server:
        assert args.database is not None and args.wire_log is not None and args.snapshots is not None and args.stop_file is not None
        fixture_server(args.database, args.wire_log, args.snapshots, args.stop_file)
    else:
        if not 0 < args.timeout <= 600:
            parser.error("--timeout must be between 0 and 600 seconds")
        print(json.dumps(verify(args.output_dir, timeout=args.timeout, exercise_streets=args.exercise_streets), ensure_ascii=False))


if __name__ == "__main__":
    main()
