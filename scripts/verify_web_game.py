"""G05: a real browser player and two installed CLI processes, with reproducible fixtures.

Run this driver, then use the actual web UI at the URL in status.json. The driver
controls only the two CLI players. Each browser checkpoint waits for a real GUI
command, never submits an HTTP poker action on the browser player's behalf.
"""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
from threading import Event, Thread
from time import monotonic
from typing import cast
from unittest.mock import patch
from uuid import uuid4

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from jsonschema import Draft202012Validator, FormatChecker
from fastapi import FastAPI
import uvicorn

from poker.application.commands import ActCommand, StartHandCommand, StateCommand
from poker.application.views import CommandResponse, PlayerView
from poker.client.parser import CommandParser
from poker.domain.models import Table
from poker.domain.types import HandPhase, PlayerId, TableId
from poker.gui import bootstrap as gui_bootstrap
from poker.persistence.snapshot import TableSnapshotCodec
from poker.persistence.sqlite import SqliteTableRepository
from poker.transport.codec import JsonLineCodec
from scripts.verify_local_game import CapturedProcess, read_requests, verify_view


def fixture_bridge(port: int, game_port: int, stop_file: Path, host: str = "127.0.0.1") -> None:
    def serve(app: FastAPI, **options: object) -> None:
        server = uvicorn.Server(uvicorn.Config(app, host=str(options["host"]), port=port, log_config=None))
        done = Event()

        def stop_when_requested() -> None:
            while not done.wait(0.05):
                if stop_file.exists():
                    server.should_exit = True
                    return

        thread = Thread(target=stop_when_requested, daemon=True)
        thread.start()
        try:
            server.run()
        finally:
            done.set()
            thread.join(timeout=5)
            assert not thread.is_alive()

    sys.argv = ["poker-gui-bridge", "--host", host, "--port", str(port), "--game-port", str(game_port)]
    with patch.object(uvicorn, "run", new=serve):
        gui_bootstrap.main()


def verify(output_dir: Path, host: str = "127.0.0.1") -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Use a fresh evidence directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    database, wire = output_dir / "final.sqlite3", output_dir / "server_commands.jsonl"
    stop_server, stop_bridge = output_dir / "stop-server", output_dir / "stop-bridge"
    run_id = f"g05-{uuid4().hex}"
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
           "APP_RUN_ID": run_id, "APP_LOG_DIR": str(output_dir / "logs"), "LOG_LEVEL": "DEBUG"}
    entry = Path(sys.executable).with_name("poker-client.exe" if os.name == "nt" else "poker-client")
    assert entry.is_file()
    captures: list[CapturedProcess] = []
    clients: list[CapturedProcess] = []
    checkpoints: list[dict[str, object]] = []
    hands: list[dict[str, object]] = []
    origin = monotonic()
    codec = JsonLineCodec()
    snapshot_codec = TableSnapshotCodec()

    def status(label: str, line: str | None = None, view: PlayerView | None = None) -> None:
        row: dict[str, object] = {"checkpoint": label, "browser_command": line,
                                  "seconds": round(monotonic() - origin, 3), "url": f"http://{host}:{http_port}"}
        if view is not None:
            row["view"] = json.loads(codec.encode_response(CommandResponse(view=view)))["view"]
        temporary = output_dir / "status.tmp"
        temporary.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output_dir / "status.json")
        checkpoints.append(row)
        print(json.dumps({"checkpoint": label, "browser_command": line, "url": row["url"]}, ensure_ascii=False), flush=True)

    server = CapturedProcess([sys.executable, "-u", "-X", "utf8", str(root / "scripts/verify_local_game.py"),
                              "--fixture-server", "--database", str(database), "--wire-log", str(wire),
                              "--stop-file", str(stop_server)], env)
    captures.append(server)
    try:
        deadline = monotonic() + 10
        while True:
            match = re.search(r"on \('127\.0\.0\.1', (\d+)\); table=([0-9a-f]+)", server.next_line(deadline))
            if match is not None:
                game_port, table_id = int(match.group(1)), TableId(match.group(2))
                break
        with socket.socket() as available:
            available.bind((host, 0))
            http_port = available.getsockname()[1]

        def current() -> Table:
            value = SqliteTableRepository(database).load(table_id)
            assert value is not None
            return value

        def wait_revision(revision: int, *, timeout: float = 600) -> Table:
            until = monotonic() + timeout
            while monotonic() < until:
                value = current()
                if value.revision == revision:
                    return value
                assert value.revision < revision, (value.revision, revision)
                assert all(p.process.poll() is None for p in captures), "A child exited early"
                Event().wait(0.02)
            raise RuntimeError(f"Revision {revision} not reached")

        for index, name in enumerate(("甲", "乙")):
            cli = CapturedProcess([str(entry), "--port", str(game_port)], env)
            clients.append(cli)
            captures.append(cli)
            cli.enter(f"join {name}")
            value = wait_revision(index + 1, timeout=10)
            joined = next(row.response.view for row in reversed(read_requests(wire))
                          if row.response.view is not None and row.response.view.revision == value.revision)
            cli.wait_frame(joined)
        bridge = CapturedProcess([sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()),
                                  "--fixture-bridge", "--host", host, "--port", str(http_port), "--game-port", str(game_port),
                                  "--stop-file", str(stop_bridge)], env)
        captures.append(bridge)
        status("join_browser", "join 丙")
        value = wait_revision(3)
        assert [p.name for p in value.players] == ["甲", "乙", "丙"]
        ids = [p.id for p in value.players]

        def synchronize(table: Table) -> PlayerView:
            views: dict[PlayerId, PlayerView] = {}
            until = monotonic() + 10
            while monotonic() < until:
                for row in read_requests(wire):
                    view = row.response.view
                    if view is not None and view.revision == table.revision:
                        views[view.me.player_id] = view
                if set(views) == set(ids):
                    break
                Event().wait(0.02)
            assert set(views) == set(ids), (table.revision, set(views))
            for pid, view in views.items():
                verify_view(view, table, pid)
            for index, cli in enumerate(clients):
                cli.wait_frame(views[ids[index]])
            return views[ids[2]]

        def action(seat: int, line: str, label: str) -> Table:
            before = current()
            gui_view = synchronize(before)
            after_row = len(read_requests(wire))
            if seat == 2:
                status(label, line, gui_view)
            else:
                clients[seat].enter(line)
            table = wait_revision(before.revision + 1)
            updates = [r for r in read_requests(wire)[after_row:] if not isinstance(r.command, StateCommand)]
            assert len(updates) == 1 and updates[0].response.ok and updates[0].player_id == ids[seat], updates
            expected = CommandParser().parse(line, before.hand.id if before.hand else None)
            assert updates[0].command == expected
            synchronize(table)
            return table

        def completed(table: Table, stacks: list[int], label: str) -> None:
            assert table.hand is not None and table.hand.phase is HandPhase.COMPLETE and table.last_result is not None
            assert [p.stack for p in table.players] == stacks
            hands.append({"hand_id": table.hand.id, "stacks": stacks, "result": json.loads(snapshot_codec.encode(table))["last_result"]})
            status(label, "start", synchronize(table))

        action(0, "start", "start_hand1")
        sequence = [(0, "raise_to 20"), (1, "call"), (2, "call"), (1, "bet_to 20"), (2, "call"),
                    (0, "fold"), (1, "bet_to 10"), (2, "call"), (1, "bet_to 10"), (2, "call")]
        for index, (seat, line) in enumerate(sequence):
            value = action(seat, line, f"hand1_step{index + 1}")
        completed(value, [980, 940, 1080], "hand1_complete")
        action(2, "start", "start_hand2")
        for seat, line in ((1, "all_in"), (2, "all_in"), (0, "all_in")):
            value = action(seat, line, f"hand2_all_in_P{seat}")
        completed(value, [80, 2820, 100], "hand2_complete")
        assert value.last_result is not None and [a.pot.amount for a in value.last_result.awards] == [2820, 80]
        assert {s.player_id: s.amount for s in value.last_result.refunds} == {ids[2]: 100}
        action(2, "start", "start_hand3")
        action(2, "fold", "hand3_fold")
        value = action(0, "fold", "hand3_finish")
        completed(value, [75, 2825, 100], "hand3_complete")
        assert value.hand is not None and value.hand.board == []
        assert value.last_result is not None and value.last_result.revealed_hands == {}
        before_idle = snapshot_codec.encode(value)
        baseline = len(read_requests(wire))
        until = monotonic() + 10
        while monotonic() < until:
            counts = {pid: sum(isinstance(r.command, StateCommand) and r.player_id == pid
                               for r in read_requests(wire)[baseline:]) for pid in ids}
            if all(count >= 2 for count in counts.values()):
                break
            Event().wait(0.02)
        assert all(count >= 2 for count in counts.values())
        assert snapshot_codec.encode(current()) == before_idle
        value = action(2, "start", "start_hand4")
        assert value.hand is not None and value.hand.phase is HandPhase.PREFLOP
        assert [p.stack for p in value.players] == [75, 2820, 90]
        assert sum(m.hand_commit for m in value.hand.players.values()) == 15
        status("hand4_started_close_browser", "close", synchronize(value))

        def log_rows() -> list[dict[str, object]]:
            return [json.loads(line) for path in (output_dir / "logs" / run_id).glob("*.jsonl*")
                    for line in path.read_text(encoding="utf-8").splitlines() if line.endswith("}")]

        until = monotonic() + 600
        while monotonic() < until:
            if any(r["event"] == "gui.session_closed" for r in log_rows()):
                break
            Event().wait(0.05)
        else:
            raise RuntimeError("Close the actual browser session to finish")
        for cli in clients:
            cli.enter("quit")
            cli.finish()
        stop_bridge.touch()
        bridge.finish()
        stop_server.touch()
        server.finish()
        for index, cli in enumerate(clients):
            (output_dir / f"cli_P{index}.log").write_text("\n".join(cli.lines) + "\n", encoding="utf-8")
        (output_dir / "server_stdout.log").write_text("\n".join(server.lines) + "\n", encoding="utf-8")
        (output_dir / "bridge_stdout.log").write_text("\n".join(bridge.lines) + "\n", encoding="utf-8")
        rows = read_requests(wire)
        assert all(r.response.ok for r in rows)
        assert sum(isinstance(r.command, ActCommand) for r in rows) == 15
        assert sum(isinstance(r.command, StartHandCommand) for r in rows) == 4
        validator = Draft202012Validator(json.loads((root / "src/shared_logging/event.schema.json").read_text(encoding="utf-8")),
                                         format_checker=FormatChecker())
        logs = log_rows()
        for row in logs:
            validator.validate(row)
            assert row["run_id"] == run_id
        client_correlations = {cast(dict[str, object], r["context"])["correlation_id"] for r in logs if r["event"] == "client.transport_sent"}
        server_correlations = {cast(dict[str, object], r["context"])["correlation_id"] for r in logs if r["event"] == "command.received"}
        assert client_correlations == server_correlations and len(client_correlations) == len(rows)
        assert {"server", "cli", "gui"} == {r["process_role"] for r in logs}
        assert len({c.process.pid for c in captures}) == 4
        assert not any(str(key).lower() in {"hole_cards", "deck", "session_id"}
                       for r in logs for section in ("context", "data") for key in cast(dict[str, object], r[section]))
        for file in (stop_server, stop_bridge):
            file.unlink()
        source_files = [p for base in (root / "src/poker", root / "src/shared_logging", root / "scripts")
                        for p in base.rglob("*.py") if "node_modules" not in p.parts]
        status("complete")
        receipt: dict[str, object] = {
            "scope": "G05 + L02/L04: browser GUI player + two installed CLI processes + real server and bridge",
            "network": {"http_host": host, "http_port": http_port, "game_host": "127.0.0.1", "game_port": game_port,
                        "browser_url": f"http://{host}:{http_port}", "second_host_tested": False},
            "date": datetime.now().date().isoformat(), "run_id": run_id, "elapsed_seconds": round(monotonic() - origin, 3),
            "controls": ["fixed four deck orders", "wire audit", "graceful server/bridge shutdown watchers"],
            "processes": [{"pid": c.process.pid, "command": c.command, "stdin": c.inputs, "exit_code": c.process.returncode} for c in captures],
            "completed_hands": hands, "checkpoints": checkpoints, "request_count": len(rows),
            "actions": 15, "starts": 4, "final_revision": value.revision,
            "final_stacks": [p.stack for p in value.players], "current_pot": 15, "chip_total": 3000,
            "logging": {"files": len(list((output_dir / "logs" / run_id).glob("*.jsonl*"))), "events": len(logs),
                        "linked_commands": len(client_correlations), "event_counts": dict(Counter(str(r["event"]) for r in logs))},
            "checks": {"all_four_processes_exit_zero": True, "private_views_match_sqlite": True,
                       "normal_four_streets": True, "main_and_side_different_winners": True,
                       "uncalled_refund": True, "uncontested_without_reveal": True,
                       "idle_complete_does_not_pay_twice": True, "explicit_hand4": True,
                       "each_request_correlated_across_processes": True, "all_logs_match_schema": True},
            "source_sha256": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
            "evidence_sha256": {p.relative_to(output_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in output_dir.rglob("*") if p.is_file()},
        }
        (output_dir / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"receipt": str(output_dir / "receipt.json"), "hands": 3, "actions": 15, "logging_events": len(logs)}
    finally:
        if server.process.poll() is None:
            stop_server.touch()
        if "bridge" in locals() and bridge.process.poll() is None:
            stop_bridge.touch()
        for capture in reversed(captures):
            capture.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".data/verification/G05"))
    parser.add_argument("--fixture-bridge", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--game-port", type=int)
    parser.add_argument("--host", default="127.0.0.1", help="HTTP interface to test; use the machine's LAN IP")
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    if args.fixture_bridge:
        assert args.port is not None and args.game_port is not None and args.stop_file is not None
        fixture_bridge(args.port, args.game_port, args.stop_file, args.host)
    else:
        print(json.dumps(verify(args.output_dir, args.host), ensure_ascii=False))


if __name__ == "__main__":
    main()
