"""I01: one real server process and three installed CLI processes, with an evidence bundle."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from random import Random
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import monotonic
from typing import cast
from unittest.mock import patch

import poker
from poker.application.commands import ActCommand, Command, SessionContext, StartHandCommand, StateCommand
from poker.application.service import TableService
from poker.application.views import CommandResponse, PlayerView
from poker.bootstrap import server_main
from poker.client.cli import PokerCli
from poker.domain.cards import Card, Deck
from poker.domain.models import Table
from poker.domain.types import HandPhase, PlayerId, PlayerStatus, TableId
from poker.engine.holdem import HoldemEngine
from poker.engine.policies import FiveCardHighEvaluator, NoLimitBettingRules, SidePotAllocator
from poker.persistence.snapshot import TableSnapshotCodec
from poker.persistence.sqlite import SqliteTableRepository
from poker.transport.codec import JsonLineCodec
from poker.transport.local_tcp import LocalTcpServer


def fixture_decks() -> tuple[Deck, ...]:
    """Control only card order; stacks, blinds, actions, ranking and payments are real."""
    pairs = (
        ("Qh Qd", "Kh Kd", "Ah Ad"),
        ("Kh Kd", "Ah Ad", "Qh Qd"),
        ("2h 4h", "6d 8d", "Tc Qc"),
        ("3h 5h", "7d 9d", "Js Ks"),
    )
    decks = []
    board = tuple(Card.parse(code) for code in "2c 3d 7h 9s Jc".split())
    for hand_index, codes in enumerate(pairs):
        holes = [tuple(Card.parse(code) for code in pair.split()) for pair in codes]
        used = set(board).union(card for pair in holes for card in pair)
        unused = [c for c in Deck.shuffled(Random(101 + hand_index)).remaining if c not in used]
        button = hand_index % 3
        order = [(button + offset) % 3 for offset in (1, 2, 3)]
        prefix = [holes[seat][circle] for circle in range(2) for seat in order]
        cards = prefix + [unused[0], *board[:3], unused[1], board[3], unused[2], board[4]] + unused[3:]
        assert len(cards) == len(set(cards)) == 52
        decks.append(Deck(cards))
    return tuple(decks)


def fixture_server(database: Path, wire_log: Path, stop_file: Path) -> None:
    """Launch the production composition root, instrumenting only fixtures/audit/stop."""
    decks = iter(fixture_decks())
    wire = JsonLineCodec()
    origin = monotonic()
    original_handle = TableService.handle
    original_serve = LocalTcpServer.serve_forever

    def recorded_handle(service: TableService, command: Command, session: SessionContext) -> CommandResponse:
        engine = service._engine
        assert type(engine) is HoldemEngine
        assert isinstance(engine, HoldemEngine)
        assert type(engine._betting) is NoLimitBettingRules
        assert type(engine._evaluator) is FiveCardHighEvaluator
        assert type(engine._pots) is SidePotAllocator
        response = original_handle(service, command, session)
        row = {
            "seconds": round(monotonic() - origin, 6), "player_id": session.player_id,
            "request_json": wire.encode_command(command), "response_json": wire.encode_response(response),
            "implementations": [type(engine).__name__, type(engine._betting).__name__,
                                type(engine._evaluator).__name__, type(engine._pots).__name__],
        }
        with wire_log.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return response

    def serve_with_stop(server: LocalTcpServer) -> None:
        done = Event()

        def stop_when_requested() -> None:
            while not done.wait(0.05):
                if stop_file.exists():
                    server.close()
                    return

        stopper = Thread(target=stop_when_requested, name="i01-stop", daemon=True)
        stopper.start()
        try:
            original_serve(server)
        finally:
            done.set()
            stopper.join(timeout=5)
            assert not stopper.is_alive()

    wire_log.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"loaded_package": poker.__file__, "server_main_module": server_main.__module__,
                      "controls": ["fixed Deck.shuffled", "request audit", "graceful stop file"]}))
    sys.argv = ["poker-server", "--port", "0", "--db", str(database)]
    with patch.object(Deck, "shuffled", side_effect=lambda: next(decks)), \
            patch.object(TableService, "handle", new=recorded_handle), \
            patch.object(LocalTcpServer, "serve_forever", new=serve_with_stop):
        server_main()


@dataclass(frozen=True)
class RecordedRequest:
    seconds: float
    player_id: PlayerId | None
    command: Command
    response: CommandResponse
    raw: dict[str, object]


def read_requests(path: Path) -> list[RecordedRequest]:
    if not path.exists():
        return []
    codec = JsonLineCodec()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if not line.endswith("\n"):
            continue  # The sole writer may still be appending the final record.
        raw = cast(dict[str, object], json.loads(line))
        assert raw["implementations"] == ["HoldemEngine", "NoLimitBettingRules",
                                          "FiveCardHighEvaluator", "SidePotAllocator"]
        pid = cast(str | None, raw["player_id"])
        rows.append(RecordedRequest(
            cast(float, raw["seconds"]), PlayerId(pid) if pid is not None else None,
            codec.decode_command(cast(str, raw["request_json"])),
            codec.decode_response(cast(str, raw["response_json"])), raw,
        ))
    return rows


class CapturedProcess:
    """Own only this subprocess's stdin/stdout; every poker request is sent by its CLI."""

    def __init__(self, command: list[str], environment: dict[str, str]) -> None:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.command = command
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", env=environment, creationflags=flags,
        )
        self.lines: list[str] = []
        self.inputs: list[str] = []
        self.pending: Queue[str | None] = Queue()
        self.frame: list[str] = []
        self.frame_revision = -1
        self.reader = Thread(target=self._capture, name=f"output-{self.process.pid}", daemon=True)
        self.reader.start()

    def _capture(self) -> None:
        stdout = self.process.stdout
        assert stdout is not None
        for line in stdout:
            clean = line.rstrip("\r\n")
            self.lines.append(clean)
            self.pending.put(clean)
        self.pending.put(None)

    def enter(self, line: str) -> None:
        assert self.process.stdin is not None and self.process.poll() is None
        self.inputs.append(line)
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def next_line(self, deadline: float) -> str:
        try:
            line = self.pending.get(timeout=max(0, deadline - monotonic()))
        except Empty as error:
            raise RuntimeError(f"Output timeout for PID {self.process.pid}: {self.lines[-25:]!r}") from error
        if line is None:
            raise RuntimeError(f"Process {self.process.pid} exited early: {self.lines[-25:]!r}")
        if line.startswith(("服务器拒绝", "输入错误", "连接结束")):
            raise RuntimeError(f"CLI rejected a step: {line}; inputs={self.inputs!r}")
        return line

    def wait_frame(self, view: PlayerView) -> list[str]:
        expected = PokerCli._render(view).lstrip("\n").splitlines()
        deadline = monotonic() + 8
        while self.frame_revision < view.revision or len(self.frame) < len(expected):
            line = self.next_line(deadline)
            match = re.fullmatch(r"牌桌: .+ \| 版本: (\d+)", line)
            if match is not None:
                self.frame_revision = int(match.group(1))
                self.frame = [line]
            elif line and self.frame:
                self.frame.append(line)
        assert self.frame_revision == view.revision, (self.frame_revision, view.revision)
        assert self.frame == expected, (self.frame, expected)
        return list(self.frame)

    def finish(self) -> None:
        assert self.process.wait(timeout=5) == 0
        self.reader.join(timeout=5)
        assert not self.reader.is_alive()

    def cleanup(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=5)
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        self.reader.join(timeout=5)
        if self.process.stdout is not None:
            self.process.stdout.close()


def verify_view(view: PlayerView, table: Table, player_id: PlayerId) -> None:
    """Independently check card visibility and chips against the persisted aggregate."""
    hand = table.hand
    assert view.table_id == table.id and view.revision == table.revision
    assert view.me.player_id == player_id
    member = hand.players.get(player_id) if hand is not None else None
    expected_holes = tuple(c.code for c in member.hole_cards) if member is not None else ()
    assert view.me.hole_cards == expected_holes
    assert view.hand_id == (hand.id if hand is not None else None)
    assert view.phase == (hand.phase if hand is not None else None)
    assert view.board == (tuple(c.code for c in hand.board) if hand is not None else ())
    assert view.actor_id == (hand.betting.actor_id if hand is not None else None)
    assert len(view.players) == len(table.players)
    reveal = (
        table.last_result.revealed_hands if table.last_result is not None and hand is not None
        and table.last_result.hand_id == hand.id and hand.phase is HandPhase.COMPLETE else {}
    )
    for public in view.players:
        player = table.player(public.player_id)
        current = hand.players.get(public.player_id) if hand is not None else None
        assert (public.seat, public.name, public.stack) == (player.seat, player.name, player.stack)
        assert public.status == (current.status if current is not None else None)
        assert public.street_commit == (current.street_commit if current is not None else 0)
        assert public.hand_commit == (current.hand_commit if current is not None else 0)
        assert public.revealed_cards == tuple(c.code for c in reveal.get(public.player_id, ()))
    if view.actor_id != player_id:
        assert not view.me.legal_actions
    if hand is not None:
        cards = hand.deck.remaining + hand.deck.burned + hand.board + [
            card for m in hand.players.values() for card in m.hole_cards
        ]
        assert len(cards) == len(set(cards)) == 52
        assert view.pot_total == sum(m.hand_commit for m in hand.players.values())
        assets = sum(table.player(pid).stack for pid in hand.players)
        if hand.phase is not HandPhase.COMPLETE:
            assets += sum(m.hand_commit for m in hand.players.values())
        assert assets == hand.chip_total_at_start == 3000


def verify(output_dir: Path) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    assert poker.__file__ is not None and Path(poker.__file__).resolve() == root / "src/poker/__init__.py"
    entry = Path(sys.executable).with_name("poker-client.exe" if os.name == "nt" else "poker-client")
    assert entry.is_file(), "Install the project's editable package first"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Choose a fresh --output-dir to preserve prior I01 evidence")
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    captures: list[CapturedProcess] = []
    clients: list[CapturedProcess] = []
    checkpoints: list[dict[str, object]] = []
    hands: list[dict[str, object]] = []
    codec = TableSnapshotCodec()
    origin = monotonic()
    with TemporaryDirectory(prefix="holdem-i01-") as directory:
        working = Path(directory).resolve()
        database = working / "poker.sqlite3"
        wire_log = working / "server_commands.jsonl"
        stop_file = working / "stop"
        server_command = [sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()),
                          "--fixture-server", "--database", str(database), "--wire-log", str(wire_log),
                          "--stop-file", str(stop_file)]
        server = CapturedProcess(server_command, environment)
        captures.append(server)
        try:
            deadline = monotonic() + 8
            while True:
                ready = server.next_line(deadline)
                match = re.search(r"on \('127\.0\.0\.1', (\d+)\); table=([0-9a-f]+)", ready)
                if match is not None:
                    port, table_id = int(match.group(1)), TableId(match.group(2))
                    break

            def snapshot() -> Table:
                saved = SqliteTableRepository(database).load(table_id)
                assert saved is not None
                return saved

            def await_revision(revision: int) -> Table:
                deadline = monotonic() + 8
                while monotonic() < deadline:
                    saved = snapshot()
                    if saved.revision == revision:
                        return saved
                    assert saved.revision < revision, (saved.revision, revision)
                    assert all(c.process.poll() is None for c in captures), "A process exited"
                    Event().wait(0.02)
                raise RuntimeError(f"SQLite revision {revision} not reached; CLI output={[c.lines[-15:] for c in clients]!r}")

            def observed_view(player_id: PlayerId, revision: int) -> PlayerView:
                deadline = monotonic() + 8
                while monotonic() < deadline:
                    for row in reversed(read_requests(wire_log)):
                        view = row.response.view
                        if row.player_id == player_id and view is not None and view.revision == revision:
                            return view
                    Event().wait(0.02)
                raise RuntimeError(f"Player {player_id} did not query revision {revision}")

            def checkpoint(label: str) -> Table:
                saved = snapshot()
                frames = []
                for seat, client in enumerate(clients):
                    player = next(p for p in saved.players if p.seat == seat)
                    view = observed_view(player.id, saved.revision)
                    verify_view(view, saved, player.id)
                    frame = client.wait_frame(view)
                    # These checks inspect the real text slot, not substring matches in UUIDs.
                    assert next(line for line in frame if line.startswith("本人底牌: ")) == \
                        "本人底牌: " + (" ".join(view.me.hole_cards) or "无")
                    for public in view.players:
                        public_line = next(line for line in frame if line.startswith(f"{public.seat} | "))
                        assert public_line.rsplit(" | ", 1)[1] == (" ".join(public.revealed_cards) or "无")
                    frames.append({"seat": seat, "player_id": player.id, "frame": frame})
                checkpoints.append({"label": label, "revision": saved.revision,
                                    "snapshot": json.loads(codec.encode(saved)), "cli_frames": frames})
                assert codec.encode(snapshot()) == codec.encode(saved), "Polling changed SQLite"
                return saved

            def send(seat: int, line: str, label: str) -> Table:
                before = snapshot()
                if line not in ("start",) and not line.startswith("join "):
                    assert before.hand is not None and before.hand.betting.actor_id is not None
                    assert before.player(before.hand.betting.actor_id).seat == seat
                clients[seat].enter(line)
                await_revision(before.revision + 1)
                return checkpoint(label)

            def finish_hand(label: str, stacks: list[int], pots: list[int], refund: tuple[int, int] | None) -> Table:
                saved = checkpoint(label)
                assert saved.hand is not None and saved.last_result is not None
                assert saved.hand.phase is HandPhase.COMPLETE
                assert [p.stack for p in sorted(saved.players, key=lambda p: p.seat)] == stacks
                assert [a.pot.amount for a in saved.last_result.awards] == pots
                if refund is None:
                    assert not saved.last_result.refunds
                else:
                    seat, amount = refund
                    assert [(saved.player(r.player_id).seat, r.amount) for r in saved.last_result.refunds] == [(seat, amount)]
                hands.append({"scenario": label, "hand_id": saved.hand.id, "stacks": stacks,
                              "result": json.loads(codec.encode(saved))["last_result"]})
                print(json.dumps({"completed": label, "revision": saved.revision, "stacks": stacks,
                                  "pots": pots, "refund": refund}, ensure_ascii=False), flush=True)
                return saved

            for seat, name in enumerate(("甲", "乙", "丙")):
                client = CapturedProcess([str(entry), "--port", str(port)], environment)
                captures.append(client)
                clients.append(client)
                send(seat, f"join {name}", f"join_{seat}")
            assert len({c.process.pid for c in captures}) == 4
            send(0, "start", "hand1_start")
            for seat, line, label in (
                (0, "raise_to 20", "preflop_raise"), (1, "call", "preflop_call_1"),
                (2, "call", "flop"), (1, "bet_to 20", "flop_bet"),
                (2, "call", "flop_call"), (0, "fold", "turn"),
                (1, "bet_to 10", "turn_bet"), (2, "call", "river"),
                (1, "bet_to 10", "river_bet"), (2, "call", "showdown"),
            ):
                send(seat, line, label)
            first = finish_hand("normal_multi_street", [980, 940, 1080], [60, 80], None)
            assert first.last_result is not None
            assert all(first.player(s.player_id).seat == 2 for a in first.last_result.awards for s in a.shares)
            assert set(first.last_result.revealed_hands) == {p.id for p in first.players if p.seat in (1, 2)}
            send(0, "start", "hand2_start")
            for seat in (1, 2, 0):
                send(seat, "all_in", f"all_in_{seat}")
            second = finish_hand("all_ins_side_pot_refund", [80, 2820, 100], [2820, 80], (2, 100))
            assert second.last_result is not None
            assert [[second.player(s.player_id).seat for s in a.shares] for a in second.last_result.awards] == [[1], [0]]
            send(1, "start", "hand3_start")
            send(2, "fold", "fold_2")
            send(0, "fold", "fold_0")
            third = finish_hand("consecutive_folds", [75, 2825, 100], [10], (1, 5))
            assert third.hand is not None and third.last_result is not None
            assert third.hand.board == [] and third.last_result.revealed_hands == {}

            # Keep all three CLIs idle and require two more autonomous reads each.
            before_idle = codec.encode(third)
            after = len(read_requests(wire_log))
            pids = {p.id for p in third.players}
            deadline = monotonic() + 8
            idle_counts: dict[PlayerId, int] = {}
            while monotonic() < deadline:
                rows = read_requests(wire_log)[after:]
                idle_counts = {pid: sum(isinstance(r.command, StateCommand) and r.player_id == pid for r in rows) for pid in pids}
                if all(count >= 2 for count in idle_counts.values()):
                    break
                Event().wait(0.02)
            assert all(count >= 2 for count in idle_counts.values())
            assert codec.encode(snapshot()) == before_idle, "Idle polling paid or advanced a completed hand"
            final = send(2, "start", "explicit_hand4")
            assert final.hand is not None and final.last_result is not None
            assert final.hand.id != third.hand.id and final.last_result.hand_id == third.hand.id
            assert final.hand.phase is HandPhase.PREFLOP and final.revision == 22
            assert [p.stack for p in final.players] == [75, 2820, 90]
            assert sum(m.hand_commit for m in final.hand.players.values()) == 15
            for client in clients:
                assert not any(line.strip() == "state" for line in client.inputs)
                client.enter("quit")
            for client in clients:
                client.finish()
            stop_file.touch()
            server.finish()

            rows = read_requests(wire_log)
            assert all(row.response.ok for row in rows)
            actions = [row for row in rows if isinstance(row.command, ActCommand)]
            starts = [row for row in rows if isinstance(row.command, StartHandCommand)]
            assert len(actions) == 15 and len(starts) == 4
            for row in actions:
                assert isinstance(row.command, ActCommand) and row.response.view is not None
                assert row.command.hand_id == row.response.view.hand_id
            polls = {p.id: sum(isinstance(r.command, StateCommand) and r.player_id == p.id for r in rows) for p in final.players}
            assert all(count > 0 for count in polls.values())
            observer = next(p.id for p in final.players if p.seat == 0)
            assert {HandPhase.TURN, HandPhase.RIVER} <= {
                r.response.view.phase for r in rows if r.player_id == observer
                and isinstance(r.command, StateCommand) and r.response.view is not None
            }
            for row in rows:
                payload = cast(dict[str, object], json.loads(cast(str, row.raw["response_json"])))
                raw_view = cast(dict[str, object], payload["view"])
                assert not {"deck", "pending", "last_action_bet"} & raw_view.keys()
                assert all("hole_cards" not in p for p in cast(list[dict[str, object]], raw_view["players"]))
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(database, output_dir / "final.sqlite3")
            shutil.copy2(wire_log, output_dir / "server_commands.jsonl")
            for index, client in enumerate(clients):
                (output_dir / f"cli_P{index}.log").write_text("\n".join(client.lines) + "\n", encoding="utf-8")
            (output_dir / "server_stdout.log").write_text("\n".join(server.lines) + "\n", encoding="utf-8")
            readback = SqliteTableRepository(output_dir / "final.sqlite3").load(table_id)
            assert readback is not None and codec.encode(readback) == codec.encode(final)
            evidence_hashes = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in output_dir.iterdir() if path.is_file()
            }
            source_paths = [root / "src/poker" / name for name in (
                "bootstrap.py", "client/cli.py", "client/parser.py", "engine/holdem.py", "engine/policies.py",
                "application/service.py", "application/views.py", "transport/local_tcp.py", "transport/codec.py",
                "messaging/local.py", "persistence/sqlite.py", "persistence/snapshot.py", "domain/models.py", "domain/cards.py",
            )]
            receipt = {
                "scope": "I01: three independent installed CLI processes + independent production server_main process",
                "date": datetime.now().date().isoformat(), "elapsed_seconds": round(monotonic() - origin, 3),
                "controls": ["fixed four 52-card deck orders", "request audit wrapper", "graceful stop watcher"],
                "starting_stacks": [1000, 1000, 1000], "table_id": table_id,
                "processes": [{"pid": c.process.pid, "command": c.command, "stdin": c.inputs,
                               "exit_code": c.process.returncode} for c in captures],
                "completed_hands": hands, "checkpoints": checkpoints,
                "request_count": len(rows), "action_count": len(actions),
                "automatic_state_counts": polls, "idle_complete_state_counts": idle_counts,
                "final_snapshot": json.loads(codec.encode(final)), "final_revision": final.revision,
                "evidence_sha256": evidence_hashes,
                "source_sha256": {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                "checks": {
                    "four_distinct_processes": True, "three_cli_joins": True, "normal_multi_street": True,
                    "all_ins_main_side_different_winners": True, "uncalled_refund": True,
                    "uncontested_no_board_no_reveal": True, "explicit_next_hand_chip_carry": True,
                    "all_cli_checkpoints_match_wire_views": True, "private_visibility_matches_sqlite": True,
                    "no_manual_state_commands": True, "idle_completed_hand_not_paid_twice": True,
                    "cards_partition_52": True, "chips_conserved_3000": True,
                    "copied_sqlite_reopened": True, "all_processes_exit_zero": True,
                },
            }
            (output_dir / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return {"receipt": str((output_dir / "receipt.json").resolve()), "completed_hands": len(hands),
                    "requests": len(rows), "actions": len(actions), "automatic_state_counts": polls,
                    "final_revision": final.revision, "final_stacks": [p.stack for p in final.players],
                    "current_pot": 15, "chip_total": 3000, "process_exit_codes": [c.process.returncode for c in captures]}
        finally:
            if server.process.poll() is None:
                stop_file.touch()
            for capture in reversed(captures):
                capture.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".data/verification/I01"))
    parser.add_argument("--fixture-server", action="store_true")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--wire-log", type=Path)
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    if args.fixture_server:
        assert args.database is not None and args.wire_log is not None and args.stop_file is not None
        fixture_server(args.database, args.wire_log, args.stop_file)
    else:
        print(json.dumps(verify(args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
