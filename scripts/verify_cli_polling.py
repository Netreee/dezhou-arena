"""C01 real CLI/TCP polling smoke, with two players and no I01 multi-hand claim."""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic

from poker.application.commands import ActCommand, Command, JoinCommand, SessionContext, StartHandCommand, StateCommand
from poker.application.interfaces import CommandHandler
from poker.application.service import TableService
from poker.application.views import CommandResponse, PlayerViewBuilder
from poker.domain.models import PlayerAction, Table
from poker.domain.types import ActionKind, HandPhase, TableId
from poker.engine.holdem import HoldemEngine
from poker.engine.policies import FiveCardHighEvaluator, NoLimitBettingRules, SidePotAllocator
from poker.messaging.local import LocalCommandQueue, QueuedCommandHandler, QueueWorker
from poker.persistence.snapshot import TableSnapshotCodec
from poker.persistence.sqlite import SqliteTableRepository
from poker.transport.codec import JsonLineCodec
from poker.transport.local_tcp import LocalTcpClient, LocalTcpServer


class RecordingHandler(CommandHandler):
    """Record completed requests in the existing single-consumer worker."""

    def __init__(self, handler: CommandHandler) -> None:
        self.handler = handler
        self.origin = monotonic()
        self.rows: list[dict[str, object]] = []
        self.codec = JsonLineCodec()

    def handle(self, command: Command, session: SessionContext) -> CommandResponse:
        response = self.handler.handle(command, session)
        self.rows.append({
            "seconds": round(monotonic() - self.origin, 6),
            "player_id": session.player_id,
            "command": json.loads(self.codec.encode_command(command)),
            "response": asdict(response),
        })
        return response


def verify(receipt_path: Path) -> dict[str, object]:
    transcript: list[str] = []
    sent_input: list[str] = []
    output: Queue[str | None] = Queue()
    process: subprocess.Popen[str] | None = None
    output_thread: Thread | None = None
    entry = Path(sys.executable).with_name("poker-client.exe" if os.name == "nt" else "poker-client")
    assert entry.is_file(), "Install the project's editable package to provide poker-client"
    with TemporaryDirectory() as directory:
        database = Path(directory) / "polling.sqlite3"
        repository = SqliteTableRepository(database)
        table = Table(TableId("cli-polling-c01"))
        repository.save(table)
        engine = HoldemEngine(NoLimitBettingRules(), FiveCardHighEvaluator(), SidePotAllocator())
        recorder = RecordingHandler(TableService(table.id, repository, engine, PlayerViewBuilder()))
        queue = LocalCommandQueue()
        worker = QueueWorker(queue, recorder)
        worker_thread = Thread(target=worker.run, name="smoke-worker", daemon=True)
        server = LocalTcpServer(QueuedCommandHandler(queue), port=0)
        server_thread = Thread(target=server.serve_forever, name="smoke-server", daemon=True)
        peer = LocalTcpClient(server.address[1])
        worker_thread.start()
        server_thread.start()
        command = [str(entry), "--port", str(server.address[1])]
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

        def snapshot() -> Table:
            saved = SqliteTableRepository(database).load(table.id)
            assert saved is not None
            return saved

        def wait_line(expected: str) -> None:
            deadline = monotonic() + 5.0
            while True:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"CLI did not display {expected!r}; output={transcript!r}")
                try:
                    line = output.get(timeout=remaining)
                except Empty as error:
                    raise RuntimeError(f"CLI output timeout for {expected!r}; output={transcript!r}") from error
                if line is None:
                    raise RuntimeError(f"CLI exited before {expected!r}; output={transcript!r}")
                if expected in line:
                    return

        def enter(line: str) -> None:
            assert process is not None and process.stdin is not None
            sent_input.append(line)
            process.stdin.write(line + "\n")
            process.stdin.flush()

        try:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", env=environment, creationflags=flags,
            )
            stdout = process.stdout
            assert stdout is not None

            def capture() -> None:
                for line in stdout:
                    clean = line.rstrip("\r\n")
                    transcript.append(clean)
                    output.put(clean)
                output.put(None)

            output_thread = Thread(target=capture, name="cli-output", daemon=True)
            output_thread.start()
            enter("join 甲")
            wait_line("阶段: 未开局")
            peer.connect()
            joined = peer.send(JoinCommand("乙"))
            assert joined.view is not None
            peer_id = joined.view.me.player_id
            wait_line("1 | 乙 | 1000 | 未参局")
            started = peer.send(StartHandCommand())
            assert started.view is not None and started.view.hand_id is not None
            hand_id = started.view.hand_id
            wait_line(f"本手: {hand_id} | 阶段: preflop")
            enter("call")
            wait_line("行动者: 乙")
            peer_state = peer.send(StateCommand())
            assert peer_state.view is not None
            assert peer_state.view.actor_id == peer_id
            advanced = peer.send(ActCommand(hand_id, PlayerAction(ActionKind.CHECK)))
            assert advanced.view is not None and advanced.view.phase is HandPhase.FLOP
            wait_line(f"本手: {hand_id} | 阶段: flop")
            wait_line("公共牌: " + " ".join(advanced.view.board))
            saved = snapshot()
            assert saved.hand is not None
            cli_player = next(p for p in saved.players if p.name == "甲")
            own_cards = " ".join(c.code for c in saved.hand.players[cli_player.id].hole_cards)
            wait_line("本人底牌: " + own_cards)
            assert saved.revision == 5
            assert [p.stack for p in saved.players] == [990, 990]
            assert sum(p.stack for p in saved.players) + sum(m.hand_commit for m in saved.hand.players.values()) == 2000
            enter("quit")
            assert process.wait(timeout=5) == 0
            output_thread.join(timeout=5)
            assert not output_thread.is_alive()
        finally:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
                if process.stdin is not None:
                    process.stdin.close()
                if output_thread is not None:
                    output_thread.join(timeout=5)
                if process.stdout is not None:
                    process.stdout.close()
            peer.close()
            server.close()
            server_thread.join(timeout=5)
            worker.stop()
            worker_thread.join(timeout=5)

        assert not server_thread.is_alive() and not worker_thread.is_alive()
        saved = snapshot()
        assert saved.hand is not None
        cli_player = next(p for p in saved.players if p.name == "甲")
        cli_rows = [row for row in recorder.rows if row["player_id"] == cli_player.id]
        commands = [row["command"] for row in cli_rows]
        state_rows = [row for row in cli_rows if row["command"] == {"command": "state"}]
        assert len(state_rows) >= 3
        assert commands[0] == {"command": "join", "name": "甲"}
        assert {"command": "act", "hand_id": saved.hand.id, "action": {"kind": "call", "to": None}} in commands
        assert sent_input == ["join 甲", "call", "quit"]
        assert not any("服务器拒绝" in line or "输入错误" in line for line in transcript)
        assert all(line.endswith("| 无") for line in transcript if line.startswith("1 | 乙 |"))
        assert set(line.removeprefix("本人底牌: ") for line in transcript if line.startswith("本人底牌: ")) <= {
            "无", " ".join(c.code for c in saved.hand.players[cli_player.id].hole_cards),
        }
        raw_snapshot = json.loads(TableSnapshotCodec().encode(saved))
        receipt: dict[str, object] = {
            "scope": "C01 actual poker-client process + two-player TCP + real HoldemEngine/queue/SQLite; not I01",
            "date": datetime.now().date().isoformat(), "entrypoint_command": command,
            "default_poll_interval_seconds": 1.0,
            "stdin_commands": sent_input, "client_stdout": transcript,
            "server_commands": recorder.rows, "client_state_requests": len(state_rows),
            "hand_id": saved.hand.id, "final_revision": saved.revision,
            "final_snapshot": raw_snapshot,
            "checks": {
                "automatic_observation_of_peer_join": True,
                "automatic_observation_of_peer_start": True,
                "action_uses_polled_hand_id": True,
                "automatic_observation_of_peer_check_and_flop": True,
                "no_manual_state_in_stdin": True, "private_cards_verified": True,
                "sqlite_reopened_readback": True, "chips_conserved": True,
                "client_exit_zero": True, "server_and_worker_stopped": True,
            },
        }
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"receipt": str(receipt_path.resolve()), "client_state_requests": len(state_rows),
                "stdin_commands": sent_input, "phase": saved.hand.phase.value,
                "final_revision": saved.revision, "stacks": [p.stack for p in saved.players],
                "pot": sum(m.hand_commit for m in saved.hand.players.values()), "chip_total": 2000}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=Path(".data/verification/C01_POLLING_RECEIPT.json"))
    args = parser.parse_args()
    print(json.dumps(verify(args.receipt), ensure_ascii=False))


if __name__ == "__main__":
    main()
