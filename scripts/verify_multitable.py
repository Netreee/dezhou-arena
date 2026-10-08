"""M01: one game listener, two tables, native CLIs, a real Agent and a browser."""

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
from threading import Event, Lock, Thread
from time import monotonic
from typing import cast
from unittest.mock import patch
from uuid import uuid4

from jsonschema import Draft202012Validator, FormatChecker
from poker.application.commands import Command, SessionContext, StateCommand
from poker.application.service import TableService
from poker.application.views import CommandResponse, PlayerView
from poker.bootstrap import server_main
from poker.domain.cards import Card, Deck
from poker.domain.models import Table
from poker.domain.types import HandPhase, PlayerId, TableId
from poker.persistence.snapshot import TableSnapshotCodec
from poker.persistence.sqlite import SqliteTableRepository
from poker.transport.codec import JsonLineCodec
from poker.transport.local_tcp import LocalTcpServer
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.verify_local_game import CapturedProcess


def fixture_server(database: Path, wire: Path, stop: Path) -> None:
    original, serve = TableService.handle, LocalTcpServer.serve_forever
    codec, snapshot = JsonLineCodec(), TableSnapshotCodec()
    lock = Lock()
    prefix = tuple(Card.parse(code) for code in "2c 3d 4c 5d".split())
    board = tuple(Card.parse(code) for code in "Ts Js Qs Ks As".split())
    unused = sorted((c for c in Deck.shuffled().remaining if c not in (*prefix, *board)), key=lambda c: c.code)
    fixed = [*prefix, unused[0], *board[:3], unused[1], board[3], unused[2], board[4], *unused[3:]]
    assert len(fixed) == len(set(fixed)) == 52

    def recorded(service: TableService, command: Command, session: SessionContext) -> CommandResponse:
        response = original(service, command, session)
        table = service._repository.load(service._table_id)
        assert table is not None
        row = {"table_id": service._table_id, "player_id": session.player_id,
               "connection_id": session.connection_id, "request": codec.encode_command(command),
               "response": codec.encode_response(response), "snapshot": snapshot.encode(table)}
        with lock, wire.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return response

    def serve_until_stop(server: LocalTcpServer) -> None:
        done = Event()
        def watch() -> None:
            while not done.wait(0.02):
                if stop.exists():
                    server.close()
                    return
        thread = Thread(target=watch, daemon=True)
        thread.start()
        try:
            serve(server)
        finally:
            done.set()
            thread.join(timeout=5)

    sys.argv = ["poker-server", "--port", "0", "--db", str(database), "--table", "1001", "--table", "1002"]
    with patch.object(Deck, "shuffled", side_effect=lambda: Deck(list(fixed))), \
         patch.object(TableService, "handle", new=recorded), \
         patch.object(LocalTcpServer, "serve_forever", new=serve_until_stop):
        server_main()


def verify(output: Path) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh evidence directory")
    output.mkdir(parents=True, exist_ok=True)
    database, wire = output / "tables.sqlite3", output / "wire.jsonl"
    stop, bridge_stop = output / "stop", output / "stop-bridge"
    run = "m01-" + uuid4().hex
    env = {**os.environ, "PYTHONIOENCODING":"utf-8", "PYTHONUNBUFFERED":"1",
           "APP_RUN_ID":run, "APP_LOG_DIR":str(output / "logs"), "LOG_LEVEL":"DEBUG"}
    captures: list[CapturedProcess] = []
    codec, snapshots = JsonLineCodec(), TableSnapshotCodec()
    origin = monotonic()
    server = CapturedProcess([sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()),
                              "--fixture-server", "--output-dir", str(output)], env)
    captures.append(server)
    checkpoints: list[dict[str, object]] = []
    try:
        deadline = monotonic() + 10
        while True:
            ready = server.next_line(deadline)
            match = re.search(r"on \('127\.0\.0\.1', (\d+)\); table=1001", ready)
            if match:
                port = int(match.group(1))
                break
        repository = SqliteTableRepository(database)
        def table(code: str) -> Table:
            value = repository.load(TableId(code))
            assert value is not None
            return value
        def wait(code: str, revision: int, timeout: float = 600) -> Table:
            until = monotonic() + timeout
            while monotonic() < until:
                value = table(code)
                if value.revision == revision:
                    return value
                assert value.revision < revision, (code, value.revision, revision)
                Event().wait(0.02)
            raise RuntimeError(f"Table {code} did not reach revision {revision}")
        def rows() -> list[dict[str, object]]:
            if not wire.exists():
                return []
            return [json.loads(line) for line in wire.read_text(encoding="utf-8").splitlines(keepends=True) if line.endswith("\n")]
        def view(code: str, name: str) -> PlayerView:
            value = table(code)
            player = next(p for p in value.players if p.name == name)
            until = monotonic() + 8
            while monotonic() < until:
                for row in reversed(rows()):
                    if row["table_id"] != code or row["player_id"] != player.id:
                        continue
                    response = codec.decode_response(cast(str, row["response"]))
                    if response.view is not None and response.view.revision == value.revision:
                        return response.view
                Event().wait(0.02)
            raise RuntimeError("Client did not observe current table")
        with socket.socket() as available:
            available.bind(("127.0.0.1", 0))
            http_port = available.getsockname()[1]
        def status(label: str, command: str | None = None) -> None:
            row: dict[str, object] = {"checkpoint":label, "browser_command":command,
                                      "url":f"http://127.0.0.1:{http_port}", "table_id":"1001"}
            if len(table("1001").players) > 1:
                row["view"] = json.loads(codec.encode_response(CommandResponse(view=view("1001", "网页玩家"))))["view"]
            (output / "status.json").write_text(json.dumps(row,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            checkpoints.append(row)
            print(json.dumps(row,ensure_ascii=False),flush=True)
        entry = Path(sys.executable).with_name("poker-client.exe" if os.name == "nt" else "poker-client")
        hosts = []
        for code, name in (("1001", "甲"),("1002", "乙")):
            host = CapturedProcess([str(entry), "--port",str(port),"--table",code],env)
            captures.append(host)
            hosts.append(host)
            host.enter(f"join {name}")
            wait(code,1,10)
        agent = CapturedProcess([str(entry.with_name("poker-agent.exe" if os.name=="nt" else "poker-agent")),
                                 "--port",str(port),"--table","1002","--name","Agent_B",
                                 "--policy","poker.agent.examples:first_legal","--max-hands","2","--session-timeout","600"],env)
        captures.append(agent)
        wait("1002",2,10)
        bridge = CapturedProcess([sys.executable,"-u","-X","utf8",str(root / "scripts/verify_web_game.py"),
                                  "--fixture-bridge","--port",str(http_port),"--game-port",str(port),
                                  "--stop-file",str(bridge_stop)],env)
        captures.append(bridge)
        status("join_browser", "join table 1001 as 网页玩家")
        wait("1001",2)
        status("start_table1001", "start")
        wait("1001",3)
        hosts[0].wait_frame(view("1001", "甲"))
        hosts[0].enter("call")
        a = wait("1001",4,10)
        assert a.hand is not None and a.hand.phase is HandPhase.PREFLOP
        before = snapshots.encode(a)
        hosts[1].enter("start")
        wait("1002",3,10)
        hosts[1].wait_frame(view("1002", "乙"))
        hosts[1].enter("call")
        b = wait("1002",5,10)
        assert b.hand is not None and b.hand.phase is HandPhase.COMPLETE
        hosts[1].enter("start")
        b = wait("1002",7,10)
        assert b.hand is not None and b.hand.phase is HandPhase.COMPLETE
        agent.finish()
        assert snapshots.encode(table("1001")) == before, "Other table altered A"
        assert [p.stack for p in b.players] == [1015,985]
        status("table1002_complete_table1001_unchanged", "check")
        a = wait("1001",5)
        for phase, revision in ((HandPhase.FLOP,5),(HandPhase.TURN,7),(HandPhase.RIVER,9)):
            assert a.hand is not None and a.hand.phase is phase
            status(phase.value, "check")
            wait("1001",revision+1)
            hosts[0].enter("check")
            a = wait("1001",revision+2,10)
        assert a.hand is not None and a.hand.phase is HandPhase.COMPLETE
        assert [p.stack for p in a.players] == [1000,1000]
        closed_before = sum(json.loads(line).get("event") == "gui.session_closed"
                            for file in (output / "logs" / run).glob("gui-*.jsonl")
                            for line in file.read_text(encoding="utf-8").splitlines())
        status("complete_close_browser", "close")
        until = monotonic() + 600
        while monotonic() < until:
            files = list((output / "logs" / run).glob("gui-*.jsonl"))
            closed = sum(json.loads(line).get("event") == "gui.session_closed"
                         for file in files for line in file.read_text(encoding="utf-8").splitlines())
            if closed > closed_before:
                break
            Event().wait(0.05)
        else:
            raise RuntimeError("Close the browser session")
        for host in hosts:
            host.enter("quit")
            host.finish()
        bridge_stop.touch()
        bridge.finish()
        stop.touch()
        server.finish()
        records = rows()
        bindings: dict[str,str] = {}
        for row in records:
            code = cast(str,row["table_id"])
            connection = cast(str,row["connection_id"])
            assert bindings.setdefault(connection,code) == code
            saved = snapshots.decode(cast(str,row["snapshot"]))
            response = codec.decode_response(cast(str,row["response"]))
            assert response.ok and response.view is not None
            received = response.view
            assert received.table_id == saved.id == code
            assert {p.player_id for p in received.players} == {p.id for p in saved.players}
            member = saved.hand.players.get(received.me.player_id) if saved.hand else None
            assert received.me.hole_cards == (tuple(c.code for c in member.hole_cards) if member else ())
            if saved.hand:
                hand = saved.hand
                cards = hand.deck.remaining + hand.deck.burned + hand.board + [c for p in hand.players.values() for c in p.hole_cards]
                assert len(cards) == len(set(cards)) == 52
                assets = sum(p.stack for p in saved.players) + (sum(p.hand_commit for p in hand.players.values()) if hand.phase is not HandPhase.COMPLETE else 0)
                assert assets == 2000
        logs = [json.loads(line) for file in (output / "logs" / run).glob("*.jsonl*") for line in file.read_text(encoding="utf-8").splitlines()]
        validator = Draft202012Validator(json.loads((root/'src/shared_logging/event.schema.json').read_text(encoding='utf-8')),format_checker=FormatChecker())
        for log in logs:
            validator.validate(log)
        assert {"1001","1002"} <= {cast(dict[str,object],log["context"]).get("table_id") for log in logs if log["event"] == "command.applied"}
        for index,capture in enumerate(captures):
            (output / f"process_{index}.log").write_text("\n".join(capture.lines)+"\n",encoding="utf-8")
        stop.unlink()
        bridge_stop.unlink()
        receipt = {"date":datetime.now(UTC).isoformat(),"scope":"One game process/port, two tables, two installed CLIs, one installed Agent and real browser GUI",
                   "game_port":port,"gui_port":http_port,"run_id":run,"elapsed_seconds":round(monotonic()-origin,3),
                   "controls":["fixed unique 52-card orders","wire/snapshot audit","graceful stop watcher"],
                   "processes":[{"pid":p.process.pid,"command":p.command,"stdin":p.inputs,"exit_code":p.process.returncode} for p in captures],
                   "request_count":len(records),"connection_table_bindings":bindings,"logging_events":len(logs),"checkpoints":checkpoints,
                   "tables":{"1001":{"revision":a.revision,"completed_hands":1,"stacks":[p.stack for p in a.players]},
                             "1002":{"revision":b.revision,"completed_hands":2,"stacks":[p.stack for p in b.players]}},
                   "checks":{"one_game_listener":True,"all_clients_same_game_port":True,"separate_table_consumers":True,
                             "other_table_unchanged_during_two_hands":True,"private_views_isolated":True,"cards_52_each":True,
                             "chips_2000_each":True,"browser_four_streets":True,"agent_table_selection":True,"all_processes_exit_zero":True,"all_logs_match_schema":True},
                   "evidence_sha256":{p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()}}
        (output / "receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        return {"receipt":str(output / "receipt.json"),"requests":len(records),"tables":receipt["tables"]}
    finally:
        if server.process.poll() is None:
            stop.touch()
        if "bridge" in locals() and bridge.process.poll() is None:
            bridge_stop.touch()
        for capture in reversed(captures):
            capture.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path('.data/verification/M01'))
    parser.add_argument("--fixture-server",action='store_true')
    args = parser.parse_args()
    if args.fixture_server:
        fixture_server(args.output_dir/'tables.sqlite3',args.output_dir/'wire.jsonl',args.output_dir/'stop')
    else:
        print(json.dumps(verify(args.output_dir),ensure_ascii=False))
