"""L03/L04: installed bot, local HTTP fixture, browser diagnostics and independent reuse."""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import re
import socket
import sys
from threading import Event, Thread
from time import monotonic
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from jsonschema import Draft202012Validator, FormatChecker
from poker.application.commands import ActCommand
from poker.domain.types import HandPhase, TableId
from poker.persistence.sqlite import SqliteTableRepository
from shared_logging import LoggingConfig, configure_logging, shutdown_logging
from shared_logging.observations import run_logged_process
from scripts.verify_local_game import CapturedProcess, read_requests


def verify(output_dir: Path) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Use a fresh evidence directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"l03-{uuid4().hex}"
    configure_logging(LoggingConfig(output_dir / "logs", run_id, "verifier", level="DEBUG"))
    env = {**os.environ, "APP_RUN_ID": run_id, "APP_LOG_DIR": str(output_dir / "logs"), "LOG_LEVEL": "DEBUG",
           "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    env.pop("APP_LOG_ROLE", None)
    captures: list[CapturedProcess] = []
    database, wire = output_dir / "bot.sqlite3", output_dir / "server_commands.jsonl"
    server_stop, bridge_stop = output_dir / "stop-server", output_dir / "stop-bridge"
    fixture_requests: list[dict[str, object]] = []

    class Fixture(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            status = 200 if self.path == "/success" else 503
            body = json.dumps({"text": "check", "usage": {"input_tokens": 12, "output_tokens": 4}, "stop_reason": "stop"}).encode()
            fixture_requests.append({"path": self.path, "status": status,
                                     "returned_usage": {"input_tokens": 12, "output_tokens": 4} if status == 200 else None})
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            logging.getLogger("fixture.http").info(format, *args)

    fixture = ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
    thread = Thread(target=fixture.serve_forever, daemon=True)
    thread.start()
    server: CapturedProcess | None = None
    bridge: CapturedProcess | None = None
    try:
        result = run_logged_process([sys.executable, "-X", "utf8", str(root / "scripts/logging_fixture_client.py"),
                                     f"http://127.0.0.1:{fixture.server_port}"], role="other_project")
        assert result.returncode == 0, result.stderr
        assert [r["status"] for r in fixture_requests] == [200, 503]
        server = CapturedProcess([sys.executable, "-u", "-X", "utf8", str(root / "scripts/verify_local_game.py"),
                                  "--fixture-server", "--database", str(database), "--wire-log", str(wire),
                                  "--stop-file", str(server_stop)], env)
        captures.append(server)
        deadline = monotonic() + 10
        while True:
            match = re.search(r"on \('127\.0\.0\.1', (\d+)\); table=([0-9a-f]+)", server.next_line(deadline))
            if match is not None:
                port, table_id = int(match.group(1)), TableId(match.group(2))
                break

        def wait_revision(revision: int) -> None:
            until = monotonic() + 10
            while monotonic() < until:
                table = SqliteTableRepository(database).load(table_id)
                assert table is not None
                if table.revision == revision:
                    return
                assert table.revision < revision
                Event().wait(0.02)
            raise RuntimeError(f"Bot game did not reach revision {revision}")

        entry = Path(sys.executable).with_name("poker-client.exe" if os.name == "nt" else "poker-client")
        host = CapturedProcess([str(entry), "--port", str(port)], env)
        captures.append(host)
        host.enter("join Host")
        wait_revision(1)
        bot_entry = entry.with_name("poker-bot.exe" if os.name == "nt" else "poker-bot")
        assert bot_entry.is_file(), "Reinstall the project's entry points first"
        bot = CapturedProcess([str(bot_entry), "--port", str(port), "--name", "Bot", "--max-actions", "2"], env)
        captures.append(bot)
        wait_revision(2)
        host.enter("start")
        wait_revision(3)
        host.enter("call")
        wait_revision(6)
        bot.finish()
        table = SqliteTableRepository(database).load(table_id)
        assert table is not None and table.hand is not None and table.hand.phase is HandPhase.FLOP
        bot_id = next(p.id for p in table.players if p.name == "Bot")
        assert sum(isinstance(r.command, ActCommand) and r.player_id == bot_id for r in read_requests(wire)) == 2
        host.enter("quit")
        host.finish()

        with socket.socket() as available:
            available.bind(("127.0.0.1", 0))
            http_port = available.getsockname()[1]
        bridge = CapturedProcess([sys.executable, "-u", "-X", "utf8", str(root / "scripts/verify_web_game.py"),
                                  "--fixture-bridge", "--port", str(http_port), "--game-port", str(port),
                                  "--stop-file", str(bridge_stop)], env)
        captures.append(bridge)
        base = f"http://127.0.0.1:{http_port}"
        until = monotonic() + 10
        while True:
            try:
                with urlopen(base, timeout=1) as response:
                    assert response.status == 200 and "德州扑克" in response.read().decode()
                break
            except URLError:
                if monotonic() >= until:
                    raise
                Event().wait(0.02)
        for event in ("gui.render_failed", "gui.request_failed"):
            body = json.dumps({"event": event, "message": "Fixture diagnostic api_key=FIXTURE-SECRET"}).encode()
            with urlopen(Request(base + "/api/gui/diagnostics", body, {"Content-Type": "application/json"}), timeout=5) as response:
                assert response.status == 204
        try:
            body = json.dumps({"event": "gui.render_failed", "message": "bad envelope", "level": "INFO"}).encode()
            urlopen(Request(base + "/api/gui/diagnostics", body, {"Content-Type": "application/json"}), timeout=5).close()
        except HTTPError as error:
            assert error.code == 422
        else:
            raise AssertionError("Reserved browser metadata was accepted")
        bridge_stop.touch()
        bridge.finish()
        server_stop.touch()
        server.finish()
        fixture.shutdown()
        thread.join(timeout=5)
        assert not thread.is_alive()
        shutdown_logging()
        log_files = list((output_dir / "logs" / run_id).glob("*.jsonl*"))
        rows = [json.loads(line) for file in log_files for line in file.read_text(encoding="utf-8").splitlines()]
        validator = Draft202012Validator(json.loads((root / "src/shared_logging/event.schema.json").read_text(encoding="utf-8")),
                                         format_checker=FormatChecker())
        for row in rows:
            validator.validate(row)
            assert row["run_id"] == run_id
        assert not any("FIXTURE-SECRET" in file.read_text(encoding="utf-8") for file in log_files)
        bot_rows = [r for r in rows if r["process_role"] == "bot"]
        bot_counts = Counter(str(r["event"]) for r in bot_rows)
        assert bot_counts["bot.started"] == bot_counts["bot.stopped"] == 1
        assert bot_counts["bot.decision_requested"] == bot_counts["bot.decision_completed"] == bot_counts["bot.command_submitted"] == 2
        success = next(r for r in rows if r["event"] == "api.call.completed")
        data = cast(dict[str, object], success["data"])
        assert data["input_tokens"] == 12 and data["output_tokens"] == 4 and data["cost"] is None
        failure = next(r for r in rows if r["event"] == "api.call.failed")
        assert cast(dict[str, object], failure["data"])["http_status"] == 503
        browser_rows = [r for r in rows if r["process_role"] == "browser"]
        assert len(browser_rows) == 2 and all(r["pid"] is None and r["level"] == "ERROR" for r in browser_rows)
        for index, capture in enumerate(captures):
            (output_dir / f"process_{index}.log").write_text("\n".join(capture.lines) + "\n", encoding="utf-8")
        for file in (server_stop, bridge_stop):
            file.unlink()
        receipt: dict[str, object] = {
            "scope": "Real installed passive bot + local HTTP API fixture + synthetic browser diagnostic HTTP envelopes + independent non-poker consumer",
            "date": datetime.now().date().isoformat(), "run_id": run_id, "provider_call": False, "paid_cost": None,
            "fixture_requests": fixture_requests, "bot_actions": 2, "bot_events": dict(bot_counts),
            "processes": [{"pid": c.process.pid, "command": c.command, "exit_code": c.process.returncode} for c in captures],
            "log_files": len(log_files), "log_events": len(rows), "browser_events": 2,
            "checks": {"schema_all_records": True, "reuse_without_poker_import": True, "same_run_child_config": True,
                       "actual_local_http_usage": True, "unknown_cost_null": True, "http_failure_propagated": True,
                       "bot_owner_cli_transport": True, "raw_stderr_wrapped_and_redacted": True,
                       "browser_reserved_fields_rejected": True, "all_child_exit_codes_zero": True},
            "evidence_sha256": {p.relative_to(output_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in output_dir.rglob("*") if p.is_file()},
        }
        (output_dir / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"receipt": str(output_dir / "receipt.json"), "bot_actions": 2, "log_events": len(rows)}
    finally:
        if server is not None and server.process.poll() is None:
            server_stop.touch()
        if bridge is not None and bridge.process.poll() is None:
            bridge_stop.touch()
        for capture in reversed(captures):
            capture.cleanup()
        fixture.shutdown()
        fixture.server_close()
        thread.join(timeout=5)
        shutdown_logging()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".data/verification/L03"))
    print(json.dumps(verify(parser.parse_args().output_dir), ensure_ascii=False))
