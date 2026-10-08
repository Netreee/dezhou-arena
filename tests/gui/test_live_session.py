from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic, sleep
import unittest

from fastapi.testclient import TestClient
from poker.application.service import TableService
from poker.application.views import PlayerViewBuilder
from poker.application.views import PlayerView
from poker.gui.models import GuiError
from poker.transport.codec import JsonLineCodec
import json
from poker.client.cli import PokerCli
from poker.client.parser import CommandParser
from poker.domain.models import Table
from poker.domain.types import TableId
from poker.gui.cli_adapter import PokerCliGuiAdapter
from poker.gui.controller import LocalGuiSessionManager
from poker.gui.http_app import create_app
from poker.messaging.local import LocalCommandQueue, QueuedCommandHandler, QueueWorker
from poker.persistence.sqlite import SqliteTableRepository
from poker.transport.local_tcp import LocalTcpClient, LocalTcpServer
from tests.engine.test_start_hand import opening_engine


class LiveGuiTests(unittest.TestCase):
    def test_two_real_cli_runtimes_join_and_act_with_isolated_views_then_close(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "game.sqlite3"
            repository = SqliteTableRepository(path)
            table = Table(TableId("gui-live"))
            repository.save(table)
            queue = LocalCommandQueue()
            worker = QueueWorker(queue, TableService(table.id, repository, opening_engine(), PlayerViewBuilder()))
            server = LocalTcpServer(QueuedCommandHandler(queue), port=0)
            wt, st = Thread(target=worker.run, daemon=True), Thread(target=server.serve_forever, daemon=True)
            wt.start()
            st.start()
            manager = LocalGuiSessionManager(lambda: PokerCliGuiAdapter(
                PokerCli(LocalTcpClient(server.address[1]), CommandParser()), poll_interval=0.02))
            try:
                with TestClient(create_app(manager)) as client:
                    a = client.post("/api/gui/sessions", json={"name": "甲"}).json()["session_id"]
                    def ready(sid: str, phase: str | None = None) -> tuple[PlayerView, GuiError | None]:
                        deadline = monotonic() + 3
                        while monotonic() < deadline:
                            response = client.get(f"/api/gui/sessions/{sid}")
                            data = response.json()
                            if data["status"] == "ready" and not data["command_pending"] and (phase is None or data["view"]["phase"] == phase):
                                value = JsonLineCodec().decode_response(json.dumps({"ok": True, "view": data["view"]})).view
                                assert value is not None
                                error = data.get("error")
                                return value, GuiError(str(error["code"]), str(error["message"])) if error else None
                            sleep(0.01)
                        self.fail("GUI did not receive a real CLI confirmation")

                    ready(a)
                    b = client.post("/api/gui/sessions", json={"name": "乙"}).json()["session_id"]
                    ready(b)
                    self.assertEqual(client.post(f"/api/gui/sessions/{a}/commands", json={"line": "start"}).status_code, 202)
                    av, bv = ready(a, "preflop")[0], ready(b, "preflop")[0]
                    self.assertNotEqual(av.me.hole_cards, bv.me.hole_cards)
                    self.assertFalse(hasattr(av, "deck"))
                    self.assertEqual(client.post(f"/api/gui/sessions/{a}/commands", json={"line": "check"}).status_code, 202)
                    rejected, error = ready(a, "preflop")
                    assert error is not None
                    self.assertEqual(error.code, "invalid_action")
                    self.assertEqual(rejected.revision, av.revision)
                    self.assertEqual(client.delete(f"/api/gui/sessions/{a}").status_code, 204)
                    self.assertEqual(client.get(f"/api/gui/sessions/{a}").status_code, 404)
            finally:
                manager.close_all()
                server.close()
                st.join(timeout=3)
                worker.stop()
                wt.join(timeout=3)
            self.assertFalse(st.is_alive())
            self.assertFalse(wt.is_alive())
