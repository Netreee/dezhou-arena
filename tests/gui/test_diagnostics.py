import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi.testclient import TestClient
from poker.gui.http_app import create_app
from shared_logging import LoggingConfig, configure_logging, shutdown_logging
from tests.gui.test_http_contract import FakeGuiManager


class BrowserDiagnosticTests(unittest.TestCase):
    def test_browser_events_receive_trusted_metadata_and_reject_reserved_fields(self) -> None:
        with TemporaryDirectory() as directory:
            path = configure_logging(LoggingConfig(Path(directory), "browser-test", "gui"))
            try:
                with TestClient(create_app(FakeGuiManager())) as client:
                    body = {"event": "gui.render_failed", "message": "Authorization: Bearer HIDDEN"}
                    self.assertEqual(client.post("/api/gui/diagnostics", json=body).status_code, 204)
                    self.assertEqual(client.post("/api/gui/diagnostics", json={**body, "level": "INFO", "service": "attacker"}).status_code, 422)
                    self.assertEqual(client.post("/api/gui/diagnostics", json={"event": "other.event", "message": "invalid"}).status_code, 422)
                    self.assertEqual(client.post("/api/gui/diagnostics", json={"event": "gui.request_failed", "message": "x" * 513}).status_code, 422)
                raw = path.read_text(encoding="utf-8")
                rows = [json.loads(line) for line in raw.splitlines()]
                event = next(row for row in rows if row["event"] == "gui.render_failed")
                self.assertEqual((event["process_role"], event["pid"], event["service"], event["level"]), ("browser", None, "holdem", "ERROR"))
                self.assertNotIn("HIDDEN", raw)
                self.assertEqual(len([r for r in rows if r["event"] == "gui.http_rejected"]), 3)
            finally:
                shutdown_logging()
