import unittest

from fastapi.testclient import TestClient

from poker.application.views import PlayerViewBuilder
from poker.domain.types import PlayerId
from poker.gui.controller import LocalGuiSessionManager
from poker.gui.http_app import create_app
from poker.gui.interfaces import CliGuiPort, GuiSessionManager
from poker.gui.models import (
    CommandAcceptance, GuiSessionId, GuiSessionNotFound, GuiSnapshot, GuiStatus,
)
from tests.fixtures import make_table


class FakeGuiManager(GuiSessionManager):
    def __init__(self) -> None:
        self.views: dict[GuiSessionId, GuiSnapshot] = {}
        self.lines: list[tuple[GuiSessionId, str]] = []

    def create(self, name: str, table_id: str | None = None) -> GuiSnapshot:
        session_id = GuiSessionId(f"gui-{len(self.views) + 1}")
        player_id = PlayerId("p1" if len(self.views) == 0 else "p2")
        value = GuiSnapshot(
            session_id, GuiStatus.READY,
            PlayerViewBuilder().build(make_table(with_hand=True), player_id),
        )
        self.views[session_id] = value
        return value

    def snapshot(self, session_id: GuiSessionId) -> GuiSnapshot:
        if session_id not in self.views:
            raise GuiSessionNotFound("Unknown GUI session")
        return self.views[session_id]

    def submit(self, session_id: GuiSessionId, line: str) -> CommandAcceptance:
        self.snapshot(session_id)
        self.lines.append((session_id, line))
        return CommandAcceptance(session_id)

    def close(self, session_id: GuiSessionId) -> None:
        self.snapshot(session_id)
        del self.views[session_id]

    def close_all(self) -> None:
        self.views.clear()


class GuiHttpContractTests(unittest.TestCase):
    def test_created_sessions_return_separate_player_views(self) -> None:
        with TestClient(create_app(FakeGuiManager())) as client:
            first = client.post("/api/gui/sessions", json={"name": "甲"})
            second = client.post("/api/gui/sessions", json={"name": "乙"})
            self.assertEqual((first.status_code, second.status_code), (201, 201))
            a, b = first.json(), second.json()
            self.assertNotEqual(a["session_id"], b["session_id"])
            self.assertNotEqual(a["view"]["me"]["hole_cards"], b["view"]["me"]["hole_cards"])
            self.assertNotIn("deck", a["view"])
            self.assertNotIn("hole_cards", a["view"]["players"][1])

    def test_http_action_is_queued_cli_text_not_game_execution(self) -> None:
        manager = FakeGuiManager()
        with TestClient(create_app(manager)) as client:
            client.post("/api/gui/sessions", json={"name": "甲"})
            response = client.post("/api/gui/sessions/gui-1/commands", json={"line": "raise_to 100"})
            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json()["stage"], "queued")
            self.assertEqual(manager.lines, [(GuiSessionId("gui-1"), "raise_to 100")])

    def test_client_cannot_choose_another_game_player_identity(self) -> None:
        with TestClient(create_app(FakeGuiManager())) as client:
            response = client.post("/api/gui/sessions", json={"name": "甲", "player_id": "p2"})
            self.assertEqual(response.status_code, 422)

    def test_close_and_unknown_session_contract(self) -> None:
        with TestClient(create_app(FakeGuiManager())) as client:
            client.post("/api/gui/sessions", json={"name": "甲"})
            self.assertEqual(client.delete("/api/gui/sessions/gui-1").status_code, 204)
            self.assertEqual(client.get("/api/gui/sessions/gui-1").status_code, 404)

    def test_unknown_session_in_real_manager_returns_404(self) -> None:
        def unused_factory() -> CliGuiPort:
            raise AssertionError("No connection should be allocated for an unknown ID")

        with TestClient(create_app(LocalGuiSessionManager(unused_factory))) as client:
            self.assertEqual(client.get("/api/gui/sessions/unknown").status_code, 404)
