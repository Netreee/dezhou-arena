import unittest
from queue import Empty, Queue

from poker.application.views import CommandResponse, ErrorInfo, PlayerViewBuilder
from poker.domain.types import ErrorCode, PlayerId, TableId
from poker.gui.controller import LocalGuiSessionManager
from poker.gui.interfaces import CliGuiPort
from poker.gui.models import (
    CliLifecycleUpdate, CliResponseUpdate, CliUpdate, GuiCommandConflict, GuiError, GuiStatus,
)
from tests.fixtures import make_table


class ControlledPort(CliGuiPort):
    def __init__(self) -> None:
        self.updates: Queue[CliUpdate] = Queue()
        self.lines: list[str] = []
        self.closed = False

    def start(self, name: str, table_id: TableId | None = None) -> None:
        self.lines.append(f"join_table {table_id} {name}" if table_id is not None else f"join {name}")

    def submit_line(self, line: str) -> None:
        self.lines.append(line)

    def next_update(self) -> CliUpdate | None:
        try:
            return self.updates.get_nowait()
        except Empty:
            return None

    def close(self) -> None:
        self.closed = True


class SessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ports: list[ControlledPort] = []

        def new_port() -> CliGuiPort:
            port = ControlledPort()
            self.ports.append(port)
            return port

        self.manager = LocalGuiSessionManager(new_port)
        self.view = PlayerViewBuilder().build(make_table(with_hand=True), PlayerId("p1"))

    def tearDown(self) -> None:
        self.manager.close_all()

    def ready(self) -> str:
        snap = self.manager.create("甲")
        self.ports[-1].updates.put(CliResponseUpdate("join 甲", CommandResponse(view=self.view)))
        self.assertEqual(self.manager.snapshot(snap.session_id).status, GuiStatus.READY)
        return snap.session_id

    def test_opening_is_not_ready_until_real_join_response(self) -> None:
        snap = self.manager.create("甲")
        self.assertEqual(snap.status, GuiStatus.OPENING)
        self.assertIsNone(snap.view)
        self.ports[0].updates.put(CliResponseUpdate("poll", CommandResponse(view=self.view)))
        self.assertEqual(self.manager.snapshot(snap.session_id).status, GuiStatus.OPENING)
        self.ports[0].updates.put(CliResponseUpdate("join 甲", CommandResponse(view=self.view)))
        confirmed = self.manager.snapshot(snap.session_id)
        self.assertEqual(confirmed.status, GuiStatus.READY)
        self.assertFalse(confirmed.command_pending)

    def test_table_selection_is_forwarded_and_confirmed_by_that_join(self) -> None:
        from dataclasses import replace
        selected = replace(self.view, table_id=TableId("1002"))
        snap = self.manager.create("甲", "1002")
        self.assertEqual(self.ports[0].lines, ["join_table 1002 甲"])
        self.ports[0].updates.put(CliResponseUpdate("join_table 1002 甲", CommandResponse(view=selected)))
        self.assertEqual(self.manager.snapshot(snap.session_id).view, selected)

    def test_foreign_table_response_is_not_cached(self) -> None:
        from dataclasses import replace
        from poker.gui.models import GuiSessionId
        sid = GuiSessionId(self.ready())
        self.ports[0].updates.put(CliResponseUpdate("poll", CommandResponse(view=replace(self.view, table_id=TableId("another")))))
        with self.assertRaises(RuntimeError):
            self.manager.snapshot(sid)
        self.assertTrue(self.ports[0].closed)

    def test_poll_does_not_confirm_pending_user_action_and_rejection_keeps_view(self) -> None:
        from poker.gui.models import GuiSessionId
        sid = GuiSessionId(self.ready())
        self.manager.submit(sid, "raise_to 100")
        self.ports[0].updates.put(CliResponseUpdate("poll", CommandResponse(view=self.view)))
        self.assertTrue(self.manager.snapshot(sid).command_pending)
        with self.assertRaises(GuiCommandConflict):
            self.manager.submit(sid, "all_in")
        self.ports[0].updates.put(CliResponseUpdate("raise_to 100", CommandResponse(error=ErrorInfo(ErrorCode.RAISE_TOO_SMALL, "too small"))))
        snap = self.manager.snapshot(sid)
        self.assertFalse(snap.command_pending)
        self.assertIs(snap.view, self.view)
        self.assertEqual(snap.error, GuiError("raise_too_small", "too small"))

    def test_independent_sessions_have_different_handles_ports_and_player_views(self) -> None:
        a, b = self.manager.create("甲"), self.manager.create("乙")
        self.assertNotEqual(a.session_id, b.session_id)
        second = PlayerViewBuilder().build(make_table(with_hand=True), PlayerId("p2"))
        self.ports[0].updates.put(CliResponseUpdate("join 甲", CommandResponse(view=self.view)))
        self.ports[1].updates.put(CliResponseUpdate("join 乙", CommandResponse(view=second)))
        self.assertNotEqual(self.manager.snapshot(a.session_id).view, self.manager.snapshot(b.session_id).view)
        self.manager.close(a.session_id)
        self.assertTrue(self.ports[0].closed)
        self.assertFalse(self.ports[1].closed)

    def test_poll_tag_cannot_confirm_an_invalid_command_with_the_same_text(self) -> None:
        from poker.gui.models import GuiSessionId
        sid = GuiSessionId(self.ready())
        self.manager.submit(sid, "poll")
        self.ports[0].updates.put(CliResponseUpdate("poll", CommandResponse(view=self.view)))
        self.assertTrue(self.manager.snapshot(sid).command_pending)
        self.ports[0].updates.put(CliLifecycleUpdate(GuiError("input_error", "Unknown command"), False, "poll"))
        snap = self.manager.snapshot(sid)
        self.assertFalse(snap.command_pending)
        self.assertEqual(snap.error, GuiError("input_error", "Unknown command"))

    def test_local_syntax_failure_confirms_only_its_pending_command(self) -> None:
        from poker.gui.models import GuiSessionId
        sid = GuiSessionId(self.ready())
        self.manager.submit(sid, "invalid")
        self.ports[0].updates.put(CliLifecycleUpdate(GuiError("input_error", "syntax"), False, "invalid"))
        snap = self.manager.snapshot(sid)
        self.assertEqual(snap.status, GuiStatus.READY)
        self.assertFalse(snap.command_pending)
        self.assertIs(snap.view, self.view)

    def test_rejected_join_closes_unbound_cli(self) -> None:
        snap = self.manager.create("甲")
        self.ports[0].updates.put(CliResponseUpdate("join 甲", CommandResponse(error=ErrorInfo(ErrorCode.TABLE_FULL, "full"))))
        result = self.manager.snapshot(snap.session_id)
        self.assertEqual(result.status, GuiStatus.CLOSED)
        self.assertFalse(result.command_pending)
        self.assertTrue(self.ports[0].closed)

    def test_foreign_player_response_is_rejected_instead_of_cached(self) -> None:
        from poker.gui.models import GuiSessionId
        sid = GuiSessionId(self.ready())
        other = PlayerViewBuilder().build(make_table(with_hand=True), PlayerId("p2"))
        self.ports[0].updates.put(CliResponseUpdate("poll", CommandResponse(view=other)))
        with self.assertRaises(RuntimeError):
            self.manager.snapshot(sid)
        self.assertTrue(self.ports[0].closed)
