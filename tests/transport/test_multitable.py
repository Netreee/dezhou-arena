from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
import unittest
from unittest.mock import patch

from poker.application.commands import ActCommand, Command, JoinCommand, SessionContext, StartHandCommand, StateCommand
from poker.application.views import CommandResponse
from poker.application.service import TableService
from poker.domain.models import PlayerAction, Table
from poker.domain.types import ActionKind, ErrorCode, TableId
from poker.engine.holdem import HoldemEngine
from poker.engine.policies import FiveCardHighEvaluator, NoLimitBettingRules, SidePotAllocator
from poker.persistence.snapshot import TableSnapshotCodec
from poker.persistence.sqlite import SqliteTableRepository
from poker.server import TableRegistry, TableRuntime
from poker.transport.local_tcp import LocalTcpClient, LocalTcpServer


def engine() -> HoldemEngine:
    return HoldemEngine(NoLimitBettingRules(), FiveCardHighEvaluator(), SidePotAllocator())


class MultiTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.repository = SqliteTableRepository(Path(self.temp.name) / "tables.sqlite3")
        self.registry = TableRegistry(tuple(TableRuntime(Table(TableId(code)), self.repository, engine()) for code in ("1001", "1002")))
        self.server = LocalTcpServer(self.registry, port=0)
        self.registry.start()
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.clients = [LocalTcpClient(self.server.address[1]) for _ in range(5)]
        for client in self.clients:
            client.connect()

    def tearDown(self) -> None:
        for client in self.clients:
            client.close()
        self.server.close()
        self.thread.join(timeout=3)
        self.registry.close()
        self.temp.cleanup()

    def test_same_port_different_tables_and_private_views_remain_isolated(self) -> None:
        for index, client in enumerate(self.clients[:4]):
            code = TableId("1001" if index < 2 else "1002")
            response = client.send(JoinCommand(f"P{index}", code))
            assert response.view is not None
            self.assertEqual(response.view.table_id, code)
        self.assertTrue(self.clients[0].send(StartHandCommand()).ok)
        a, b = self.clients[0].send(StateCommand()), self.clients[2].send(StateCommand())
        assert a.view is not None and b.view is not None and a.view.hand_id is not None
        self.assertIsNone(b.view.hand_id)
        self.assertEqual([p.name for p in a.view.players], ["P0", "P1"])
        self.assertEqual([p.name for p in b.view.players], ["P2", "P3"])
        self.assertFalse(b.view.me.hole_cards)
        before = self.repository.load(TableId("1002"))
        assert before is not None
        self.assertTrue(self.clients[0].send(ActCommand(a.view.hand_id, PlayerAction(ActionKind.FOLD))).ok)
        after = self.repository.load(TableId("1002"))
        assert after is not None
        self.assertEqual(TableSnapshotCodec().encode(before), TableSnapshotCodec().encode(after))
        self.assertTrue(self.clients[2].send(StartHandCommand()).ok)
        bv = self.clients[2].send(StateCommand()).view
        assert bv is not None
        rejected = self.clients[2].send(ActCommand(a.view.hand_id, PlayerAction(ActionKind.FOLD)))
        assert rejected.error is not None
        self.assertEqual(rejected.error.code, ErrorCode.HAND_MISMATCH)
        self.assertEqual(self.clients[2].send(StateCommand()).view, bv)

    def test_missing_table_does_not_bind_connection_and_default_join_remains_valid(self) -> None:
        response = self.clients[0].send(JoinCommand("甲", TableId("missing")))
        assert response.error is not None
        self.assertEqual(response.error.code, ErrorCode.TABLE_NOT_FOUND)
        joined = self.clients[0].send(JoinCommand("甲"))
        assert joined.view is not None
        self.assertEqual(joined.view.table_id, TableId("1001"))
        denied = self.clients[0].send(JoinCommand("甲", TableId("1002")))
        assert denied.error is not None
        self.assertEqual(denied.error.code, ErrorCode.ALREADY_SEATED)
        unjoined = self.clients[4].send(StateCommand())
        assert unjoined.error is not None
        self.assertEqual(unjoined.error.code, ErrorCode.NOT_SEATED)

    def test_each_table_has_its_own_consumer(self) -> None:
        for index, code in enumerate(("1001", "1002")):
            self.assertTrue(self.clients[index].send(JoinCommand("甲", TableId(code))).ok)
        blocked, release = Event(), Event()
        original = TableService.handle
        def delayed(service: TableService, command: Command, session: SessionContext) -> CommandResponse:
            if service._table_id == "1001":
                blocked.set()
                assert release.wait(3)
            return original(service, command, session)
        with patch.object(TableService, "handle", new=delayed), ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(self.clients[0].send, StateCommand())
            try:
                self.assertTrue(blocked.wait(1))
                response = self.clients[1].send(StateCommand())
                assert response.view is not None
                self.assertEqual(response.view.table_id, TableId("1002"))
                self.assertFalse(first.done())
            finally:
                release.set()
            self.assertTrue(first.result().ok)

    def test_unstarted_listener_can_be_closed_without_waiting(self) -> None:
        listener = LocalTcpServer(self.registry, port=0)
        listener.close()

    def test_restarting_runtime_cannot_reset_an_active_snapshot(self) -> None:
        self.assertTrue(self.clients[0].send(JoinCommand("甲", TableId("1001"))).ok)
        before = self.repository.load(TableId("1001"))
        assert before is not None
        with self.assertRaises(RuntimeError):
            self.registry.start()
        after = self.repository.load(TableId("1001"))
        assert after is not None
        self.assertEqual(TableSnapshotCodec().encode(before), TableSnapshotCodec().encode(after))

    def test_second_listener_failure_does_not_replace_live_tables(self) -> None:
        self.assertTrue(self.clients[0].send(JoinCommand("甲", TableId("1001"))).ok)
        before = self.repository.load(TableId("1001"))
        assert before is not None
        candidate = TableRegistry((TableRuntime(Table(TableId("1001")), self.repository, engine()),))
        with self.assertRaises(OSError):
            LocalTcpServer(candidate, self.server.address[1])
        candidate.close()
        after = self.repository.load(TableId("1001"))
        assert after is not None
        self.assertEqual(TableSnapshotCodec().encode(before), TableSnapshotCodec().encode(after))
