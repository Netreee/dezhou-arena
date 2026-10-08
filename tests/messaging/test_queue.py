import unittest

from poker.application.commands import JoinCommand, SessionContext
from poker.application.service import TableService
from poker.application.views import PlayerViewBuilder
from poker.domain.models import Table
from poker.domain.types import TableId
from poker.messaging.interfaces import CommandEnvelope
from poker.messaging.local import LocalCommandQueue, QueueWorker
from tests.fixtures import FakeEngine, MemoryRepository


class QueueTests(unittest.TestCase):
    def test_queue_preserves_fifo(self) -> None:
        queue = LocalCommandQueue()
        first = CommandEnvelope(JoinCommand("甲"), SessionContext())
        second = CommandEnvelope(JoinCommand("乙"), SessionContext())
        queue.put(first)
        queue.put(second)
        self.assertIs(queue.get(), first)
        self.assertIs(queue.get(), second)

    def test_worker_returns_each_response_to_its_own_reply_queue(self) -> None:
        table = Table(TableId("t"))
        queue = LocalCommandQueue()
        service = TableService(table.id, MemoryRepository(table), FakeEngine(), PlayerViewBuilder())
        worker = QueueWorker(queue, service)
        first = CommandEnvelope(JoinCommand("甲"), SessionContext())
        second = CommandEnvelope(JoinCommand("乙"), SessionContext())
        queue.put(first)
        queue.put(second)
        self.assertTrue(worker.process_next())
        self.assertTrue(worker.process_next())
        a = first.reply.get_nowait()
        b = second.reply.get_nowait()
        assert a.view is not None and b.view is not None
        self.assertNotEqual(a.view.me.player_id, b.view.me.player_id)
        self.assertEqual(len(a.view.players), 1)
        self.assertEqual(len(b.view.players), 2)

    def test_normal_stop_marker_ends_worker(self) -> None:
        table = Table(TableId("t"))
        queue = LocalCommandQueue()
        worker = QueueWorker(queue, TableService(table.id, MemoryRepository(table), FakeEngine(), PlayerViewBuilder()))
        worker.stop()
        self.assertFalse(worker.process_next())
