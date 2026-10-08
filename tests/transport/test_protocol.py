import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from unittest.mock import MagicMock

from poker.application.commands import ActCommand, JoinCommand, StartHandCommand, StateCommand
from poker.application.service import TableService
from poker.application.views import CommandResponse, ErrorInfo, PlayerViewBuilder
from poker.domain.models import ChipShare, HandResult, PlayerAction, Pot, PotAward, Table
from poker.domain.types import ActionKind, ErrorCode, HandId, HandPhase, PlayerId, TableId
from poker.messaging.local import LocalCommandQueue, QueuedCommandHandler, QueueWorker
from poker.persistence.sqlite import SqliteTableRepository
from poker.transport.codec import JsonLineCodec
from poker.transport.local_tcp import LocalTcpClient, LocalTcpServer
from tests.fixtures import FakeEngine, make_table


class ProtocolTests(unittest.TestCase):
    def test_table_is_only_selected_at_join(self) -> None:
        codec = JsonLineCodec()
        selected = JoinCommand("甲", TableId("1002"))
        self.assertEqual(codec.decode_command(codec.encode_command(selected)), selected)
        for raw in ('{"command":"state","table_id":"1002"}', '{"command":"join","name":"甲","table_id":true}',
                    '{"command":"join","name":"甲","table_id":"bad/table"}'):
            with self.assertRaises(ValueError):
                codec.decode_command(raw)

    def test_optional_correlation_id_is_metadata_and_rejects_invalid_values(self) -> None:
        codec = JsonLineCodec()
        command = StateCommand()
        self.assertEqual(codec.decode_request(codec.encode_command(command, correlation_id="trace-1")), (command, "trace-1"))
        self.assertEqual(codec.decode_request(codec.encode_command(command)), (command, None))
        for raw in ('{"command":"state","correlation_id":false}', '{"command":"state","correlation_id":""}',
                    '{"command":"state","correlation_id":"' + "x" * 97 + '"}'):
            with self.assertRaises(ValueError):
                codec.decode_request(raw)

    def test_socket_is_closed_even_when_stream_flush_fails(self) -> None:
        client = LocalTcpClient()
        stream, connection = MagicMock(), MagicMock()
        stream.close.side_effect = OSError("broken stream")
        client._stream, client._socket = stream, connection
        with self.assertRaises(OSError):
            client.close()
        connection.close.assert_called_once()
        client.close()
        connection.close.assert_called_once()

    def test_all_command_variants_round_trip(self) -> None:
        codec = JsonLineCodec()
        commands = (
            JoinCommand("Ryan 甲"), StateCommand(), StartHandCommand(),
            ActCommand(HandId("h1"), PlayerAction(ActionKind.RAISE_TO, 100)),
            ActCommand(HandId("h1"), PlayerAction(ActionKind.CHECK)),
        )
        for command in commands:
            self.assertEqual(codec.decode_command(codec.encode_command(command)), command)

    def test_client_cannot_supply_player_identity(self) -> None:
        with self.assertRaises(ValueError):
            JsonLineCodec().decode_command('{"command":"state","player_id":"someone-else"}')

    def test_boolean_is_not_a_chip_amount(self) -> None:
        with self.assertRaises(ValueError):
            JsonLineCodec().decode_command(
                '{"command":"act","hand_id":"h1","action":{"kind":"raise_to","to":true}}',
            )

    def test_private_view_response_round_trip(self) -> None:
        table = make_table(with_hand=True)
        response = CommandResponse(view=PlayerViewBuilder().build(table, PlayerId("p1")))
        codec = JsonLineCodec()
        self.assertEqual(codec.decode_response(codec.encode_response(response)), response)

    def test_error_response_round_trip(self) -> None:
        response = CommandResponse(error=ErrorInfo(ErrorCode.NOT_YOUR_TURN, "还没轮到你"))
        codec = JsonLineCodec()
        self.assertEqual(codec.decode_response(codec.encode_response(response)), response)

    def test_completed_result_and_revealed_cards_round_trip(self) -> None:
        table = make_table(with_hand=True)
        assert table.hand is not None
        table.hand.phase = HandPhase.COMPLETE
        table.last_result = HandResult(
            table.hand.id,
            (PotAward(Pot(20, (PlayerId("p1"), PlayerId("p2"))),
                      (ChipShare(PlayerId("p1"), 20),)),),
            (ChipShare(PlayerId("p2"), 5),),
            {pid: member.hole_cards for pid, member in table.hand.players.items()},
        )
        response = CommandResponse(view=PlayerViewBuilder().build(table, PlayerId("p1")))
        codec = JsonLineCodec()
        self.assertEqual(codec.decode_response(codec.encode_response(response)), response)

    def test_real_tcp_queue_sqlite_pipeline_with_fake_engine(self) -> None:
        """Actual local sockets/SQL/queue; deliberately not a poker E2E test."""
        with TemporaryDirectory() as directory:
            repository = SqliteTableRepository(Path(directory) / "poker.sqlite3")
            table = Table(TableId("socket-test"))
            repository.save(table)
            service = TableService(table.id, repository, FakeEngine(), PlayerViewBuilder())
            queue = LocalCommandQueue()
            worker = QueueWorker(queue, service)
            worker_thread = Thread(target=worker.run, daemon=True)
            server = LocalTcpServer(QueuedCommandHandler(queue), port=0)
            server_thread = Thread(target=server.serve_forever, daemon=True)
            worker_thread.start()
            server_thread.start()
            clients = [LocalTcpClient(server.address[1]), LocalTcpClient(server.address[1])]
            try:
                for client, name in zip(clients, ("甲", "乙"), strict=True):
                    client.connect()
                    self.assertTrue(client.send(JoinCommand(name)).ok)
                self.assertTrue(clients[0].send(StartHandCommand()).ok)
                first = clients[0].send(StateCommand())
                second = clients[1].send(StateCommand())
                assert first.view is not None and second.view is not None
                self.assertNotEqual(first.view.me.player_id, second.view.me.player_id)
                self.assertNotEqual(first.view.me.hole_cards, second.view.me.hole_cards)
                self.assertEqual(first.view.players, second.view.players)
                self.assertEqual(first.view.board, second.view.board)
                self.assertTrue(all(not p.revealed_cards for p in first.view.players))
            finally:
                for client in clients:
                    client.close()
                server.close()
                server_thread.join(timeout=3)
                worker.stop()
                worker_thread.join(timeout=3)
            self.assertFalse(server_thread.is_alive())
            self.assertFalse(worker_thread.is_alive())
