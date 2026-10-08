from threading import Event, get_ident
import unittest

from poker.agent.cli_bridge import (
    CliActionResult, CliObservation, CliStopped, PokerCliAgentAdapter,
)
from poker.application.views import CommandResponse
from poker.client.cli import PokerCli
from poker.client.guarded import ViewToken
from poker.client.parser import CommandParser
from poker.domain.models import PlayerAction
from poker.domain.types import ActionKind
from tests.client.test_polling import FakeClock, RecordingClient, server_view


class AgentCliBridgeTests(unittest.TestCase):
    def test_adapter_correlates_actions_and_keeps_socket_on_one_owner(self) -> None:
        view = server_view("h1", 4)
        client = RecordingClient(FakeClock(), [CommandResponse(view=view)] * 3)
        adapter = PokerCliAgentAdapter(PokerCli(client, CommandParser()), poll_interval=60)
        try:
            adapter.start("Agent")
            joined = adapter.next_event(1)
            self.assertIsInstance(joined, CliObservation)
            adapter.submit(PlayerAction(ActionKind.CHECK), ViewToken.from_view(view), "d1")
            with self.assertRaisesRegex(ValueError, "already submitted"):
                adapter.submit(PlayerAction(ActionKind.CHECK), ViewToken.from_view(view), "d1")
            observed = [adapter.next_event(1) for _ in range(3)]
            self.assertIsInstance(observed[0], CliObservation)
            self.assertIsInstance(observed[1], CliObservation)
            self.assertIsInstance(observed[2], CliActionResult)
            result = observed[2]
            assert isinstance(result, CliActionResult)
            self.assertEqual(result.result.request_id, "d1")
            self.assertIsNotNone(result.result.response)
        finally:
            self.assertTrue(adapter.close())
        self.assertEqual(adapter.next_event(1), CliStopped())
        self.assertIsNone(adapter.next_event())
        self.assertEqual(len(set(client.owners)), 1)
        self.assertNotIn(get_ident(), client.owners)
        self.assertEqual((client.connects, client.closes), (1, 1))
        self.assertTrue(adapter.close())
        with self.assertRaises(RuntimeError):
            adapter.start("Again")
        with self.assertRaises(RuntimeError):
            adapter.start_hand()

    def test_connection_error_and_unexpected_failure_are_terminal_events(self) -> None:
        for error in (OSError("offline"), ValueError("bad response")):
            with self.subTest(error=error):
                client = RecordingClient(FakeClock(), [error])
                adapter = PokerCliAgentAdapter(PokerCli(client, CommandParser()))
                adapter.start("Agent")
                event = adapter.next_event(1)
                self.assertIsInstance(event, CliStopped)
                assert isinstance(event, CliStopped)
                self.assertIsNotNone(event.error)
                self.assertTrue(adapter.close())
                self.assertEqual(client.closes, 1)
                self.assertIsNone(adapter.next_event())

    def test_close_reports_unfinished_owner_and_can_be_checked_again(self) -> None:
        entered, release = Event(), Event()

        class BlockingClient(RecordingClient):
            def connect(self) -> None:
                entered.set()
                release.wait(2)
                super().connect()

        client = BlockingClient(FakeClock(), [CommandResponse(view=server_view())])
        adapter = PokerCliAgentAdapter(PokerCli(client, CommandParser()), close_timeout=0.001)
        try:
            adapter.start("Agent")
            self.assertTrue(entered.wait(1))
            self.assertFalse(adapter.close())
        finally:
            release.set()
        # Drain join plus stopped; cleanup timeout is deliberately short above.
        self.assertIsInstance(adapter.next_event(1), CliObservation)
        self.assertEqual(adapter.next_event(1), CliStopped())
        self.assertTrue(adapter.close())

    def test_invalid_names_and_timeouts_fail_before_start(self) -> None:
        client = RecordingClient(FakeClock(), [])
        adapter = PokerCliAgentAdapter(PokerCli(client, CommandParser()))
        for name in ("", "\nstart", "x" * 65):
            with self.subTest(name=name), self.assertRaises(ValueError):
                adapter.start(name)
        for timeout in (-1.0, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                adapter.next_event(timeout)
        self.assertTrue(adapter.close())
        self.assertEqual(client.connects, 0)
