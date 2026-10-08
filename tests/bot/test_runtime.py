from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from poker.application.commands import ActCommand, JoinCommand
from poker.application.views import CommandResponse, ErrorInfo
from poker.bot.runtime import passive_decision, run_bot
from poker.client.cli import PokerCli
from poker.client.parser import CommandParser
from poker.domain.models import ActionOption
from poker.domain.types import ActionKind, ErrorCode
from shared_logging import LoggingConfig, configure_logging, shutdown_logging
from tests.client.test_polling import FakeClock, RecordingClient, server_view


class BotRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.path = configure_logging(LoggingConfig(Path(self.temp.name), "bot-test", "bot"))

    def tearDown(self) -> None:
        shutdown_logging()
        self.temp.cleanup()

    def events(self) -> list[str]:
        return [json.loads(line)["event"] for line in self.path.read_text(encoding="utf-8").splitlines()]

    def test_passive_bot_only_chooses_server_offered_options(self) -> None:
        view = server_view("h1")
        offered = replace(view, me=replace(view.me, legal_actions=(ActionOption(ActionKind.FOLD), ActionOption(ActionKind.CALL, pay=10))))
        self.assertEqual(passive_decision(offered), "call")
        self.assertIsNone(passive_decision(view))

    def test_one_cli_owner_submits_decision_and_exits_after_confirmation(self) -> None:
        view = server_view("h1")
        client = RecordingClient(FakeClock(), [CommandResponse(view=view), CommandResponse(view=view)])
        run_bot(PokerCli(client, CommandParser()), "Bot", lambda _: "check", max_actions=1)
        self.assertEqual([type(c) for c in client.commands], [JoinCommand, ActCommand])
        self.assertEqual((client.connects, client.closes), (1, 1))
        self.assertEqual(self.events().count("bot.decision_completed"), 1)
        self.assertEqual(self.events().count("bot.command_submitted"), 1)
        self.assertEqual(self.events()[-1], "bot.stopped")

    def test_network_failure_and_join_rejection_exit_with_bot_failure(self) -> None:
        for client in (RecordingClient(FakeClock(), [], OSError("offline")),
                       RecordingClient(FakeClock(), [CommandResponse(error=ErrorInfo(ErrorCode.TABLE_FULL, "full"))])):
            run_bot(PokerCli(client, CommandParser()), "Bot")
            self.assertEqual(client.closes, 1)
        self.assertEqual(self.events().count("bot.failed"), 2)

    def test_decision_failure_propagates_and_closes_cli(self) -> None:
        client = RecordingClient(FakeClock(), [CommandResponse(view=server_view("h1"))])
        def fail(_: object) -> str:
            raise RuntimeError("decision failed")
        with self.assertRaisesRegex(RuntimeError, "decision failed"):
            run_bot(PokerCli(client, CommandParser()), "Bot", fail)
        self.assertEqual(client.closes, 1)
        self.assertIn("bot.failed", self.events())
