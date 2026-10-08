import unittest

from poker.application.commands import ActCommand, Command, JoinCommand, StateCommand
from poker.application.views import CommandResponse, PlayerViewBuilder
from poker.client.cli import PokerCli
from poker.client.parser import CommandParser
from poker.domain.types import ActionKind, HandId, PlayerId, TableId
from poker.transport.interfaces import CommandClient
from tests.fixtures import make_table


class FakeClient(CommandClient):
    def __init__(self, response: CommandResponse) -> None:
        self.response = response
        self.commands: list[Command] = []

    def connect(self) -> None:
        pass

    def send(self, command: Command) -> CommandResponse:
        self.commands.append(command)
        return self.response

    def close(self) -> None:
        pass


class CliTests(unittest.TestCase):
    def test_table_selection_preserves_names_and_default_join(self) -> None:
        self.assertEqual(CommandParser(TableId("1002")).parse("join Sample Player"), JoinCommand("Sample Player", TableId("1002")))
        self.assertEqual(CommandParser().parse("join_table 1001 Sample Player"), JoinCommand("Sample Player", TableId("1001")))
        self.assertEqual(CommandParser().parse("join --table Person"), JoinCommand("--table Person"))
        for line in ("join_table", "join_table 1001", "join_table bad/table 甲"):
            with self.assertRaises(ValueError):
                CommandParser().parse(line)

    def test_name_and_raise_total_are_parsed(self) -> None:
        parser = CommandParser()
        self.assertEqual(parser.parse("join Sample Player"), JoinCommand("Sample Player"))
        command = parser.parse("raise_to 100", HandId("h1"))
        assert isinstance(command, ActCommand)
        self.assertEqual((command.action.kind, command.action.to), (ActionKind.RAISE_TO, 100))

    def test_action_requires_current_hand(self) -> None:
        with self.assertRaises(ValueError):
            CommandParser().parse("fold")

    def test_extra_or_invalid_arguments_are_rejected(self) -> None:
        parser = CommandParser()
        for line in ("check 10", "raise_to zero", "raise_to -2", ""):
            with self.assertRaises(ValueError):
                parser.parse(line, HandId("h1"))

    def test_refresh_then_submit_attaches_server_hand_id(self) -> None:
        view = PlayerViewBuilder().build(make_table(with_hand=True), PlayerId("p1"))
        client = FakeClient(CommandResponse(view=view))
        cli = PokerCli(client, CommandParser())
        cli.refresh()
        cli.submit("check")
        self.assertIsInstance(client.commands[0], StateCommand)
        command = client.commands[1]
        assert isinstance(command, ActCommand)
        self.assertEqual(command.hand_id, view.hand_id)
        self.assertEqual(cli.latest, view)
