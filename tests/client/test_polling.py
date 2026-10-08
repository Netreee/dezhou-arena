import unittest
from dataclasses import replace
from queue import Empty, Queue
from threading import get_ident
from unittest.mock import patch

from poker.application.commands import ActCommand, Command, JoinCommand, StateCommand
from poker.application.views import (
    AwardView, CommandResponse, ErrorInfo, PlayerView, PlayerViewBuilder, ResultView, ShareView,
)
from poker.client.cli import PokerCli
from poker.client.parser import CommandParser
from poker.domain.models import ActionOption, HandResult
from poker.domain.types import ActionKind, ErrorCode, HandId, HandPhase, PlayerId
from poker.transport.interfaces import CommandClient
from tests.fixtures import make_table


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class TimedInputs(Queue[str | None]):
    """Jump time to input or timeout; no real sleep or scheduling library."""

    def __init__(self, clock: FakeClock, events: list[tuple[float, str | None]]) -> None:
        super().__init__()
        self.clock = clock
        self.events = list(events)

    def get(self, block: bool = True, timeout: float | None = None) -> str | None:
        assert self.events, "Every test input script must end with quit or EOF"
        when, line = self.events[0]
        deadline = self.clock.now + timeout if timeout is not None else float("inf")
        if when <= deadline:
            self.clock.now = max(self.clock.now, when)
            self.events.pop(0)
            return line
        self.clock.now = deadline
        raise Empty


class RecordingClient(CommandClient):
    def __init__(
        self, clock: FakeClock, responses: list[CommandResponse | BaseException],
        connect_error: OSError | None = None,
    ) -> None:
        self.clock = clock
        self.responses = list(responses)
        self.connect_error = connect_error
        self.commands: list[Command] = []
        self.times: list[float] = []
        self.owners: list[int] = []
        self.connects = 0
        self.closes = 0
        self.sending = False

    def connect(self) -> None:
        self.connects += 1
        self.owners.append(get_ident())
        if self.connect_error is not None:
            raise self.connect_error

    def send(self, command: Command) -> CommandResponse:
        assert not self.sending, "Concurrent client access"
        self.sending = True
        try:
            self.owners.append(get_ident())
            self.commands.append(command)
            self.times.append(self.clock())
            assert self.responses, "Unexpected network send"
            response = self.responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return response
        finally:
            self.sending = False

    def close(self) -> None:
        self.closes += 1
        self.owners.append(get_ident())


def server_view(hand_id: str | None = None, revision: int = 1) -> PlayerView:
    table = make_table(with_hand=hand_id is not None)
    if table.hand is not None:
        assert hand_id is not None
        table.hand.id = HandId(hand_id)
    table.revision = revision
    return PlayerViewBuilder().build(table, PlayerId("p1"))


class PollingTests(unittest.TestCase):
    def test_structured_callbacks_preserve_sources_rejection_input_error_and_close(self) -> None:
        clock = FakeClock()
        joined = CommandResponse(view=server_view("h1"))
        rejected = CommandResponse(error=ErrorInfo(ErrorCode.INVALID_ACTION, "拒绝"))
        client = RecordingClient(clock, [joined, joined, rejected])
        responses: list[tuple[str, CommandResponse]] = []
        lifecycle: list[tuple[str, str | None, str | None]] = []
        PokerCli(client, CommandParser()).run(
            input_queue=TimedInputs(clock, [(0, "join 甲"), (0.2, "raise_to zero"), (1.1, "check"), (1.2, "quit")]),
            clock=clock, write=lambda _: None,
            on_response=lambda source, response: responses.append((source, response)),
            on_lifecycle=lambda code, message, source: lifecycle.append((code, message, source)),
        )
        self.assertEqual(responses, [("join 甲", joined), ("poll", joined), ("check", rejected)])
        self.assertEqual([item[0] for item in lifecycle], ["input_error", "closed"])
        self.assertEqual(lifecycle[0][2], "raise_to zero")
        self.assertEqual(set(client.owners), {get_ident()})

    def test_structured_connection_failure_is_followed_by_closed(self) -> None:
        clock = FakeClock()
        client = RecordingClient(clock, [], OSError("offline"))
        lifecycle: list[str] = []
        PokerCli(client, CommandParser()).run(
            input_queue=TimedInputs(clock, [(0, "quit")]), clock=clock, write=lambda _: None,
            on_lifecycle=lambda code, message, source: lifecycle.append(code),
        )
        self.assertEqual(lifecycle, ["connection_failed", "closed"])

    def test_no_poll_before_join_then_periodic_state_without_duplicate_display(self) -> None:
        clock = FakeClock()
        response = CommandResponse(view=server_view())
        client = RecordingClient(clock, [response, response, response])
        cli = PokerCli(client, CommandParser())
        output: list[str] = []
        cli.run(input_queue=TimedInputs(clock, [(2, "join 甲"), (4.5, "quit")]),
                clock=clock, write=output.append)
        self.assertEqual([type(c) for c in client.commands], [JoinCommand, StateCommand, StateCommand])
        self.assertEqual(client.times, [2, 3, 4])
        self.assertEqual(sum("牌桌:" in block for block in output), 1)
        self.assertEqual((client.connects, client.closes), (1, 1))

    def test_failed_join_does_not_start_polling(self) -> None:
        clock = FakeClock()
        response = CommandResponse(error=ErrorInfo(ErrorCode.TABLE_FULL, "满员"))
        client = RecordingClient(clock, [response])
        cli = PokerCli(client, CommandParser())
        output: list[str] = []
        cli.run(input_queue=TimedInputs(clock, [(0, "join 甲"), (8, "quit")]),
                clock=clock, write=output.append)
        self.assertEqual(len(client.commands), 1)
        self.assertIsNone(cli.latest)
        self.assertIn("服务器拒绝 [table_full]: 满员", output)

    def test_due_poll_precedes_action_and_attaches_latest_hand_id(self) -> None:
        clock = FakeClock()
        current = server_view("new-hand", 2)
        client = RecordingClient(clock, [CommandResponse(view=server_view()),
                                         CommandResponse(view=current), CommandResponse(view=current)])
        cli = PokerCli(client, CommandParser())
        cli.run(input_queue=TimedInputs(clock, [(0, "join 甲"), (1, "check"), (1.1, "quit")]),
                clock=clock, write=lambda _: None)
        self.assertEqual([type(c) for c in client.commands], [JoinCommand, StateCommand, ActCommand])
        action = client.commands[-1]
        assert isinstance(action, ActCommand)
        self.assertEqual(action.hand_id, HandId("new-hand"))
        self.assertEqual(client.times, [0, 1, 1])
        self.assertEqual(set(client.owners), {get_ident()})

    def test_failed_poll_and_action_preserve_last_successful_view(self) -> None:
        clock = FakeClock()
        current = server_view("h-latest", 2)
        error = CommandResponse(error=ErrorInfo(ErrorCode.INVALID_ACTION, "拒绝"))
        client = RecordingClient(clock, [CommandResponse(view=server_view()),
                                         CommandResponse(view=current), error, error])
        cli = PokerCli(client, CommandParser())
        output: list[str] = []
        cli.run(input_queue=TimedInputs(clock, [(0, "join 甲"), (2.1, "check"), (2.2, "quit")]),
                clock=clock, write=output.append)
        self.assertIs(cli.latest, current)
        action = client.commands[-1]
        assert isinstance(action, ActCommand)
        self.assertEqual(action.hand_id, current.hand_id)
        self.assertEqual(sum("服务器拒绝" in line for line in output), 2)
        self.assertEqual([type(c) for c in client.commands], [JoinCommand, StateCommand, StateCommand, ActCommand])

    def test_input_traffic_does_not_postpone_polling_deadline(self) -> None:
        clock = FakeClock()
        response = CommandResponse(view=server_view("h1"))
        client = RecordingClient(clock, [response] * 5)
        cli = PokerCli(client, CommandParser())
        cli.run(input_queue=TimedInputs(clock, [(0, "join 甲"), (0.4, "state"), (0.9, "check"),
                                                (1.2, "state"), (1.3, "quit")]),
                clock=clock, write=lambda _: None)
        self.assertEqual([type(c) for c in client.commands],
                         [JoinCommand, StateCommand, ActCommand, StateCommand, StateCommand])
        self.assertAlmostEqual(client.times[3], 1)
        self.assertEqual(set(client.owners), {get_ident()})

    def test_syntax_error_does_not_send_and_next_command_still_works(self) -> None:
        clock = FakeClock()
        response = CommandResponse(view=server_view("h1"))
        client = RecordingClient(clock, [response, response])
        cli = PokerCli(client, CommandParser())
        output: list[str] = []
        cli.run(input_queue=TimedInputs(clock, [(0, "join 甲"), (0.1, "raise_to zero"),
                                                (0.2, "check"), (0.3, None)]),
                clock=clock, write=output.append)
        self.assertEqual([type(c) for c in client.commands], [JoinCommand, ActCommand])
        self.assertEqual(sum("输入错误:" in line for line in output), 1)
        self.assertEqual(client.closes, 1)

    def test_real_input_thread_only_produces_text_and_main_thread_owns_client(self) -> None:
        clock = FakeClock()
        response = CommandResponse(view=server_view())
        client = RecordingClient(clock, [response])
        cli = PokerCli(client, CommandParser())
        lines = iter(("join 甲", "quit"))
        input_owners: list[int] = []

        def read() -> str:
            input_owners.append(get_ident())
            return next(lines)

        with patch("builtins.input", side_effect=read):
            cli.run(clock=clock, write=lambda _: None)
        self.assertEqual(len(client.commands), 1)
        self.assertEqual(set(client.owners), {get_ident()})
        self.assertEqual(len(set(input_owners)), 1)
        self.assertNotIn(get_ident(), input_owners)
        self.assertEqual((client.connects, client.closes), (1, 1))

    def test_console_eof_closes_client_without_polling(self) -> None:
        clock = FakeClock()
        client = RecordingClient(clock, [])
        with patch("builtins.input", side_effect=EOFError):
            PokerCli(client, CommandParser()).run(clock=clock, write=lambda _: None)
        self.assertEqual(client.commands, [])
        self.assertEqual((client.connects, client.closes), (1, 1))

    def test_connect_failure_closes_client_and_does_not_retry(self) -> None:
        clock = FakeClock()
        client = RecordingClient(clock, [], OSError("offline"))
        output: list[str] = []
        PokerCli(client, CommandParser()).run(input_queue=TimedInputs(clock, [(0, "quit")]),
                                             clock=clock, write=output.append)
        self.assertEqual((client.connects, client.closes), (1, 1))
        self.assertEqual(client.commands, [])
        self.assertEqual(output, ["连接结束: offline"])

    def test_connection_loss_stops_without_resending_or_further_polling(self) -> None:
        clock = FakeClock()
        client = RecordingClient(clock, [CommandResponse(view=server_view()), OSError("closed")])
        output: list[str] = []
        PokerCli(client, CommandParser()).run(input_queue=TimedInputs(clock, [(0, "join 甲"), (3, "quit")]),
                                             clock=clock, write=output.append)
        self.assertEqual([type(c) for c in client.commands], [JoinCommand, StateCommand])
        self.assertEqual((client.connects, client.closes), (1, 1))
        self.assertIn("连接结束: closed", output)

    def test_malformed_response_is_not_misreported_as_input_syntax(self) -> None:
        clock = FakeClock()
        client = RecordingClient(clock, [ValueError("malformed response")])
        output: list[str] = []
        with self.assertRaisesRegex(ValueError, "malformed response"):
            PokerCli(client, CommandParser()).run(input_queue=TimedInputs(clock, [(0, "join 甲"), (1, "quit")]),
                                                 clock=clock, write=output.append)
        self.assertEqual(client.closes, 1)
        self.assertFalse(any("输入错误" in line for line in output))

    def test_keyboard_interrupt_closes_client(self) -> None:
        clock = FakeClock()
        client = RecordingClient(clock, [KeyboardInterrupt()])
        PokerCli(client, CommandParser()).run(input_queue=TimedInputs(clock, [(0, "join 甲"), (1, "quit")]),
                                             clock=clock, write=lambda _: None)
        self.assertEqual(client.closes, 1)

    def test_invalid_interval_is_rejected_before_connect(self) -> None:
        for interval in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(interval=interval):
                clock = FakeClock()
                client = RecordingClient(clock, [])
                with self.assertRaises(ValueError):
                    PokerCli(client, CommandParser()).run(poll_interval=interval)
                self.assertEqual(client.connects, 0)


class DisplayTests(unittest.TestCase):
    def test_latest_public_and_private_fields_and_server_action_amounts_are_displayed(self) -> None:
        table = make_table(with_hand=True)
        assert table.hand is not None
        options = (ActionOption(ActionKind.CALL, pay=35),
                   ActionOption(ActionKind.RAISE_TO, min_to=100, max_to=260),
                   ActionOption(ActionKind.ALL_IN, pay=85))
        view = PlayerViewBuilder().build(table, PlayerId("p1"), options)
        output = PokerCli._render(view)
        self.assertIn("公共牌: " + " ".join(view.board), output)
        self.assertIn("本人底牌: " + " ".join(view.me.hole_cards), output)
        self.assertIn("行动者: 甲", output)
        self.assertIn("0 | 甲 (你) | 990 | active | 0 | 10", output)
        self.assertIn("1 | 乙 | 990 | active | 0 | 10", output)
        self.assertIn("call (支付 35)", output)
        self.assertIn("raise_to 100..260 (本街总额)", output)
        self.assertIn("all_in (支付 85)", output)
        for card in table.hand.players[PlayerId("p2")].hole_cards:
            self.assertNotIn(card.code, output)

    def test_explicit_public_reveal_is_displayed(self) -> None:
        table = make_table(with_hand=True)
        assert table.hand is not None
        table.hand.phase = HandPhase.COMPLETE
        table.last_result = HandResult(table.hand.id, (), revealed_hands={
            PlayerId("p2"): table.hand.players[PlayerId("p2")].hole_cards,
        })
        view = PlayerViewBuilder().build(table, PlayerId("p1"))
        for card in table.hand.players[PlayerId("p2")].hole_cards:
            self.assertIn(card.code, PokerCli._render(view))

    def test_previous_result_and_refunds_use_server_values_without_revealing_new_cards(self) -> None:
        view = server_view("new")
        result = ResultView(HandId("old"),
                            (AwardView(13, (PlayerId("p1"), PlayerId("p2")),
                                       (ShareView(PlayerId("p1"), 7), ShareView(PlayerId("p2"), 6))),),
                            (ShareView(PlayerId("p1"), 3),))
        output = PokerCli._render(replace(view, result=result))
        self.assertIn("上一手结果: old", output)
        self.assertIn("池 1: 13 | 分配: 甲: 7, 乙: 6", output)
        self.assertIn("退款: 甲: 3", output)
