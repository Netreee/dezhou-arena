from dataclasses import replace
from queue import Queue
from threading import get_ident
import unittest

from poker.application.commands import ActCommand, JoinCommand, StateCommand
from poker.application.views import CommandResponse, ErrorInfo
from poker.client.cli import PokerCli
from poker.client.guarded import GuardedInput, GuardedResult, ViewToken
from poker.client.parser import CommandParser
from poker.domain.types import ErrorCode
from tests.client.test_polling import FakeClock, RecordingClient, server_view


class GuardedCliTests(unittest.TestCase):
    def run_guarded(
        self, client: RecordingClient, item: GuardedInput,
    ) -> tuple[list[GuardedResult], list[str]]:
        inputs: Queue[str | None] = Queue()
        guards: Queue[GuardedInput] = Queue()
        inputs.put("join Agent")
        results: list[GuardedResult] = []
        sources: list[str] = []

        def response(source: str, result: CommandResponse) -> None:
            sources.append(source)
            if source == "join Agent":
                guards.put(item)

        def completed(result: GuardedResult) -> None:
            results.append(result)
            inputs.put("quit")

        PokerCli(client, CommandParser()).run(
            input_queue=inputs, guarded_queue=guards,
            on_response=response, on_guarded_result=completed,
            clock=client.clock, write=lambda _: None,
        )
        return results, sources

    def test_fresh_action_uses_frozen_hand_and_atomic_revision(self) -> None:
        view = server_view("h1", 7)
        client = RecordingClient(FakeClock(), [CommandResponse(view=view)] * 3)
        results, sources = self.run_guarded(client, GuardedInput("r1", "check", ViewToken.from_view(view)))
        self.assertEqual([type(c) for c in client.commands], [JoinCommand, StateCommand, ActCommand])
        command = client.commands[-1]
        assert isinstance(command, ActCommand)
        self.assertEqual((command.hand_id, command.expected_revision), (view.hand_id, 7))
        self.assertEqual(results, [GuardedResult("r1", response=CommandResponse(view=view))])
        self.assertEqual(sources, ["join Agent", "guarded_refresh", "check"])
        self.assertEqual(set(client.owners), {get_ident()})
        self.assertEqual((client.connects, client.closes), (1, 1))

    def test_old_decision_never_gets_attached_to_new_hand(self) -> None:
        old, new = server_view("h1", 7), server_view("h2", 8)
        client = RecordingClient(FakeClock(), [CommandResponse(view=old), CommandResponse(view=new)])
        results, sources = self.run_guarded(client, GuardedInput("r1", "check", ViewToken.from_view(old)))
        self.assertEqual(results, [GuardedResult("r1", error="stale_observation")])
        self.assertEqual([type(c) for c in client.commands], [JoinCommand, StateCommand])
        self.assertEqual(sources[-1], "guarded_refresh")

    def test_changed_revision_or_actor_rejects_before_send(self) -> None:
        old = server_view("h1", 7)
        for new in (replace(old, revision=8), replace(old, actor_id=None)):
            with self.subTest(new=new):
                client = RecordingClient(FakeClock(), [CommandResponse(view=old), CommandResponse(view=new)])
                results, _ = self.run_guarded(client, GuardedInput("r1", "check", ViewToken.from_view(old)))
                self.assertEqual(results[0].error, "stale_observation")
                self.assertEqual(len(client.commands), 2)

    def test_only_action_grammar_is_allowed(self) -> None:
        view = server_view("h1", 7)
        for line in ("start", "state", "join Other", "raise_to no", "quit"):
            with self.subTest(line=line):
                client = RecordingClient(FakeClock(), [CommandResponse(view=view)])
                results, _ = self.run_guarded(client, GuardedInput("r1", line, ViewToken.from_view(view)))
                self.assertEqual(results[0].error, "invalid_action")
                self.assertEqual(len(client.commands), 1)

    def test_refresh_failure_does_not_reuse_cached_view(self) -> None:
        view = server_view("h1", 7)
        error = CommandResponse(error=ErrorInfo(ErrorCode.NOT_SEATED, "not seated"))
        client = RecordingClient(FakeClock(), [CommandResponse(view=view), error])
        results, _ = self.run_guarded(client, GuardedInput("r1", "check", ViewToken.from_view(view)))
        self.assertEqual(results[0].error, "refresh_failed")
        self.assertEqual(len(client.commands), 2)

    def test_server_rejection_is_a_correlated_response_not_local_error(self) -> None:
        view = server_view("h1", 7)
        rejection = CommandResponse(error=ErrorInfo(ErrorCode.INVALID_ACTION, "illegal"))
        client = RecordingClient(FakeClock(), [CommandResponse(view=view), CommandResponse(view=view), rejection])
        results, _ = self.run_guarded(client, GuardedInput("r1", "check", ViewToken.from_view(view)))
        self.assertEqual(results, [GuardedResult("r1", response=rejection)])
        self.assertEqual(len(client.commands), 3)

    def test_connection_failure_reports_pending_action_once_without_retry(self) -> None:
        view = server_view("h1", 7)
        client = RecordingClient(FakeClock(), [CommandResponse(view=view)] * 2 + [OSError("lost response")])
        results, _ = self.run_guarded(client, GuardedInput("r1", "check", ViewToken.from_view(view)))
        self.assertEqual(results, [GuardedResult("r1", error="connection_failed")])
        self.assertEqual(len(client.commands), 3)
        self.assertEqual(client.closes, 1)

    def test_result_requires_exactly_one_outcome(self) -> None:
        with self.assertRaises(ValueError):
            GuardedResult("r1")
        with self.assertRaises(ValueError):
            GuardedResult("r1", response=CommandResponse(view=server_view()), error="bad")
