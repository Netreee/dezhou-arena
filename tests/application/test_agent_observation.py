"""Public agent observations use exactly the ordinary player's projection."""

from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from poker.application.commands import ActCommand, SessionContext, StartHandCommand, StateCommand
from poker.application.service import TableService
from poker.application.views import CommandResponse, PlayerViewBuilder
from poker.domain.errors import RuleViolation
from poker.domain.models import PlayerAction
from poker.domain.types import ActionKind, ErrorCode, HandPhase, PlayerId
from poker.persistence.snapshot import TableSnapshotCodec
from poker.persistence.sqlite import SqliteTableRepository
from poker.transport.codec import JsonLineCodec
from tests.engine.test_start_hand import hand_of, opening_engine, opening_table
from tests.fixtures import MemoryRepository


class PublicHistoryTests(unittest.TestCase):
    def test_two_through_six_players_have_complete_blinds_and_safe_shared_history(self) -> None:
        for count in range(2, 7):
            with self.subTest(players=count):
                table = opening_table((1000,) * count)
                engine = opening_engine()
                engine.start_hand(table, PlayerId("p0"))
                hand = hand_of(table)
                first = PlayerViewBuilder().build(table, PlayerId("p0"))
                second = PlayerViewBuilder().build(table, PlayerId("p1"))
                self.assertTrue(first.history_complete)
                self.assertEqual(first.config, table.config)
                self.assertEqual(first.action_history, second.action_history)
                self.assertEqual([event.sequence for event in first.action_history], [1, 2])
                self.assertEqual([event.kind for event in first.action_history], ["small_blind", "big_blind"])
                self.assertEqual([event.pay for event in first.action_history], [5, 10])
                self.assertEqual([event.to for event in first.action_history], [5, 10])
                self.assertEqual([event.stack for event in first.action_history], [995, 990])
                self.assertTrue(all(event.phase is HandPhase.PREFLOP for event in first.action_history))
                allowed = {"sequence", "phase", "player_id", "kind", "pay", "to", "stack"}
                self.assertTrue(all(set(asdict(event)) == allowed for event in first.action_history))
                self.assertNotEqual(first.me.hole_cards, second.me.hole_cards)
                self.assertEqual(tuple(hand.action_history), first.action_history)

    def test_short_blinds_record_actual_payments_even_when_opening_runs_out(self) -> None:
        table = opening_table((2, 3))
        opening_engine().start_hand(table, PlayerId("p0"))
        hand = hand_of(table)
        self.assertEqual(hand.phase, HandPhase.COMPLETE)
        self.assertEqual([event.pay for event in hand.action_history], [2, 3])
        self.assertEqual([event.to for event in hand.action_history], [2, 3])
        self.assertEqual([event.stack for event in hand.action_history], [0, 0])
        self.assertTrue(hand.history_complete)

    def test_action_payments_are_preserved_across_street_reset_and_missed_polls(self) -> None:
        table = opening_table((1000, 1000))
        engine = opening_engine()
        engine.start_hand(table, PlayerId("p0"))
        hand = hand_of(table)
        before = PlayerViewBuilder().build(table, PlayerId("p0"))
        engine.act(table, PlayerId("p0"), hand.id, PlayerAction(ActionKind.CALL))
        engine.act(table, PlayerId("p1"), hand.id, PlayerAction(ActionKind.CHECK))
        engine.act(table, PlayerId("p1"), hand.id, PlayerAction(ActionKind.BET_TO, 20))
        after = PlayerViewBuilder().build(table, PlayerId("p0"))
        self.assertEqual(len(before.action_history), 2)
        self.assertEqual([event.sequence for event in after.action_history], [1, 2, 3, 4, 5])
        call, check, bet = after.action_history[2:]
        self.assertEqual((call.phase, call.pay, call.to, call.stack), (HandPhase.PREFLOP, 5, 10, 990))
        self.assertEqual((check.phase, check.pay, check.to, check.stack), (HandPhase.PREFLOP, 0, 10, 990))
        self.assertEqual((bet.phase, bet.pay, bet.to, bet.stack), (HandPhase.FLOP, 20, 20, 970))
        self.assertEqual(hand.players[PlayerId("p0")].street_commit, 0)

    def test_fold_settlement_does_not_rewrite_raise_before_refund(self) -> None:
        table = opening_table((1000, 1000))
        engine = opening_engine()
        engine.start_hand(table, PlayerId("p0"))
        hand = hand_of(table)
        engine.act(table, PlayerId("p0"), hand.id, PlayerAction(ActionKind.RAISE_TO, 100))
        engine.act(table, PlayerId("p1"), hand.id, PlayerAction(ActionKind.FOLD))
        raised, folded = hand.action_history[2:]
        self.assertEqual((raised.pay, raised.to, raised.stack), (95, 100, 900))
        self.assertEqual((folded.pay, folded.to, folded.stack), (0, 10, 990))
        self.assertEqual(hand.phase, HandPhase.COMPLETE)
        self.assertEqual(hand.players[PlayerId("p0")].hand_commit, 10)
        self.assertEqual(table.player(PlayerId("p0")).stack, 1010)
        self.assertFalse(any(player.revealed_cards for player in PlayerViewBuilder().build(table, PlayerId("p1")).players))
        engine.start_hand(table, PlayerId("p0"))
        self.assertEqual(len(hand_of(table).action_history), 2)
        self.assertTrue(hand_of(table).history_complete)
        self.assertNotEqual(hand_of(table).id, hand.id)

    def test_rejected_action_does_not_append_history(self) -> None:
        table = opening_table((1000, 1000))
        engine = opening_engine()
        engine.start_hand(table, PlayerId("p0"))
        hand = hand_of(table)
        previous = tuple(hand.action_history)
        with self.assertRaises(RuleViolation):
            engine.act(table, PlayerId("p0"), hand.id, PlayerAction(ActionKind.CHECK))
        self.assertEqual(tuple(hand.action_history), previous)

    def test_history_and_config_survive_wire_and_sqlite_reopen(self) -> None:
        table = opening_table((1000, 1000))
        engine = opening_engine()
        engine.start_hand(table, PlayerId("p0"))
        hand = hand_of(table)
        engine.act(table, PlayerId("p0"), hand.id, PlayerAction(ActionKind.CALL))
        response = CommandResponse(view=PlayerViewBuilder().build(table, PlayerId("p1")))
        codec = JsonLineCodec()
        self.assertEqual(codec.decode_response(codec.encode_response(response)), response)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            SqliteTableRepository(path).save(table)
            saved = SqliteTableRepository(path).load(table.id)
            self.assertEqual(saved, table)

    def test_legacy_snapshot_history_is_explicitly_incomplete_and_stays_incomplete(self) -> None:
        table = opening_table((1000, 1000))
        engine = opening_engine()
        engine.start_hand(table, PlayerId("p0"))
        data = json.loads(TableSnapshotCodec().encode(table))
        del data["hand"]["action_history"]
        del data["hand"]["history_complete"]
        restored = TableSnapshotCodec().decode(json.dumps(data))
        hand = hand_of(restored)
        self.assertFalse(hand.history_complete)
        self.assertEqual(hand.action_history, [])
        engine.act(restored, PlayerId("p0"), hand.id, PlayerAction(ActionKind.CALL))
        self.assertFalse(hand.history_complete)
        self.assertEqual(len(hand.action_history), 1)
        self.assertFalse(PlayerViewBuilder().build(restored, PlayerId("p0")).history_complete)

    def test_legacy_wire_view_does_not_invent_config_or_complete_history(self) -> None:
        table = opening_table((1000, 1000))
        opening_engine().start_hand(table, PlayerId("p0"))
        codec = JsonLineCodec()
        data = json.loads(codec.encode_response(CommandResponse(view=PlayerViewBuilder().build(table, PlayerId("p0")))))
        for key in ("config", "action_history", "history_complete"):
            del data["view"][key]
        decoded = codec.decode_response(json.dumps(data))
        assert decoded.view is not None
        self.assertIsNone(decoded.view.config)
        self.assertEqual(decoded.view.action_history, ())
        self.assertFalse(decoded.view.history_complete)


class RevisionGuardTests(unittest.TestCase):
    def test_revision_and_hand_id_are_independent_guards_across_hands(self) -> None:
        table = opening_table((1000, 1000))
        service = TableService(table.id, MemoryRepository(table), opening_engine(), PlayerViewBuilder())
        session = SessionContext(PlayerId("p0"))
        first = service.handle(StartHandCommand(), session)
        assert first.view is not None and first.view.hand_id is not None
        self.assertTrue(service.handle(ActCommand(first.view.hand_id, PlayerAction(ActionKind.FOLD), 1), session).ok)
        second = service.handle(StartHandCommand(), session)
        assert second.view is not None and second.view.hand_id is not None
        actor = SessionContext(second.view.actor_id)
        stale = service.handle(ActCommand(second.view.hand_id, PlayerAction(ActionKind.CALL), 1), actor)
        assert stale.error is not None
        self.assertEqual(stale.error.code, ErrorCode.STALE_STATE)
        wrong_hand = service.handle(ActCommand(first.view.hand_id, PlayerAction(ActionKind.CALL), second.view.revision), actor)
        assert wrong_hand.error is not None
        self.assertEqual(wrong_hand.error.code, ErrorCode.HAND_MISMATCH)

    def test_stale_action_is_rejected_without_mutation_then_fresh_action_succeeds(self) -> None:
        table = opening_table((1000, 1000))
        repository = MemoryRepository(table)
        service = TableService(table.id, repository, opening_engine(), PlayerViewBuilder())
        session = SessionContext(PlayerId("p0"))
        started = service.handle(StartHandCommand(), session)
        assert started.view is not None and started.view.hand_id is not None
        hand_id = started.view.hand_id
        saved = repository.load(table.id)
        stale = service.handle(ActCommand(hand_id, PlayerAction(ActionKind.CALL), expected_revision=0), session)
        assert stale.error is not None
        self.assertEqual(stale.error.code, ErrorCode.STALE_STATE)
        self.assertEqual(repository.load(table.id), saved)
        self.assertEqual(repository.saves, 1)
        accepted = service.handle(ActCommand(hand_id, PlayerAction(ActionKind.CALL), expected_revision=1), session)
        assert accepted.view is not None
        self.assertEqual(accepted.view.revision, 2)
        self.assertEqual(len(accepted.view.action_history), 3)
        # Repeating a previously confirmed action cannot apply it again.
        duplicate = service.handle(ActCommand(hand_id, PlayerAction(ActionKind.CALL), expected_revision=1), session)
        assert duplicate.error is not None
        self.assertEqual(duplicate.error.code, ErrorCode.STALE_STATE)
        self.assertEqual(repository.saves, 2)
        # Human commands omit the guard and keep their previous behavior.
        self.assertTrue(service.handle(ActCommand(hand_id, PlayerAction(ActionKind.CHECK)), SessionContext(PlayerId("p1"))).ok)
        observed = service.handle(StateCommand(), session)
        assert observed.view is not None
        self.assertEqual(observed.view.phase, HandPhase.FLOP)

    def test_guard_roundtrips_and_legacy_command_omits_the_optional_field(self) -> None:
        table = opening_table((1000, 1000))
        opening_engine().start_hand(table, PlayerId("p0"))
        hand_id = hand_of(table).id
        codec = JsonLineCodec()
        guarded = ActCommand(hand_id, PlayerAction(ActionKind.CALL), 0)
        self.assertEqual(codec.decode_command(codec.encode_command(guarded)), guarded)
        legacy = ActCommand(hand_id, PlayerAction(ActionKind.CALL))
        raw = codec.encode_command(legacy)
        self.assertNotIn("expected_revision", json.loads(raw))
        self.assertEqual(codec.decode_command(raw), legacy)
        for invalid in (True, -1, 1.5, "1"):
            data = json.loads(raw)
            data["expected_revision"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                codec.decode_command(json.dumps(data))
