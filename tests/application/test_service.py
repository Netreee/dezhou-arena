import unittest
from dataclasses import asdict

from poker.application.commands import ActCommand, JoinCommand, SessionContext, StartHandCommand, StateCommand
from poker.application.service import TableService
from poker.application.views import PlayerViewBuilder
from poker.domain.models import HandResult, PlayerAction, Table
from poker.domain.types import ActionKind, ErrorCode, HandId, HandPhase, PlayerId, TableId
from poker.engine.holdem import HoldemEngine
from poker.engine.policies import FiveCardHighEvaluator, NoLimitBettingRules, SidePotAllocator
from tests.fixtures import FakeEngine, MemoryRepository, make_table


class ApplicationTests(unittest.TestCase):
    def test_join_binds_server_session_after_save(self) -> None:
        repository = MemoryRepository(Table(TableId("t")))
        service = TableService(TableId("t"), repository, FakeEngine(), PlayerViewBuilder())
        session = SessionContext()
        response = service.handle(JoinCommand("甲"), session)
        self.assertTrue(response.ok)
        self.assertEqual(session.player_id, PlayerId("p1"))
        self.assertEqual(repository.saves, 1)

    def test_observe_is_read_only(self) -> None:
        table = make_table(with_hand=True)
        repository = MemoryRepository(table)
        service = TableService(table.id, repository, FakeEngine(), PlayerViewBuilder())
        response = service.handle(StateCommand(), SessionContext(PlayerId("p1")))
        assert response.view is not None
        self.assertEqual(response.view.revision, 0)
        self.assertEqual(repository.saves, 0)

    def test_unbound_connection_is_rejected(self) -> None:
        table = make_table()
        service = TableService(table.id, MemoryRepository(table), FakeEngine(), PlayerViewBuilder())
        response = service.handle(StateCommand(), SessionContext())
        assert response.error is not None
        self.assertEqual(response.error.code, ErrorCode.NOT_SEATED)

    def test_rejected_working_copy_is_not_saved(self) -> None:
        table = make_table(with_hand=True)
        repository = MemoryRepository(table)
        engine = FakeEngine()
        engine.reject_after_mutation = True
        service = TableService(table.id, repository, engine, PlayerViewBuilder())
        response = service.handle(
            ActCommand(HandId("fixture-hand"), PlayerAction(ActionKind.CHECK)),
            SessionContext(PlayerId("p1")),
        )
        saved = repository.load(table.id)
        assert saved is not None
        self.assertFalse(response.ok)
        self.assertEqual(saved.player(PlayerId("p1")).stack, 990)
        self.assertEqual(repository.saves, 0)

    def test_projection_has_only_own_hole_cards_and_no_deck_fields(self) -> None:
        table = make_table(with_hand=True)
        assert table.hand is not None
        views = PlayerViewBuilder()
        first = views.build(table, PlayerId("p1"))
        second = views.build(table, PlayerId("p2"))
        self.assertNotEqual(first.me.hole_cards, second.me.hole_cards)
        self.assertEqual(first.players, second.players)
        self.assertTrue(all(not player.revealed_cards for player in first.players))
        self.assertNotIn("deck", asdict(first))
        self.assertNotIn("hole_cards", asdict(first.players[1]))

    def test_current_actor_can_start_and_read_legal_actions(self) -> None:
        table = make_table()
        repository = MemoryRepository(table)
        engine = HoldemEngine(NoLimitBettingRules(), FiveCardHighEvaluator(), SidePotAllocator())
        service = TableService(table.id, repository, engine, PlayerViewBuilder())
        response = service.handle(StartHandCommand(), SessionContext(PlayerId("p1")))
        assert response.view is not None
        self.assertEqual(response.view.actor_id, PlayerId("p1"))
        self.assertIn(ActionKind.CALL, [option.kind for option in response.view.me.legal_actions])
        self.assertEqual(repository.saves, 1)
        saved = repository.load(table.id)
        assert saved is not None and saved.hand is not None
        self.assertEqual(saved.hand.phase, HandPhase.PREFLOP)
        self.assertEqual(saved.revision, 1)

    def test_reveal_requires_matching_completed_hand_and_explicit_public_cards(self) -> None:
        table = make_table(with_hand=True)
        assert table.hand is not None
        table.last_result = HandResult(
            table.hand.id, (), (), {PlayerId("p1"): table.hand.players[PlayerId("p1")].hole_cards},
        )
        builder = PlayerViewBuilder()
        self.assertTrue(all(not p.revealed_cards for p in builder.build(table, PlayerId("p2")).players))
        table.hand.phase = HandPhase.COMPLETE
        view = builder.build(table, PlayerId("p2"))
        self.assertTrue(view.players[0].revealed_cards)
        self.assertFalse(view.players[1].revealed_cards)
        table.last_result = HandResult(HandId("another-hand"), (), (), table.last_result.revealed_hands)
        self.assertTrue(all(not p.revealed_cards for p in builder.build(table, PlayerId("p2")).players))
