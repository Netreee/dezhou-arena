"""Reproducible E01-E06 application/SQLite receipt; not TCP/CLI I01 evidence."""

import argparse
import json
import platform
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from unittest.mock import patch

from poker.application.commands import ActCommand, Command, SessionContext, StartHandCommand, StateCommand
from poker.application.service import TableService
from poker.application.views import PlayerView, PlayerViewBuilder
from poker.domain.cards import Card, Deck
from poker.domain.models import Player, PlayerAction, Table
from poker.domain.types import ActionKind, ErrorCode, HandPhase, PlayerId, TableId
from poker.engine.holdem import HoldemEngine
from poker.engine.policies import FiveCardHighEvaluator, NoLimitBettingRules, SidePotAllocator
from poker.persistence.snapshot import TableSnapshotCodec
from poker.persistence.sqlite import SqliteTableRepository
from poker.transport.codec import JsonLineCodec


def fixed_deck(button: int, pairs: tuple[str, ...]) -> Deck:
    holes = [tuple(Card.parse(code) for code in pair.split()) for pair in pairs]
    board = tuple(Card.parse(code) for code in "2c 3d 7h 9s Jc".split())
    used = set(board).union(card for pair in holes for card in pair)
    unused = [c for c in Deck.shuffled(Random(99)).remaining if c not in used]
    order = [(button + offset) % 3 for offset in (1, 2, 3)]
    dealt = [holes[seat][circle] for circle in range(2) for seat in order]
    cards = dealt + [unused[0], *board[:3], unused[1], board[3], unused[2], board[4]] + unused[3:]
    return Deck(cards)


def verify(receipt_path: Path) -> dict[str, object]:
    table = Table(TableId("engine-e01-e06-20261007"))
    for seat, stack in enumerate((500, 100, 300)):
        table.add_player(Player(PlayerId(f"p{seat}"), f"P{seat}", seat, stack))
    engine = HoldemEngine(NoLimitBettingRules(), FiveCardHighEvaluator(), SidePotAllocator())
    wire = JsonLineCodec()
    codec = TableSnapshotCodec()
    commands: list[dict[str, object]] = []
    completed: list[dict[str, object]] = []
    with TemporaryDirectory() as directory:
        database_path = Path(directory) / "engine.sqlite3"
        repository = SqliteTableRepository(database_path)
        repository.save(table)
        service = TableService(table.id, repository, engine, PlayerViewBuilder())

        def snapshot() -> Table:
            saved = SqliteTableRepository(database_path).load(table.id)
            assert saved is not None
            return saved

        def send(player_id: PlayerId, command: Command) -> PlayerView:
            before = codec.encode(snapshot())
            response = service.handle(command, SessionContext(player_id))
            assert response.view is not None, response.error
            if isinstance(command, StateCommand):
                assert codec.encode(snapshot()) == before, "state changed the SQLite snapshot"
            commands.append({
                "player_id": player_id,
                "command": json.loads(wire.encode_command(command)),
                "view": asdict(response.view),
            })
            return response.view

        def check_views() -> Table:
            saved = snapshot()
            hand = saved.hand
            assert hand is not None
            stack_total = sum(saved.player(pid).stack for pid in hand.players)
            assets = stack_total if hand.phase is HandPhase.COMPLETE else stack_total + sum(
                m.hand_commit for m in hand.players.values()
            )
            assert assets == hand.chip_total_at_start == 900
            cards = hand.deck.remaining + hand.deck.burned + hand.board + [
                c for m in hand.players.values() for c in m.hole_cards
            ]
            assert len(cards) == len(set(cards)) == 52
            views = [send(p.id, StateCommand()) for p in saved.players]
            assert all(v.players == views[0].players and v.board == views[0].board for v in views)
            for view in views:
                assert view.me.hole_cards == tuple(c.code for c in hand.players[view.me.player_id].hole_cards)
                for public in view.players:
                    revealed = (
                        saved.last_result.revealed_hands.get(public.player_id, ())
                        if hand.phase is HandPhase.COMPLETE and saved.last_result is not None else ()
                    )
                    assert public.revealed_cards == tuple(c.code for c in revealed)
            assert codec.encode(snapshot()) == codec.encode(saved)
            return saved

        def act(kind: ActionKind, to: int | None = None) -> None:
            hand = snapshot().hand
            assert hand is not None and hand.betting.actor_id is not None
            send(hand.betting.actor_id, ActCommand(hand.id, PlayerAction(kind, to)))
            check_views()

        def finish(label: str, expected_stacks: list[int], expected_pots: list[int]) -> None:
            saved = check_views()
            assert saved.hand is not None and saved.last_result is not None
            assert saved.hand.phase is HandPhase.COMPLETE
            assert [p.stack for p in saved.players] == expected_stacks
            assert [a.pot.amount for a in saved.last_result.awards] == expected_pots
            completed.append({
                "scenario": label, "hand_id": saved.hand.id, "revision": saved.revision,
                "stacks": expected_stacks, "board": [c.code for c in saved.hand.board],
                "result": asdict(saved.last_result),
                "private_snapshot": json.loads(codec.encode(saved)),
            })

        with patch("poker.engine.holdem.Deck.shuffled", return_value=fixed_deck(0, ("As Ad", "Kh Kd", "Qh Qd"))):
            send(PlayerId("p0"), StartHandCommand())
        check_views()
        for kind in (ActionKind.CALL, ActionKind.CALL, ActionKind.CHECK):
            act(kind)
        for kind, total in ((ActionKind.BET_TO, 20), (ActionKind.CALL, None), (ActionKind.CALL, None)):
            act(kind, total)
        for _ in range(6):
            act(ActionKind.CHECK)
        finish("normal_multi_street_betting", [560, 70, 270], [90])

        with patch("poker.engine.holdem.Deck.shuffled", return_value=fixed_deck(1, ("Qh Qd", "As Ad", "Kh Kd"))):
            send(PlayerId("p0"), StartHandCommand())
        # Let the deep stack shove while the middle stack can still respond.
        # After all opponents are all-in, an excess shove would be illegal.
        for kind in (ActionKind.ALL_IN, ActionKind.CALL, ActionKind.ALL_IN, ActionKind.ALL_IN):
            act(kind)
        finish("three_all_ins_main_side_and_refund", [290, 210, 400], [210, 400])
        result = snapshot().last_result
        assert result is not None
        assert [(r.player_id, r.amount) for r in result.refunds] == [(PlayerId("p0"), 290)]

        with patch("poker.engine.holdem.Deck.shuffled", return_value=fixed_deck(2, ("As Ad", "Kh Kd", "Qh Qd"))):
            send(PlayerId("p0"), StartHandCommand())
        act(ActionKind.FOLD)
        act(ActionKind.FOLD)
        finish("consecutive_folds_without_reveal", [285, 215, 400], [10])
        ended = snapshot()
        assert ended.hand is not None and ended.last_result is not None
        assert ended.hand.board == [] and ended.last_result.revealed_hands == {}
        rejected = ActCommand(ended.hand.id, PlayerAction(ActionKind.ALL_IN))
        response = service.handle(rejected, SessionContext(PlayerId("p1")))
        assert response.error is not None and response.error.code is ErrorCode.INVALID_ACTION
        assert codec.encode(snapshot()) == codec.encode(ended)
        commands.append({"player_id": "p1", "command": json.loads(wire.encode_command(rejected)),
                         "error": asdict(response.error)})
        for _ in range(3):
            check_views()

        with patch("poker.engine.holdem.Deck.shuffled", return_value=fixed_deck(0, ("As Ad", "Kh Kd", "Qh Qd"))):
            send(PlayerId("p0"), StartHandCommand())
        final = check_views()
        assert final.hand is not None and final.last_result is not None
        assert final.hand.id != ended.hand.id and final.hand.phase is HandPhase.PREFLOP
        assert final.last_result.hand_id == ended.hand.id
        assert [p.stack for p in final.players] == [285, 210, 390]
        assert final.revision == 22
        receipt: dict[str, object] = {
            "scope": "E01-E06 real engine + application + SQLite; no TCP/CLI I01 claim",
            "date": datetime.now().date().isoformat(), "python_version": platform.python_version(),
            "initial_stacks": [500, 100, 300], "fixed_decks": True,
            "completed_hands": completed, "commands": commands,
            "final_snapshot": json.loads(codec.encode(final)),
            "checks": {
                "three_completed_hands": True, "explicit_next_hand": True,
                "sqlite_reopened_readback": True, "state_reads_do_not_write_or_pay": True,
                "private_views_verified": True, "chips_conserved": True,
                "cards_partition_52": True, "complete_rejects_actions": True,
            },
        }
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"receipt": str(receipt_path.resolve()), "completed_hands": len(completed),
                "recorded_commands": len(commands), "final_revision": final.revision,
                "final_stacks": [p.stack for p in final.players], "current_pot": 15, "chip_total": 900}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=Path(".data/verification/E01_E06_RECEIPT.json"))
    args = parser.parse_args()
    print(json.dumps(verify(args.receipt), ensure_ascii=False))


if __name__ == "__main__":
    main()
