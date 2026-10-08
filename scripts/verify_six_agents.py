"""Run six distinct Policy mechanisms as independent ordinary CLI players.

The fixed-deck scenario validates mechanism integration; a seeded random-deck
scenario is also available. Neither scenario measures competitive strength.
Every run owns a fresh server, six agent processes, SQLite and evidence folder.
"""

import argparse
from datetime import UTC, datetime
import hashlib
import json
from math import isclose, isfinite
import os
from pathlib import Path
from random import Random
import re
import secrets
import subprocess
import sys
from threading import Event, Thread
from time import monotonic
from typing import cast
from unittest.mock import patch
from uuid import uuid4

# Support both direct script execution and supervisor imports from the project.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from jsonschema import Draft202012Validator

from scripts.verify_agent_runtime import CapturedProcess, RecordedRequest, read_requests
from poker.agent.naive_language import DEFAULT_CODEX_MODEL
from poker.application.commands import ActCommand, Command, JoinCommand, SessionContext, StartHandCommand, StateCommand
from poker.application.service import TableService
from poker.application.views import CommandResponse, PlayerView
from poker.bootstrap import server_main
from poker.domain.cards import Card, Deck
from poker.domain.models import Table
from poker.domain.types import ActionKind, HandPhase, PlayerId, TableId
from poker.engine.holdem import HoldemEngine
from poker.engine.policies import FiveCardHighEvaluator, NoLimitBettingRules, SidePotAllocator
from poker.persistence.snapshot import TableSnapshotCodec
from poker.persistence.sqlite import SqliteTableRepository
from poker.transport.codec import JsonLineCodec
from poker.transport.local_tcp import LocalTcpServer

MECHANISMS = ("rules", "mixed", "lookup", "stateful", "solver", "natural_language")
DEFAULT_LANGUAGE_FACTORY = "poker.agent.naive_language:natural_language"


def fixed_cards() -> tuple[Card, ...]:
    """Twelve distinct hole cards followed by a shared royal-flush runout."""
    holes = tuple(Card.parse(code) for code in "2c 3d 4c 5d 6c 7d 2h 3h 4h 5h 6h 7h".split())
    board = tuple(Card.parse(code) for code in "Ts Js Qs Ks As".split())
    remaining = sorted((card for card in Deck.shuffled(Random(0)).remaining if card not in (*holes, *board)), key=lambda card: card.code)
    cards = (*holes, remaining[0], *board[:3], remaining[1], board[3], remaining[2], board[4], *remaining[3:])
    assert len(cards) == len(set(cards)) == 52
    return cards


def fixture_server(database: Path, wire_path: Path, snapshot_path: Path, stop_path: Path,
                   *, deck_mode: str, deck_seed: int) -> None:
    wire, snapshots = JsonLineCodec(), TableSnapshotCodec()
    original_handle, original_serve = TableService.handle, LocalTcpServer.serve_forever
    original_shuffle = Deck.shuffled
    predetermined, rng = fixed_cards(), Random(deck_seed)
    saved_revisions: set[int] = set()

    def next_deck() -> Deck:
        return Deck(list(predetermined)) if deck_mode == "fixed" else original_shuffle(rng)

    def recorded_handle(service: TableService, command: Command, session: SessionContext) -> CommandResponse:
        engine = service._engine
        assert type(engine) is HoldemEngine and isinstance(engine, HoldemEngine)
        assert type(engine._betting) is NoLimitBettingRules
        assert type(engine._evaluator) is FiveCardHighEvaluator
        assert type(engine._pots) is SidePotAllocator
        response = original_handle(service, command, session)
        table = service._repository.load(service._table_id)
        assert table is not None
        if table.revision not in saved_revisions:
            with snapshot_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(snapshots.encode(table) + "\n")
            saved_revisions.add(table.revision)
        row = {"player_id": session.player_id, "request_json": wire.encode_command(command),
               "response_json": wire.encode_response(response), "revision": table.revision,
               "implementations": [type(engine).__name__, type(engine._betting).__name__,
                                   type(engine._evaluator).__name__, type(engine._pots).__name__]}
        with wire_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return response

    def serve_with_stop(server: LocalTcpServer) -> None:
        done = Event()

        def stop_when_requested() -> None:
            while not done.wait(0.05):
                if stop_path.exists():
                    server.close()
                    return

        watcher = Thread(target=stop_when_requested, name="six-agent-server-stop", daemon=True)
        watcher.start()
        try:
            original_serve(server)
        finally:
            done.set()
            watcher.join(timeout=5)
            assert not watcher.is_alive()

    sys.argv = ["poker-server", "--port", "0", "--db", str(database)]
    with patch.object(Deck, "shuffled", side_effect=next_deck), \
            patch.object(TableService, "handle", new=recorded_handle), \
            patch.object(LocalTcpServer, "serve_forever", new=serve_with_stop):
        server_main()


def _snapshots(path: Path) -> dict[int, Table]:
    if not path.exists():
        return {}
    codec = TableSnapshotCodec()
    tables = [codec.decode(line) for line in path.read_text(encoding="utf-8").splitlines(keepends=True) if line.endswith("\n")]
    return {table.revision: table for table in tables}


def _completed(snapshots: dict[int, Table]) -> dict[str, Table]:
    return {str(table.hand.id): table for table in snapshots.values()
            if table.hand is not None and table.hand.phase is HandPhase.COMPLETE}


def verify_view(view: PlayerView, table: Table, player_id: PlayerId) -> None:
    hand = table.hand
    assert (view.table_id, view.revision, view.me.player_id) == (table.id, table.revision, player_id)
    assert view.config == table.config
    assert view.hand_id == (hand.id if hand else None)
    assert view.phase == (hand.phase if hand else None)
    assert view.board == (tuple(card.code for card in hand.board) if hand else ())
    assert view.actor_id == (hand.betting.actor_id if hand else None)
    member = hand.players.get(player_id) if hand else None
    assert view.me.hole_cards == (tuple(card.code for card in member.hole_cards) if member else ())
    assert len(view.players) == len(table.players)
    reveal = (table.last_result.revealed_hands if hand and hand.phase is HandPhase.COMPLETE
              and table.last_result and table.last_result.hand_id == hand.id else {})
    for public in view.players:
        player = table.player(public.player_id)
        current = hand.players.get(public.player_id) if hand else None
        assert (public.name, public.seat, public.stack) == (player.name, player.seat, player.stack)
        assert public.status == (current.status if current else None)
        assert public.street_commit == (current.street_commit if current else 0)
        assert public.hand_commit == (current.hand_commit if current else 0)
        assert public.revealed_cards == tuple(card.code for card in reveal.get(public.player_id, ()))
    if view.actor_id != player_id:
        assert not view.me.legal_actions
    if hand is None:
        assert not view.action_history and not view.history_complete
        return
    assert len(hand.players) == 6, "This acceptance run requires six funded participants in every hand"
    assert view.history_complete and hand.history_complete
    assert view.action_history == tuple(hand.action_history)
    assert [event.sequence for event in view.action_history] == list(range(1, len(view.action_history) + 1))
    assert [event.kind for event in view.action_history[:2]] == ["small_blind", "big_blind"]
    assert view.pot_total == sum(member.hand_commit for member in hand.players.values())
    cards = hand.deck.remaining + hand.deck.burned + hand.board + [card for member in hand.players.values() for card in member.hole_cards]
    assert len(cards) == len(set(cards)) == 52
    assets = sum(table.player(pid).stack for pid in hand.players)
    if hand.phase is not HandPhase.COMPLETE:
        assets += sum(member.hand_commit for member in hand.players.values())
    assert assets == hand.chip_total_at_start == 6000


def audit_wire(rows: list[RecordedRequest], snapshots: dict[int, Table], *, hands: int,
               deck_mode: str) -> dict[str, object]:
    assert rows and all(row.response.ok for row in rows), "A game request was rejected"
    joins = [row for row in rows if isinstance(row.command, JoinCommand)]
    assert len(joins) == len({row.player_id for row in joins}) == 6
    assert {cast(JoinCommand, row.command).name for row in joins} == set(MECHANISMS)
    starts = [row for row in rows if isinstance(row.command, StartHandCommand)]
    assert len(starts) == hands
    starter = next(row.player_id for row in joins if cast(JoinCommand, row.command).name == "rules")
    assert all(row.player_id == starter for row in starts)
    for row in rows:
        view = row.response.view
        assert view is not None
        verify_view(view, snapshots[row.revision], row.player_id)
        raw_view = cast(dict[str, object], json.loads(cast(str, row.raw["response_json"]))["view"])
        assert not {"deck", "pending", "last_action_bet"} & raw_view.keys()
        assert all("hole_cards" not in player for player in cast(list[dict[str, object]], raw_view["players"]))
        if isinstance(row.command, ActCommand):
            assert row.command.expected_revision == row.revision - 1
            assert row.command.hand_id == view.hand_id
            event = view.action_history[-1]
            previous = snapshots[row.revision - 1]
            assert previous.hand is not None
            assert (event.player_id, event.kind) == (row.player_id, row.command.action.kind.value)
            assert event.stack + event.pay == previous.player(row.player_id).stack
            assert event.to == previous.hand.players[row.player_id].street_commit + event.pay
            if row.command.action.kind in (ActionKind.BET_TO, ActionKind.RAISE_TO):
                assert event.to == row.command.action.to
            if row.command.action.kind in (ActionKind.CHECK, ActionKind.FOLD):
                assert event.pay == 0
    completed = _completed(snapshots)
    assert len(completed) == hands
    for hand_id, table in completed.items():
        assert table.hand is not None
        action_count = sum(isinstance(row.command, ActCommand) and row.command.hand_id == hand_id for row in rows)
        assert len(table.hand.action_history) == action_count + 2
        if deck_mode == "fixed":
            assert len(table.hand.board) == 5
            assert {event.phase for event in table.hand.action_history[2:]} == {
                HandPhase.PREFLOP, HandPhase.FLOP, HandPhase.TURN, HandPhase.RIVER,
            }
    actions_by_player = {row.player_id: sum(isinstance(item.command, ActCommand) and item.player_id == row.player_id for item in rows) for row in joins}
    assert all(count >= 1 for count in actions_by_player.values()), "Every mechanism must actually act"
    polls = {row.player_id: sum(isinstance(item.command, StateCommand) and item.player_id == row.player_id for item in rows) for row in joins}
    assert all(count >= 1 for count in polls.values())
    return {"requests": len(rows), "actions": sum(actions_by_player.values()), "actions_by_player": actions_by_player,
            "automatic_state_counts": polls, "completed_hands": list(completed)}


def _logs(output_dir: Path, run_id: str) -> list[dict[str, object]]:
    return [cast(dict[str, object], json.loads(line)) for path in sorted((output_dir / "logs" / run_id).glob("*.jsonl*"))
            for line in path.read_text(encoding="utf-8").splitlines()]


def model_evidence(output_dir: Path) -> dict[str, object]:
    paths = sorted((output_dir / "model_calls").glob("*.json"))
    records = [cast(dict[str, object], json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    return {"attempts": len(records),
            "completed": sum(record.get("status") == "completed" and record.get("success") is True for record in records),
            "failed": sum(record.get("status") == "failed" for record in records),
            "pending": sum(record.get("status") == "started" for record in records),
            "records": [{"file": str(path.relative_to(output_dir)).replace("\\", "/"), **record}
                        for path, record in zip(paths, records, strict=True)]}


def audit_logs(output_dir: Path, run_id: str, agents: list[CapturedProcess], rows: list[RecordedRequest],
               *, hands: int, language_backend: str) -> dict[str, object]:
    schema = json.loads((_ROOT / "src/shared_logging/event.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    events = _logs(output_dir, run_id)
    for event in events:
        validator.validate(event)
        assert event["run_id"] == run_id
        assert event["level"] not in ("ERROR", "CRITICAL"), event
    summaries: list[dict[str, object]] = []
    for agent in agents:
        joined = next(row for row in rows if isinstance(row.command, JoinCommand) and row.command.name == agent.label)
        runtime_pids = {cast(int, event["pid"]) for event in events if event["event"] == "client.joined"
                        and cast(dict[str, object], event["context"]).get("player_id") == joined.player_id}
        assert len(runtime_pids) == 1
        agent.runtime_pid = runtime_pids.pop()
        own = [event for event in events if event["pid"] == agent.runtime_pid]
        named = {name: [event for event in own if event["event"] == name] for name in (
            "agent.started", "agent.decision.started", "agent.action.queued", "agent.action.confirmed", "agent.stopped", "naive.policy.decided",
        )}
        assert all(named[name] for name in named), (agent.label, "Missing mechanism or runtime evidence")
        queued = {cast(dict[str, object], event["data"])["decision_id"]: cast(dict[str, object], event["data"])
                  for event in named["agent.action.queued"]}
        confirmed = [cast(dict[str, object], event["data"])["decision_id"] for event in named["agent.action.confirmed"]]
        assert len(queued) == len(named["agent.action.queued"]) == len(confirmed) == len(set(confirmed))
        assert set(queued) == set(confirmed)
        applied = [row for row in rows if row.player_id == joined.player_id and isinstance(row.command, ActCommand)]
        assert len(applied) == len(confirmed) >= 1
        expected_actions = {(str(row.command.hand_id), row.command.expected_revision, row.command.action.kind.value, row.command.action.to)
                            for row in applied if isinstance(row.command, ActCommand)}
        submitted_actions = {(data["hand_id"], data["revision"], data["action"], data.get("to")) for data in queued.values()}
        assert expected_actions == submitted_actions
        mechanism_events = [cast(dict[str, object], event["data"]) for event in named["naive.policy.decided"]]
        assert all(event.get("representation") == agent.label for event in mechanism_events)
        expected_mechanism = "prompt_tool_loop" if agent.label == "natural_language" else agent.label
        assert all(event.get("mechanism") == expected_mechanism for event in mechanism_events)
        evidence_ids = {event["decision_id"] for event in mechanism_events}
        assert set(confirmed) <= evidence_ids, "An executed action is missing its policy mechanism evidence"
        if agent.label == "mixed":
            non_degenerate = False
            for event in mechanism_events:
                distribution = cast(list[dict[str, object]], event["distribution"])
                keys = {(entry["action"], entry.get("to")) for entry in distribution}
                probabilities = [entry["probability"] for entry in distribution]
                assert len(keys) == len(distribution) >= 1
                assert all(type(value) in (int, float) and isfinite(cast(float, value)) and cast(float, value) >= 0 for value in probabilities)
                assert isclose(sum(cast(list[float], probabilities)), 1.0, abs_tol=1e-9, rel_tol=0.0)
                positive = {(entry["action"], entry.get("to")) for entry in distribution if cast(float, entry["probability"]) > 0}
                non_degenerate = non_degenerate or len(positive) >= 2
                submitted = queued.get(event["decision_id"])
                if submitted is not None:
                    assert (submitted["action"], submitted.get("to")) in positive
                    assert (submitted["revision"], submitted["hand_id"]) == (event["revision"], event["hand_id"])
            assert non_degenerate, "Mixed policy never emitted two different actions with positive probability"
        if agent.label == "lookup":
            assert any(event.get("table_hit") is True for event in mechanism_events)
        if agent.label == "stateful":
            assert any(cast(int, event.get("confirmed_memory", 0)) >= 1 for event in mechanism_events)
            assert len({event.get("memory_mode") for event in mechanism_events}) >= 2
            if hands >= 2:
                assert any(cast(int, event.get("completed_memory", 0)) >= 1 for event in mechanism_events)
        if agent.label == "solver":
            assert all(cast(int, event.get("rollouts", 0)) >= 1 and cast(int, event.get("evaluated_hands", 0)) >= 1 for event in mechanism_events)
            multiple_candidates = False
            for event in mechanism_events:
                scores = cast(list[dict[str, object]], event["candidate_evs"])
                assert scores and all(type(score["estimated_ev"]) in (int, float) and isfinite(cast(float, score["estimated_ev"])) for score in scores)
                multiple_candidates = multiple_candidates or len({(score["action"], score.get("to")) for score in scores}) >= 2
                submitted = queued.get(event["decision_id"])
                if submitted is not None:
                    matches = [score for score in scores if (score["action"], score.get("to")) == (submitted["action"], submitted.get("to"))]
                    assert len(matches) == 1
                    assert cast(float, matches[0]["estimated_ev"]) == max(cast(float, score["estimated_ev"]) for score in scores)
            assert multiple_candidates, "Solver never evaluated two different candidate actions"
        if agent.label == "natural_language":
            assert all(event.get("backend") == language_backend for event in mechanism_events)
            assert any(cast(int, event.get("rounds", 0)) >= 2 for event in mechanism_events)
            assert sum(event["event"] == "agent.tool.completed" for event in own) >= 1
            calls = [event for event in own if event["event"] == "api.call.completed"]
            if language_backend == "codex":
                assert calls and all(cast(dict[str, object], event["data"]).get("provider") == "codex-cli" for event in calls)
                assert all(cast(int, event.get("live_model_calls", 0)) >= 1 for event in mechanism_events)
                evidence = model_evidence(output_dir)
                attempts = [event for event in own if event["event"] == "api.call.started"]
                assert evidence["attempts"] == evidence["completed"] == len(calls) == len(attempts)
                assert evidence["failed"] == evidence["pending"] == 0
                assert sum(cast(int, event["live_model_calls"]) for event in mechanism_events) == len(calls)
                for record in cast(list[dict[str, object]], evidence["records"]):
                    assert record["live"] is True and record["backend"] == "codex"
                    assert record["status"] == "completed" and record["success"] is True and record["exit_code"] == 0
                    assert record["environment_tool_items"] == 0
                    assert "turn.completed" in cast(list[str], record["event_types"])
                    assert isinstance(record.get("answer"), dict)
                    assert record["cost"] is None, "No unverified monetary cost should be invented"
            else:
                assert not calls and not list((output_dir / "model_calls").glob("*.json"))
                assert all(event.get("live_model_calls", 0) == 0 for event in mechanism_events)
        stop = cast(dict[str, object], named["agent.stopped"][-1]["data"])
        assert stop["stop_reason"] == "max_hands" and stop["observed_hands"] == hands
        assert stop["confirmed_actions"] == len(confirmed)
        assert stop["ok"] is True and stop["error"] is None
        assert stop["cli_closed"] is True and stop["policy_closed"] is True
        summaries.append({"mechanism": agent.label, "player_id": joined.player_id, "runtime_pid": agent.runtime_pid,
                          "launcher_pid": agent.process.pid, "confirmed_actions": len(confirmed),
                          "decisions": len(named["agent.decision.started"]),
                          "tool_calls": sum(event["event"] == "agent.tool.completed" for event in own),
                          "mechanism_evidence": mechanism_events, "stop": stop})
    assert any(cast(int, summary["tool_calls"]) >= 1 for summary in summaries)
    server_pids = {cast(int, event["pid"]) for event in events if event["process_role"] == "server"}
    assert len(server_pids) == 1
    return {"events": len(events), "files": len(list((output_dir / "logs" / run_id).glob("*.jsonl*"))),
            "server_pid": server_pids.pop(), "agents": summaries, "language_backend": language_backend}


def _hashes(output_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    packages = ("agent", "client", "application", "domain", "engine", "transport", "messaging", "persistence")
    sources = [path for package in packages for path in (_ROOT / "src/poker" / package).rglob("*.py")]
    sources.extend((_ROOT / "src/shared_logging").rglob("*.py"))
    sources.extend(_ROOT / path for path in (
        "pyproject.toml", "scripts/__init__.py", "scripts/verify_six_agents.py", "scripts/verify_agent_runtime.py",
        "src/poker/__init__.py", "src/poker/bootstrap.py", "src/poker/persistence/schema.sql", "src/shared_logging/event.schema.json",
    ))
    source_hashes = {str(path.relative_to(_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(set(sources))}
    evidence_hashes = {str(path.relative_to(output_dir)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(output_dir.rglob("*")) if path.is_file() and path.name not in ("receipt.json", "failure.json")}
    return source_hashes, evidence_hashes


def verify(output_dir: Path, *, hands: int = 2, timeout: float = 900.0, seed: int = 7,
           language_backend: str = "stub", deck_mode: str = "fixed", decision_timeout: float = 180.0,
           language_factory: str = DEFAULT_LANGUAGE_FACTORY, seat_rotation: int = 0,
           language_max_calls: int = 32, language_model: str | None = DEFAULT_CODEX_MODEL,
           deck_seed: int | None = None) -> dict[str, object]:
    if type(hands) is not int or hands < 1 or not 10 < timeout <= 43200 or not 0 < decision_timeout < timeout:
        raise ValueError("Require hands >= 1, 10 < timeout <= 43200, and 0 < decision_timeout < timeout")
    if deck_mode not in ("fixed", "random"):
        raise ValueError("deck_mode must be fixed or random")
    if language_backend not in ("stub", "codex"):
        raise ValueError("language_backend must be stub or codex")
    if type(seat_rotation) is not int or not 0 <= seat_rotation < 6:
        raise ValueError("seat_rotation must be an integer between zero and five")
    if type(language_max_calls) is not int or language_max_calls < 1:
        raise ValueError("language_max_calls must be positive")
    if deck_seed is not None and (type(deck_seed) is not int or deck_seed < 0):
        raise ValueError("The private audit deck seed must be a nonnegative integer")
    server_deck_seed = secrets.randbits(64) if deck_seed is None else deck_seed
    output_dir = output_dir.resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Choose a new or empty output directory; previous evidence is never overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)
    database, wire_path = output_dir / "final.sqlite3", output_dir / "server_commands.jsonl"
    snapshot_path, stop_path = output_dir / "arena_snapshots.jsonl", output_dir / "stop"
    config_path = output_dir / "policy_config.json"
    config: dict[str, object] = {"verification_continuation": True, "solver_rollouts": 24,
                                "language_backend": language_backend, "language_max_calls": language_max_calls,
                                "language_evidence_dir": str(output_dir / "model_calls")}
    if language_model is not None:
        config["language_model"] = language_model
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    run_id = f"six-agent-{uuid4().hex}"
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                   "APP_RUN_ID": run_id, "APP_LOG_DIR": str(output_dir / "logs"), "LOG_LEVEL": "INFO",
                   "LOG_MAX_BYTES": "50000000", "LOG_BACKUP_COUNT": "3"}
    environment.pop("APP_LOG_ROLE", None)
    # Capture before any production server/agent child imports its modules.
    # A later hash alone cannot describe code actually loaded during the run.
    source_hashes_start, _ = _hashes(output_dir)
    captures: list[CapturedProcess] = []
    agents: list[CapturedProcess] = []
    started = monotonic()
    server: CapturedProcess | None = None
    table_id: TableId | None = None
    error_message: str | None = None
    error_type: str | None = None
    result: dict[str, object] = {}
    try:
        server = CapturedProcess("server", [sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()),
                                 "--fixture-server", "--database", str(database), "--wire-log", str(wire_path),
                                 "--snapshots", str(snapshot_path), "--stop-file", str(stop_path),
                                 "--deck-mode", deck_mode, "--deck-seed", str(server_deck_seed)], environment, _ROOT)
        captures.append(server)
        ready_deadline = monotonic() + 20
        while True:
            match = re.search(r"on \('127\.0\.0\.1', (\d+)\); table=([0-9a-f]+)", server.next_line(ready_deadline))
            if match:
                port, table_id = int(match.group(1)), TableId(match.group(2))
                break
        seat_order = MECHANISMS[seat_rotation:] + MECHANISMS[:seat_rotation]
        for index, mechanism in enumerate(seat_order):
            factory = language_factory if mechanism == "natural_language" else f"poker.agent.naive_policies:{mechanism}"
            command = [sys.executable, "-u", "-X", "utf8", "-m", "poker.agent", "--port", str(port),
                       "--name", mechanism, "--policy", factory, "--config", str(config_path), "--min-players", "6",
                       "--max-hands", str(hands), "--poll-interval", "0.2", "--decision-timeout", str(decision_timeout),
                       "--session-timeout", str(timeout - 5), "--seed", str(seed + MECHANISMS.index(mechanism))]
            if mechanism == "rules":
                command.append("--auto-start")
            agent = CapturedProcess(mechanism, command, environment, _ROOT)
            captures.append(agent)
            agents.append(agent)
            # Join in a known order; concurrent interpreter startup does not
            # otherwise guarantee the requested seat rotation.
            join_deadline = monotonic() + 20
            while True:
                joined_snapshots = _snapshots(snapshot_path)
                joined_table = joined_snapshots[max(joined_snapshots)] if joined_snapshots else None
                player = next((player for player in joined_table.players if player.name == mechanism), None) if joined_table else None
                if player is not None:
                    assert player.seat == index
                    break
                if agent.process.poll() is not None:
                    raise RuntimeError(f"{mechanism} exited during join: {agent.lines[-5:]}")
                if monotonic() >= join_deadline:
                    raise TimeoutError(f"{mechanism} did not join before the startup deadline")
                Event().wait(0.05)
        deadline = monotonic() + timeout
        while any(agent.process.poll() is None for agent in agents):
            for agent in agents:
                code = agent.process.poll()
                if code not in (None, 0):
                    raise RuntimeError(f"{agent.label} exited {code}: {agent.lines[-5:]}")
            if server.process.poll() is not None:
                raise RuntimeError(f"Server exited before players: {server.lines[-5:]}")
            current_snapshots = _snapshots(snapshot_path)
            if current_snapshots:
                latest = current_snapshots[max(current_snapshots)]
                if len(latest.players) == 6 and latest.between_hands and len(_completed(current_snapshots)) < hands:
                    busted = [player.name for player in latest.players if player.stack == 0]
                    if busted:
                        raise RuntimeError(f"Six-player session cannot continue: busted={busted}; completed={len(_completed(current_snapshots))}/{hands}")
            if monotonic() >= deadline:
                raise TimeoutError("Six-agent session exceeded its bounded verification timeout")
            Event().wait(0.2)
        for agent in agents:
            agent.finish()
        stop_path.touch()
        server.finish()
        rows, snapshots = read_requests(wire_path), _snapshots(snapshot_path)
        wire_summary = audit_wire(rows, snapshots, hands=hands, deck_mode=deck_mode)
        logging = audit_logs(output_dir, run_id, agents, rows, hands=hands, language_backend=language_backend)
        server.runtime_pid = cast(int, logging["server_pid"])
        assert len({capture.runtime_pid for capture in captures}) == 7
        assert table_id is not None
        final = SqliteTableRepository(database).load(table_id)
        assert final is not None and final.hand is not None and final.hand.phase is HandPhase.COMPLETE
        assert final == snapshots[max(snapshots)]
        assert sum(player.stack for player in final.players) == 6000
        result = {"wire": wire_summary, "logging": logging, "completed_hands": len(_completed(snapshots)),
                  "final_snapshot": json.loads(TableSnapshotCodec().encode(final)),
                  "checks": {"seven_distinct_runtime_processes": True, "six_distinct_player_identities": True,
                             "six_mechanisms_actually_decided_and_confirmed": True, "mixed_non_degenerate_evidence": True,
                             "all_views_match_sqlite_revision": True, "private_cards_isolated": True,
                             "complete_public_action_history": True, "actions_match_queued_and_confirmed_ids": True,
                             "all_target_hands_completed_with_six_players": True, "single_auto_start_owner": True,
                             "chips_conserved_6000": True, "cards_partition_52": True, "sqlite_reopened": True,
                             "logs_schema_validated": True, "all_processes_exit_zero": True}}
    except (Exception, KeyboardInterrupt) as error:
        error_message, error_type = str(error), type(error).__name__
    finally:
        if error_message is not None and os.name == "nt":
            for capture in reversed(agents):
                if capture.process.poll() is None:
                    # Only process trees launched and still owned by this run.
                    try:
                        subprocess.run(["taskkill", "/PID", str(capture.process.pid), "/T", "/F"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                                       creationflags=subprocess.CREATE_NO_WINDOW, timeout=15)
                    except (OSError, subprocess.TimeoutExpired):
                        # Continue the other cleanup attempts and still preserve
                        # the original failure and all available output files.
                        pass
        if server is not None and server.process.poll() is None:
            stop_path.touch()
            try:
                server.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        for capture in reversed(captures):
            try:
                capture.cleanup()
            except Exception as error:
                error_message = error_message or f"Process cleanup failed: {type(error).__name__}"
                error_type = error_type or type(error).__name__
            (output_dir / f"{capture.label}_stdout.log").write_text("\n".join(capture.lines) + "\n", encoding="utf-8")
    final_snapshots = _snapshots(snapshot_path)
    last = final_snapshots[max(final_snapshots)] if final_snapshots else None
    model_calls = model_evidence(output_dir)
    source_hashes, evidence_hashes = _hashes(output_dir)
    changed_source_files = sorted(path for path in set(source_hashes_start) | set(source_hashes)
                                  if source_hashes_start.get(path) != source_hashes.get(path))
    if changed_source_files and error_message is None:
        error_message = "Source files changed during verification; game results are retained without frozen-source acceptance"
        error_type = "SourceChangedDuringRun"
    checks = cast(dict[str, object], result.setdefault("checks", {}))
    checks["source_unchanged_during_run"] = not changed_source_files
    summary: dict[str, object] = {
        "ok": error_message is None, "scope": "Six independent Policy mechanisms using ordinary CLI players and an independent production Arena",
        "timestamp_utc": datetime.now(UTC).isoformat(), "elapsed_seconds": round(monotonic() - started, 3),
        "run_id": run_id, "table_id": table_id, "target_hands": hands, "deck_mode": deck_mode, "seed": seed,
        "seed_scope": "agent_policy_sampling_only",
        "server_deck_seed": server_deck_seed if deck_mode == "random" else None,
        "deck_seed_scope": "server_and_private_audit_only; never provided to policies or model requests",
        "seat_rotation": seat_rotation,
        "language_backend": language_backend, "language_is_stub": language_backend == "stub",
        "competitive_strength_claim": False, "verification_continuation": True,
        "controls": ["fixed 52-card order with royal-flush board" if deck_mode == "fixed" else "seeded random Deck.shuffled",
                     "request and SQL snapshot audit", "graceful stop watcher", "bounded agent session deadline"],
        "processes": [{"label": capture.label, "launcher_pid": capture.process.pid, "runtime_pid": capture.runtime_pid,
                       "command": capture.command, "exit_code": capture.process.returncode} for capture in captures],
        "server_completed_hands": list(_completed(final_snapshots)),
        "busted_players": [player.name for player in last.players if player.stack == 0] if last else [],
        "funded_players": sum(player.stack > 0 for player in last.players) if last else 0,
        "error_type": error_type, "error": error_message, "source_sha256": source_hashes,
        "source_sha256_start": source_hashes_start, "source_changed": bool(changed_source_files),
        "changed_source_files": changed_source_files,
        "evidence_sha256": evidence_hashes, "model_calls": model_calls, **result,
    }
    destination = output_dir / ("receipt.json" if summary["ok"] else "failure.json")
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"ok": summary["ok"], "receipt": str(destination), "completed_hands": len(_completed(final_snapshots)),
            "requests": cast(dict[str, object], result.get("wire", {})).get("requests"),
            "actions": cast(dict[str, object], result.get("wire", {})).get("actions"),
            "process_exit_codes": [capture.process.returncode for capture in captures],
            "runtime_pids": [capture.runtime_pid for capture in captures], "error": error_message,
            "source_changed": bool(changed_source_files), "changed_source_files": changed_source_files,
            "model_calls": {key: value for key, value in model_calls.items() if key != "records"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".data/verification/six-agents"))
    parser.add_argument("--hands", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--decision-timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--deck-mode", choices=("fixed", "random"), default="fixed")
    parser.add_argument("--language-backend", default="stub")
    parser.add_argument("--language-factory", default=DEFAULT_LANGUAGE_FACTORY)
    parser.add_argument("--seat-rotation", type=int, default=0)
    parser.add_argument("--language-max-calls", type=int, default=32)
    parser.add_argument("--language-model", default=DEFAULT_CODEX_MODEL,
                        help=f"Explicit Codex model (project default: {DEFAULT_CODEX_MODEL}; reasoning: low)")
    parser.add_argument("--fixture-server", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--database", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--wire-log", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--snapshots", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--stop-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--deck-seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.fixture_server:
        assert args.database is not None and args.wire_log is not None and args.snapshots is not None and args.stop_file is not None
        assert args.deck_seed is not None
        fixture_server(args.database, args.wire_log, args.snapshots, args.stop_file, deck_mode=args.deck_mode, deck_seed=args.deck_seed)
        return
    outcome = verify(args.output_dir, hands=args.hands, timeout=args.timeout, seed=args.seed,
                     language_backend=args.language_backend, deck_mode=args.deck_mode,
                     decision_timeout=args.decision_timeout, language_factory=args.language_factory,
                     seat_rotation=args.seat_rotation, language_max_calls=args.language_max_calls,
                     language_model=args.language_model, deck_seed=args.deck_seed)
    print(json.dumps(outcome, ensure_ascii=False))
    raise SystemExit(0 if outcome["ok"] else 1)


if __name__ == "__main__":
    main()
