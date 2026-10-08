"""Five deliberately small policy representations for process integration tests.

None is a competitive poker strategy. ``verification_continuation`` restricts the
candidate set to calls/checks and affordable minimum bets, with at most one
voluntary increase per street when complete public history is available. The
individual policy still computes and selects its own action; no wrapper rewrites
the returned decision. A finite mixture may have only one candidate in a forced
continuation state, and records that fact explicitly.

The solver samples uniform unknown holdings and runouts, then compares immediate
call-to-showdown payoffs assuming opponents match a minimum increase. It ignores
future betting, fold equity and side-pot eligibility. It is neither CFR nor GTO.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import cast

from shared_logging import JsonObject, JsonValue, get_logger

from poker.agent.context import DecisionContext
from poker.agent.models import (
    ActionDistribution, ActionFeedback, Decision, DecisionRequest, HandCompleted,
    ObservationChanged, PolicyError, PolicyEvent, PolicyIdentity, WeightedAction,
)
from poker.agent.policy import Policy
from poker.application.views import PlayerView, PublicPlayerView
from poker.domain.cards import Card
from poker.domain.models import PlayerAction
from poker.domain.types import ActionKind, HandPhase, PlayerStatus, Rank, Suit
from poker.engine.policies import FiveCardHighEvaluator


_SIZED = (ActionKind.BET_TO, ActionKind.RAISE_TO)
_INCREASES = {"bet_to", "raise_to", "all_in"}
_PASSIVE = (ActionKind.CHECK, ActionKind.CALL)


def _continuation(config: JsonObject) -> bool:
    value = config.get("verification_continuation", False)
    if type(value) is not bool:
        raise ValueError("verification_continuation must be a boolean")
    return value


def _hero(view: PlayerView) -> PublicPlayerView:
    return next(player for player in view.players if player.player_id == view.me.player_id)


def _candidates(view: PlayerView, continuation: bool) -> tuple[PlayerAction, ...]:
    """Use server legality; apply only explicitly documented demonstration limits."""
    hero = _hero(view)
    blind = view.config.big_blind if view.config is not None else 10
    cap = max(blind, min(2 * blind, max(1, hero.stack // 20)))
    already_increased = any(record.phase == view.phase and record.kind in _INCREASES
                            for record in view.action_history)
    result: list[PlayerAction] = []
    for option in view.me.legal_actions:
        if option.kind in _PASSIVE:
            result.append(PlayerAction(option.kind))
        elif option.kind in _SIZED and option.min_to is not None:
            payment = option.min_to - hero.street_commit
            # Even outside verification these examples never deliberately jam.
            if payment < hero.stack and payment <= cap and not already_increased:
                result.append(PlayerAction(option.kind, option.min_to))
        elif option.kind is ActionKind.FOLD and not continuation:
            result.append(PlayerAction(option.kind))
    if not result:
        raise PolicyError("No candidate remains under the naive demonstration limits")
    return tuple(result)


def _prefer(candidates: tuple[PlayerAction, ...], priorities: tuple[ActionKind, ...]) -> PlayerAction:
    for kind in priorities:
        action = next((candidate for candidate in candidates if candidate.kind is kind), None)
        if action is not None:
            return action
    return candidates[0]


def _passive(candidates: tuple[PlayerAction, ...]) -> PlayerAction:
    return _prefer(candidates, (*_PASSIVE, ActionKind.FOLD, *_SIZED))


def _price(view: PlayerView) -> int:
    return next((option.pay or 0 for option in view.me.legal_actions if option.kind is ActionKind.CALL), 0)


def _record(representation: str, request: DecisionRequest, choice: PlayerAction | ActionDistribution,
            details: JsonObject) -> Decision:
    data: JsonObject = {
        "representation": representation, "mechanism": representation,
        "decision_id": request.decision_id, "revision": request.observation.view.revision,
        "hand_id": request.observation.view.hand_id,
        "phase": request.observation.view.phase.value if request.observation.view.phase else None,
        **details,
    }
    if isinstance(choice, PlayerAction):
        data["selected_action"] = choice.kind.value
        data["selected_to"] = choice.to
    get_logger("agent.naive").bind(correlation_id=request.decision_id,
                                   player_id=request.observation.view.me.player_id,
                                   hand_id=request.observation.view.hand_id).emit(
        "INFO", "naive.policy.decided", "Naive policy mechanism completed", data,
    )
    return Decision(choice, dict(data))


class RulesPolicy(Policy):
    def __init__(self, *, verification_continuation: bool = False) -> None:
        self.continuation = verification_continuation

    @property
    def identity(self) -> PolicyIdentity:
        return PolicyIdentity("naive.rules", "1")

    def decide(self, request: DecisionRequest, context: DecisionContext) -> Decision:
        context.check_cancelled()
        view = request.observation.view
        candidates = _candidates(view, self.continuation)
        # Ordered, executable conditions over public price/street/stack features.
        if _price(view) > max(1, _hero(view).stack // 5) and not self.continuation:
            selected = _prefer(candidates, (ActionKind.FOLD, *_PASSIVE))
            rule = 1
        elif view.phase is HandPhase.TURN and _price(view) == 0 and view.pot_total < _hero(view).stack // 2:
            selected = _prefer(candidates, (ActionKind.BET_TO, ActionKind.CHECK))
            rule = 2
        else:
            selected = _passive(candidates)
            rule = 3
        return _record("rules", request, selected, {"candidate_count": len(candidates),
                        "rules_evaluated": rule, "matched_rule": rule,
                        "verification_continuation": self.continuation})


class MixedPolicy(Policy):
    def __init__(self, *, verification_continuation: bool = False) -> None:
        self.continuation = verification_continuation

    @property
    def identity(self) -> PolicyIdentity:
        return PolicyIdentity("naive.mixed", "1")

    def decide(self, request: DecisionRequest, context: DecisionContext) -> Decision:
        context.check_cancelled()
        candidates = _candidates(request.observation.view, self.continuation)
        weights = {ActionKind.CHECK: 0.85, ActionKind.CALL: 0.90, ActionKind.FOLD: 0.01,
                   ActionKind.BET_TO: 0.15, ActionKind.RAISE_TO: 0.10}
        total = sum(weights[action.kind] for action in candidates)
        distribution = ActionDistribution(tuple(WeightedAction(action, weights[action.kind] / total)
                                                  for action in candidates))
        evidence: list[JsonValue] = [{"action": entry.action.kind.value, "to": entry.action.to,
                                     "probability": entry.probability}
                                    for entry in distribution.candidates]
        return _record("mixed", request, distribution, {"candidate_count": len(candidates),
                        "positive_candidates": len(candidates), "degenerate": len(candidates) == 1,
                        "distribution": evidence, "verification_continuation": self.continuation})


_DEFAULT_LOOKUP: dict[str, tuple[ActionKind, ...]] = {
    "preflop:free": (ActionKind.CHECK, ActionKind.RAISE_TO),
    "preflop:price": (ActionKind.CALL, ActionKind.FOLD),
    "flop:free": (ActionKind.CHECK, ActionKind.BET_TO),
    "flop:price": (ActionKind.CALL, ActionKind.FOLD),
    "turn:free": (ActionKind.CHECK, ActionKind.BET_TO),
    "turn:price": (ActionKind.CALL, ActionKind.FOLD),
    "river:free": (ActionKind.BET_TO, ActionKind.CHECK),
    "river:price": (ActionKind.CALL, ActionKind.FOLD),
}


class LookupPolicy(Policy):
    def __init__(self, table: Mapping[str, tuple[ActionKind, ...]] | None = None,
                 *, verification_continuation: bool = False) -> None:
        self.table = dict(_DEFAULT_LOOKUP if table is None else table)
        self.continuation = verification_continuation
        self._cache: dict[str, tuple[ActionKind, ...]] = {}
        self._hits = 0

    @property
    def identity(self) -> PolicyIdentity:
        return PolicyIdentity("naive.lookup", "1")

    def decide(self, request: DecisionRequest, context: DecisionContext) -> Decision:
        context.check_cancelled()
        view = request.observation.view
        candidates = _candidates(view, self.continuation)
        key = f"{view.phase.value if view.phase else 'none'}:{'price' if _price(view) else 'free'}"
        cached = key in self._cache
        hit = key in self.table
        if hit:
            self._hits += 1
            self._cache.setdefault(key, self.table[key])
            selected = _prefer(candidates, self._cache[key])
        else:
            selected = _passive(candidates)
        return _record("lookup", request, selected, {"candidate_count": len(candidates),
                        "lookup_key": key, "table_version": "naive-public-price-v1",
                        "table_entries": len(self.table), "table_hit": hit, "cache_hit": cached,
                        "cache_count": len(self._cache), "lookup_hits": self._hits,
                        "verification_continuation": self.continuation})


class StatefulPolicy(Policy):
    def __init__(self, *, verification_continuation: bool = False) -> None:
        self.continuation = verification_continuation
        self.confirmed = 0
        self.rejected = 0
        self.completed = 0
        self.observations = 0
        self._feedback_seen: set[str] = set()
        self._results_seen: set[str] = set()

    @property
    def identity(self) -> PolicyIdentity:
        return PolicyIdentity("naive.stateful", "1")

    def observe(self, event: PolicyEvent) -> None:
        if isinstance(event, ActionFeedback) and event.decision_id not in self._feedback_seen:
            self._feedback_seen.add(event.decision_id)
            if event.confirmed:
                self.confirmed += 1
            else:
                self.rejected += 1
        elif isinstance(event, HandCompleted) and event.result.hand_id not in self._results_seen:
            self._results_seen.add(event.result.hand_id)
            self.completed += 1
        elif isinstance(event, ObservationChanged):
            self.observations += 1

    def decide(self, request: DecisionRequest, context: DecisionContext) -> Decision:
        context.check_cancelled()
        candidates = _candidates(request.observation.view, self.continuation)
        mode = (self.confirmed + self.completed) % 3
        probe = mode == 2 and self.rejected == 0
        selected = (_prefer(candidates, (*_SIZED, *_PASSIVE)) if probe else _passive(candidates))
        return _record("stateful", request, selected, {"candidate_count": len(candidates),
                        "observations_seen": self.observations, "confirmed_memory": self.confirmed,
                        "rejected_memory": self.rejected, "completed_memory": self.completed,
                        "memory_mode": mode, "probe_selected_by_memory": probe,
                        "verification_continuation": self.continuation})


@dataclass(frozen=True, slots=True)
class EquityEstimate:
    equity: float
    rollouts: int
    opponent_count: int
    evaluated_hands: int


def estimate_equity(view: PlayerView, context: DecisionContext, rollouts: int) -> EquityEstimate:
    """Uniform information-set sampling, using only player-visible card values."""
    if type(rollouts) is not int or not 1 <= rollouts <= 256:
        raise ValueError("solver_rollouts must be an integer from 1 through 256")
    hero = tuple(Card.parse(code) for code in view.me.hole_cards)
    board = tuple(Card.parse(code) for code in view.board)
    if len(hero) != 2 or len(board) > 5:
        raise PolicyError("Equity requires two private cards and at most five board cards")
    opponents = tuple(player for player in view.players if player.player_id != view.me.player_id
                      and player.status is not None and player.status is not PlayerStatus.FOLDED)
    known_opponents = {player.player_id: tuple(Card.parse(code) for code in player.revealed_cards)
                       for player in opponents if player.revealed_cards}
    if any(len(cards) != 2 for cards in known_opponents.values()):
        raise PolicyError("A publicly revealed holding must have two cards")
    all_reveals = tuple(Card.parse(code) for player in view.players if player.player_id != view.me.player_id
                        for code in player.revealed_cards)
    visible = (*hero, *board, *all_reveals)
    if len(visible) != len(set(visible)):
        raise PolicyError("Visible card values are inconsistent")
    unknown = [Card(rank, suit) for suit in Suit for rank in Rank if Card(rank, suit) not in visible]
    missing_board = 5 - len(board)
    unknown_players = tuple(player for player in opponents if player.player_id not in known_opponents)
    need = missing_board + 2 * len(unknown_players)
    if need > len(unknown):
        raise PolicyError("Insufficient unseen cards for the visible game")
    evaluator = FiveCardHighEvaluator()
    share = 0.0
    for _ in range(rollouts):
        context.check_cancelled()
        sampled = context.random.sample(unknown, need)
        full_board = (*board, *sampled[:missing_board])
        hero_value = evaluator.evaluate_best((*hero, *full_board))
        holdings = dict(known_opponents)
        for index, player in enumerate(unknown_players):
            offset = missing_board + 2 * index
            holdings[player.player_id] = tuple(sampled[offset:offset + 2])
        other_values = [evaluator.evaluate_best((*holdings[player.player_id], *full_board))
                        for player in opponents]
        if not other_values or hero_value >= max(other_values):
            share += 1.0 / (1 + sum(value == hero_value for value in other_values))
    context.check_cancelled()
    return EquityEstimate(share / rollouts, rollouts, len(opponents), rollouts * (len(opponents) + 1))


def approximate_payoff(view: PlayerView, action: PlayerAction, equity: float) -> float:
    """Local showdown payoff in chips; documented uniform-call approximation."""
    if not isfinite(equity) or not 0 <= equity <= 1:
        raise ValueError("Equity must be a finite probability")
    if action.kind is ActionKind.FOLD:
        return 0.0
    hero = _hero(view)
    payment = _price(view) if action.kind is ActionKind.CALL else 0
    extra = 0
    if action.kind in _SIZED:
        if action.to is None:
            raise PolicyError("A sized candidate needs a street total")
        payment = action.to - hero.street_commit
        extra = sum(min(player.stack, max(0, action.to - player.street_commit))
                    for player in view.players if player.player_id != view.me.player_id
                    and player.status is PlayerStatus.ACTIVE)
    return equity * (view.pot_total + payment + extra) - payment


class SolverPolicy(Policy):
    def __init__(self, *, rollouts: int = 24, verification_continuation: bool = False) -> None:
        if type(rollouts) is not int or not 1 <= rollouts <= 256:
            raise ValueError("solver_rollouts must be an integer from 1 through 256")
        self.rollouts = rollouts
        self.continuation = verification_continuation

    @property
    def identity(self) -> PolicyIdentity:
        return PolicyIdentity("naive.solver", "uniform-call-monte-carlo-v1")

    def decide(self, request: DecisionRequest, context: DecisionContext) -> Decision:
        view = request.observation.view
        candidates = _candidates(view, self.continuation)
        estimate = estimate_equity(view, context, self.rollouts)
        scored = tuple((action, approximate_payoff(view, action, estimate.equity)) for action in candidates)
        selected, value = max(scored, key=lambda item: item[1])
        scores: list[JsonValue] = [{"action": action.kind.value, "to": action.to, "estimated_ev": score}
                                   for action, score in scored]
        return _record("solver", request, selected, {"candidate_count": len(candidates),
                        "rollouts": estimate.rollouts, "opponent_count": estimate.opponent_count,
                        "evaluated_hands": estimate.evaluated_hands, "equity_estimate": estimate.equity,
                        "candidate_evs": scores, "selected_ev": value,
                        "solver_model": "uniform_unknown_cards_then_call_to_showdown",
                        "verification_continuation": self.continuation})


def rules(config: JsonObject) -> Policy:
    return RulesPolicy(verification_continuation=_continuation(config))


def mixed(config: JsonObject) -> Policy:
    return MixedPolicy(verification_continuation=_continuation(config))


def lookup(config: JsonObject) -> Policy:
    raw = config.get("lookup_table")
    table: dict[str, tuple[ActionKind, ...]] | None = None
    if raw is not None:
        if not isinstance(raw, dict):
            raise ValueError("lookup_table must map keys to lists of action names")
        table = {}
        for key, value in raw.items():
            if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
                raise ValueError("Each lookup entry needs a nonempty list of action names")
            table[key] = tuple(ActionKind(cast(str, item)) for item in value)
    return LookupPolicy(table, verification_continuation=_continuation(config))


def stateful(config: JsonObject) -> Policy:
    return StatefulPolicy(verification_continuation=_continuation(config))


def solver(config: JsonObject) -> Policy:
    rollouts = config.get("solver_rollouts", 24)
    if type(rollouts) is not int:
        raise ValueError("solver_rollouts must be an integer from 1 through 256")
    return SolverPolicy(rollouts=rollouts, verification_continuation=_continuation(config))
