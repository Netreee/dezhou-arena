"""Provider-independent values exchanged by policies and the agent runtime."""

from dataclasses import dataclass, field
from enum import StrEnum
from math import isclose, isfinite
from random import Random
from typing import TypeAlias

from poker.application.views import PlayerView, ResultView
from poker.domain.models import PlayerAction
from poker.domain.types import ActionKind
from shared_logging import JsonObject


class PolicyError(RuntimeError):
    """A policy cannot produce a decision; the runtime must not invent a move."""


class DecisionCancelled(PolicyError):
    pass


class DecisionDeadlineExceeded(PolicyError):
    pass


class ToolBudgetExceeded(PolicyError):
    pass


class InvalidDecision(PolicyError):
    pass


@dataclass(frozen=True, slots=True)
class PolicyIdentity:
    name: str
    version: str = "unspecified"


@dataclass(frozen=True, slots=True)
class PolicySession:
    agent_id: str
    player_name: str


@dataclass(frozen=True, slots=True)
class Observation:
    """An immutable player projection received from the CLI, never a Table."""

    view: PlayerView
    received_at: float


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    decision_id: str
    observation: Observation
    deadline: float


@dataclass(frozen=True, slots=True)
class WeightedAction:
    action: PlayerAction
    probability: float


@dataclass(frozen=True, slots=True)
class ActionDistribution:
    """Optional finite mixture, not an enumeration of the entire betting space."""

    candidates: tuple[WeightedAction, ...]


@dataclass(frozen=True, slots=True)
class Decision:
    choice: PlayerAction | ActionDistribution
    # Optional diagnostics; not sent to the Arena or logged by default.
    annotations: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ObservationChanged:
    observation: Observation


@dataclass(frozen=True, slots=True)
class ActionFeedback:
    decision_id: str
    action: PlayerAction
    confirmed: bool
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class HandCompleted:
    """A result actually observed through the CLI; no inferred reward."""

    result: ResultView


PolicyEvent: TypeAlias = ObservationChanged | ActionFeedback | HandCompleted


class StopReason(StrEnum):
    REQUESTED = "requested"
    MAX_ACTIONS = "max_actions"
    MAX_HANDS = "max_hands"
    SESSION_LIMIT = "session_limit"
    CLI_FAILURE = "cli_failure"
    POLICY_FAILURE = "policy_failure"
    INVALID_DECISION = "invalid_decision"
    DEADLINE = "deadline"
    TOOL_BUDGET = "tool_budget"
    SERVER_REJECTED = "server_rejected"


def validate_action(action: PlayerAction, view: PlayerView) -> None:
    if not isinstance(action, PlayerAction) or not isinstance(action.kind, ActionKind):
        raise InvalidDecision("Expected a PlayerAction with an ActionKind")
    if view.actor_id != view.me.player_id:
        raise InvalidDecision("The observation does not offer this player a turn")
    option = next((item for item in view.me.legal_actions if item.kind is action.kind), None)
    if option is None:
        raise InvalidDecision(f"Action {action.kind.value} is not offered by the CLI")
    if action.kind in (ActionKind.BET_TO, ActionKind.RAISE_TO):
        if type(action.to) is not int:
            raise InvalidDecision("A sized action requires an integer street total")
        if option.min_to is None or option.max_to is None or not option.min_to <= action.to <= option.max_to:
            raise InvalidDecision("Action amount is outside the CLI bounds")
    elif action.to is not None:
        raise InvalidDecision("This action must not carry an amount")


def select_action(decision: Decision, view: PlayerView, random: Random) -> PlayerAction:
    """Validate every candidate before sampling; never repair a policy's output."""
    if not isinstance(decision, Decision):
        raise InvalidDecision("Policy.decide must return Decision")
    choice = decision.choice
    if isinstance(choice, PlayerAction):
        validate_action(choice, view)
        return choice
    if not isinstance(choice, ActionDistribution) or not choice.candidates:
        raise InvalidDecision("Expected an action or a nonempty finite distribution")
    seen: set[tuple[ActionKind, int | None]] = set()
    probabilities: list[float] = []
    for item in choice.candidates:
        if not isinstance(item, WeightedAction):
            raise InvalidDecision("Distribution entries must be WeightedAction values")
        validate_action(item.action, view)
        key = (item.action.kind, item.action.to)
        if key in seen:
            raise InvalidDecision("A distribution cannot contain duplicate actions")
        seen.add(key)
        probability = item.probability
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not isfinite(probability) or probability < 0:
            raise InvalidDecision("Probabilities must be finite nonnegative numbers")
        probabilities.append(probability)
    if not isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise InvalidDecision("Probabilities must sum to one")
    draw = random.random()
    cumulative = 0.0
    last_positive: PlayerAction | None = None
    for item in choice.candidates:
        cumulative += item.probability
        if item.probability > 0:
            last_positive = item.action
        if draw < cumulative:
            return item.action
    assert last_positive is not None
    return last_positive
