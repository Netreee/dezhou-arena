"""Local evidence tools: calculate estimates and suggestions, never pick a move."""

from collections.abc import Mapping
import hashlib
import json
from random import Random

from shared_logging import JsonObject, JsonValue

from poker.agent.context import DecisionContext
from poker.agent.models import InvalidDecision, validate_action
from poker.agent.naive_policies import approximate_payoff, estimate_equity
from poker.agent.tools import Tool, ToolContext, ToolRegistry, ToolSpec, view_json
from poker.application.views import PlayerView
from poker.domain.models import PlayerAction
from poker.domain.types import ActionKind


def _default_candidates(view: PlayerView) -> tuple[PlayerAction, ...]:
    candidates: list[PlayerAction] = []
    for option in view.me.legal_actions:
        if option.kind in (ActionKind.BET_TO, ActionKind.RAISE_TO):
            for amount in dict.fromkeys((option.min_to, option.max_to)):
                if amount is not None:
                    candidates.append(PlayerAction(option.kind, amount))
        else:
            candidates.append(PlayerAction(option.kind))
    return tuple(candidates)


class EquityTool(Tool):
    def __init__(self, rollouts: int = 24, seed: int = 0) -> None:
        if type(rollouts) is not int or not 1 <= rollouts <= 256:
            raise ValueError("rollouts must be an integer from 1 through 256")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.rollouts = rollouts
        self.seed = seed

    @property
    def spec(self) -> ToolSpec:
        action: JsonObject = {"type": "string", "enum": [kind.value for kind in ActionKind]}
        candidate: JsonObject = {"anyOf": [
            {"type": "object", "properties": {"action": action},
             "required": ["action"], "additionalProperties": False},
            {"type": "object", "properties": {"action": action, "to": {"type": ["integer", "null"]}},
             "required": ["action", "to"], "additionalProperties": False},
        ]}
        arguments: JsonObject = {"anyOf": [
            {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            {"type": "object", "properties": {"candidates": {
                "type": "array", "minItems": 1, "maxItems": 32, "items": candidate}},
             "required": ["candidates"], "additionalProperties": False},
        ]}
        return ToolSpec("equity", "Estimate equity and coarse candidate EVs from visible cards; no action is selected. "
                        "Uniform opponents, no fold equity/future betting/side-pot eligibility; all-in EV unsupported.",
                        arguments, {"type": "object"})

    def invoke(self, arguments: JsonObject, context: ToolContext) -> JsonValue:
        context.control.check()
        view = context.observation.view
        candidates = _default_candidates(view)
        if "candidates" in arguments:
            raw_candidates = arguments["candidates"]
            if not isinstance(raw_candidates, list) or not 1 <= len(raw_candidates) <= 32:
                raise ValueError("candidates must be a bounded list of concrete actions")
            parsed: list[PlayerAction] = []
            for raw in raw_candidates:
                if not isinstance(raw, dict) or not isinstance(raw.get("action"), str) or set(raw) - {"action", "to"}:
                    raise ValueError("Each candidate requires an action and optional street total")
                kind = raw["action"]
                assert isinstance(kind, str)
                amount = raw.get("to")
                if amount is not None and type(amount) is not int:
                    raise ValueError("A street total must be an integer")
                parsed.append(PlayerAction(ActionKind(kind), amount))
            candidates = tuple(parsed)
        for candidate in candidates:
            try:
                validate_action(candidate, view)
            except InvalidDecision as error:
                # Invalid analysis arguments are recoverable tool errors; they
                # are not the model's final action and must not abort the loop.
                raise ValueError("An analysis candidate is outside the visible legal actions") from error
        if len(set(candidates)) != len(candidates):
            raise ValueError("Candidate actions must be unique")
        # The RNG seed derives only from public/player-visible input, never the
        # private server deck seed or Arena objects. Repeat calls are reproducible.
        seed_material = json.dumps({"seed": self.seed, "view": view_json(context.observation)}, sort_keys=True)
        seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest(), "big")
        tools = ToolRegistry(()).bind(context.observation, context.control, max_calls=0, decision_id="equity-analysis")
        estimate = estimate_equity(view, DecisionContext(tools, Random(seed), context.control), self.rollouts)
        scores: list[JsonValue] = []
        for candidate in candidates:
            scores.append({"action": candidate.kind.value, "to": candidate.to,
                           "estimated_ev": (None if candidate.kind is ActionKind.ALL_IN
                                            else approximate_payoff(view, candidate, estimate.equity)),
                           "unsupported_reason": "all_in_side_pot_payoff" if candidate.kind is ActionKind.ALL_IN else None})
        context.control.check()
        return {"equity_estimate": estimate.equity, "rollouts": estimate.rollouts,
                "opponent_count": estimate.opponent_count, "evaluated_hands": estimate.evaluated_hands,
                "candidate_evs": scores, "model": "uniform_unknown_cards_then_call_to_showdown",
                "limitations": ["Uniform unknown opponent holdings, not inferred opponent ranges.",
                                "No future betting or fold equity; opponents assumed to match candidate increases.",
                                "Side-pot eligibility is ignored; all-in EV is not calculated.",
                                "Default sized candidates are legal minimum and maximum, not an exhaustive action space."]}


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


class LookupTool(Tool):
    def __init__(self, table: Mapping[str, tuple[ActionKind, ...]] | None = None) -> None:
        self.table = dict(_DEFAULT_LOOKUP if table is None else table)
        if any(not isinstance(key, str) or not isinstance(value, tuple) or not value
               or any(not isinstance(kind, ActionKind) for kind in value) for key, value in self.table.items()):
            raise ValueError("Lookup entries must map string keys to nonempty tuples of ActionKind values")

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec("lookup", "Look up a naive phase/price action preference table as advice, not a final action or GTO strategy.",
                        {"type": "object", "properties": {}, "required": [], "additionalProperties": False}, {"type": "object"})

    def invoke(self, arguments: JsonObject, context: ToolContext) -> JsonValue:
        context.control.check()
        view = context.observation.view
        price = next((option.pay or 0 for option in view.me.legal_actions if option.kind is ActionKind.CALL), 0)
        key = f"{view.phase.value if view.phase else 'none'}:{'price' if price else 'free'}"
        offered = {option.kind for option in view.me.legal_actions}
        priorities = self.table.get(key, ())
        return {"lookup_key": key, "table_version": "naive-public-price-v1", "table_hit": key in self.table,
                "suggested_priorities": [kind.value for kind in priorities],
                "offered_suggestions": [kind.value for kind in priorities if kind in offered],
                "limitation": "Illustrative lookup preferences; neither equilibrium strategy nor evaluated playing strength."}
