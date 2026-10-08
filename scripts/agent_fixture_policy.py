"""Verification-only plugins: traverse every street without claiming poker skill."""

from typing import cast

from shared_logging import JsonObject
from poker.agent.context import DecisionContext
from poker.agent.models import ActionDistribution, Decision, DecisionRequest, PolicyError, PolicyIdentity, WeightedAction
from poker.agent.policy import Policy
from poker.domain.models import PlayerAction
from poker.domain.types import ActionKind


class CheckCallTools(Policy):
    @property
    def identity(self) -> PolicyIdentity:
        return PolicyIdentity("fixture.check_call.tools", "1")

    def decide(self, request: DecisionRequest, context: DecisionContext) -> Decision:
        observed = context.tools.call("observation", {})
        history = context.tools.call("public_history", {})
        offered = context.tools.call("legal_actions", {})
        if not observed.ok or not history.ok or not offered.ok:
            raise PolicyError("Verification tool call failed")
        if not isinstance(observed.value, dict) or not isinstance(history.value, dict) or not isinstance(offered.value, list):
            raise PolicyError("Verification tool returned the wrong shape")
        if observed.value["revision"] != request.observation.view.revision or history.value["complete"] is not True:
            raise PolicyError("Tool observation changed or public history was incomplete")
        options = {str(cast(JsonObject, option)["kind"]) for option in offered.value}
        for name in ("check", "call", "fold"):
            if name in options:
                return Decision(PlayerAction(ActionKind(name)))
        raise PolicyError("Verification fixture found no continuation action")


class CheckCallDistribution(Policy):
    @property
    def identity(self) -> PolicyIdentity:
        return PolicyIdentity("fixture.check_call.distribution", "1")

    def decide(self, request: DecisionRequest, context: DecisionContext) -> Decision:
        context.check_cancelled()
        options = {option.kind for option in request.observation.view.me.legal_actions}
        for kind in (ActionKind.CHECK, ActionKind.CALL, ActionKind.FOLD):
            if kind in options:
                # A degenerate distribution tests the same sampler/output path
                # while deterministically exercising all four betting streets.
                return Decision(ActionDistribution((WeightedAction(PlayerAction(kind), 1.0),)))
        raise PolicyError("Verification fixture found no continuation action")


def tool_check_call(config: JsonObject) -> Policy:
    return CheckCallTools()


def distribution_check_call(config: JsonObject) -> Policy:
    return CheckCallDistribution()
