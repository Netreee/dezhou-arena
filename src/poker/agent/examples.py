"""Connection examples only; none claims poker strength."""

from shared_logging import JsonObject
from poker.agent.context import DecisionContext
from poker.agent.models import ActionDistribution, Decision, DecisionRequest, PolicyError, PolicyIdentity, WeightedAction
from poker.agent.policy import Policy
from poker.domain.models import ActionOption, PlayerAction
from poker.domain.types import ActionKind


def _minimum(option: ActionOption) -> PlayerAction:
    return PlayerAction(option.kind, option.min_to if option.kind in (ActionKind.BET_TO, ActionKind.RAISE_TO) else None)


class FirstLegalPolicy(Policy):
    @property
    def identity(self) -> PolicyIdentity:
        return PolicyIdentity("example.first_legal", "1")

    def decide(self, request: DecisionRequest, context: DecisionContext) -> Decision:
        options = request.observation.view.me.legal_actions
        if not options:
            raise PolicyError("No legal action was supplied")
        return Decision(_minimum(options[0]))


class UniformPolicy(Policy):
    @property
    def identity(self) -> PolicyIdentity:
        return PolicyIdentity("example.uniform", "1")

    def decide(self, request: DecisionRequest, context: DecisionContext) -> Decision:
        options = request.observation.view.me.legal_actions
        if not options:
            raise PolicyError("No legal action was supplied")
        return Decision(ActionDistribution(tuple(WeightedAction(_minimum(option), 1.0 / len(options))
                                                 for option in options)))


class ToolFirstPolicy(Policy):
    @property
    def identity(self) -> PolicyIdentity:
        return PolicyIdentity("example.tool_first", "1")

    def decide(self, request: DecisionRequest, context: DecisionContext) -> Decision:
        result = context.tools.call("legal_actions", {})
        if not result.ok or not isinstance(result.value, list) or not result.value:
            raise PolicyError("Legal-actions tool did not return actions")
        first = result.value[0]
        if not isinstance(first, dict):
            raise PolicyError("Invalid action description")
        raw_kind = first.get("kind")
        if not isinstance(raw_kind, str):
            raise PolicyError("Invalid action kind")
        kind = ActionKind(raw_kind)
        amount = first.get("min_to")
        if kind in (ActionKind.BET_TO, ActionKind.RAISE_TO):
            if type(amount) is not int:
                raise PolicyError("Missing minimum street total")
            return Decision(PlayerAction(kind, amount))
        return Decision(PlayerAction(kind))


def first_legal(config: JsonObject) -> Policy:
    return FirstLegalPolicy()


def uniform(config: JsonObject) -> Policy:
    return UniformPolicy()


def tool_first(config: JsonObject) -> Policy:
    return ToolFirstPolicy()
