"""Action-producing port used by the infrastructure, including mechanical baselines."""

from abc import ABC, abstractmethod

from poker.agent.context import DecisionContext
from poker.agent.models import Decision, DecisionRequest, PolicyEvent, PolicyIdentity, PolicySession, StopReason


class DecisionEngine(ABC):
    """One instance per agent session; all callbacks run serially on one worker.

    Implementations own their state, weights, prompts and model clients. They may
    use the supplied analysis tools, but receive no CLI or Arena command channel.
    All blocking I/O should use context.remaining_seconds() as its timeout.
    """

    @property
    def identity(self) -> PolicyIdentity:
        return PolicyIdentity(type(self).__qualname__)

    def open(self, session: PolicySession) -> None:
        """Optional initialization; invoked once before any observations."""

    def observe(self, event: PolicyEvent) -> None:
        """Optional memory update, including actual action acknowledgements."""

    @abstractmethod
    def decide(self, request: DecisionRequest, context: DecisionContext) -> Decision:
        """Propose exactly one action or one finite mixture before the deadline."""
        raise NotImplementedError

    def close(self, reason: StopReason) -> None:
        """Optional cleanup, serialized after the last running callback returns."""
