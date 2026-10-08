from abc import ABC, abstractmethod
from itertools import combinations

from poker.domain.cards import Card
from poker.domain.models import (
    ActionOption, HandResult, HandValue, PlayerAction, Pot, Table,
)
from poker.domain.types import HandId, PlayerId


class BettingRules(ABC):
    @abstractmethod
    def legal_actions(self, table: Table, actor_id: PlayerId) -> tuple[ActionOption, ...]:
        """Return options for this player only; never change table."""
        raise NotImplementedError

    @abstractmethod
    def apply(self, table: Table, actor_id: PlayerId, action: PlayerAction) -> None:
        """Validate and update chips, pending actions and raise rights."""
        raise NotImplementedError

    @abstractmethod
    def round_complete(self, table: Table) -> bool:
        raise NotImplementedError


class HandEvaluator(ABC):
    @abstractmethod
    def evaluate_five(self, cards: tuple[Card, ...]) -> HandValue:
        """Exactly five unique cards; compare category then kickers."""
        raise NotImplementedError

    def evaluate_best(self, cards: tuple[Card, ...]) -> HandValue:
        """Shared parent algorithm; seven cards yield 21 five-card evaluations."""
        if not 5 <= len(cards) <= 7 or len(set(cards)) != len(cards):
            raise ValueError("Expected five to seven unique cards")
        return max(self.evaluate_five(group) for group in combinations(cards, 5))


class PotAllocator(ABC):
    @abstractmethod
    def refund_uncalled(self, table: Table) -> None:
        """Return unmatched chips and reduce the relevant contributions."""
        raise NotImplementedError

    @abstractmethod
    def build_pots(self, table: Table) -> tuple[Pot, ...]:
        """Include folded contributions but exclude folded eligibility."""
        raise NotImplementedError

    @abstractmethod
    def settle(self, table: Table, evaluator: HandEvaluator) -> HandResult:
        """Pay each pot once, save the result, and end this hand."""
        raise NotImplementedError


class GameEngine(ABC):
    @abstractmethod
    def join(self, table: Table, name: str) -> PlayerId:
        raise NotImplementedError

    @abstractmethod
    def start_hand(self, table: Table, actor_id: PlayerId) -> None:
        """Create the hand and advance until waiting or complete."""
        raise NotImplementedError

    @abstractmethod
    def act(
        self, table: Table, actor_id: PlayerId, hand_id: HandId, action: PlayerAction,
    ) -> None:
        """Apply one player action, then advance automatically."""
        raise NotImplementedError

    @abstractmethod
    def legal_actions(self, table: Table, actor_id: PlayerId) -> tuple[ActionOption, ...]:
        raise NotImplementedError
