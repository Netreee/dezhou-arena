from dataclasses import dataclass, field
from random import Random, SystemRandom

from poker.domain.types import Rank, Suit

_SYMBOLS = "23456789TJQKA"


@dataclass(frozen=True, slots=True)
class Card:
    rank: Rank
    suit: Suit

    @property
    def code(self) -> str:
        return _SYMBOLS[int(self.rank) - 2] + self.suit.value

    @classmethod
    def parse(cls, code: str) -> "Card":
        if len(code) != 2 or code[0] not in _SYMBOLS:
            raise ValueError(f"Invalid card code: {code}")
        return cls(Rank(_SYMBOLS.index(code[0]) + 2), Suit(code[1]))


@dataclass(slots=True)
class Deck:
    """Owns remaining and burned cards. Dealt cards belong to Hand."""

    remaining: list[Card]
    burned: list[Card] = field(default_factory=list)

    def __post_init__(self) -> None:
        cards = self.remaining + self.burned
        if len(cards) != len(set(cards)):
            raise ValueError("Deck contains duplicate cards")

    @classmethod
    def shuffled(cls, rng: Random | None = None) -> "Deck":
        cards = [Card(rank, suit) for suit in Suit for rank in Rank]
        (rng if rng is not None else SystemRandom()).shuffle(cards)
        return cls(cards)

    def draw(self, count: int = 1) -> tuple[Card, ...]:
        if count < 1 or count > len(self.remaining):
            raise ValueError("Invalid draw count")
        drawn = tuple(self.remaining[:count])
        del self.remaining[:count]
        return drawn

    def burn(self) -> None:
        self.burned.extend(self.draw())
