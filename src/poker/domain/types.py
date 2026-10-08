"""Small shared types. IDs are static types rather than wrapper objects."""

from enum import IntEnum, StrEnum
from typing import NewType

TableId = NewType("TableId", str)
PlayerId = NewType("PlayerId", str)
HandId = NewType("HandId", str)


class Suit(StrEnum):
    CLUBS = "c"
    DIAMONDS = "d"
    HEARTS = "h"
    SPADES = "s"


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


class HandPhase(StrEnum):
    SETUP = "setup"
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"
    SETTLEMENT = "settlement"
    COMPLETE = "complete"


class PlayerStatus(StrEnum):
    ACTIVE = "active"
    FOLDED = "folded"
    ALL_IN = "all_in"


class ActionKind(StrEnum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET_TO = "bet_to"
    RAISE_TO = "raise_to"
    ALL_IN = "all_in"


class HandCategory(IntEnum):
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8


class ErrorCode(StrEnum):
    BAD_REQUEST = "bad_request"
    ALREADY_SEATED = "already_seated"
    NOT_SEATED = "not_seated"
    TABLE_FULL = "table_full"
    TABLE_NOT_FOUND = "table_not_found"
    HAND_IN_PROGRESS = "hand_in_progress"
    NOT_ENOUGH_PLAYERS = "not_enough_players"
    HAND_MISMATCH = "hand_mismatch"
    STALE_STATE = "stale_state"
    NOT_YOUR_TURN = "not_your_turn"
    INVALID_ACTION = "invalid_action"
    RAISE_TOO_SMALL = "raise_too_small"
    RAISE_NOT_REOPENED = "raise_not_reopened"
