"""Private snapshot format. This codec must never serve network responses."""

import json
from dataclasses import asdict
from typing import NotRequired, TypedDict, cast

from poker.domain.cards import Card, Deck
from poker.domain.models import (
    BettingRound, ChipShare, Hand, HandPlayer, HandResult, Player, Pot,
    PotAward, PublicActionRecord, Table, TableConfig,
)
from poker.domain.types import HandId, HandPhase, PlayerId, PlayerStatus, Rank, Suit, TableId


class ConfigData(TypedDict):
    small_blind: int
    big_blind: int
    starting_stack: int
    max_players: int


class CardData(TypedDict):
    rank: int
    suit: str


class DeckData(TypedDict):
    remaining: list[CardData]
    burned: list[CardData]


class PlayerData(TypedDict):
    id: str
    name: str
    seat: int
    stack: int


class MemberData(TypedDict):
    hole_cards: list[CardData]
    status: str
    street_commit: int
    hand_commit: int


class BettingData(TypedDict):
    actor_id: str | None
    current_bet: int
    last_full_raise: int
    full_opening_established: bool
    pending: list[str]
    last_action_bet: dict[str, int]


class PublicActionData(TypedDict):
    sequence: int
    phase: str
    player_id: str
    kind: str
    pay: int
    to: int
    stack: int


class HandData(TypedDict):
    id: str
    phase: str
    button_seat: int
    small_blind_seat: int
    big_blind_seat: int
    deck: DeckData
    chip_total_at_start: int
    players: dict[str, MemberData]
    board: list[CardData]
    betting: BettingData
    refunds: list["ShareData"]
    action_history: NotRequired[list[PublicActionData]]
    history_complete: NotRequired[bool]


class ShareData(TypedDict):
    player_id: str
    amount: int


class PotData(TypedDict):
    amount: int
    eligible_ids: list[str]


class AwardData(TypedDict):
    pot: PotData
    shares: list[ShareData]


class ResultData(TypedDict):
    hand_id: str
    awards: list[AwardData]
    refunds: list[ShareData]
    revealed_hands: dict[str, list[CardData]]


class TableData(TypedDict):
    id: str
    config: ConfigData
    players: list[PlayerData]
    button_seat: int | None
    hand: HandData | None
    last_result: ResultData | None
    revision: int


class TableSnapshotCodec:
    def encode(self, table: Table) -> str:
        return json.dumps(asdict(table), ensure_ascii=False, separators=(",", ":"))

    def decode(self, payload: str) -> Table:
        # Owned SQLite format, not an untrusted wire schema or migration system.
        data = cast(TableData, json.loads(payload))
        return Table(
            TableId(data["id"]), TableConfig(**data["config"]),
            [Player(PlayerId(p["id"]), p["name"], p["seat"], p["stack"])
             for p in data["players"]],
            data["button_seat"],
            self._hand(data["hand"]) if data["hand"] is not None else None,
            self._result(data["last_result"]) if data["last_result"] is not None else None,
            data["revision"],
        )

    @staticmethod
    def _cards(data: list[CardData]) -> tuple[Card, ...]:
        return tuple(Card(Rank(c["rank"]), Suit(c["suit"])) for c in data)

    def _hand(self, data: HandData) -> Hand:
        betting = data["betting"]
        return Hand(
            HandId(data["id"]), HandPhase(data["phase"]),
            data["button_seat"], data["small_blind_seat"], data["big_blind_seat"],
            Deck(list(self._cards(data["deck"]["remaining"])),
                 list(self._cards(data["deck"]["burned"]))),
            data["chip_total_at_start"],
            {PlayerId(pid): HandPlayer(
                self._cards(member["hole_cards"]), PlayerStatus(member["status"]),
                member["street_commit"], member["hand_commit"],
            ) for pid, member in data["players"].items()},
            list(self._cards(data["board"])),
            BettingRound(
                PlayerId(betting["actor_id"]) if betting["actor_id"] is not None else None,
                betting["current_bet"], betting["last_full_raise"],
                betting["full_opening_established"],
                [PlayerId(pid) for pid in betting["pending"]],
                {PlayerId(pid): amount for pid, amount in betting["last_action_bet"].items()},
            ),
            [self._share(refund) for refund in data["refunds"]],
            [PublicActionRecord(
                event["sequence"], HandPhase(event["phase"]), PlayerId(event["player_id"]),
                event["kind"], event["pay"], event["to"], event["stack"],
            ) for event in data.get("action_history", [])],
            data.get("history_complete", False) if "action_history" in data else False,
        )

    @staticmethod
    def _share(data: ShareData) -> ChipShare:
        return ChipShare(PlayerId(data["player_id"]), data["amount"])

    def _result(self, data: ResultData) -> HandResult:
        return HandResult(
            HandId(data["hand_id"]),
            tuple(PotAward(
                Pot(a["pot"]["amount"], tuple(PlayerId(pid) for pid in a["pot"]["eligible_ids"])),
                tuple(self._share(share) for share in a["shares"]),
            ) for a in data["awards"]),
            tuple(self._share(share) for share in data["refunds"]),
            {PlayerId(pid): self._cards(cards) for pid, cards in data["revealed_hands"].items()},
        )
