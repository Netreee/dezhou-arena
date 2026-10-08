import json
from dataclasses import asdict
from typing import cast

from poker.application.commands import (
    ActCommand, Command, CommandKind, JoinCommand, StartHandCommand, StateCommand,
)
from poker.application.views import (
    AwardView, CommandResponse, ErrorInfo, PlayerView, PrivatePlayerView,
    PublicPlayerView, ResultView, ShareView,
)
from poker.domain.models import ActionOption, PlayerAction, PublicActionRecord, TableConfig
from poker.domain.types import (
    ActionKind, ErrorCode, HandId, HandPhase, PlayerId, PlayerStatus, TableId,
)

JsonObject = dict[str, object]


def _object(value: object) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return cast(JsonObject, value)


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected a string")
    return value


def _integer(value: object) -> int:
    if type(value) is not int:
        raise ValueError("Expected an integer")
    return value


def _boolean(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError("Expected a boolean")
    return value


def _optional_integer(value: object) -> int | None:
    return None if value is None else _integer(value)


def _optional_string(value: object) -> str | None:
    return None if value is None else _string(value)


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("Expected an array")
    return cast(list[object], value)


def _strings(value: object) -> tuple[str, ...]:
    return tuple(_string(item) for item in _list(value))


def _keys(data: JsonObject, allowed: set[str]) -> None:
    if set(data) - allowed:
        raise ValueError("Unknown fields in command")


class JsonLineCodec:
    """JSON values only. The TCP adapter adds/removes the newline frame."""

    def encode_command(self, command: Command, *, correlation_id: str | None = None) -> str:
        data: JsonObject = {"command": command.kind.value}
        if isinstance(command, JoinCommand):
            data["name"] = command.name
            if command.table_id is not None:
                data["table_id"] = command.table_id
        elif isinstance(command, ActCommand):
            data["hand_id"] = command.hand_id
            data["action"] = asdict(command.action)
            if command.expected_revision is not None:
                data["expected_revision"] = command.expected_revision
        if correlation_id is not None:
            data["correlation_id"] = correlation_id
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    def decode_command(self, raw: str) -> Command:
        return self.decode_request(raw)[0]

    def decode_request(self, raw: str) -> tuple[Command, str | None]:
        data = _object(json.loads(raw))
        correlation = _optional_string(data.pop("correlation_id", None))
        if correlation is not None and (not correlation or len(correlation) > 96):
            raise ValueError("Invalid correlation identifier")
        return self._command(data), correlation

    def _command(self, data: JsonObject) -> Command:
        kind = CommandKind(_string(data.get("command")))
        if kind is CommandKind.JOIN:
            _keys(data, {"command", "name", "table_id"})
            table_id = _optional_string(data.get("table_id"))
            return JoinCommand(_string(data.get("name")), TableId(table_id) if table_id is not None else None)
        if kind is CommandKind.STATE:
            _keys(data, {"command"})
            return StateCommand()
        if kind is CommandKind.START_HAND:
            _keys(data, {"command"})
            return StartHandCommand()
        _keys(data, {"command", "hand_id", "action", "expected_revision"})
        action = _object(data.get("action"))
        _keys(action, {"kind", "to"})
        return ActCommand(
            HandId(_string(data.get("hand_id"))),
            PlayerAction(ActionKind(_string(action.get("kind"))),
                         _optional_integer(action.get("to"))),
            _optional_integer(data.get("expected_revision")),
        )

    def encode_response(self, response: CommandResponse) -> str:
        data: JsonObject = {"ok": response.ok}
        if response.view is not None:
            data["view"] = asdict(response.view)
        if response.error is not None:
            data["error"] = asdict(response.error)
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    def decode_response(self, raw: str) -> CommandResponse:
        data = _object(json.loads(raw))
        ok = data.get("ok")
        if type(ok) is not bool:
            raise ValueError("Expected a boolean ok field")
        if not ok:
            error = _object(data.get("error"))
            return CommandResponse(error=ErrorInfo(
                ErrorCode(_string(error.get("code"))), _string(error.get("message")),
            ))
        return CommandResponse(view=self._view(_object(data.get("view"))))

    def _view(self, data: JsonObject) -> PlayerView:
        members: list[PublicPlayerView] = []
        for item in _list(data.get("players")):
            member = _object(item)
            status = _optional_string(member.get("status"))
            members.append(PublicPlayerView(
                PlayerId(_string(member.get("player_id"))), _string(member.get("name")),
                _integer(member.get("seat")), _integer(member.get("stack")),
                PlayerStatus(status) if status is not None else None,
                _integer(member.get("street_commit")), _integer(member.get("hand_commit")),
                _strings(member.get("revealed_cards")),
            ))
        me = _object(data.get("me"))
        options: list[ActionOption] = []
        for item in _list(me.get("legal_actions")):
            option = _object(item)
            options.append(ActionOption(
                ActionKind(_string(option.get("kind"))), _optional_integer(option.get("pay")),
                _optional_integer(option.get("min_to")), _optional_integer(option.get("max_to")),
            ))
        hand_id = _optional_string(data.get("hand_id"))
        phase = _optional_string(data.get("phase"))
        actor_id = _optional_string(data.get("actor_id"))
        result = data.get("result")
        config = data.get("config")
        return PlayerView(
            TableId(_string(data.get("table_id"))), _integer(data.get("revision")),
            HandId(hand_id) if hand_id is not None else None,
            HandPhase(phase) if phase is not None else None,
            _strings(data.get("board")), _optional_integer(data.get("button_seat")),
            PlayerId(actor_id) if actor_id is not None else None,
            _integer(data.get("current_bet")), _integer(data.get("pot_total")),
            tuple(members),
            PrivatePlayerView(PlayerId(_string(me.get("player_id"))),
                              _strings(me.get("hole_cards")), tuple(options)),
            self._result(_object(result)) if result is not None else None,
            self._config(_object(config)) if config is not None else None,
            tuple(self._public_action(_object(event)) for event in _list(data.get("action_history", []))),
            _boolean(data.get("history_complete", False)) if "action_history" in data else False,
        )

    @staticmethod
    def _config(data: JsonObject) -> TableConfig:
        return TableConfig(
            _integer(data.get("small_blind")), _integer(data.get("big_blind")),
            _integer(data.get("starting_stack")), _integer(data.get("max_players")),
        )

    @staticmethod
    def _public_action(data: JsonObject) -> PublicActionRecord:
        return PublicActionRecord(
            _integer(data.get("sequence")), HandPhase(_string(data.get("phase"))),
            PlayerId(_string(data.get("player_id"))), _string(data.get("kind")),
            _integer(data.get("pay")), _integer(data.get("to")), _integer(data.get("stack")),
        )

    @staticmethod
    def _share(data: JsonObject) -> ShareView:
        return ShareView(PlayerId(_string(data.get("player_id"))), _integer(data.get("amount")))

    def _result(self, data: JsonObject) -> ResultView:
        awards: list[AwardView] = []
        for item in _list(data.get("awards")):
            award = _object(item)
            awards.append(AwardView(
                _integer(award.get("amount")),
                tuple(PlayerId(pid) for pid in _strings(award.get("eligible_ids"))),
                tuple(self._share(_object(share)) for share in _list(award.get("shares"))),
            ))
        return ResultView(
            HandId(_string(data.get("hand_id"))), tuple(awards),
            tuple(self._share(_object(share)) for share in _list(data.get("refunds"))),
        )
