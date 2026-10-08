"""Public logging contract; no business or backend dependencies."""

from abc import ABC, abstractmethod
from typing import TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class EventLogger(ABC):
    @abstractmethod
    def bind(self, **context: JsonScalar) -> "EventLogger":
        """Return a logger carrying additional scalar context."""

    @abstractmethod
    def emit(self, level: str, event: str, message: str, data: JsonObject | None = None) -> None:
        """Write one event through the process logging configuration."""

    @abstractmethod
    def exception(self, event: str, message: str, data: JsonObject | None = None, *, error: BaseException | None = None) -> None:
        """Write an error event with the current or explicitly supplied exception."""
