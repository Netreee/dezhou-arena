"""Bounded factual memory, populated only by actual runtime feedback events."""

from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
import json
from typing import cast

from shared_logging import JsonObject

from poker.agent.models import ActionFeedback, HandCompleted, Observation, PolicyEvent


class Memory(ABC):
    @abstractmethod
    def read(self, observation: Observation) -> JsonObject:
        """Return a detached player-visible memory snapshot for this decision."""
        raise NotImplementedError

    @abstractmethod
    def observe(self, event: PolicyEvent) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Optional local cleanup after the last callback."""


class NullMemory(Memory):
    def read(self, observation: Observation) -> JsonObject:
        return {}

    def observe(self, event: PolicyEvent) -> None:
        pass


class BoundedEventMemory(Memory):
    """Keep confirmed/rejected actions and completed-hand results without inference.

    Deduplication covers the retained window, so both records and ID tracking are
    bounded. ObservationChanged, including an old result in a new observation,
    never manufactures a feedback event or a newly completed hand.
    """

    def __init__(self, max_events: int = 64) -> None:
        if type(max_events) is not int or max_events < 1:
            raise ValueError("max_events must be a positive integer")
        self.max_events = max_events
        self._records: OrderedDict[tuple[str, str], JsonObject] = OrderedDict()

    def read(self, observation: Observation) -> JsonObject:
        return {"records": [deepcopy(record) for record in self._records.values()],
                "capacity": self.max_events, "deduplication_scope": "retained_records"}

    def observe(self, event: PolicyEvent) -> None:
        record: JsonObject
        if isinstance(event, ActionFeedback):
            key = ("action_feedback", event.decision_id)
            record = {"type": "action_feedback", "decision_id": event.decision_id,
                      "action": {"kind": event.action.kind.value, "to": event.action.to},
                      "confirmed": event.confirmed, "error_code": event.error_code}
        elif isinstance(event, HandCompleted):
            key = ("hand_completed", event.result.hand_id)
            record = {"type": "hand_completed", "result": cast(JsonObject, json.loads(json.dumps(asdict(event.result))))}
        else:
            return
        if key in self._records:
            return
        self._records[key] = record
        if len(self._records) > self.max_events:
            self._records.popitem(last=False)
