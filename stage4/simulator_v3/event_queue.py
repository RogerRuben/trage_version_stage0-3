"""Heap-backed event queue."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .enums import EventType


EVENT_PRIORITY = {
    EventType.LEG_COMPLETED: 1,
    EventType.SERVICE_COMPLETED: 2,
    EventType.HV_SESSION_END: 3,
    EventType.HV_SESSION_START: 4,
    EventType.REQUEST_REVEALED: 5,
    EventType.DRIVER_RESPONSE: 6,
    EventType.RESERVATION_EXPIRED: 7,
    EventType.PLAN_INVALIDATED: 8,
    EventType.DECISION_EPOCH: 9,
}


@dataclass(order=True)
class SimulationEvent:
    event_time: pd.Timestamp
    priority: int
    sequence: int
    event_type: EventType = field(compare=False)
    entity_id: str = field(compare=False)
    payload: dict[str, Any] = field(default_factory=dict, compare=False)


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[SimulationEvent] = []
        self._seq = 0

    def push(self, event_time: pd.Timestamp, event_type: EventType, entity_id: str, payload: dict | None = None) -> None:
        self._seq += 1
        event = SimulationEvent(
            event_time=pd.Timestamp(event_time),
            priority=EVENT_PRIORITY[event_type],
            sequence=self._seq,
            event_type=event_type,
            entity_id=entity_id,
            payload=payload or {},
        )
        heapq.heappush(self._heap, event)

    def pop(self) -> SimulationEvent:
        return heapq.heappop(self._heap)

    def peek_time(self) -> pd.Timestamp | None:
        return self._heap[0].event_time if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)

    def to_records(self) -> list[dict]:
        return [
            {
                "event_time": str(e.event_time),
                "priority": e.priority,
                "sequence": e.sequence,
                "event_type": e.event_type.value,
                "entity_id": e.entity_id,
                "payload": e.payload,
            }
            for e in sorted(self._heap)
        ]

