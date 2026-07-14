"""Mutable simulation state container."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .entities import RequestState, VehicleState


@dataclass
class SystemState:
    current_time: pd.Timestamp
    requests: dict[str, RequestState] = field(default_factory=dict)
    vehicles: dict[str, VehicleState] = field(default_factory=dict)
    pending_request_ids: set[str] = field(default_factory=set)
    offered_request_ids: set[str] = field(default_factory=set)
    reserved_request_ids: set[str] = field(default_factory=set)
    completed_request_ids: set[str] = field(default_factory=set)
    cancelled_request_ids: set[str] = field(default_factory=set)
    routing_query_count: int = 0
    routing_cache_hit_count: int = 0

