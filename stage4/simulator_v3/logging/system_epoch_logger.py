"""System epoch log helper."""

from __future__ import annotations

import pandas as pd

from ..system_state import SystemState


def epoch_record(state: SystemState, decision_time: pd.Timestamp, extra: dict) -> dict:
    base = {
        "decision_time": str(decision_time),
        "pending_orders": len(state.pending_request_ids),
        "offered_orders": len(state.offered_request_ids),
        "reserved_orders": len(state.reserved_request_ids),
        "idle_HV": sum(v.vehicle_type == "HV" and v.execution_status.value == "IDLE" for v in state.vehicles.values()),
        "idle_AV": sum(v.vehicle_type == "AV" and v.execution_status.value == "IDLE" for v in state.vehicles.values()),
        "busy_HV": sum(v.vehicle_type == "HV" and v.execution_status.value not in {"IDLE", "OFFLINE"} for v in state.vehicles.values()),
        "busy_AV": sum(v.vehicle_type == "AV" and v.execution_status.value not in {"IDLE", "OFFLINE"} for v in state.vehicles.values()),
        "repositioning_HV": sum(v.vehicle_type == "HV" and v.execution_status.value == "REPOSITIONING" for v in state.vehicles.values()),
        "rebalancing_AV": sum(v.vehicle_type == "AV" and v.execution_status.value == "REBALANCING" for v in state.vehicles.values()),
        "routing_queries": state.routing_query_count,
        "routing_cache_hit_rate": state.routing_cache_hit_count / state.routing_query_count if state.routing_query_count else 0.0,
    }
    base.update(extra)
    return base

