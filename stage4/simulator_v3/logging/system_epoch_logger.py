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
        "idle_HV": len(state.idle_hv_ids),
        "idle_AV": len(state.idle_av_ids),
        "busy_HV": len(state.busy_hv_ids),
        "busy_AV": len(state.busy_av_ids),
        "repositioning_HV": len(state.repositioning_hv_ids),
        "rebalancing_AV": len(state.rebalancing_av_ids),
        "full_vehicle_scan_count": state.full_vehicle_scan_count,
        "decision_epoch_full_scan_count": state.decision_epoch_full_scan_count,
        "routing_queries": state.routing_query_count,
        "routing_cache_hit_rate": state.routing_cache_hit_count / state.routing_query_count if state.routing_query_count else 0.0,
    }
    base.update(extra)
    return base
