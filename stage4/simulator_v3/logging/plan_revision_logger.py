"""Plan revision log helpers."""

from __future__ import annotations

import pandas as pd

from ..entities import VehiclePlan


def plan_revision_record(old_version: int, new_plan: VehiclePlan, revision_time: pd.Timestamp, validation_status: str) -> dict:
    return {
        "vehicle_id": new_plan.vehicle_id,
        "old_plan_version": old_version,
        "new_plan_version": new_plan.plan_version,
        "revision_time": str(revision_time),
        "trigger": new_plan.trigger,
        "added_request_id": ",".join(new_plan.assigned_request_ids + new_plan.reserved_request_ids),
        "removed_request_id": "",
        "old_plan_stop_count": 0,
        "new_plan_stop_count": len(new_plan.stops),
        "objective_before": 0.0,
        "objective_after": new_plan.objective_value,
        "validation_status": validation_status,
    }

