"""Plan revision log helpers."""

from __future__ import annotations

import pandas as pd

from ..entities import VehiclePlan


def plan_revision_record(old_plan: VehiclePlan, new_plan: VehiclePlan, revision_time: pd.Timestamp, validation_status: str) -> dict:
    old_assigned = set(old_plan.assigned_request_ids)
    new_assigned = set(new_plan.assigned_request_ids)
    old_reserved = set(old_plan.reserved_request_ids)
    new_reserved = set(new_plan.reserved_request_ids)
    added = sorted((new_assigned | new_reserved) - (old_assigned | old_reserved))
    removed = sorted((old_assigned | old_reserved) - (new_assigned | new_reserved))
    return {
        "vehicle_id": new_plan.vehicle_id,
        "old_plan_version": old_plan.plan_version,
        "new_plan_version": new_plan.plan_version,
        "revision_time": str(revision_time),
        "trigger": new_plan.trigger,
        "added_request_id": ",".join(added),
        "removed_request_id": ",".join(removed),
        "old_plan_stop_count": len(old_plan.stops),
        "new_plan_stop_count": len(new_plan.stops),
        "old_assigned_requests": ",".join(old_plan.assigned_request_ids),
        "new_assigned_requests": ",".join(new_plan.assigned_request_ids),
        "old_reserved_requests": ",".join(old_plan.reserved_request_ids),
        "new_reserved_requests": ",".join(new_plan.reserved_request_ids),
        "objective_before": old_plan.objective_value,
        "objective_after": new_plan.objective_value,
        "locked_stop_count_before": sum(1 for s in old_plan.stops if s.locked),
        "locked_stop_count_after": sum(1 for s in new_plan.stops if s.locked),
        "validation_status": validation_status,
    }
