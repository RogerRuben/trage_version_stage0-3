"""Service-route ODD lookup."""

from __future__ import annotations


class ServiceODDChecker:
    def __init__(self, capability_rows: dict[str, dict]):
        self.capability_rows = capability_rows

    def check(self, order_id: str, vehicle_type: str, condition_available: bool) -> tuple[bool, str]:
        if vehicle_type != "AV":
            return True, "not_applicable_hv"
        if not condition_available:
            return False, "unknown_condition_av_disabled"
        row = self.capability_rows.get(str(order_id))
        if row is None:
            return False, "missing_capability_row"
        return bool(row.get("service_feasible", False)), "service_capability_mapping"

