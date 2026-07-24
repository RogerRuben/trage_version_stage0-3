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

    def audit_metadata(self, order_id: str, vehicle_type: str) -> dict:
        """Return immutable mapping provenance for assignment logging."""
        if vehicle_type != "AV":
            return {
                "capability_profile": "not_applicable_hv",
                "capability_mapping_version": "not_applicable_hv",
            }
        row = self.capability_rows.get(str(order_id), {})
        return {
            "capability_profile": str(row.get("vehicle_profile", "missing_capability_row")),
            "capability_mapping_version": str(row.get("capability_mapping_version", "missing_capability_mapping_version")),
        }
