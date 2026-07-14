"""Pickup-route ODD conservative proxy."""

from __future__ import annotations


class PickupODDChecker:
    def __init__(self, zone_pair_map: dict[tuple[str, str], bool]):
        self.zone_pair_map = zone_pair_map

    def check(self, vehicle_type: str, vehicle_zone: str, pickup_zone: str, condition_available: bool) -> tuple[bool, str]:
        if vehicle_type != "AV":
            return True, "not_applicable_hv"
        if not condition_available:
            return False, "unknown_condition_av_disabled"
        key = (str(vehicle_zone), str(pickup_zone))
        if key not in self.zone_pair_map:
            return False, "pickup_odd_proxy_unknown_zone_pair"
        return bool(self.zone_pair_map[key]), "pickup_odd_proxy_v1"

