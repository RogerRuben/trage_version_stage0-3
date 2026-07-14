"""Idle movement policies for Simulator v3.

This module implements a conservative operational package for O1/O3:

* HV vehicles use an empirical idle-repositioning proxy label.  The first
  implementation uses stable operational-zone centers and deterministic
  time/vehicle hashing rather than test-day future demand.
* AV vehicles use a platform rebalancing label with the same no-future-demand
  restriction.

All movements are published as vehicle plans and executed by VehicleExecutor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .entities import PlanStop, VehiclePlan, VehicleState
from .enums import StopType


@dataclass
class IdleMovementDecision:
    vehicle: VehicleState
    plan: VehiclePlan
    movement_reason: str


class IdleMovementManager:
    def __init__(self, zone_system: dict, interval_sec: int = 300, max_share_per_epoch: float = 0.02):
        self.zone_system = zone_system
        self.interval_sec = interval_sec
        self.max_share_per_epoch = max_share_per_epoch
        self.records: list[dict] = []

    def should_run(self, now: pd.Timestamp) -> bool:
        seconds = int(now.timestamp())
        return seconds % self.interval_sec == 0

    def zone_center(self, zone: str) -> tuple[float, float]:
        match = re.match(r"z(\d+)_(\d+)", str(zone))
        grid = float(self.zone_system.get("grid_size", 0.02))
        min_lon = float(self.zone_system.get("min_lon", 108.9))
        min_lat = float(self.zone_system.get("min_lat", 34.2))
        if not match:
            return min_lon + grid / 2, min_lat + grid / 2
        x = int(match.group(1))
        y = int(match.group(2))
        return min_lon + (x + 0.5) * grid, min_lat + (y + 0.5) * grid

    def target_zone(self, current_zone: str, now: pd.Timestamp, vehicle_id: str) -> str:
        match = re.match(r"z(\d+)_(\d+)", str(current_zone))
        if not match:
            return current_zone
        x = int(match.group(1))
        y = int(match.group(2))
        step = (sum(ord(c) for c in vehicle_id) + int(now.hour)) % 4
        if step == 0:
            x += 1
        elif step == 1:
            y += 1
        elif step == 2:
            x = max(0, x - 1)
        else:
            y = max(0, y - 1)
        return f"z{x}_{y}"

    def build_plans(self, vehicles: list[VehicleState], now: pd.Timestamp, vehicle_type: str) -> list[IdleMovementDecision]:
        if not self.should_run(now):
            return []
        limit = max(1, int(len(vehicles) * self.max_share_per_epoch)) if vehicles else 0
        decisions: list[IdleMovementDecision] = []
        for vehicle in sorted(vehicles, key=lambda v: v.vehicle_id)[:limit]:
            target_zone = self.target_zone(vehicle.current_zone, now, vehicle.vehicle_id)
            if target_zone == vehicle.current_zone:
                continue
            lon, lat = self.zone_center(target_zone)
            version = vehicle.plan_version + 1
            stop_type = StopType.AV_REBALANCE if vehicle_type == "AV" else StopType.HV_REPOSITION
            reason = "platform_rebalancing" if vehicle_type == "AV" else "empirical_idle_repositioning_proxy"
            plan = VehiclePlan(
                vehicle_id=vehicle.vehicle_id,
                plan_version=version,
                stops=[
                    PlanStop(
                        stop_id=f"{vehicle.vehicle_id}:{version}:{stop_type.value}:{target_zone}",
                        stop_type=stop_type,
                        request_id=None,
                        lon=lon,
                        lat=lat,
                        zone=target_zone,
                        earliest_start=None,
                        latest_start=None,
                        planned_arrival=now,
                        planned_departure=now,
                        locked=False,
                    )
                ],
                created_time=now,
                trigger=reason,
                feasible=True,
                objective_value=0.0,
            )
            decisions.append(IdleMovementDecision(vehicle, plan, reason))
            self.records.append({
                "vehicle_id": vehicle.vehicle_id,
                "vehicle_type": vehicle_type,
                "movement_time": str(now),
                "origin_zone": vehicle.current_zone,
                "target_zone": target_zone,
                "movement_reason": reason,
            })
        return decisions

