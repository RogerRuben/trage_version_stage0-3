"""Mutable simulation state container."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .entities import RequestState, VehicleState
from .enums import VehicleExecutionStatus


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
    offline_vehicle_ids: set[str] = field(default_factory=set)
    idle_hv_ids: set[str] = field(default_factory=set)
    idle_av_ids: set[str] = field(default_factory=set)
    busy_hv_ids: set[str] = field(default_factory=set)
    busy_av_ids: set[str] = field(default_factory=set)
    pickup_vehicle_ids: set[str] = field(default_factory=set)
    service_vehicle_ids: set[str] = field(default_factory=set)
    repositioning_hv_ids: set[str] = field(default_factory=set)
    rebalancing_av_ids: set[str] = field(default_factory=set)
    full_vehicle_scan_count: int = 0
    decision_epoch_full_scan_count: int = 0

    def initialize_vehicle_indexes(self) -> None:
        self.offline_vehicle_ids = set(self.vehicles)
        self.idle_hv_ids.clear()
        self.idle_av_ids.clear()
        self.busy_hv_ids.clear()
        self.busy_av_ids.clear()
        self.pickup_vehicle_ids.clear()
        self.service_vehicle_ids.clear()
        self.repositioning_hv_ids.clear()
        self.rebalancing_av_ids.clear()

    def set_vehicle_status(self, vehicle_id: str, new_status: VehicleExecutionStatus) -> None:
        vehicle = self.vehicles[vehicle_id]
        for group in [
            self.offline_vehicle_ids,
            self.idle_hv_ids,
            self.idle_av_ids,
            self.busy_hv_ids,
            self.busy_av_ids,
            self.pickup_vehicle_ids,
            self.service_vehicle_ids,
            self.repositioning_hv_ids,
            self.rebalancing_av_ids,
        ]:
            group.discard(vehicle_id)
        vehicle.execution_status = new_status
        if new_status == VehicleExecutionStatus.OFFLINE:
            self.offline_vehicle_ids.add(vehicle_id)
        elif new_status == VehicleExecutionStatus.IDLE:
            if vehicle.vehicle_type == "AV":
                self.idle_av_ids.add(vehicle_id)
            else:
                self.idle_hv_ids.add(vehicle_id)
        elif new_status == VehicleExecutionStatus.WAITING:
            (self.busy_av_ids if vehicle.vehicle_type == "AV" else self.busy_hv_ids).add(vehicle_id)
        elif new_status == VehicleExecutionStatus.PICKUP:
            self.pickup_vehicle_ids.add(vehicle_id)
            (self.busy_av_ids if vehicle.vehicle_type == "AV" else self.busy_hv_ids).add(vehicle_id)
        elif new_status == VehicleExecutionStatus.SERVICE:
            self.service_vehicle_ids.add(vehicle_id)
            (self.busy_av_ids if vehicle.vehicle_type == "AV" else self.busy_hv_ids).add(vehicle_id)
        elif new_status == VehicleExecutionStatus.REPOSITIONING:
            self.repositioning_hv_ids.add(vehicle_id)
            self.busy_hv_ids.add(vehicle_id)
        elif new_status == VehicleExecutionStatus.REBALANCING:
            self.rebalancing_av_ids.add(vehicle_id)
            self.busy_av_ids.add(vehicle_id)
        else:
            (self.busy_av_ids if vehicle.vehicle_type == "AV" else self.busy_hv_ids).add(vehicle_id)
