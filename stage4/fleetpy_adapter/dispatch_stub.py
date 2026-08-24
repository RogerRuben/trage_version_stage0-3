"""Transparent deterministic dispatch stub for the FleetPy compatibility spike."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .mixed_fleet_adapter import VehicleRuntime
from .replay_service_time_adapter import FleetPyLifecycleAdapter
from .test31_demand_adapter import SpikeRequest
from .upstream import FleetPyCompatibilityError
from .valhalla_time_adapter import PickupEstimate, ValhallaPickupTimeAdapter


@dataclass
class SpikeRunResult:
    request_log: pd.DataFrame
    vehicle_log: pd.DataFrame
    assignment_log: pd.DataFrame
    summary: dict[str, Any]


class MinCorrectedPickupEtaDispatchStub:
    """Sequential minimum corrected-pickup-ETA policy; not a research baseline."""

    def __init__(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        requests: list[SpikeRequest],
        vehicles: list[VehicleRuntime],
        eta_adapter: ValhallaPickupTimeAdapter,
        lifecycle: FleetPyLifecycleAdapter,
        native_output: list[dict[str, Any]],
    ) -> None:
        self.start = start
        self.end = end
        self.requests = requests
        self.request_by_id = {request.order_id: request for request in requests}
        self.vehicles = vehicles
        self.vehicle_by_id = {
            vehicle.fixture.vehicle_id: vehicle for vehicle in vehicles
        }
        self.eta_adapter = eta_adapter
        self.lifecycle = lifecycle
        self.native_output = native_output
        self.waiting: list[str] = []
        self.events: list[tuple[float, int, str, str, str | None]] = []
        self.event_sequence = 0
        self.request_rows: dict[str, dict[str, Any]] = {}
        self.vehicle_rows: list[dict[str, Any]] = []
        self.assignment_rows: list[dict[str, Any]] = []
        self.hv_session_end_exclusions: set[tuple[float, str, str]] = set()
        self.av_availability_violations = 0
        self.position_transition_failures = 0
        self.request_timing_failures = 0
        self.routing_failures = 0

    def _seconds(self, timestamp: pd.Timestamp) -> float:
        return float((timestamp - self.start).total_seconds())

    def _timestamp(self, seconds: float) -> pd.Timestamp:
        return self.start + pd.Timedelta(seconds=float(seconds))

    def _push(
        self,
        seconds: float,
        event_type: str,
        vehicle_id: str,
        order_id: str | None = None,
    ) -> None:
        priority = {
            "DROPOFF": 0,
            "PICKUP": 1,
            "ACTIVATE": 2,
            "EXPIRE": 3,
            "REQUEST": 4,
        }[event_type]
        self.event_sequence += 1
        heapq.heappush(
            self.events,
            (float(seconds), priority, event_type, vehicle_id, order_id),
        )

    def _vehicle_event(
        self, vehicle: VehicleRuntime, timestamp: pd.Timestamp, event: str
    ) -> None:
        self.vehicle_rows.append(
            {
                "event_time": timestamp,
                "simulation_time_s": self._seconds(timestamp),
                "vehicle_id": vehicle.fixture.vehicle_id,
                "vehicle_type": vehicle.fixture.vehicle_type,
                "event": event,
                "state": vehicle.state,
                "current_lon_wgs84": vehicle.current_lon_wgs84,
                "current_lat_wgs84": vehicle.current_lat_wgs84,
                "active_order_id": vehicle.active_order_id,
                "availability_start_time": vehicle.fixture.availability_start_time,
                "availability_end_time": vehicle.fixture.availability_end_time,
                "availability_policy": (
                    "FULL_SIMULATION_HORIZON"
                    if vehicle.fixture.vehicle_type == "AV"
                    else "RECONSTRUCTED_S0_SESSION_WINDOW"
                ),
            }
        )

    def _initialize(self) -> None:
        for request in self.requests:
            self._push(request.sim_time_s, "REQUEST", "", request.order_id)
            self.request_rows[request.order_id] = {
                "order_id": request.order_id,
                "profile_id": request.profile_id,
                "historical_request_time": request.request_time,
                "request_enter_time": pd.NaT,
                "selected_vehicle_id": None,
                "selected_vehicle_type": None,
                "corrected_pickup_eta_s": np.nan,
                "pickup_time": pd.NaT,
                "service_start_time": pd.NaT,
                "service_end_time": pd.NaT,
                "dropoff_lon_wgs84": request.dropoff_lon_wgs84,
                "dropoff_lat_wgs84": request.dropoff_lat_wgs84,
                "next_availability_time": pd.NaT,
                "hard_state": request.hard_state,
                "evidence_complete": request.evidence_complete,
                "rho_static": request.rho_static,
                "rho_dynamic": request.rho_dynamic,
                "rho_speed": request.rho_speed,
                "status": "NOT_INJECTED",
            }
        for vehicle in self.vehicles:
            start_s = self._seconds(vehicle.fixture.availability_start_time)
            end_s = self._seconds(vehicle.fixture.availability_end_time)
            if start_s > 0:
                vehicle.state = "INACTIVE"
                self._push(start_s, "ACTIVATE", vehicle.fixture.vehicle_id)
            else:
                vehicle.state = "AVAILABLE"
                self._vehicle_event(vehicle, self.start, "ACTIVATED")
            self._push(end_s, "EXPIRE", vehicle.fixture.vehicle_id)

    def _candidate_estimate(
        self, vehicle: VehicleRuntime, request: SpikeRequest, timestamp: pd.Timestamp
    ) -> PickupEstimate | None:
        if not vehicle.available_at(timestamp):
            return None
        if vehicle.fixture.vehicle_type == "AV" and not request.av_smoke_eligible:
            return None
        try:
            estimate = self.eta_adapter.estimate(
                vehicle.current_lon_wgs84,
                vehicle.current_lat_wgs84,
                request.pickup_lon_wgs84,
                request.pickup_lat_wgs84,
                timestamp,
            )
        except Exception:
            self.routing_failures += 1
            return None
        if vehicle.fixture.vehicle_type == "HV":
            predicted_end = timestamp + pd.Timedelta(
                seconds=estimate.corrected_pickup_eta_s
                + request.predicted_service_time_s
            )
            if predicted_end > vehicle.fixture.availability_end_time:
                self.hv_session_end_exclusions.add(
                    (
                        self._seconds(timestamp),
                        vehicle.fixture.vehicle_id,
                        request.order_id,
                    )
                )
                return None
        return estimate

    def _dispatch(self, timestamp: pd.Timestamp) -> None:
        if timestamp >= self.end or not self.waiting:
            return
        retained: list[str] = []
        for order_id in self.waiting:
            request = self.request_by_id[order_id]
            candidates: list[tuple[float, str, VehicleRuntime, PickupEstimate]] = []
            for vehicle in self.vehicles:
                estimate = self._candidate_estimate(vehicle, request, timestamp)
                if estimate is not None:
                    candidates.append(
                        (
                            estimate.corrected_pickup_eta_s,
                            vehicle.fixture.vehicle_id,
                            vehicle,
                            estimate,
                        )
                    )
            if not candidates:
                retained.append(order_id)
                continue
            _, _, vehicle, estimate = min(
                candidates, key=lambda item: (item[0], item[1])
            )
            if vehicle.fixture.vehicle_type == "AV" and not (
                vehicle.fixture.availability_start_time == self.start
                and vehicle.fixture.availability_end_time == self.end
                and not vehicle.fixture.av_source_session_end_inherited
            ):
                self.av_availability_violations += 1
            sim_time_s = self._seconds(timestamp)
            self.lifecycle.assign(vehicle, request, estimate, sim_time_s)
            pickup_time = timestamp + pd.Timedelta(
                seconds=estimate.corrected_pickup_eta_s
            )
            service_end = pickup_time + pd.Timedelta(
                seconds=request.realized_service_time_s
            )
            vehicle.next_event_time = pickup_time
            self._push(
                self._seconds(pickup_time),
                "PICKUP",
                vehicle.fixture.vehicle_id,
                order_id,
            )
            self._push(
                self._seconds(service_end),
                "DROPOFF",
                vehicle.fixture.vehicle_id,
                order_id,
            )
            row = self.request_rows[order_id]
            row.update(
                {
                    "selected_vehicle_id": vehicle.fixture.vehicle_id,
                    "selected_vehicle_type": vehicle.fixture.vehicle_type,
                    "corrected_pickup_eta_s": estimate.corrected_pickup_eta_s,
                    "pickup_time": pickup_time,
                    "service_start_time": pickup_time,
                    "service_end_time": service_end,
                    "next_availability_time": service_end,
                    "status": "ASSIGNED",
                }
            )
            self.assignment_rows.append(
                {
                    "assignment_time": timestamp,
                    "order_id": order_id,
                    "vehicle_id": vehicle.fixture.vehicle_id,
                    "vehicle_type": vehicle.fixture.vehicle_type,
                    "pickup_eta_s": estimate.corrected_pickup_eta_s,
                    "valhalla_time_s": estimate.valhalla_time_s,
                    "beta": estimate.beta,
                    "time_bin_index": estimate.time_bin_index,
                    "pickup_route_distance_m": estimate.route_distance_m,
                    "predicted_service_time_s": request.predicted_service_time_s,
                    "realized_service_time_s": request.realized_service_time_s,
                    "hv_session_end_time": (
                        vehicle.fixture.availability_end_time
                        if vehicle.fixture.vehicle_type == "HV"
                        else pd.NaT
                    ),
                    "dispatch_policy": "MIN_CORRECTED_PICKUP_ETA",
                }
            )
            self._vehicle_event(vehicle, timestamp, "ASSIGNED")
        self.waiting = retained

    def _process_event(
        self, seconds: float, event_type: str, vehicle_id: str, order_id: str | None
    ) -> None:
        timestamp = self._timestamp(seconds)
        if event_type == "REQUEST":
            if order_id is None:
                raise FleetPyCompatibilityError("request event missing order_id")
            request = self.request_by_id[order_id]
            self.waiting.append(order_id)
            row = self.request_rows[order_id]
            row["request_enter_time"] = timestamp
            row["status"] = "WAITING"
            if timestamp != request.request_time:
                self.request_timing_failures += 1
            return
        vehicle = self.vehicle_by_id[vehicle_id]
        if event_type == "ACTIVATE":
            if vehicle.state == "INACTIVE":
                vehicle.state = "AVAILABLE"
                self._vehicle_event(vehicle, timestamp, "ACTIVATED")
        elif event_type == "EXPIRE":
            if vehicle.state == "AVAILABLE":
                vehicle.state = "EXPIRED"
            self._vehicle_event(vehicle, timestamp, "WINDOW_END")
        elif event_type == "PICKUP":
            request = self.request_by_id[str(order_id)]
            self.lifecycle.arrive_pickup(vehicle, request, seconds)
            vehicle.next_event_time = self.request_rows[request.order_id][
                "service_end_time"
            ]
            self._vehicle_event(vehicle, timestamp, "PICKUP_COMPLETED")
        elif event_type == "DROPOFF":
            request = self.request_by_id[str(order_id)]
            self.lifecycle.arrive_dropoff(vehicle, request, seconds)
            if not (
                np.isclose(vehicle.current_lon_wgs84, request.dropoff_lon_wgs84)
                and np.isclose(vehicle.current_lat_wgs84, request.dropoff_lat_wgs84)
            ):
                self.position_transition_failures += 1
            row = self.request_rows[request.order_id]
            if not np.isclose(
                (row["service_end_time"] - row["service_start_time"]).total_seconds(),
                request.realized_service_time_s,
            ):
                self.request_timing_failures += 1
            row["status"] = "COMPLETED"
            vehicle.state = (
                "AVAILABLE"
                if timestamp < vehicle.fixture.availability_end_time
                else "EXPIRED"
            )
            self._vehicle_event(vehicle, timestamp, "SERVICE_COMPLETED")

    def run(self) -> SpikeRunResult:
        self._initialize()
        while self.events:
            seconds = self.events[0][0]
            same_time: list[tuple[float, int, str, str, str | None]] = []
            while self.events and self.events[0][0] == seconds:
                same_time.append(heapq.heappop(self.events))
            same_time.sort(key=lambda item: (item[1], item[3], str(item[4])))
            for event in same_time:
                self._process_event(event[0], event[2], event[3], event[4])
            timestamp = self._timestamp(seconds)
            if timestamp <= self.end:
                self._dispatch(timestamp)
        for order_id in self.waiting:
            self.request_rows[order_id]["status"] = "WAITING_UNASSIGNED"

        request_log = pd.DataFrame(self.request_rows.values()).sort_values(
            ["historical_request_time", "order_id"], kind="mergesort"
        )
        vehicle_log = pd.DataFrame(self.vehicle_rows).sort_values(
            ["event_time", "vehicle_id", "event"], kind="mergesort"
        )
        assignment_log = pd.DataFrame(self.assignment_rows).sort_values(
            ["assignment_time", "order_id"], kind="mergesort"
        )
        completed = request_log["status"].eq("COMPLETED")
        assigned = request_log["selected_vehicle_id"].notna()
        native_completed = sum(
            request.native_request.do_time is not None for request in self.requests
        )
        native_reconciled = native_completed == int(completed.sum()) and len(
            self.native_output
        ) == 4 * int(completed.sum())
        pickup_values = (
            assignment_log["pickup_eta_s"]
            if len(assignment_log)
            else pd.Series(dtype=float)
        )
        summary = {
            "requests_injected": int(len(request_log)),
            "requests_assigned": int(assigned.sum()),
            "requests_completed": int(completed.sum()),
            "hv_assignments": int(
                assignment_log["vehicle_type"].eq("HV").sum()
                if len(assignment_log)
                else 0
            ),
            "av_assignments": int(
                assignment_log["vehicle_type"].eq("AV").sum()
                if len(assignment_log)
                else 0
            ),
            "mean_corrected_pickup_eta_s": float(pickup_values.mean())
            if len(pickup_values)
            else None,
            "max_corrected_pickup_eta_s": float(pickup_values.max())
            if len(pickup_values)
            else None,
            "vehicles_activated": int(
                vehicle_log["event"]
                .eq("ACTIVATED")
                .groupby(vehicle_log["vehicle_id"])
                .any()
                .sum()
            ),
            "hv_session_end_exclusions": int(len(self.hv_session_end_exclusions)),
            "av_availability_violations": int(self.av_availability_violations),
            "position_transition_failures": int(self.position_transition_failures),
            "request_timing_failures": int(self.request_timing_failures),
            "valhalla_route_failures": int(self.routing_failures),
            "valhalla_calls": int(len(self.eta_adapter.call_log)),
            "valhalla_cache_hits": int(self.eta_adapter.cache_hit_count),
            "fleetpy_native_vehicle_log_rows": int(len(self.native_output)),
            "fleetpy_native_completed_requests": int(native_completed),
            "native_output_reconciles": bool(native_reconciled),
        }
        if self.av_availability_violations or self.position_transition_failures:
            raise FleetPyCompatibilityError("spike lifecycle invariant failed")
        if self.request_timing_failures or not native_reconciled:
            raise FleetPyCompatibilityError(
                "FleetPy/adapter output reconciliation failed"
            )
        return SpikeRunResult(request_log, vehicle_log, assignment_log, summary)
