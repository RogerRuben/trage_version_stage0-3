"""Minimal FleetControlBase hook for the native FleetPy rolling shell."""

from __future__ import annotations

from math import isfinite
from typing import Any

import numpy as np
import pandas as pd

from .mixed_fleet_adapter import VehicleRuntime
from .replay_service_time_adapter import _haversine_m
from .test31_demand_adapter import SpikeRequest
from .upstream import FleetPyBindings, FleetPyCompatibilityError
from .valhalla_time_adapter import PickupEstimate, ValhallaPickupTimeAdapter


class _NativeFleetControlCore:
    """Deterministic dispatch stub driven by FleetPy-native demand and vehicle state."""

    def __init__(
        self,
        bindings: FleetPyBindings,
        vehicles: list[VehicleRuntime],
        requests: list[SpikeRequest],
        demand: Any,
        network: Any,
        eta_adapter: ValhallaPickupTimeAdapter,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> None:
        self.bindings = bindings
        self.op_id = 0
        self.sim_vehicles = [runtime.native_vehicle for runtime in vehicles]
        self.runtime_by_vid = {
            runtime.fixture.native_id: runtime for runtime in vehicles
        }
        self.request_by_rid = {request.native_id: request for request in requests}
        self.demand = demand
        self.routing_engine = network
        self.eta_adapter = eta_adapter
        self.start = start
        self.end = end
        self.sim_time = 0
        self.rq_dict: dict[int, Any] = {}
        self.offers: dict[int, Any] = {}
        self.rid_to_assigned_vid: dict[int, int] = {}
        self.assignment_rows: list[dict[str, Any]] = []
        self.activation_rows: list[dict[str, Any]] = []
        self.vehicle_rows: list[dict[str, Any]] = []
        self.status_update_rows: list[dict[str, Any]] = []
        self.hv_session_end_exclusions: set[tuple[int, int, int]] = set()
        self.hv_realized_overrun_seconds: list[float] = []
        self.candidate_arc_evaluations = 0
        self.routing_failures = 0
        self.av_availability_violations = 0
        self.position_reconciliation_failures = 0
        self.request_state_reconciliation_failures = 0
        self.vehicle_state_reconciliation_failures = 0
        self.completed_rids: set[int] = set()
        self.cancelled_rids: set[int] = set()

    def _timestamp(self, sim_time: float) -> pd.Timestamp:
        return self.start + pd.Timedelta(seconds=float(sim_time))

    def _fixture_seconds(self, timestamp: pd.Timestamp) -> float:
        return float((timestamp - self.start).total_seconds())

    def _available(self, runtime: VehicleRuntime, sim_time: int) -> bool:
        timestamp = self._timestamp(sim_time)
        fixture = runtime.fixture
        native = runtime.native_vehicle
        native_free = (
            native.status == self.bindings.states.IDLE and not native.assigned_route
        )
        inside_window = (
            timestamp >= fixture.availability_start_time
            and timestamp < fixture.availability_end_time
        )
        return native_free and inside_window

    def user_request(self, rq: Any, simulation_time: int) -> None:
        rid = int(rq.get_rid_struct())
        request = self.request_by_rid[rid]
        self.rq_dict[rid] = rq
        self.activation_rows.append(
            {
                "native_request_id": rid,
                "order_id": request.order_id,
                "historical_request_time": request.request_time,
                "native_activation_time": self._timestamp(simulation_time),
                "activation_lag_s": float(simulation_time - request.sim_time_s),
            }
        )
        self.offers[rid] = self.bindings.traveller_offer(
            rid,
            self.op_id,
            0.0,
            request.predicted_service_time_s,
            0,
        )

    def get_current_offer(self, rid: int) -> Any:
        return self.offers.get(int(rid))

    def user_confirms_booking(self, rid: int, simulation_time: int) -> None:
        del simulation_time
        rid = int(rid)
        if rid in self.rq_dict:
            self.rq_dict[rid].chosen_operator_id = self.op_id

    def user_cancels_request(self, rid: int, simulation_time: int) -> None:
        del simulation_time
        self.cancelled_rids.add(int(rid))

    def _candidate_estimate(
        self,
        runtime: VehicleRuntime,
        request: SpikeRequest,
        simulation_time: int,
    ) -> PickupEstimate | None:
        if not self._available(runtime, simulation_time):
            return None
        if runtime.fixture.vehicle_type == "AV" and not request.av_smoke_eligible:
            return None
        self.candidate_arc_evaluations += 1
        lon, lat = self.routing_engine.return_position_coordinates(
            runtime.native_vehicle.pos
        )
        try:
            estimate = self.eta_adapter.estimate(
                lon,
                lat,
                request.pickup_lon_wgs84,
                request.pickup_lat_wgs84,
                self._timestamp(simulation_time),
            )
        except Exception:
            self.routing_failures += 1
            return None
        if runtime.fixture.vehicle_type == "HV":
            predicted_end = simulation_time + (
                estimate.corrected_pickup_eta_s + request.predicted_service_time_s
            )
            window_end = self._fixture_seconds(runtime.fixture.availability_end_time)
            if predicted_end > window_end:
                self.hv_session_end_exclusions.add(
                    (int(simulation_time), runtime.fixture.native_id, request.native_id)
                )
                return None
        return estimate

    def _assign(
        self,
        runtime: VehicleRuntime,
        request: SpikeRequest,
        estimate: PickupEstimate,
        simulation_time: int,
    ) -> None:
        native = runtime.native_vehicle
        if native.status != self.bindings.states.IDLE or native.assigned_route:
            raise FleetPyCompatibilityError("native assignment requires idle vehicle")
        sim_vid_id = (self.op_id, runtime.fixture.native_id)
        self.routing_engine.register_vehicle_leg(
            sim_vid_id,
            native.pos,
            request.pickup_position,
            estimate.corrected_pickup_eta_s,
            estimate.route_distance_m,
        )
        service_distance = _haversine_m(
            request.pickup_lon_wgs84,
            request.pickup_lat_wgs84,
            request.dropoff_lon_wgs84,
            request.dropoff_lat_wgs84,
        )
        self.routing_engine.register_vehicle_leg(
            sim_vid_id,
            request.pickup_position,
            request.dropoff_position,
            request.realized_service_time_s,
            service_distance,
        )
        leg = self.bindings.vehicle_route_leg
        states = self.bindings.states
        route_legs = [
            leg(states.REPOSITION, request.pickup_position, {}),
            leg(
                states.BOARDING,
                request.pickup_position,
                {1: [request.native_request]},
                duration=0.0,
                locked=True,
            ),
            leg(states.ROUTE, request.dropoff_position, {}),
            leg(
                states.BOARDING,
                request.dropoff_position,
                {-1: [request.native_request]},
                duration=0.0,
                locked=True,
            ),
        ]
        native.assign_vehicle_plan(route_legs, simulation_time)
        self.rid_to_assigned_vid[request.native_id] = runtime.fixture.native_id
        runtime.active_order_id = request.order_id
        runtime.state = "NATIVE_ASSIGNED"
        if runtime.fixture.vehicle_type == "AV" and not (
            runtime.fixture.availability_start_time == self.start
            and runtime.fixture.availability_end_time == self.end
            and not runtime.fixture.av_source_session_end_inherited
        ):
            self.av_availability_violations += 1
        self.assignment_rows.append(
            {
                "assignment_time": self._timestamp(simulation_time),
                "simulation_time_s": int(simulation_time),
                "native_request_id": request.native_id,
                "order_id": request.order_id,
                "vehicle_id": runtime.fixture.vehicle_id,
                "native_vehicle_id": runtime.fixture.native_id,
                "vehicle_type": runtime.fixture.vehicle_type,
                "pickup_eta_s": estimate.corrected_pickup_eta_s,
                "valhalla_time_s": estimate.valhalla_time_s,
                "pickup_route_distance_m": estimate.route_distance_m,
                "beta": estimate.beta,
                "time_bin_index": estimate.time_bin_index,
                "predicted_service_time_s": request.predicted_service_time_s,
                "realized_service_time_s": request.realized_service_time_s,
                "hv_session_end_time": (
                    runtime.fixture.availability_end_time
                    if runtime.fixture.vehicle_type == "HV"
                    else pd.NaT
                ),
                "pickup_time": pd.NaT,
                "service_end_time": pd.NaT,
                "completed": False,
                "dispatch_policy": "MIN_CORRECTED_PICKUP_ETA_STUB",
            }
        )

    def time_trigger(self, simulation_time: int) -> None:
        self.sim_time = int(simulation_time)
        if self._timestamp(simulation_time) >= self.end:
            return
        waiting = sorted(
            self.demand.waiting_rq.items(),
            key=lambda item: (
                self.request_by_rid[int(item[0])].request_time,
                int(item[0]),
            ),
        )
        for rid, _native_request in waiting:
            rid = int(rid)
            if rid in self.rid_to_assigned_vid:
                continue
            request = self.request_by_rid[rid]
            candidates: list[tuple[float, str, VehicleRuntime, PickupEstimate]] = []
            for runtime in self.runtime_by_vid.values():
                estimate = self._candidate_estimate(runtime, request, simulation_time)
                if estimate is not None:
                    candidates.append(
                        (
                            estimate.corrected_pickup_eta_s,
                            runtime.fixture.vehicle_id,
                            runtime,
                            estimate,
                        )
                    )
            if candidates:
                _, _, runtime, estimate = min(
                    candidates, key=lambda item: (item[0], item[1])
                )
                self._assign(runtime, request, estimate, simulation_time)

    def receive_status_update(
        self,
        vid: int,
        simulation_time: int,
        list_finished_vrl: list[Any],
        force_update: bool = True,
    ) -> None:
        del force_update
        for route_leg in list_finished_vrl:
            if not hasattr(route_leg, "status"):
                continue
            self.status_update_rows.append(
                {
                    "simulation_time_s": int(simulation_time),
                    "native_vehicle_id": int(vid),
                    "native_status": route_leg.status.display_name,
                    "destination_position": str(route_leg.destination_pos),
                    "boarding_request_ids": [
                        int(request.get_rid_struct())
                        for request in route_leg.rq_dict.get(1, [])
                    ],
                    "alighting_request_ids": [
                        int(request.get_rid_struct())
                        for request in route_leg.rq_dict.get(-1, [])
                    ],
                }
            )

    def acknowledge_boarding(self, rid: int, vid: int, simulation_time: float) -> None:
        rid = int(rid)
        for row in reversed(self.assignment_rows):
            if row["native_request_id"] == rid:
                row["pickup_time"] = self._timestamp(simulation_time)
                break
        expected = self.rid_to_assigned_vid.get(rid)
        if expected != int(vid):
            self.request_state_reconciliation_failures += 1

    def acknowledge_alighting(self, rid: int, vid: int, simulation_time: float) -> None:
        rid = int(rid)
        request = self.request_by_rid[rid]
        runtime = self.runtime_by_vid[int(vid)]
        lon, lat = self.routing_engine.return_position_coordinates(
            runtime.native_vehicle.pos
        )
        if not (
            np.isclose(lon, request.dropoff_lon_wgs84)
            and np.isclose(lat, request.dropoff_lat_wgs84)
        ):
            self.position_reconciliation_failures += 1
        expected = self.rid_to_assigned_vid.get(rid)
        if expected != int(vid):
            self.request_state_reconciliation_failures += 1
        for row in reversed(self.assignment_rows):
            if row["native_request_id"] == rid:
                row["service_end_time"] = self._timestamp(simulation_time)
                row["completed"] = True
                row["completion_lon_wgs84"] = lon
                row["completion_lat_wgs84"] = lat
                break
        fixture_end_s = self._fixture_seconds(runtime.fixture.availability_end_time)
        if runtime.fixture.vehicle_type == "HV" and simulation_time > fixture_end_s:
            self.hv_realized_overrun_seconds.append(
                float(simulation_time - fixture_end_s)
            )
        self.completed_rids.add(rid)
        runtime.active_order_id = None
        runtime.state = (
            "OFFLINE_AFTER_COMPLETION"
            if runtime.fixture.vehicle_type == "HV" and simulation_time >= fixture_end_s
            else "NATIVE_AVAILABLE"
        )

    def record_tick(self, simulation_time: int) -> None:
        timestamp = self._timestamp(simulation_time)
        for vid, runtime in sorted(self.runtime_by_vid.items()):
            native = runtime.native_vehicle
            lon, lat = self.routing_engine.return_position_coordinates(native.pos)
            inside_window = (
                timestamp >= runtime.fixture.availability_start_time
                and timestamp < runtime.fixture.availability_end_time
            )
            native_free = (
                native.status == self.bindings.states.IDLE and not native.assigned_route
            )
            if runtime.fixture.vehicle_type == "HV" and not inside_window:
                availability_state = "OUTSIDE_HV_SESSION"
            elif native_free:
                availability_state = "AVAILABLE"
            else:
                availability_state = "BUSY"
            if runtime.active_order_id is not None and native_free:
                self.vehicle_state_reconciliation_failures += 1
            self.vehicle_rows.append(
                {
                    "native_tick_time": timestamp,
                    "simulation_time_s": int(simulation_time),
                    "vehicle_id": runtime.fixture.vehicle_id,
                    "native_vehicle_id": vid,
                    "vehicle_type": runtime.fixture.vehicle_type,
                    "native_status": native.status.display_name,
                    "availability_state": availability_state,
                    "current_lon_wgs84": lon,
                    "current_lat_wgs84": lat,
                    "active_order_id": runtime.active_order_id,
                    "availability_start_time": runtime.fixture.availability_start_time,
                    "availability_end_time": runtime.fixture.availability_end_time,
                    "availability_policy": (
                        "FULL_SIMULATION_HORIZON"
                        if runtime.fixture.vehicle_type == "AV"
                        else "RECONSTRUCTED_S0_SESSION_WINDOW"
                    ),
                }
            )

    def reconcile(self) -> None:
        assignment_by_rid = {
            int(row["native_request_id"]): row for row in self.assignment_rows
        }
        for rid, row in assignment_by_rid.items():
            request = self.request_by_rid[rid]
            native = request.native_request
            if row["completed"]:
                if (
                    native.do_time is None
                    or native.service_vid != row["native_vehicle_id"]
                ):
                    self.request_state_reconciliation_failures += 1
                if native.do_pos != request.dropoff_position:
                    self.position_reconciliation_failures += 1
        for runtime in self.runtime_by_vid.values():
            if runtime.fixture.vehicle_type == "AV" and (
                runtime.fixture.availability_start_time != self.start
                or runtime.fixture.availability_end_time != self.end
                or runtime.fixture.av_source_session_end_inherited
            ):
                self.av_availability_violations += 1
        values = [
            row["pickup_eta_s"]
            for row in self.assignment_rows
            if isfinite(row["pickup_eta_s"])
        ]
        if len(values) != len(self.assignment_rows):
            raise FleetPyCompatibilityError("non-finite assigned pickup ETA")

    def _call_time_trigger_request_batch(self, simulation_time: int) -> None:
        self.time_trigger(simulation_time)

    def _create_user_offer(
        self,
        prq: Any,
        simulation_time: int,
        assigned_vehicle_plan: Any = None,
        offer_dict_without_plan: dict | None = None,
    ) -> Any:
        del simulation_time, assigned_vehicle_plan, offer_dict_without_plan
        return self.offers[int(prq.get_rid_struct())]

    def assign_vehicle_plan(
        self,
        veh_obj: Any,
        vehicle_plan: Any,
        sim_time: int,
        force_assign: bool = False,
        assigned_charging_task: Any = None,
        add_arg: Any = None,
    ) -> None:
        del assigned_charging_task, add_arg
        veh_obj.assign_vehicle_plan(
            vehicle_plan, sim_time, force_ignore_lock=force_assign
        )

    def change_prq_time_constraints(
        self, sim_time: int, rid: int, new_lpt: int, new_ept: int | None = None
    ) -> None:
        del sim_time, rid, new_lpt, new_ept

    def lock_current_vehicle_plan(self, vid: int) -> None:
        del vid

    def _lock_vid_rid_pickup(self, sim_time: int, vid: int, rid: int) -> None:
        del sim_time, vid, rid

    def _prq_from_reservation_to_immediate(
        self, rid: int, simulation_time: int
    ) -> None:
        del rid, simulation_time

    def inform_network_travel_time_update(self, simulation_time: int) -> None:
        del simulation_time


def create_native_fleet_control(
    bindings: FleetPyBindings,
    vehicles: list[VehicleRuntime],
    requests: list[SpikeRequest],
    demand: Any,
    network: Any,
    eta_adapter: ValhallaPickupTimeAdapter,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Any:
    """Create a concrete pinned FleetControlBase subclass."""
    fleet_control_class = type(
        "Stage4NativeFleetControl",
        (_NativeFleetControlCore, bindings.fleet_control_base),
        {},
    )
    return fleet_control_class(
        bindings, vehicles, requests, demand, network, eta_adapter, start, end
    )
