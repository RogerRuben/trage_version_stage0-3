from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from stage4.fleetpy_adapter.mixed_fleet_adapter import (
    VehicleFixture,
    VehicleRuntime,
    load_mixed_fleet_fixture,
)
from stage4.fleetpy_adapter.native_demand import create_native_demand
from stage4.fleetpy_adapter.native_fleet_control import (
    _NativeFleetControlCore,
    create_native_fleet_control,
)
from stage4.fleetpy_adapter.native_network import create_native_network
from stage4.fleetpy_adapter.native_simulation import create_native_simulation
from stage4.fleetpy_adapter.test31_demand_adapter import (
    SpikeRequest,
    attach_fleetpy_requests,
)
from stage4.fleetpy_adapter.upstream import (
    CoordinateRegistry,
    FLEETPY_COMMIT,
    load_fleetpy_bindings,
)
from stage4.fleetpy_adapter.valhalla_time_adapter import (
    PickupEstimate,
    ValhallaPickupTimeAdapter,
)

ROOT = Path(__file__).resolve().parents[2]
START = pd.Timestamp("2016-10-31 08:00:00", tz="Asia/Shanghai")
END = pd.Timestamp("2016-10-31 09:00:00", tz="Asia/Shanghai")


def _bindings():
    root = os.environ.get("FLEETPY_ROOT")
    if not root:
        raise RuntimeError("FLEETPY_ROOT is required for pinned FleetPy tests")
    return load_fleetpy_bindings(root)


def _request(**updates) -> SpikeRequest:
    values = {
        "native_id": 0,
        "order_id": "test-order",
        "request_time": START + pd.Timedelta(seconds=17),
        "sim_time_s": 17,
        "pickup_lon_wgs84": 108.90,
        "pickup_lat_wgs84": 34.20,
        "dropoff_lon_wgs84": 108.91,
        "dropoff_lat_wgs84": 34.21,
        "realized_service_time_s": 321.0,
        "predicted_service_time_s": 280.0,
        "profile_id": "M",
        "hard_state": "FEASIBLE",
        "evidence_complete": True,
        "rho_static": 0.7,
        "rho_dynamic": 0.8,
        "rho_speed": 0.6,
    }
    values.update(updates)
    return SpikeRequest(**values)


def _fixture(vehicle_type="HV", start=START, end=END) -> VehicleFixture:
    return VehicleFixture(
        vehicle_id=f"{vehicle_type}_000",
        native_id=0,
        vehicle_type=vehicle_type,
        initial_lon_wgs84=108.89,
        initial_lat_wgs84=34.19,
        availability_start_time=start,
        availability_end_time=end,
        source_session_id="fixture",
        av_source_session_end_inherited=False,
    )


def test_native_shell_config_is_engineering_only():
    config = json.loads((ROOT / "stage4/config/fleetpy_native_shell.json").read_text())
    assert config["simulation_time_step_s"] == 60
    assert config["engineering_request_count"] == 200
    assert config["engineering_hv_count"] == 40
    assert config["engineering_av_count"] == 10
    assert config["fleetpy_commit"] == FLEETPY_COMMIT
    text = json.dumps(config).lower()
    assert "gurobi" not in text and "gamma" not in text


def test_entry_point_instantiates_real_fleetpy_simulation_class():
    bindings = _bindings()
    vehicle = SimpleNamespace(vid=0)
    operator = SimpleNamespace(sim_time=0, record_tick=lambda value: None)
    demand = SimpleNamespace(future_requests={})
    network = SimpleNamespace()
    simulation = create_native_simulation(
        bindings,
        simulation_end_s=3600,
        time_step_s=60,
        demand=demand,
        vehicles=[vehicle],
        fleet_control=operator,
        network=network,
        native_output=[],
    )
    assert isinstance(simulation, bindings.immediate_simulation)
    assert simulation.run.__func__.__qualname__ == "FleetSimulationBase.run"
    assert simulation.step.__func__.__qualname__ == "ImmediateDecisionsSimulation.step"


def test_formal_path_does_not_use_s1_event_loop():
    from stage4.fleetpy_adapter import native_shell_runner

    source = inspect.getsource(native_shell_runner)
    assert "MinCorrectedPickupEtaDispatchStub" not in source
    assert "dispatch_stub" not in source
    assert "heapq" not in source


def test_network_bridge_is_official_subclass_and_moves_natively():
    bindings = _bindings()
    registry = CoordinateRegistry()
    origin = registry.position_for(108.90, 34.20)
    destination = registry.position_for(108.91, 34.21)
    registry.set_leg_metrics(origin, destination, 100.0, 1000.0)
    network = create_native_network(bindings, registry)
    assert isinstance(network, bindings.network_base)
    partial, distance, arrival, passed, _ = network.move_along_route(
        [origin[0], destination[0]], origin, 40.0, new_sim_time=0
    )
    assert partial[:2] == (origin[0], destination[0])
    assert np.isclose(partial[2], 0.4)
    assert np.isclose(distance, 400.0)
    assert arrival == -1 and passed == []
    final, distance, arrival, passed, _ = network.move_along_route(
        [destination[0]], partial, 60.0, new_sim_time=40
    )
    assert final == destination
    assert np.isclose(distance, 600.0)
    assert np.isclose(arrival, 100.0)
    assert passed == [destination[0]]


def test_requests_enter_through_fleetpy_demand_lifecycle(tmp_path):
    bindings = _bindings()
    registry = CoordinateRegistry()
    request = _request()
    attach_fleetpy_requests([request], bindings, registry)
    network = create_native_network(bindings, registry)
    demand = create_native_demand(bindings, [request], registry, network, tmp_path)
    assert request.native_id in demand.future_requests[17]
    activated = demand.get_new_travelers(60, since=0)
    assert [rid for rid, _ in activated] == [request.native_id]
    assert request.native_id in demand.rq_db
    assert 17 not in demand.future_requests


def test_fleet_control_is_official_subclass():
    bindings = _bindings()
    fleet_control = create_native_fleet_control(
        bindings,
        [],
        [],
        SimpleNamespace(waiting_rq={}),
        SimpleNamespace(),
        SimpleNamespace(),
        START,
        END,
    )
    assert isinstance(fleet_control, bindings.fleet_control_base)


def test_av_fixture_is_full_horizon_and_never_inherits_hv_end():
    fixtures = load_mixed_fleet_fixture(
        ROOT,
        start=START,
        end=END,
        hv_count=2,
        av_count=2,
        seed=20260824,
    )
    av = [fixture for fixture in fixtures if fixture.vehicle_type == "AV"]
    assert len(av) == 2
    assert all(item.availability_start_time == START for item in av)
    assert all(item.availability_end_time == END for item in av)
    assert not any(item.av_source_session_end_inherited for item in av)


def test_hv_is_unavailable_outside_reconstructed_window():
    idle = object()
    fixture = _fixture(
        start=START + pd.Timedelta(seconds=120),
        end=START + pd.Timedelta(seconds=600),
    )
    native = SimpleNamespace(status=idle, assigned_route=[])
    runtime = VehicleRuntime(fixture, native, "", 108.89, 34.19)
    control = object.__new__(_NativeFleetControlCore)
    control.start = START
    control.bindings = SimpleNamespace(states=SimpleNamespace(IDLE=idle))
    assert not control._available(runtime, 119)
    assert control._available(runtime, 120)
    assert not control._available(runtime, 600)


def test_hv_predicted_session_end_admission_is_enforced():
    idle = object()
    fixture = _fixture(end=START + pd.Timedelta(seconds=500))
    native = SimpleNamespace(status=idle, assigned_route=[], pos=(0, None, None))
    runtime = VehicleRuntime(fixture, native, "", 108.89, 34.19)
    request = _request(predicted_service_time_s=450.0)
    control = object.__new__(_NativeFleetControlCore)
    control.start = START
    control.end = END
    control.bindings = SimpleNamespace(states=SimpleNamespace(IDLE=idle))
    control.routing_engine = SimpleNamespace(
        return_position_coordinates=lambda position: (108.89, 34.19)
    )
    control.eta_adapter = SimpleNamespace(
        estimate=lambda *args: PickupEstimate(100.0, 100.0, 1000.0, 1.0, 32, False)
    )
    control.candidate_arc_evaluations = 0
    control.routing_failures = 0
    control.hv_session_end_exclusions = set()
    assert control._candidate_estimate(runtime, request, 0) is None
    assert len(control.hv_session_end_exclusions) == 1


class _Actor:
    def __init__(self):
        self.requests = []

    def route(self, request):
        self.requests.append(request)
        return {
            "trip": {
                "status": 0,
                "legs": [{}],
                "summary": {"time": 100.0, "length": 1.5},
            }
        }


def test_pickup_routing_uses_wgs84_and_frozen_15min_beta():
    actor = _Actor()
    adapter = ValhallaPickupTimeAdapter(ROOT, actor=actor)
    timestamp = START + pd.Timedelta(minutes=16)
    estimate = adapter.estimate(108.89, 34.19, 108.90, 34.20, timestamp)
    index, beta = adapter.beta_for(timestamp)
    assert actor.requests[0]["costing"] == "auto"
    assert actor.requests[0]["locations"][0]["lon"] == 108.89
    assert actor.requests[0]["locations"][0]["lat"] == 34.19
    assert estimate.time_bin_index == index
    assert np.isclose(estimate.corrected_pickup_eta_s, 100.0 * beta)


def test_native_assignment_keeps_predicted_and_realized_times_separate():
    idle = "IDLE"
    states = SimpleNamespace(
        IDLE=idle, REPOSITION="REPOSITION", BOARDING="BOARDING", ROUTE="ROUTE"
    )

    def leg(status, destination, rq_dict, **kwargs):
        return SimpleNamespace(
            status=status, destination_pos=destination, rq_dict=rq_dict, **kwargs
        )

    class Native:
        status = idle
        pos = (0, None, None)
        assigned_route = []

        def assign_vehicle_plan(self, route, simulation_time):
            self.assigned_route = route
            self.assignment_time = simulation_time

    class Network:
        def __init__(self):
            self.registrations = []

        def register_vehicle_leg(self, *args):
            self.registrations.append(args)

    fixture = _fixture()
    native = Native()
    runtime = VehicleRuntime(fixture, native, "", 108.89, 34.19)
    request = _request()
    request.pickup_position = (1, None, None)
    request.dropoff_position = (2, None, None)
    request.native_request = SimpleNamespace(get_rid=lambda: 0)
    network = Network()
    control = object.__new__(_NativeFleetControlCore)
    control.bindings = SimpleNamespace(states=states, vehicle_route_leg=leg)
    control.op_id = 0
    control.routing_engine = network
    control.rid_to_assigned_vid = {}
    control.assignment_rows = []
    control.start = START
    control.end = END
    control.av_availability_violations = 0
    estimate = PickupEstimate(100.0, 120.0, 1500.0, 1.2, 32, False)
    control._assign(runtime, request, estimate, 60)
    assert network.registrations[0][-2] == estimate.corrected_pickup_eta_s
    assert network.registrations[1][-2] == request.realized_service_time_s
    assert control.assignment_rows[0]["predicted_service_time_s"] == 280.0
    assert control.assignment_rows[0]["realized_service_time_s"] == 321.0


def test_hv_realized_overrun_finishes_then_goes_offline():
    fixture = _fixture(end=START + pd.Timedelta(seconds=100))
    native = SimpleNamespace(pos=(2, None, None))
    runtime = VehicleRuntime(
        fixture, native, "NATIVE_ASSIGNED", 108.89, 34.19, "test-order"
    )
    request = _request()
    request.dropoff_position = (2, None, None)
    request.native_request = SimpleNamespace(
        do_time=125.0, service_vid=0, do_pos=request.dropoff_position
    )
    control = object.__new__(_NativeFleetControlCore)
    control.start = START
    control.request_by_rid = {0: request}
    control.runtime_by_vid = {0: runtime}
    control.routing_engine = SimpleNamespace(
        return_position_coordinates=lambda position: (
            request.dropoff_lon_wgs84,
            request.dropoff_lat_wgs84,
        )
    )
    control.rid_to_assigned_vid = {0: 0}
    control.assignment_rows = [
        {
            "native_request_id": 0,
            "native_vehicle_id": 0,
            "completed": False,
            "pickup_eta_s": 10.0,
        }
    ]
    control.position_reconciliation_failures = 0
    control.request_state_reconciliation_failures = 0
    control.hv_realized_overrun_seconds = []
    control.completed_rids = set()
    control.av_availability_violations = 0
    control.acknowledge_alighting(0, 0, 125.0)
    assert control.assignment_rows[0]["completed"]
    assert control.hv_realized_overrun_seconds == [25.0]
    assert runtime.state == "OFFLINE_AFTER_COMPLETION"
    assert runtime.active_order_id is None
    control.reconcile()
    assert control.position_reconciliation_failures == 0
    assert control.request_state_reconciliation_failures == 0
