from dataclasses import replace
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from stage4.dispatch.flexibility_model import (
    CurrentPickup, FuturePickup, ModelLimits, Problem, Request, Scenario, Vehicle, solve_dispatch,
)
from stage4.dispatch.flexibility_native import (
    NativeFlexibilityAdapter, position_key, predicted_vehicle_states,
)
from stage4.dispatch.flexibility_v3 import solve_dispatch_v3
from stage4.dispatch.rolling_or_control import _RollingORFleetControlCore
from stage4.dispatch.solver import AssignmentArc
from stage4.fleetpy_adapter.mixed_fleet_adapter import VehicleFixture, VehicleRuntime
from stage4.fleetpy_adapter.test31_demand_adapter import SpikeRequest
from stage4.fleetpy_adapter.upstream import FleetPyCompatibilityError
from stage4.fleetpy_adapter.valhalla_time_adapter import PickupEstimate


START = pd.Timestamp("2016-10-31", tz="Asia/Shanghai")
POLICY = "STOP_ADMISSION_FINISH_COMMITTED"


class EtaAdapter:
    routing_time_s = 0.0

    def __init__(self):
        self.budgets = []
        self.last_certified_prunes = []

    def estimate(self, *args):
        return PickupEstimate(20, 20, 100, 1, 0, False)

    def estimate_epoch(self, batches, eta_budgets=None):
        self.budgets = eta_budgets
        self.last_certified_prunes = [{} for _ in batches]
        return [{v.native_vehicle_id: self.estimate() for v in batch[0]} for batch in batches]


def controller(kind="HV", policy=POLICY):
    states = NS(IDLE=0, REPOSITION=1, BOARDING=2, ROUTE=3)
    bindings = NS(states=states, traveller_offer=lambda *args: None,
                  vehicle_route_leg=lambda status, destination, requests, **kw:
                  NS(status=status, destination_pos=destination, rq_dict=requests, **kw))
    fixture = VehicleFixture("slot-0", 0, kind, 108.93, 34.24,
        START + pd.Timedelta(seconds=60), START + pd.Timedelta(seconds=100),
        "slot-0", kind == "AV", availability_policy=policy)
    native = NS(status=states.IDLE, pos=(108.93, 34.24), assigned_route=[])

    def assign(route, now):
        native.assigned_route = list(route)
        native.status = states.REPOSITION

    native.assign_vehicle_plan = assign
    runtime = VehicleRuntime(fixture, native, "AVAILABLE", 108.93, 34.24)
    source = SpikeRequest(0, "request-0", START + pd.Timedelta(seconds=60), 60,
        108.93, 34.24, 108.94, 34.25, 200, 120, "M", "FEASIBLE", True, .5, .5, .5)
    source.pickup_position, source.dropoff_position = (108.93, 34.24), (108.94, 34.25)
    source.native_request = NS(get_rid_struct=lambda: 0, do_time=None, service_vid=None, do_pos=None)
    demand = NS(waiting_rq={0: source.native_request}, rq_db={0: source.native_request})
    network = NS(return_position_coordinates=lambda pos: pos, register_vehicle_leg=lambda *args: None)
    eta = EtaAdapter()
    cfg = dict(dispatch_interval_s=30, max_pickup_wait_s=300, matching_end_s=1000,
        benchmark_runtime_guard_s=600, candidate_top_k=20, search_radius_initial_m=2000,
        search_radius_step_m=1000, search_radius_cap_m=8000, profile_id="M",
        controlled_replay_v2=True, pre_route_session_certificate=True,
        epoch_routing_queue=True, certified_eta_pruning=True, passenger_acceptance_rate=1)
    c = _RollingORFleetControlCore(bindings, [runtime], [source], demand, network, eta,
                                 START, START + pd.Timedelta(seconds=1000), cfg)
    return c, runtime, source, eta


@pytest.mark.parametrize("kind", ["HV", "AV"])
def test_native_half_open_window_preserves_customer_commitment(kind):
    c, runtime, source, eta = controller(kind)
    assert not c._available(runtime, 59)
    assert c._available(runtime, 60) and c._available(runtime, 99)
    assert not c._available(runtime, 100)

    runtime.native_vehicle.status = c.bindings.states.REPOSITION
    runtime.native_vehicle.assigned_route = ["empty move"]
    c.repositioning_manager = NS(is_interruptible=lambda *args: True)
    assert c._available(runtime, 99)
    assert predicted_vehicle_states(c, 99, 600, {})[0].ready_position == position_key(108.93, 34.24)
    runtime.active_order_id = "customer"
    assert not c._available(runtime, 99)
    runtime.active_order_id = None
    runtime.native_vehicle.status = c.bindings.states.IDLE
    runtime.native_vehicle.assigned_route = []

    assert c._candidate_estimate(runtime, source, 99) == eta.estimate()
    source.sim_time_s = -300
    assert c._candidate_estimate(runtime, source, 99) is None
    source.sim_time_s = 60
    c._assign(runtime, source, eta.estimate(), 99)
    committed_route = list(runtime.native_vehicle.assigned_route)
    assert not c._available(runtime, 100)
    assert runtime.native_vehicle.assigned_route == committed_route and runtime.active_order_id == source.order_id
    c.acknowledge_boarding(0, 0, 119)
    runtime.native_vehicle.pos = source.dropoff_position
    runtime.native_vehicle.status = c.bindings.states.IDLE
    runtime.native_vehicle.assigned_route = []
    source.native_request.do_time, source.native_request.service_vid = 319, 0
    source.native_request.do_pos = source.dropoff_position
    c.acknowledge_alighting(0, 0, 319)
    assert runtime.state == "OFFLINE_AFTER_COMPLETION"
    assert c.assignment_rows[0]["pickup_time"] == START + pd.Timedelta(seconds=119)
    assert c.assignment_rows[0]["service_end_time"] == START + pd.Timedelta(seconds=319)
    assert c.assignment_rows[0]["completed"] and not c.cancelled_rids
    assert c.assignment_rows[0]["availability_end_time"] == runtime.fixture.availability_end_time
    c.repositioning_manager = None
    c.reconcile()
    assert c.av_availability_violations == 0
    with pytest.raises(FleetPyCompatibilityError, match="outside admission window"):
        c._assign(runtime, source, eta.estimate(), 100)

    legacy, legacy_runtime, legacy_request, _ = controller("HV", "EMPIRICAL_SESSION")
    assert legacy._candidate_estimate(legacy_runtime, legacy_request, 99) is None


@pytest.mark.parametrize("solver", [solve_dispatch, solve_dispatch_v3])
def test_model_current_commitment_and_future_admission_have_distinct_cutoffs(solver):
    limits = ModelLimits(horizon_s=600, rolling_step_s=30, recourse_mode="FLOW_RELAXED")
    vehicle = Vehicle(0, "HV", "origin", 60, 100, strict_completion_deadline=False)
    request = Request(0, 60, 360, 120, "dropoff")
    future = Request(1, 95, 395, 120, "future-dropoff")
    scene = Scenario("train", 1, -86400, (future,), (
        FuturePickup(0, 0, 1, 0, "dropoff"),))
    problem = Problem(60, (vehicle,), (request,), (CurrentPickup(0, 0, 20),), (scene,), limits)
    selected = solver(problem, "SERVICE_PRESERVING_LOOKAHEAD")
    assert selected.selected_pairs == ((0, 0),)
    assert selected.expected_next_service_count == 0  # Current completion at 200 is beyond b=100.
    at_end = replace(problem, waiting_requests=(replace(request, predicted_service_time_s=20),))
    selected_at_end = solver(at_end, "SERVICE_PRESERVING_LOOKAHEAD")
    assert selected_at_end.selected_pairs == ((0, 0),)
    assert selected_at_end.expected_next_service_count == 0  # Ready exactly at b cannot admit a next service.
    legacy = replace(problem, vehicles=(replace(vehicle, strict_completion_deadline=True),))
    assert solver(legacy, "SERVICE_PRESERVING_LOOKAHEAD").selected_pairs == ()
    boundary = replace(problem, now_s=100, scenarios=())
    assert solver(boundary, "SERVICE_PRESERVING_LOOKAHEAD").selected_pairs == ()

    for release, expected in ((99, 1), (100, 0)):
        next_request = replace(future, release_time_s=release, pickup_deadline_s=release + 300)
        scene = replace(scene, new_requests=(next_request,),
                        pickups=(FuturePickup(0, None, 1, 20, "origin"),))
        next_problem = replace(problem, waiting_requests=(), current_pickups=(), scenarios=(scene,))
        assert solver(next_problem, "SERVICE_PRESERVING_LOOKAHEAD").expected_next_service_count == expected


def test_same_slot_hv_a_label_swap_is_neutral_when_passenger_accepts():
    vehicles=(Vehicle(0,"HV","p0",60,100,False),Vehicle(1,"HV","p1",60,100,False))
    requests=(Request(0,60,360,120,"d0"),Request(1,60,360,120,"d1"))
    pickups=(CurrentPickup(0,0,20),CurrentPickup(0,1,40),CurrentPickup(1,0,50),CurrentPickup(1,1,10))
    problem=Problem(60,vehicles,requests,pickups,limits=ModelLimits(recourse_mode="FLOW_RELAXED"))
    before=solve_dispatch_v3(problem,"SERVICE_PRESERVING_LOOKAHEAD")
    swapped=replace(problem,vehicles=(replace(vehicles[0],profile_id="A"),vehicles[1]))
    after=solve_dispatch_v3(swapped,"SERVICE_PRESERVING_LOOKAHEAD")
    assert before.selected_pairs==after.selected_pairs==((0,0),(1,1))
    assert before.immediate_service_count==after.immediate_service_count==2


@pytest.mark.parametrize("fast_graph", [False, True])
def test_rolling_certificates_and_future_graph_use_admission_not_completion(fast_graph):
    c, runtime, source, eta = controller()
    c.user_request(source.native_request, 60)
    c.request_meta[0].update(research_route_compatible=True, research_base_eligible=True)
    future = (Request(1, 99, 399, 120, "future-dropoff"), Request(2, 100, 400, 120, "future-dropoff"))
    origin = position_key(108.93, 34.24)
    forecast = NS(global_pace=1, pace_by_slot={}, scenarios=lambda *args: [
        (Scenario("train", 1, -86400, future, ()), {1: origin, 2: origin})])
    cfg = dict(forecast_horizon_s=600, max_model_variables=1000, max_model_nonzeros=10000,
        solver_time_limit_s=10, future_search_radius_m=2000, future_top_k_vehicles=20,
        fast_future_graph=fast_graph)
    adapter = NativeFlexibilityAdapter("SERVICE_PRESERVING_LOOKAHEAD", forecast, cfg)
    adapter.stage_timings = {}
    problem = adapter._problem(c, [AssignmentArc(0, 0, 20, False, False)], [0], 60)
    assert not problem.vehicles[0].strict_completion_deadline
    pickups = problem.scenarios[0].pickups
    assert any(p.request_id == 1 and p.after_request_id is None for p in pickups)
    assert not any(p.request_id == 2 or p.after_request_id == 0 for p in pickups)

    c.time_trigger(99)
    assert len(c.assignment_rows) == 1
    assert c.epoch_rows[-1]["zero_eta_session_certified_prunes"] == 0
    assert c.epoch_rows[-1]["hv_window_arc_exclusions"] == 0
    assert eta.budgets[0][0].empirical_session_end_s is None
    assert eta.budgets[0][0].rejection(20) is None
    assert predicted_vehicle_states(c, 99, 600, {
        0: dict(simulation_time_s=99, pickup_eta_s=20, predicted_service_time_s=120, native_request_id=0)
    }) == ()

    legacy, _, legacy_request, _ = controller("HV", "EMPIRICAL_SESSION")
    legacy.user_request(legacy_request.native_request, 60)
    legacy.time_trigger(99)
    assert not legacy.assignment_rows
    assert legacy.epoch_rows[-1]["zero_eta_session_certified_prunes"] == 1
