"""Synthetic native-hook checks; no Actor, full city vehicle construction, or experiment."""

from types import SimpleNamespace as NS

import pandas as pd
import pytest

from stage4.dispatch.fleet_normalization import FleetScenario
from stage4.fleetpy_adapter.mixed_fleet_adapter import VehicleRuntime
from stage4.fleetpy_adapter.native_fleet_control import _NativeFleetControlCore
from stage4.fleetpy_adapter.research_native_bridge import ResearchNativeBridge
from stage4.fleetpy_adapter.research_world import episode_from_frames
from stage4.fleetpy_adapter.upstream import CoordinateRegistry, FleetPyCompatibilityError
from stage4.fleetpy_adapter.valhalla_time_adapter import PickupEstimate

START = pd.Timestamp("2016-10-31T00:00:00+08:00")


class BasicRequest:
    def __init__(self, row, registry, step, config):
        self.rid = int(row.request_id)
        self.rq_time = row.rq_time
        self.do_time = self.service_vid = self.do_pos = None
        for key, value in row.items():
            setattr(self, key, value)

    def get_rid_struct(self):
        return self.rid


def fixture(*, missing_truth=False):
    records = [("carry", 8.), ("inside", 10.), ("later", 15.), ("excluded_future", 20.)]
    base = pd.DataFrame([dict(order_id=oid, request_time=START + pd.Timedelta(seconds=t),
        pickup_lon_wgs84=108.9, pickup_lat_wgs84=34.2, dropoff_lon_wgs84=108.91, dropoff_lat_wgs84=34.21,
        realized_service_time_s=77.) for oid, t in records])
    routes = pd.DataFrame([dict(order_id=oid, common_eligible=True, predicted_route_time_p50_s=5.,
        compatible_C=True, compatible_M=True, compatible_A=True, compatible_HV=True) for oid, t in records])
    table = pd.DataFrame([dict(vehicle_id=f"slot-{vid}", native_id=vid, slot_id=f"session-{vid}", vehicle_type="AV",
        availability_start_time=START, availability_end_time=START + pd.Timedelta(seconds=end),
        initial_lon_wgs84=108.9, initial_lat_wgs84=34.2,
        availability_policy="STOP_ADMISSION_FINISH_COMMITTED") for vid, end in [(0, 12.), (1, 9.)]])
    episode = episode_from_frames(base, routes, FleetScenario(table, [], {}), benchmark_start=START,
        patience_s=10., passenger_acceptance_rate=1., truth_frame=base[["order_id"]] if missing_truth else None)
    states = NS(IDLE=0, REPOSITION=1, BOARDING=2, ROUTE=3)
    bindings = NS(basic_request=BasicRequest, states=states, traveller_offer=lambda *args: None,
        demand=lambda *args, **kwargs: NS(future_requests={}, waiting_rq={}, rq_db={}),
        vehicle_route_leg=lambda status, pos, rq, **kw: NS(status=status, destination_pos=pos, rq_dict=rq, **kw))
    registry = CoordinateRegistry()
    bridge = ResearchNativeBridge(episode, bindings, registry, window_start_s=10,
                                 measurement_end_s=20, admission_end_s=30)
    network_calls = []
    network = NS(return_position_coordinates=registry.return_position_coordinates,
        register_vehicle_leg=lambda *args: network_calls.append(args))
    demand = bridge.create_demand(network, ".")
    runtimes = []
    for plan in bridge.native_fixtures():
        native = NS(status=states.IDLE, assigned_route=[], pos=registry.position_for(108.9, 34.2), vid=plan.native_id)
        def assign(legs, now, native=native):
            native.assigned_route = list(legs)
            native.status = states.REPOSITION
        native.assign_vehicle_plan = assign
        runtimes.append(VehicleRuntime(plan, native, "AVAILABLE", 108.9, 34.2))
    c = _NativeFleetControlCore(bindings, runtimes, [], demand, network, NS(), START, START + pd.Timedelta(seconds=100))
    bridge.install(c)
    return episode, bridge, c, network_calls


def activate(bridge, c, second):
    for native_request in c.demand.future_requests.pop(second, {}).values():
        native_request.set_direct_route_travel_infos(NS(return_travel_costs_1to1=lambda *args: pytest.fail("physical query before commitment")))
        c.user_request(native_request, second)


def test_demand_activation_is_prediction_only_and_retains_carry_in_deadline():
    episode, bridge, c, calls = fixture()
    assert calls == [] and episode.execution_truth._values is None
    assert bridge.input_counts()["carry_in_requests"] == 1
    assert bridge.diagnostics()["deferred_truth_calls"] == 0
    assert bridge.requests == {} and c.request_by_rid == {}
    assert len(bridge.native_fixtures()) == 1  # Ended slot omitted without relabeling surviving slot.
    assert 20 not in c.demand.future_requests
    activate(bridge, c, 10)
    carry = next(r for r in bridge.requests.values() if r.order_id == "carry")
    assert carry.native_request.rq_time == 8 and carry.native_request.latest_decision_time == 18
    assert carry.source_cohort == "CARRY_IN" and carry.native_request.direct_route_travel_time == 5.
    assert not hasattr(carry, "realized_service_time_s") and not hasattr(carry.native_request, "realized_service_time_s")
    assert "later" not in {r.order_id for r in bridge.requests.values()}
    assert calls == [] and episode.execution_truth._values is None


def test_assign_opens_truth_after_commit_and_native_completion_syncs_episode(monkeypatch):
    episode, bridge, c, calls = fixture()
    activate(bridge, c, 10)
    record = next(r for r in bridge.requests.values() if r.order_id == "inside")
    original_lookup = episode.execution_truth.lookup
    def guarded_lookup(oid):
        assert episode.requests.was_committed(oid)
        return original_lookup(oid)
    monkeypatch.setattr(episode.execution_truth, "lookup", guarded_lookup)
    monkeypatch.setattr(episode, "snapshot", lambda *args: pytest.fail("commit must not scan all supply rows"))
    c._assign(c.runtime_by_vid[0], record, PickupEstimate(1., 1., 100., 1., 0, False), 10)
    assert calls[-1][3] == 77. and episode.requests.was_committed("inside")
    assert "realized_service_time_s" not in c.assignment_rows[0]
    assert bridge.environment_assignments().iloc[0]["realized_service_time_s"] == 77.
    assert not hasattr(c.request_by_rid[record.native_id], "realized_service_time_s")
    c.acknowledge_boarding(record.native_id, 0, 11.)
    native = c.runtime_by_vid[0].native_vehicle
    native.pos, native.status, native.assigned_route = record.dropoff_position, c.bindings.states.IDLE, []
    episode.reveal_until(100.)  # Another vehicle's later callback can have advanced the feed clock.
    c.acknowledge_alighting(record.native_id, 0, 88.)
    assert "inside" in episode.requests._completed
    assert c.runtime_by_vid[0].state == "OFFLINE_AFTER_COMPLETION"
    assert episode._locations["slot-0"] == (108.91, 34.21)
    assert bridge.environment_assignments().iloc[0]["service_end_time_s"] == 88.
    assert episode.requests._now_s == 100. and bridge.diagnostics()["deferred_truth_calls"] == 1


def test_missing_truth_aborts_after_commit_instead_of_using_p50():
    episode, bridge, c, calls = fixture(missing_truth=True)
    activate(bridge, c, 10)
    record = next(r for r in bridge.requests.values() if r.order_id == "inside")
    with pytest.raises(FleetPyCompatibilityError, match="P50 fallback prohibited"):
        c._assign(c.runtime_by_vid[0], record, PickupEstimate(1., 1., 100., 1., 0, False), 10)
    assert episode.requests.was_committed("inside") and calls == []


def test_pickup_geometry_reuses_shared_bridge_after_old_assign_and_releases_on_boarding():
    episode, bridge, c, calls = fixture()
    activate(bridge, c, 10)
    record = next(r for r in bridge.requests.values() if r.order_id == "inside")
    events = []
    geometry = NS(register=lambda *args: events.append(("register", bool(c.assignment_rows))),
                  release=lambda vid: events.append(("release", vid)))
    c.repositioning_manager = NS(geometry=geometry, active={})
    c.city_pickup_paths = {(0, record.native_id): [(108.9, 34.2), (108.91, 34.21)]}
    c._assign(c.runtime_by_vid[0], record, PickupEstimate(1., 1., 100., 1., 0, False), 10)
    assert events == [("register", True)]
    c.acknowledge_boarding(record.native_id, 0, 11.)
    assert events[-1] == ("release", 0) and c.repositioning_manager.active == {}


def test_factory_retains_midnight_fixture_offsets_and_only_changes_native_start_clock():
    episode, bridge, c, calls = fixture()
    bridge.bindings.immediate_simulation = type("PinnedSimulation", (), {"step": lambda self, t: None})
    bridge.bindings.broker_basic = lambda *args: NS()
    simulation = bridge.create_simulation(vehicles=list(c.runtime_by_vid.values()), fleet_control=c,
                                          network=c.routing_engine, native_output=[],
                                          simulation_end_s=100, time_step_s=1)
    assert simulation.start_time == 10 and simulation.scenario_parameters["start_time"] == 10
    assert c.start == START and c.fixture_windows_s[0] == (0., 12.)
    assert simulation.step.__func__ is bridge.bindings.immediate_simulation.step
    assert calls == [] and episode.execution_truth._values is None
