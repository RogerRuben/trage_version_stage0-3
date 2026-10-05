"""Joint checks of exact acceleration contracts, not a new experiment grid."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from stage4.analysis.flexibility_small_instances import small_case
from stage4.dispatch.candidate_graph import SparseCandidateIndex, SpatialVehicle
from stage4.dispatch.deterministic_routing import ArcDeterministicValhallaAdapter, SINGLE_SOURCE_MATRIX
from stage4.dispatch.flexibility_model import CurrentServiceFace, FuturePickup, Vehicle, solve_dispatch
from stage4.dispatch.flexibility_native import NativeFlexibilityAdapter, TrainDemandForecast
from stage4.dispatch.runtime_assets import PositionCatalogBuilder, build_forecast_product, DrawTapeForecast
from stage4.dispatch.runtime_calendar import FleetEventCalendar
from stage4.dispatch.solver import AssignmentArc, solve_lexicographic
from stage4.dispatch.sparse_route_cache import SparseRouteDiskCache
from stage4.tests.test_acceleration import objective_vector
from stage4.tests.test_flexibility_native import config, templates
from stage4.tests.test_routing_determinism import BatchSensitiveActor, _vehicles

ROOT = Path(__file__).resolve().parents[2]


def test_cached_candidate_geometry_and_session_calendar_are_exact():
    rng = np.random.default_rng(20261005)
    vehicles = [SpatialVehicle(f"V{i:04d}", i, "HV" if i % 3 else "AV", 108.9 + rng.uniform(-.05, .05),
                               34.2 + rng.uniform(-.05, .05)) for i in range(200)]
    slow, fast = SparseCandidateIndex(vehicles), SparseCandidateIndex(vehicles, cached_geometry=True)
    for i in range(30):
        lon, lat = 108.9 + rng.uniform(-.04, .04), 34.2 + rng.uniform(-.04, .04)
        for radius in (2000, 8000):
            assert slow.query(lon, lat, radius, 20, bool(i % 2)) == fast.query(lon, lat, radius, 20, bool(i % 2))
    windows = {3: (0, 100), 1: (30, 80), 5: (90, 120), 4: (200, 400)}
    calendar = FleetEventCalendar(windows, windows)
    for now in (0, 29, 30, 80, 90, 99, 100, 120, 200, 400):
        assert calendar.active_ids(now) == tuple(v for v, (a, b) in windows.items() if a <= now < b)
        assert calendar.horizon_ids(now, 60) == tuple(sorted(v for v, (a, b) in windows.items() if b > now and a <= now + 60))


def test_offline_draw_tape_is_exact_and_crn_rate_switchable(tmp_path):
    cfg = {**config(), "rolling_step_s": 30, "last_dispatch_s": 210}
    frame = templates()
    frame.loc[1, "compatible_C"] = False
    catalog = PositionCatalogBuilder()
    build_forecast_product(tmp_path / "forecast", frame, cfg, 160, 123, catalog)
    catalog.write(tmp_path)
    tape = DrawTapeForecast(tmp_path / "forecast")
    reference = TrainDemandForecast(frame, {**cfg, "fast_forecast": True}, 160)
    for rate in (0., .4, .7, 1.):
        for now in range(0, 211, 30):
            assert tape.scenarios(now, "M", rate, 123) == reference.scenarios(now, "M", rate, 123)
    assert len(tape._chunks) <= 2
    with pytest.raises(ValueError, match="seed/rate"):
        tape.scenarios(0, "M", .7, 124)


def test_fast_future_graph_is_exact_including_known_waiting_and_duplicate_pickups():
    cfg = {**config(), "fast_forecast": True, "max_model_variables": 20000, "max_model_nonzeros": 150000,
           "solver_time_limit_s": 10, "future_search_radius_m": 2000, "future_top_k_vehicles": 5}
    c = NS(config={"profile_id": "M"}, dispatch_interval_s=30, max_pickup_wait_s=300,
        _fixture_seconds=lambda x: x, _pickup_overhead_s=lambda: 0,
        bindings=NS(states=NS(IDLE=0)), assignment_rows=[],
        routing_engine=NS(return_position_coordinates=lambda pos: pos))
    c.runtime_by_vid = {i: NS(fixture=NS(vehicle_type="HV" if i % 2 else "AV", availability_start_time=0,
        availability_end_time=5000), native_vehicle=NS(status=0, assigned_route=[], pos=(108.93 + i * .0001, 34.24)))
        for i in range(1, 16)}
    c.request_by_rid = {i: NS(order_id=str(i), sim_time_s=0, predicted_service_time_s=180,
        pickup_lon_wgs84=108.93, pickup_lat_wgs84=34.24, dropoff_lon_wgs84=108.95, dropoff_lat_wgs84=34.25) for i in (10, 11)}
    c.request_meta = {i: dict(research_route_compatible=True, pickup_deadline_s=300, passenger_accepts_av=True, carry_over_flag=True) for i in (10, 11)}
    c.acceptance_rate, c.acceptance_seed = .7, 123
    slow = NativeFlexibilityAdapter("SERVICE_PRESERVING_LOOKAHEAD", TrainDemandForecast(templates(), cfg, 1000), cfg)
    fast = NativeFlexibilityAdapter("SERVICE_PRESERVING_LOOKAHEAD", slow.forecast, {**cfg, "fast_future_graph": True})
    arcs = [AssignmentArc(1, 10, 30, False, True), AssignmentArc(2, 11, 20, False, True)]
    for adapter in (slow, fast):
        adapter.stage_timings = {}
    reference = slow._problem(c, arcs, [10, 11], 60)
    actual = fast._problem(c, arcs, [10, 11], 60)
    assert actual == reference


def test_compressed_model_preserves_all_lexicographic_objectives_and_face():
    for name in ("flexibility_reservation", "location_counterexample", "forecast_error_counterexample"):
        base, _ = small_case(name)
        idle = tuple(Vehicle(100 + i, "HV", "IDLE", 0, 1200) for i in range(8))
        scenarios = tuple(replace(s, pickups=s.pickups + tuple(FuturePickup(v.vehicle_id, None, s.new_requests[0].request_id, 5, "IDLE") for v in idle)) for s in base.scenarios)
        base = replace(base, vehicles=base.vehicles + idle, scenarios=scenarios, limits=replace(base.limits, recourse_mode="FLOW_RELAXED"))
        arcs = [AssignmentArc(p.vehicle_id, p.request_id, p.pickup_eta_s,
            next(r for r in base.waiting_requests if r.request_id == p.request_id).critical,
            next(r for r in base.waiting_requests if r.request_id == p.request_id).carry_over) for p in base.current_pickups]
        face = CurrentServiceFace.from_arcs(arcs, solve_lexicographic(arcs))
        fast = replace(base, limits=replace(base.limits, lock_current_face=True, compress_fixed_states=True, trim_flow_rows=True))
        for policy in ("LOOKAHEAD", "SERVICE_PRESERVING_LOOKAHEAD"):
            reference = solve_dispatch(base, policy)
            actual = solve_dispatch(fast, policy, current_face=face)
            assert objective_vector(base, actual, policy) == pytest.approx(objective_vector(base, reference, policy), abs=1e-6)
            assert actual.eliminated_fixed_variables == len(idle)
            assert actual.integer_variable_count == reference.integer_variable_count - len(idle)
        wrong = replace(face, signature=())
        with pytest.raises(ValueError, match="different sparse graph"):
            solve_dispatch(fast, "SERVICE_PRESERVING_LOOKAHEAD", current_face=wrong)


def test_disk_raw_cache_is_bounded_and_full_precision(tmp_path):
    cache = SparseRouteDiskCache(tmp_path / "raw.sqlite3", {"tiles": "frozen"}, max_entries=3, limit_mib=16)
    keys = [cache.key(("matrix", float(i).hex(), "minute")) for i in range(7)]
    for i, key in enumerate(keys):
        cache.remember(key, float(i), 100.)
    cache.flush()
    assert cache.count <= 3 and cache.path.stat().st_size <= 16 * 2**20
    assert keys[-1] in cache.get_many(keys)
    assert cache.key(("matrix", (1. + 1e-9).hex(), "minute")) != keys[1]
    cache.close()


def test_epoch_queue_replays_legacy_aliases_and_reuses_raw_disk_without_beta(tmp_path):
    class CoordinateSensitiveActor:
        def matrix(self, request):
            return {"sources_to_targets": [[{"time": request["sources"][0]["lon"] * 1000., "distance": .5}]]}
    timestamp = pd.Timestamp("2016-10-31T08:00:00+08:00")
    v = _vehicles()[0]
    nearby = replace(v, native_vehicle_id=3, lon_wgs84=v.lon_wgs84 + 1e-9)
    batches = [([v, nearby], 108.92, 34.22, timestamp), ([nearby], 108.92, 34.22, timestamp)]
    serial = ArcDeterministicValhallaAdapter(ROOT, actor=CoordinateSensitiveActor(), routing_mode=SINGLE_SOURCE_MATRIX,
        persistent_cache_size=2)
    serial.estimate_many([nearby], 108.92, 34.22, timestamp)
    serial.cache.clear()
    expected = [serial.estimate_many(*batch) for batch in batches]
    cache_path = tmp_path / "routes.sqlite3"
    fast = ArcDeterministicValhallaAdapter(ROOT, actor=CoordinateSensitiveActor(), routing_mode=SINGLE_SOURCE_MATRIX,
        disk_cache_path=cache_path, routing_context={"tiles": "same"}, persistent_cache_size=2)
    fast.estimate_many([nearby], 108.92, 34.22, timestamp)
    fast.cache.clear()
    actual = fast.estimate_epoch(batches)
    assert actual == expected
    raw = actual[0][v.native_vehicle_id].valhalla_time_s
    beta = actual[0][v.native_vehicle_id].beta
    fast.close()
    warm = ArcDeterministicValhallaAdapter(ROOT, actor=CoordinateSensitiveActor(), routing_mode=SINGLE_SOURCE_MATRIX,
        disk_cache_path=cache_path, routing_context={"tiles": "same"}, persistent_cache_size=2)
    index, _ = warm.beta_for(timestamp)
    warm._beta[index] = beta * 2
    found = warm.estimate_epoch([batches[0]])[0]
    assert found[v.native_vehicle_id].corrected_pickup_eta_s == raw * beta * 2
    assert warm.routing_queries == 0 and warm.disk_cache_hits == 2
    warm.close()
