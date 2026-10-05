"""A few joint equivalence checks for opt-in v3, not an audit expansion."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stage4.analysis.acceleration_benchmark import sparse_case, vector
from stage4.analysis.flexibility_small_instances import small_case
from stage4.dispatch.flexibility_model import (Vehicle, Request, CurrentPickup, FuturePickup, Scenario, Problem,
    ModelLimits, CurrentServiceFace, solve_dispatch)
from stage4.dispatch.flexibility_v3 import solve_dispatch_v3
from stage4.dispatch.deterministic_routing import ArcDeterministicValhallaAdapter, SINGLE_SOURCE_MATRIX
from stage4.dispatch.routing_v3 import DemandRoutingAdapter, _ShadowLRU
from stage4.dispatch.route_bank_v3 import BufferedRouteBank
from stage4.tests.test_routing_determinism import _vehicles

ROOT = Path(__file__).resolve().parents[2]


def test_decomposition_matches_original_full_objective_and_recovers_capacity():
    cases = [small_case(name)[0] for name in ("flexibility_reservation", "location_counterexample", "forecast_error_counterexample")]
    cases += [sparse_case(12, 8, 12), sparse_case(36, 24, 30)]
    for p in cases:
        p = replace(p, limits=replace(p.limits, recourse_mode="FLOW_RELAXED"))
        reference = solve_dispatch(p, "SERVICE_PRESERVING_LOOKAHEAD")
        actual = solve_dispatch_v3(p, "SERVICE_PRESERVING_LOOKAHEAD")
        assert vector(p, actual) == pytest.approx(vector(p, reference), abs=1e-6)
        for scene in p.scenarios:
            pairs = [(v, r) for sid, v, r in actual.recourse_pairs if sid == scene.scenario_id]
            assert len({v for v, _ in pairs}) == len(pairs) == len({r for _, r in pairs})


def test_pure_future_is_constant_but_future_requests_merge_current_components():
    vehicles = tuple(Vehicle(i, "HV", f"V{i}", 0, 1200) for i in range(1, 5))
    current = (Request(10, 0, 300, 30, "D10"), Request(11, 0, 300, 30, "D11"))
    future = (Request(20, 90, 390, 30, "D20"), Request(21, 90, 390, 30, "D21"))
    p = Problem(0, vehicles, current, (CurrentPickup(1, 10, 0), CurrentPickup(2, 11, 0)),
        (Scenario("S", 1., -100, future, (FuturePickup(1, None, 20, 10, "V1"),
            FuturePickup(2, None, 20, 10, "V2"), FuturePickup(3, None, 21, 10, "V3"),
            FuturePickup(4, None, 21, 10, "V4"))),), ModelLimits(recourse_mode="FLOW_RELAXED"))
    d = solve_dispatch_v3(p, "SERVICE_PRESERVING_LOOKAHEAD")
    assert d.mip_component_count == 1 and d.pure_future_component_count == 1
    assert d.pure_future_edges == 2 and d.pure_future_expected_count == 1
    assert vector(p, d) == pytest.approx(vector(p, solve_dispatch(p, "SERVICE_PRESERVING_LOOKAHEAD")))
    # One fixed physical vehicle may be used independently in different scenes.
    s2 = replace(p.scenarios[0], scenario_id="S2", probability=.5)
    p = replace(p, scenarios=(replace(p.scenarios[0], probability=.5), s2))
    assert vector(p, solve_dispatch_v3(p, "SERVICE_PRESERVING_LOOKAHEAD")) == pytest.approx(vector(p, solve_dispatch(p, "SERVICE_PRESERVING_LOOKAHEAD")))


def test_v3_resource_caps_precede_component_allocation():
    p = replace(small_case("flexibility_reservation")[0], limits=ModelLimits(recourse_mode="FLOW_RELAXED", max_variables=1))
    with pytest.raises(ValueError, match="resource cap"):
        solve_dispatch_v3(p, "SERVICE_PRESERVING_LOOKAHEAD")


def test_shadow_lru_matches_real_lru_with_eviction():
    from collections import OrderedDict
    actual = OrderedDict((i, i) for i in range(4))
    shadow = _ShadowLRU(actual.copy(), 4)
    for operation, key in (("get", 0), ("put", 4), ("get", 1), ("put", 5), ("get", 2), ("put", 0), ("put", 6)):
        if operation == "get":
            value = actual.get(key)
            if value is not None:
                actual.move_to_end(key)
            assert shadow.get(key) == value
        else:
            actual[key] = key; actual.move_to_end(key)
            while len(actual) > 4:
                actual.popitem(last=False)
            shadow.put(key, key)
    for key in range(7):
        assert shadow.get(key) == actual.get(key)


def test_demand_planning_preserves_rounded_alias_precedence_without_prefetch():
    class Actor:
        def matrix(self, request):
            return {"sources_to_targets": [[{"time": s["lon"] * 1000., "distance": .5} for _ in request["targets"]] for s in request["sources"]]}
    t = pd.Timestamp("2016-10-31T08:00:00+08:00")
    v = _vehicles()[0]
    near = replace(v, native_vehicle_id=3, lon_wgs84=v.lon_wgs84 + 1e-9)
    batches = [([v], 108.92, 34.22, t), ([near], 108.92, 34.22, t)]
    old = ArcDeterministicValhallaAdapter(ROOT, actor=Actor(), routing_mode=SINGLE_SOURCE_MATRIX, persistent_cache_size=2)
    new = DemandRoutingAdapter(ROOT, actor=Actor(), routing_mode=SINGLE_SOURCE_MATRIX, persistent_cache_size=2)
    assert new.estimate_epoch(batches) == [old.estimate_many(*batch) for batch in batches]
    assert new.routing_arc_evaluations == 1 and new.raw_prefetch_avoided == 1
    old.cache.clear(); new.cache.clear()
    batches = [([near, v], 108.92, 34.22, t), ([v], 108.92, 34.22, t)]
    assert new.estimate_epoch(batches) == [old.estimate_many(*batch) for batch in batches]


def test_buffered_bank_capacity_and_raw_reuse_are_independent_of_beta(tmp_path):
    path = tmp_path / "cache.sqlite3"
    bank = BufferedRouteBank(path, {"tiles": "frozen"}, max_entries=4, limit_mib=16)
    keys = [bank.key(("MATRIX", float(i).hex(), "MINUTE")) for i in range(8)]
    for i, key in enumerate(keys):
        bank.remember(key, i, 100.)
    bank.close()
    reader = BufferedRouteBank(path, {"tiles": "frozen"}, max_entries=4, limit_mib=16)
    assert reader.entry_count == 4 and len(reader.get_many(keys)) == 4
    reader.close()
