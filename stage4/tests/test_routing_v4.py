"""Joint checks for raw-time reuse, original aliases, and admission bounds."""
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from stage4.analysis.symmetric_flexibility_full_day import acceleration_settings
from stage4.dispatch.deterministic_routing import SINGLE_SOURCE_MATRIX
from stage4.dispatch.routing_v3 import DemandRoutingAdapter
from stage4.dispatch.routing_v4 import (STATIC_TAG, PickupEtaBudget, StaticRawRoutingAdapter,
                                       static_matrix_certificate)
from stage4.tests.test_routing_determinism import _vehicles

ROOT = Path(__file__).resolve().parents[2]
STAMP = pd.Timestamp("2016-10-31T08:00:00+08:00")


class StaticActor:
    def __init__(self):
        self.calls = []

    def matrix(self, request):
        self.calls.append(request)
        return {"sources_to_targets": [[dict(time=s["lon"] * 1000., distance=.5 + t["lat"] / 100.)
            for t in request["targets"]] for s in request["sources"]]}


def values(rows):
    return [{vid: (e.valhalla_time_s, e.corrected_pickup_eta_s, e.route_distance_m, e.beta, e.time_bin_index)
             for vid, e in row.items()} for row in rows]


def make(actor, **kwargs):
    return StaticRawRoutingAdapter(ROOT, actor=actor, routing_mode=SINGLE_SOURCE_MATRIX,
        persistent_cache_size=8, **kwargs)


def test_cross_minute_raw_reuse_recomputes_beta_and_keeps_precise_coordinates():
    actor = StaticActor()
    r = make(actor, grouped_sources=False, raw_od_cache_size=8)
    v = _vehicles()[0]
    r._beta[32], r._beta[33] = 1., 2.
    answer = []
    for minutes in (0, 1, 15):
        r.cache.clear()
        answer.append(r.estimate_many([v], 108.92, 34.22, STAMP + pd.Timedelta(minutes=minutes))[v.native_vehicle_id])
    assert len(actor.calls) == 1
    assert [e.valhalla_time_s for e in answer] == [answer[0].valhalla_time_s] * 3
    assert answer[2].corrected_pickup_eta_s == 2 * answer[0].corrected_pickup_eta_s
    assert r.raw_od_cross_minute_reuses == 2 and r.raw_od_beta_bin_reuses == 1
    near = replace(v, lon_wgs84=v.lon_wgs84 + 1e-9)
    local = STAMP + pd.Timedelta(minutes=16)
    assert r._raw_key(v, 108.92, 34.22, local) != r._raw_key(near, 108.92, 34.22, local)
    assert r._raw_key(v, 108.92, 34.22, local)[-1] == STATIC_TAG
    r.close()


def test_near_zero_keeps_minute_and_is_never_mixed_into_group():
    class ClockActor(StaticActor):
        def matrix(self, request):
            result = super().matrix(request)
            for row in result["sources_to_targets"]:
                for e in row:
                    e["time"] = float(request["date_time"]["value"][-2:]) + 1.
            return result
    actor, v = ClockActor(), _vehicles()[0]
    r = make(actor, grouped_sources=True)
    answers = []
    for minute in (0, 1):
        r.cache.clear()
        answers.append(r.estimate_many([v], v.lon_wgs84, v.lat_wgs84,
            STAMP + pd.Timedelta(minutes=minute))[v.native_vehicle_id].valhalla_time_s)
    assert answers == [1., 2.] and len(actor.calls) == 2
    r.cache.clear()
    r.estimate_epoch([([v], v.lon_wgs84, v.lat_wgs84, STAMP + pd.Timedelta(minutes=2)),
                      ([v], 108.92, 34.22, STAMP + pd.Timedelta(minutes=2))])
    assert all(len(q["targets"]) == 1 for q in actor.calls)
    r.close()


def test_rounded_alias_precedence_and_cached_pruning_never_refill_topk():
    v = _vehicles()[0]
    near = replace(v, native_vehicle_id=30, lon_wgs84=v.lon_wgs84 + 1e-9)
    old = DemandRoutingAdapter(ROOT, actor=StaticActor(), routing_mode=SINGLE_SOURCE_MATRIX, persistent_cache_size=8)
    new = make(StaticActor(), grouped_sources=True)
    for minute in (0, 1):
        old.cache.clear(); new.cache.clear()
        stamp = STAMP + pd.Timedelta(minutes=minute)
        batches = [([v, near], 108.92, 34.22, stamp), ([v], 108.92, 34.22, stamp)]
        assert values(old.estimate_epoch(batches)) == values(new.estimate_epoch(batches))
    new.cache.clear()
    stamp = STAMP + pd.Timedelta(minutes=2)
    budgets = [{v.native_vehicle_id: PickupEtaBudget(0., 0., 30.)}]
    assert new.estimate_epoch([([v], 108.92, 34.22, stamp)], eta_budgets=budgets) == [{}]
    assert new.last_certified_prunes == [{v.native_vehicle_id: "PATIENCE"}]
    assert new.certified_eta_prunes == new.known_eta_prunes_before_backend == 1
    assert new._matrix_key(v, 108.92, 34.22, stamp, new.beta_for(stamp)[0]) in new.cache
    old.close(); new.close()


def test_budget_keeps_original_boundary_order_and_does_not_prune_unknown_route():
    b = PickupEtaBudget(100., 900., 30., 1030.)
    assert b.rejection(100.) is None
    assert replace(b, remaining_patience_s=200.).rejection(100.00001) == "HV_SESSION_END"
    assert b.rejection(0.) is None
    assert PickupEtaBudget(100., 0., 0.).rejection(100.) is None
    assert PickupEtaBudget(100., 0., 0.).rejection(100.00001) == "PATIENCE"
    assert PickupEtaBudget(100., 0., float("nan"), 300.).rejection(0.) == "HV_SESSION_EVIDENCE"
    actor, v = StaticActor(), _vehicles()[0]
    r = make(actor)
    budgets = [{v.native_vehicle_id: PickupEtaBudget(0., 0., 30.)}]
    first = r.estimate_epoch([([v], 108.92, 34.22, STAMP)], eta_budgets=budgets)
    assert v.native_vehicle_id in first[0] and r.last_certified_prunes == [{}]
    assert len(actor.calls) == 1
    r.cache.clear()
    assert r.estimate_epoch([([v], 108.92, 34.22, STAMP + pd.Timedelta(minutes=1))], eta_budgets=budgets) == [{}]
    assert len(actor.calls) == 1
    r.close()


def test_bounded_lru_and_disk_raw_reuse_have_separate_time_specific_beta(tmp_path):
    actor, v = StaticActor(), _vehicles()[0]
    r = make(actor, raw_od_cache_size=1)
    for minute, lon in enumerate((108.92, 108.93, 108.92)):
        r.cache.clear()
        r.estimate_many([v], lon, 34.22, STAMP + pd.Timedelta(minutes=minute))
    assert len(r._raw_od_cache) == 1 and len(actor.calls) == 3
    r.close()
    path = tmp_path / "raw.sqlite3"
    context = {"frozen": "unit"}
    first = make(StaticActor(), disk_cache_path=path, routing_context=context, disk_cache_limit_mib=16)
    first._beta[32] = 1.
    e = first.estimate_many([v], 108.92, 34.22, STAMP)[v.native_vehicle_id]
    first.close()
    actor = StaticActor()
    second = make(actor, disk_cache_path=path, routing_context=context, disk_cache_limit_mib=16)
    second._beta[33] = 2.
    e2 = second.estimate_many([v], 108.92, 34.22, STAMP + pd.Timedelta(minutes=15))[v.native_vehicle_id]
    assert not actor.calls and second.disk_cache_hits == 1
    assert e2.valhalla_time_s == e.valhalla_time_s and e2.corrected_pickup_eta_s == 2 * e.corrected_pickup_eta_s
    second.cache.clear(); second._persistent_cache.clear()
    def disk_read_forbidden(keys):
        raise AssertionError("known raw RAM hit must precede disk lookup")
    second._disk_cache.get_many = disk_read_forbidden
    e3 = second.estimate_many([v], 108.92, 34.22, STAMP + pd.Timedelta(minutes=16))[v.native_vehicle_id]
    assert e3.corrected_pickup_eta_s == e2.corrected_pickup_eta_s
    assert second.disk_cache_hits == 1 and second.raw_od_memory_queries_avoided == 1
    second.close()


def test_static_context_is_narrow_and_v4_config_does_not_touch_science(tmp_path):
    spec = ROOT / "stage4/config/symmetric_flexibility_acceleration_v4.json"
    settings, _, limit = acceleration_settings(ROOT, spec)
    assert settings["route_workers"] == 2 and limit == 2048
    assert settings["grouped_sources"] and settings["static_raw_od"] and settings["certified_eta_pruning"]
    assert "forecast_horizon_s" not in settings and "current_top_k" not in settings
    cfg = {"service_limits": {"max_timedep_distance_matrix": 1}, "thor": {"source_to_target_algorithm": "select_optimal"}}
    local = tmp_path / "routing.json"
    local.write_text(json.dumps(cfg))
    (tmp_path / "stage3/config").mkdir(parents=True)
    (tmp_path / "stage3/config/stage3_finalization.json").write_text(json.dumps({"valhalla_config": str(local)}))
    with pytest.raises(ValueError, match="inspected frozen"):
        static_matrix_certificate(tmp_path)
