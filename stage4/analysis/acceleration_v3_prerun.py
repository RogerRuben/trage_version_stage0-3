"""Six historical decision snapshots, bounded real routing, no native replay.

Completion EVENTS at/before t determine observed idle/busy state. Future
completion timestamps/durations never enter prediction or optimization inputs.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa

from stage3.scripts.traffic_state_batch1 import sha, write_json
from stage4.analysis.acceleration_benchmark import vector
from stage4.analysis.symmetric_research_prepare import CONFIG
from stage4.dispatch.acceptance import passenger_acceptance
from stage4.dispatch.candidate_graph import SpatialVehicle, SparseCandidateIndex, search_radius_m
from stage4.dispatch.deterministic_routing import ArcDeterministicValhallaAdapter, SINGLE_SOURCE_MATRIX
from stage4.dispatch.flexibility_model import solve_dispatch, CurrentServiceFace
from stage4.dispatch.flexibility_v3 import solve_dispatch_v3
from stage4.dispatch.flexibility_native import NativeFlexibilityAdapter
from stage4.dispatch.remaining_time import TrainRemainingTime
from stage4.dispatch.runtime_assets import RuntimeAssets, CompactResearchRoutePolicy, DrawTapeForecast
from stage4.dispatch.routing_v3 import DemandRoutingAdapter
from stage4.dispatch.solver import AssignmentArc, solve_lexicographic
from stage4.fleetpy_adapter.upstream import CoordinateRegistry

STATES = (28800, 36000, 43200, 54000, 64800, 75600)
DOC = Path("stage4/docs/flexibility_dispatch/acceleration_v3")
OUT = Path("stage4/output/runtime_acceleration_v3/prerun")


class SnapshotActor:
    """Read existing raw bank; resolve only missing 1x1 primitives lazily."""
    def __init__(self, root, context):
        self.root = root
        self.context = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.db = sqlite3.connect((root / "stage4/output/runtime_acceleration_v2/routing/cache.sqlite3").as_uri() + "?mode=ro", uri=True, timeout=.25)
        self.values = {}
        self.actor = None
        self.hits = self.new_queries = 0

    @staticmethod
    def payload(vehicle, lon, lat, minute):
        return (SINGLE_SOURCE_MATRIX, float(vehicle.lon_wgs84).hex(), float(vehicle.lat_wgs84).hex(),
                float(lon).hex(), float(lat).hex(), minute)

    def preload(self, batches):
        keys = {}
        for vehicles, lon, lat, stamp in batches:
            minute = pd.Timestamp(stamp).tz_convert("Asia/Shanghai").strftime("%Y-%m-%dT%H:%M")
            for v in vehicles:
                key = self.payload(v, lon, lat, minute)
                digest = hashlib.sha256((self.context + "|" + "|".join(key)).encode()).digest()
                keys[digest] = key
        digests = list(keys)
        for left in range(0, len(digests), 400):
            chunk = digests[left:left + 400]
            for digest, raw, distance in self.db.execute("SELECT key,raw,distance FROM routes WHERE key IN (" + ",".join("?" for _ in chunk) + ")", chunk):
                self.values[keys[digest]] = (raw, distance)

    def matrix(self, request):
        if len(request["sources"]) != 1 or len(request["targets"]) != 1:
            raise ValueError("snapshot lookup is independent 1x1 only")
        source, target = request["sources"][0], request["targets"][0]
        key = (SINGLE_SOURCE_MATRIX, float(source["lon"]).hex(), float(source["lat"]).hex(),
               float(target["lon"]).hex(), float(target["lat"]).hex(), request["date_time"]["value"])
        if key in self.values:
            self.hits += 1
            raw, distance = self.values[key]
            return {"sources_to_targets": [[dict(time=raw, distance=distance / 1000.)]]}
        if self.actor is None:
            from valhalla import Actor
            config = json.loads((self.root / "stage3/config/stage3_finalization.json").read_text())["valhalla_config"]
            self.actor = Actor(config)
        self.new_queries += 1
        answer = self.actor.matrix(request)
        cell = answer["sources_to_targets"][0][0]
        if cell.get("time") is not None:
            self.values[key] = (float(cell["time"]), float(cell["distance"]) * 1000.)
        return answer

    def close(self):
        self.db.close()
        self.actor = None


def snapshot(root, assets, assignments, cfg, runtime, now):
    start = pd.Timestamp("2016-10-31T00:00:00+08:00")
    stamp = start + pd.Timedelta(seconds=now)
    requests = assets.requests()
    requests = [r for r in requests if assets.orders[assets.by_order[r.order_id]]["common"]]
    by_rid = {r.native_id: r for r in requests}
    past = assignments.loc[assignments.simulation_time_s.lt(now)]
    # Observed event prefix, NOT a predicted finish made from future truth.
    completed = set(assignments.loc[pd.to_datetime(assignments.service_end_time).le(stamp), "native_request_id"].astype(int))
    booked = set(past.native_request_id.astype(int))
    waiting = [r.native_id for r in requests if r.sim_time_s <= now < r.sim_time_s + cfg["patience_s"] and r.native_id not in booked]
    waiting.sort(key=lambda rid: (by_rid[rid].request_time, rid))
    registry = CoordinateRegistry()
    for r in requests:
        r.pickup_position = registry.position_for(r.pickup_lon_wgs84, r.pickup_lat_wgs84)
        r.dropoff_position = registry.position_for(r.dropoff_lon_wgs84, r.dropoff_lat_wgs84)
    fleet = assets.fleet()
    latest = past.sort_values("simulation_time_s").groupby("native_vehicle_id", sort=False).tail(1).set_index("native_vehicle_id")
    runtimes = {}
    for f in fleet.native_fixtures:
        pos = registry.position_for(f.initial_lon_wgs84, f.initial_lat_wgs84)
        busy = False
        if f.native_id in latest.index:
            rid = int(latest.loc[f.native_id, "native_request_id"])
            busy = rid not in completed
            if not busy:
                pos = by_rid[rid].dropoff_position
        runtimes[f.native_id] = NS(fixture=f, native_vehicle=NS(status=int(busy), assigned_route=[1] if busy else [], pos=pos))
    profiles = json.loads((root / "stage3/config/stage3_av_capability_profiles.json").read_text())
    policy = CompactResearchRoutePolicy(assets, "M", profiles)
    meta = {}
    for rid in waiting:
        r = by_rid[rid]
        decision = passenger_acceptance(r.order_id, runtime["passenger_acceptance_rate"], runtime["passenger_acceptance_seed"])
        first = int(np.ceil(r.sim_time_s / cfg["rolling_step_s"]) * cfg["rolling_step_s"])
        previous_attempts = max(0, (now - first) // cfg["rolling_step_s"])
        meta[rid] = dict(pickup_deadline_s=r.sim_time_s + cfg["patience_s"],
            passenger_accepts_av=decision.passenger_accepts_av, carry_over_flag=previous_attempts > 0,
            failed_round_count=previous_attempts, **policy.evaluate(r))
    c = NS(config={"profile_id": "M"}, dispatch_interval_s=cfg["rolling_step_s"], max_pickup_wait_s=cfg["patience_s"],
        _fixture_seconds=lambda stamp: float((stamp - start).total_seconds()), _pickup_overhead_s=lambda: 0.,
        bindings=NS(states=NS(IDLE=0)), runtime_by_vid=runtimes, request_by_rid=by_rid,
        routing_engine=registry, request_meta=meta, acceptance_rate=runtime["passenger_acceptance_rate"],
        acceptance_seed=runtime["passenger_acceptance_seed"],
        assignment_rows=past[["simulation_time_s", "native_vehicle_id", "native_request_id", "pickup_eta_s", "predicted_service_time_s"]].to_dict("records"))
    windows = assets.windows()
    available = [v for v in runtimes.values() if not v.native_vehicle.assigned_route and windows[v.fixture.native_id][0] <= now < windows[v.fixture.native_id][1]]
    spatial = [SpatialVehicle(v.fixture.vehicle_id, v.fixture.native_id, v.fixture.vehicle_type,
                              *registry.return_position_coordinates(v.native_vehicle.pos)) for v in available]
    index = SparseCandidateIndex(spatial, cached_geometry=True)
    batches, plans = [], []
    for rid in waiting:
        r, info = by_rid[rid], meta[rid]
        radius = search_radius_m(info["failed_round_count"], runtime["search_radius_initial_m"], runtime["search_radius_step_m"], runtime["search_radius_cap_m"])
        candidates, _ = index.query(r.pickup_lon_wgs84, r.pickup_lat_wgs84, radius, cfg["current_top_k"],
            info["research_route_compatible"] and info["passenger_accepts_av"])
        vehicles = [v for v, _ in candidates]
        batches.append((vehicles, r.pickup_lon_wgs84, r.pickup_lat_wgs84, stamp)); plans.append(rid)
    proxy = SnapshotActor(root, assets.manifest["routing_context"])
    proxy.preload(batches)
    router = ArcDeterministicValhallaAdapter(root, actor=proxy, routing_mode=SINGLE_SOURCE_MATRIX, persistent_cache_size=50000)
    arcs = []
    for rid, batch in zip(plans, batches):
        r, info = by_rid[rid], meta[rid]
        estimates = router.estimate_many(*batch)
        for vehicle in batch[0]:
            eta = estimates.get(vehicle.native_vehicle_id)
            if eta is None or eta.corrected_pickup_eta_s > info["pickup_deadline_s"] - now:
                continue
            fixture = runtimes[vehicle.native_vehicle_id].fixture
            if fixture.availability_policy == "EMPIRICAL_SESSION" and now + eta.corrected_pickup_eta_s + r.predicted_service_time_s > windows[fixture.native_id][1]:
                continue
            remaining = info["pickup_deadline_s"] - now
            arcs.append(AssignmentArc(vehicle.native_vehicle_id, rid, eta.corrected_pickup_eta_s,
                0 < remaining <= cfg["rolling_step_s"], info["carry_over_flag"]))
    routing_stats = dict(reused_raw_queries=proxy.hits, newly_resolved_raw_queries=proxy.new_queries, topk_arc_lookups=sum(len(b[0]) for b in batches))
    proxy.close(); router.close(); gc.collect()
    graph = NativeFlexibilityAdapter("SERVICE_PRESERVING_LOOKAHEAD", DrawTapeForecast(assets.directory / "forecast"),
        {**cfg, "fast_future_graph": True, "recourse_mode": "FLOW_RELAXED"},
        TrainRemainingTime(json.loads((root / cfg["remaining_time_model"]).read_text())))
    graph.stage_timings = {}
    problem = graph._problem(c, arcs, waiting, now)
    return problem, arcs, batches, dict(tick=now, waiting=len(waiting), current_arcs=len(arcs),
        predicted_vehicles=len(problem.vehicles), forecast_requests=sum(len(s.new_requests) for s in problem.scenarios),
        future_edges=sum(len(s.pickups) for s in problem.scenarios), **routing_stats)


def run_models(root, assets, assignments, cfg, runtime):
    rows, route_batches = [], []
    for i, now in enumerate(STATES):
        prepared = time.perf_counter()
        p, arcs, batches, info = snapshot(root, assets, assignments, cfg, runtime, now)
        info["snapshot_preparation_s"] = time.perf_counter() - prepared
        oracle = solve_lexicographic(arcs)
        face = CurrentServiceFace.from_arcs(arcs, oracle)
        selection = tuple((arcs[n].vehicle_id, arcs[n].request_id) for n in oracle.selected_indices)
        variants = ("V1_FLOW", "V2_COMPRESSED", "V3_DECOMPOSED")
        order = variants if i % 2 == 0 else tuple(reversed(variants))
        results = {}
        for name in order:
            limits = replace(p.limits, recourse_mode="FLOW_RELAXED", lock_current_face=name != "V1_FLOW",
                compress_fixed_states=name == "V2_COMPRESSED", trim_flow_rows=name == "V2_COMPRESSED")
            case = replace(p, limits=limits)
            t = time.perf_counter()
            try:
                d = (solve_dispatch_v3(case, "SERVICE_PRESERVING_LOOKAHEAD", current_face=face, current_selection=selection)
                     if name == "V3_DECOMPOSED" else solve_dispatch(case, "SERVICE_PRESERVING_LOOKAHEAD", current_face=face if limits.lock_current_face else None))
                results[name] = dict(status="OPTIMAL", wall_s=time.perf_counter() - t, objective=vector(case, d),
                    variables=d.variable_count, integer_variables=d.integer_variable_count, nonzeros=d.constraint_nonzeros,
                    build_s=d.model_build_time_s, optimize_s=d.solve_time_s, recovery_s=d.recourse_recovery_time_s,
                    components=getattr(d, "component_count", 0), pure_future_edges=getattr(d, "pure_future_edges", 0))
            except RuntimeError as error:
                if "timeout" not in str(error) and "time limit" not in str(error).casefold():
                    raise
                results[name] = dict(status="NOT_PROVEN_WITHIN_10S", wall_s=time.perf_counter()-t, error=str(error))
        finished = [r for r in results.values() if r["status"] == "OPTIMAL"]
        same = len(finished) == len(results) and all(np.allclose(finished[0]["objective"], r["objective"], rtol=0, atol=1e-6) for r in finished)
        if len(finished) > 1 and not all(np.allclose(finished[0]["objective"], r["objective"], rtol=0, atol=1e-6) for r in finished):
            raise RuntimeError("real-state model objectives disagree")
        rows.append(dict(**info, all_objectives_equal=same, variants=results))
        # Bound real routing comparison to 20 waiting batches per state.
        route_batches.extend(batches[:20])
        write_json(root / OUT / "progress.json", dict(completed_states=len(rows), total_states=len(STATES), latest=rows[-1]))
        print(json.dumps(rows[-1]), flush=True)
        gc.collect()
    return rows, route_batches


def run_routes(root, batches):
    rows, reference = {}, None
    for name in ("V2_EAGER_1X1", "V3_DEMAND_1X1", "V3_GROUPED_CANDIDATE"):
        router = (ArcDeterministicValhallaAdapter(root, routing_mode=SINGLE_SOURCE_MATRIX, route_workers=2,
                    persistent_cache_size=50000, lazy_worker_actor=True)
                  if name == "V2_EAGER_1X1" else DemandRoutingAdapter(root, routing_mode=SINGLE_SOURCE_MATRIX,
                    route_workers=2, persistent_cache_size=50000, lazy_worker_actor=True,
                    grouped_sources=name == "V3_GROUPED_CANDIDATE"))
        try:
            # Explicit warmup is excluded and caches reset for the cold inputs.
            if batches:
                router.estimate_epoch(batches[:1]); router.cache.clear(); router._persistent_cache.clear()
            t = time.perf_counter()
            found = []
            for stamp in sorted({b[3] for b in batches}):
                group = [b for b in batches if b[3] == stamp]
                found.extend(router.estimate_epoch(group)); router.cache.clear()
            elapsed = time.perf_counter() - t
            values = [{vid: (v.valhalla_time_s, v.corrected_pickup_eta_s, v.route_distance_m, v.beta, v.time_bin_index)
                       for vid, v in row.items()} for row in found]
            if reference is None:
                reference = values
            equal = values == reference
            rows[name] = dict(wall_s=elapsed, exact_answers_equal=equal, query_batches=len(batches),
                arc_lookups=sum(len(b[0]) for b in batches), routing_arcs=router.routing_arc_evaluations,
                backend_queries=router.routing_queries, raw_prefetch_avoided=getattr(router, "raw_prefetch_avoided", 0),
                process_group_rss_mib=router.process_group_rss_mib())
            if name == "V3_DEMAND_1X1" and not equal:
                raise RuntimeError("strict demand queue changed routing answers")
        finally:
            router.close(); router.actor = None; gc.collect()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-routing", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    (root / OUT).mkdir(parents=True, exist_ok=True)
    (root / DOC).mkdir(parents=True, exist_ok=True)
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    # No heavy comparison overlaps a native full-day experiment.
    for process in psutil.process_iter(["cmdline"]):
        try:
            if process.pid != psutil.Process().pid and "stage4.analysis.symmetric_flexibility_full_day" in " ".join(process.info["cmdline"] or []):
                raise RuntimeError("wait for the existing native run to finish before speed comparison")
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    cfg = json.loads((root / CONFIG).read_text())
    assets = RuntimeAssets(root / "stage4/output/runtime_acceleration_v2/assets")
    base = json.loads((root / "stage4/output/final_experiments" / cfg["source_scenario"] / "scenario_config.json").read_text())["runtime_configuration"]
    source = root / "stage4/output/symmetric_flexibility_v1/accelerated_full_day/SERVICE_PRESERVING_LOOKAHEAD/assignments.parquet"
    assignments = pd.read_parquet(source)
    started = time.perf_counter()
    states, batches = run_models(root, assets, assignments, cfg, base)
    routing = None if args.skip_routing else run_routes(root, batches)
    result = dict(status="COMPLETE", scope="SIX_ASOF_V1_PHYSICAL_STATES_NO_NATIVE_REPLAY", states=states,
        routing=routing, model_objectives_equal=all(r["all_objectives_equal"] for r in states),
        grouped_source_enabled_by_default=False, native_scientific_conditions_added=0,
        future_completion_times_used_for_prediction=False, gpu_used=False, dense_matrix=False,
        runtime_s=time.perf_counter()-started,
        peak_parent_rss_mib=psutil.Process().memory_info().peak_wset / 2**20,
        source_assignments_sha256=sha(source), science_config_sha256=sha(root / CONFIG))
    write_json(root / DOC / "prerun.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
