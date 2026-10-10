"""Eight fixed, earlier-history C connectors: engineering equivalence only."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
from time import perf_counter, process_time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import psutil

from stage4.analysis.capability_chain_instances import atomic_json, sha
from stage4.analysis.scheme_a_engineering_benchmark import load_reference
from stage4.dispatch import scheme_a_city_connections as optimized
from stage4.dispatch.city_pickup_support import Test31PickupJoinEvidence
from stage4.dispatch.controlled_routes import ControlledEmptyRouter
from stage4.dispatch.deterministic_routing import SINGLE_SOURCE_MATRIX
from stage4.dispatch.routing_v4 import StaticRawRoutingAdapter
from stage4.dispatch.scheme_a_city_graph import ChainTask, CoarseHistoricalLibrary, RouteState, _xy


def setup_side(module, root, templates):
    midnight = pd.Timestamp("2016-10-31", tz="Asia/Shanghai")
    eta = StaticRawRoutingAdapter(root, routing_mode=SINGLE_SOURCE_MATRIX, route_workers=1,
        defer_actor=True, grouped_sources=True, persistent_cache_size=20000, raw_od_cache_size=20000)
    router = ControlledEmptyRouter(root, eta)
    evidence = Test31PickupJoinEvidence(router)
    control = SimpleNamespace(eta_adapter=eta, sim_time=61200., request_by_rid={},
        runtime_by_vid={}, assignment_rows=[], completed_rids=set(),
        config=dict(scheme_a_fast_forecast_hv=True),
        _timestamp=lambda second: midnight + pd.Timedelta(seconds=second))
    connector = module.SchemeACityConnectors(control, router,
        SimpleNamespace(seam_evidence=evidence), templates, geometry_timestamp=midnight)
    return connector, router, eta, midnight


def run_side(module, root, pairs, templates):
    started = perf_counter()
    connector, router, eta, midnight = setup_side(module, root, templates)
    setup_s = perf_counter()-started
    outputs = []
    direct_join_outputs = []
    direct_join_s = 0.
    query_started = perf_counter()
    try:
        # One cold and one departure-time-retimed visit, identical input order.
        for departure in (61200., 62100.):
            for source, target in pairs:
                if perf_counter()-started > 60.:
                    raise TimeoutError("bounded connector engineering check exceeded 60 s")
                if psutil.Process().memory_info().rss > 1536 * 2**20:
                    raise MemoryError("connector check exceeded 1536 MiB RSS")
                state = RouteState(source.dropoff, departure, 90000., "C", dict(kind="CUSTOMER", task=source))
                result = connector(state, target, departure, "C")
                outputs.append({name: result[name] for name in (
                    "supported", "compatible_profiles", "travel_time_s", "empty_distance_m", "reason_codes")})
                # Independently exercise the join parser on the SAME cached
                # physical route, including routes rejected before that stage.
                # This is branch equivalence, NOT an eligibility override.
                routed = router.route(*source.dropoff, *target.pickup, midnight)
                if not routed.get("common_supported"):
                    raise ValueError("fixed join parser fixture has no complete common route evidence")
                parse_started = perf_counter()
                direct_join_outputs.append(connector._join(state, target, routed))
                direct_join_s += perf_counter()-parse_started
        result = dict(setup_s=setup_s, query_wall_s=perf_counter()-query_started,
            direct_join_s=direct_join_s, outputs=outputs, direct_join_outputs=direct_join_outputs,
            connector=connector.diagnostics(), router=router.diagnostics())
    finally:
        eta.close()
    return result


def compare_warm_parser(baseline, root, pairs, templates):
    """Alternating same-path kernel checks avoid cold/O.S. phase ordering."""
    started = perf_counter()
    sides = [setup_side(module, root, templates) for module in (baseline, optimized)]
    elapsed, cpu = [0., 0.], [0., 0.]
    try:
        fixtures = []
        for source, target in pairs:
            state = RouteState(source.dropoff, 61200., 90000., "C", dict(kind="CUSTOMER", task=source))
            routes, outputs = [], []
            for connector, router, _, midnight in sides:
                routed = router.route(*source.dropoff, *target.pickup, midnight)
                routes.append(routed)
                outputs.append(connector._join(state, target, routed))
            if outputs[0] != outputs[1]:
                raise AssertionError("warm join fixture changed parser output")
            fixtures.append((state, target, routes))
        for repeat in range(2):
            for state, target, routes in fixtures:
                if perf_counter()-started > 60. or psutil.Process().memory_info().rss > 1536 * 2**20:
                    raise RuntimeError("warm parser kernel check exceeded its fixed resource budget")
                outputs = [None, None]
                for side in ((0, 1) if repeat == 0 else (1, 0)):
                    connector = sides[side][0]
                    wall_start, cpu_start = perf_counter(), process_time()
                    outputs[side] = connector._join(state, target, routes[side])
                    cpu[side] += process_time()-cpu_start
                    elapsed[side] += perf_counter()-wall_start
                if outputs[0] != outputs[1]:
                    raise AssertionError("warm cache changed parser output")
        return dict(status="WARM_PARSER_EQUIVALENCE_PASS", paired_calls_per_side=16,
            old_wall_s=elapsed[0], optimized_wall_s=elapsed[1],
            old_process_cpu_s=cpu[0], optimized_process_cpu_s=cpu[1],
            alternating_order=True, real_new_routes_in_timed_kernel=0,
            optimized_parse_cache_hits=sides[1][0].diagnostics()["join_parse_cache_hits"],
            speed_ratio=elapsed[0]/elapsed[1], whole_day_speed_ratio_not_identified=True)
    finally:
        for _, _, eta, _ in sides:
            eta.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-commit", help="Use immutable baseline Git code with the existing data root.")
    args = parser.parse_args(argv)
    root = args.reference_root.resolve()
    if psutil.virtual_memory().available < 1536 * 2**20:
        raise MemoryError("connector check requires 1536 MiB available RAM")
    config_path = root / "stage4/config/scheme_a_city_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    path = root / config["source_history_templates"]
    frame = pd.read_parquet(path, columns=list(CoarseHistoricalLibrary._COLUMNS))
    candidates = frame.loc[frame.compatible_C].sort_values(["date", "order_id"], kind="stable")
    source_rows = candidates.loc[candidates.date.astype(str).eq(config["forecast_train_dates"][0])].head(4)
    target_rows = list(candidates.itertuples(index=False))
    pairs, used = [], {}
    def task(row):
        key = str(row.date), str(row.order_id)
        if key not in used:
            rid = 1000000 + len(used)
            used[key] = ChainTask(rid, 61200., 63900., float(row.predicted_route_time_p50_s),
                (float(row.start_lon_wgs84), float(row.start_lat_wgs84)),
                (float(row.end_lon_wgs84), float(row.end_lat_wgs84)), frozenset({"HV", "C"}),
                source_date=key[0], source_order_id=key[1])
        return used[key]
    for source in source_rows.itertuples(index=False):
        origin = np.asarray(_xy((source.end_lon_wgs84, source.end_lat_wgs84)))
        nearest = []
        for target in target_rows:
            if (str(source.date), str(source.order_id)) == (str(target.date), str(target.order_id)):
                continue
            gap = float(np.linalg.norm(origin - np.asarray(_xy((target.start_lon_wgs84, target.start_lat_wgs84)))))
            if gap <= 2000.:
                nearest.append((gap, str(target.date), str(target.order_id), target))
        nearest.sort(key=lambda item: item[:3])
        pairs.extend((task(source), task(target[-1])) for target in nearest[:2])
    if len(pairs) != 8:
        raise ValueError("the fixed eight-connector engineering workload is incomplete")
    templates = candidates.loc[[((str(row.date), str(row.order_id)) in used)
                               for row in candidates.itertuples(index=False)]].copy()
    del frame, candidates, target_rows
    gc.collect()
    baseline = load_reference("_engineering_reference_connections", root / "stage4/dispatch/scheme_a_city_connections.py",
        root=root, commit=args.reference_commit)
    old = run_side(baseline, root, pairs, templates)
    old_outputs = old.pop("outputs")
    old_joins = old.pop("direct_join_outputs")
    gc.collect()
    new = run_side(optimized, root, pairs, templates)
    new_outputs = new.pop("outputs")
    new_joins = new.pop("direct_join_outputs")
    if old_outputs != new_outputs:
        raise AssertionError("optimized real connectors changed support, attribution, time or distance")
    if old_joins != new_joins:
        raise AssertionError("static input cache changed full directed join parser output")
    gc.collect()
    warm = compare_warm_parser(baseline, root, pairs, templates)
    info = psutil.Process().memory_info()
    outcomes = Counter("SUPPORTED" if row["supported"] else "REJECTED" for row in new_outputs[:8])
    reasons = Counter(reason for row in new_outputs[:8] for reason in row["reason_codes"])
    result = dict(status="REAL_CONNECTOR_EQUIVALENCE_PASS", native_simulation=False,
        whole_day_experiment=False, fixed_pairs=8, visits_per_pair=2,
        source_scope="EARLIER_HISTORY_IDENTITIES_ONLY_NO_TEST31_REQUESTS_OR_TRUTH",
        time_window_is_engineering_fixture_not_reconstructed_request_times=True,
        direct_parser_outputs_equal=True, direct_parser_checks=16,
        direct_parser_invocation_is_not_an_eligibility_override=True,
        outcomes=dict(outcomes), rejection_reasons=dict(reasons), old=old, optimized=new,
        alternating_warm_parser=warm,
        input_sha256=dict(config=sha(config_path), templates=sha(path)),
        peak_rss_mib=getattr(info, "peak_wset", info.rss)/2**20,
        cold_cache_time_comparison_not_a_full_day_speed_estimate=True)
    atomic_json(args.output, result)
    print(json.dumps(result, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
