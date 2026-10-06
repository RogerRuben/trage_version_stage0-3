"""Historical-chain/ETA diagnostics and one-reference synthesis; no tuning."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa

from stage3.scripts.traffic_state_batch1 import sha, write_json, write_parquet
from stage4.analysis.replay_calibration_contract import CONFIG, load_calibration
from stage4.dispatch.acceptance import passenger_acceptance
from stage4.dispatch.candidate_graph import SpatialVehicle
from stage4.dispatch.deterministic_routing import SINGLE_SOURCE_MATRIX
from stage4.dispatch.routing_v4 import StaticRawRoutingAdapter
from stage4.replay_foundation import FULL_ORDERS_REL, load_full_test31_orders

REFERENCE = Path("stage4/output/symmetric_flexibility_v1/accelerated_v4_full_day/SERVICE_PRESERVING_LOOKAHEAD")
ASSETS = Path("stage4/output/runtime_acceleration_v2/assets")
CALIBRATION = Path("stage4/input/replay_foundation/pickup_eta_calibration_15min.parquet")


def distribution(values):
    s = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return {"count": int(len(s)), "mean": float(s.mean()) if len(s) else None,
        **{f"p{int(q*100):02d}": float(s.quantile(q)) if len(s) else None for q in (0, .25, .5, .75, .9, 1)}}


def chain_pairs(full, max_gap_s):
    """Immediate full-history predecessor, with cumulative-overlap protection."""
    keys = ["order_id", "driver_id", "request_time", "arrival_time",
        "start_lon_wgs84", "start_lat_wgs84", "end_lon_wgs84", "end_lat_wgs84"]
    valid = full["valid_session_row"].fillna(False) if "valid_session_row" in full else np.ones(len(full), bool)
    work = full.loc[valid, keys].sort_values(["driver_id", "request_time", "order_id"], kind="mergesort").copy()
    grouped = work.groupby("driver_id", sort=False)
    for source, dest in {"order_id": "previous_order_id", "arrival_time": "previous_arrival_time",
        "end_lon_wgs84": "previous_lon_wgs84", "end_lat_wgs84": "previous_lat_wgs84"}.items():
        work[dest] = grouped[source].shift()
    previous_running_end = grouped.arrival_time.cummax().groupby(work.driver_id, sort=False).shift()
    work["inter_trip_gap_s"] = (work.request_time - work.previous_arrival_time).dt.total_seconds()
    work["gap_from_running_end_s"] = (work.request_time - previous_running_end).dt.total_seconds()
    starts = previous_running_end.isna() | work.gap_from_running_end_s.gt(max_gap_s)
    seq = starts.groupby(work.driver_id, sort=False).cumsum().astype(int)
    work["source_session_id"] = work.driver_id.astype(str) + "__S" + seq.astype(str).str.zfill(3)
    work["has_predecessor"] = work.previous_order_id.notna()
    work["history_overlap"] = work.has_predecessor & work.gap_from_running_end_s.lt(0)
    work["session_break"] = work.has_predecessor & work.gap_from_running_end_s.gt(max_gap_s)
    coords = work[["previous_lon_wgs84", "previous_lat_wgs84", "start_lon_wgs84", "start_lat_wgs84"]].to_numpy(float)
    geographic = np.isfinite(coords).all(axis=1)
    geographic &= (np.abs(coords[:, [0, 2]]) <= 180).all(axis=1)
    geographic &= (np.abs(coords[:, [1, 3]]) <= 90).all(axis=1)
    work["coordinate_valid"] = geographic
    work["chain_routing_eligible"] = (work.has_predecessor & ~work.history_overlap & ~work.session_break
        & work.inter_trip_gap_s.between(0, max_gap_s) & work.coordinate_valid)
    lon1, lat1, lon2, lat2 = np.deg2rad(coords).T
    h = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    work["empty_chord_distance_m"] = 12_742_017.6 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    return work


def witness_flags(gap, raw, corrected, patience=300.0):
    valid = all(np.isfinite(v) for v in (gap, raw, corrected)) and gap >= 0 and raw >= 0 and corrected >= 0
    return dict(raw_fits_historical_gap=bool(valid and raw <= gap),
        corrected_fits_historical_gap=bool(valid and corrected <= gap),
        timing_idle_hold_conflict=bool(valid and corrected <= gap and corrected > patience),
        raw_timing_idle_hold_conflict=bool(valid and raw <= gap and raw > patience),
        beta_only_gap_conflict=bool(valid and raw <= gap and corrected > gap),
        beta_only_patience_conflict=bool(valid and raw <= patience and corrected > patience))


def stable_sample(frame, size, seed):
    work = frame.copy()
    work["sample_hash"] = work.order_id.map(lambda o: hashlib.sha256(f"{seed}|{o}".encode()).hexdigest())
    return work.sort_values(["sample_hash", "order_id"], kind="mergesort").head(size).reset_index(drop=True)


def dispatch_diagnostics(root, directory):
    outcomes = pd.read_parquet(directory / "cohort_outcomes.parquet")
    assets = np.load(root / ASSETS / "orders.npy", mmap_mode="r", allow_pickle=False)
    mask = pd.DataFrame({"order_id": assets["order_id"], "compatible_M": (assets["mask"] & 2) != 0})
    work = outcomes.merge(mask, on="order_id", validate="one_to_one")
    work = work.loc[work.common_eligible].copy()
    accepts = [passenger_acceptance(str(o), .7, 20260827).passenger_accepts_av for o in work.order_id]
    work["mixed_av_eligible"] = work.compatible_M & np.asarray(accepts, dtype=bool)
    groups = []
    for name, flag in (("M_P70_AV_ELIGIBLE", work.mixed_av_eligible), ("M_P70_HV_ONLY", ~work.mixed_av_eligible)):
        g = work.loc[flag]
        groups.append(dict(group=name, orders=len(g), matched=int(g.matched.sum()),
            expired=int(g.expired.sum()), service_rate=float(g.matched.mean())))
    epochs = pd.read_parquet(directory / "epochs.parquet")
    columns = ["candidate_topk_pairs", "valid_or_arcs", "patience_arc_exclusions",
        "hv_window_arc_exclusions", "zero_eta_session_certified_prunes"]
    totals = {c: int(epochs[c].sum()) for c in columns}
    totals["candidate_partition_reconciled"] = totals["candidate_topk_pairs"] == sum(totals[c] for c in columns[1:])
    a = pd.read_parquet(directory / "assignments.parquet")
    start = pd.Timestamp("2016-10-31T00:00:00+08:00")
    fleet_path = directory / "fleet_fixtures.json"
    if fleet_path.is_file():
        fixtures = pd.DataFrame(json.loads(fleet_path.read_text())["fixtures"]).set_index("native_id")
    else:
        fixtures = pd.DataFrame(json.loads((root / ASSETS / "fleet.json").read_text())["fixtures"]).set_index("native_id")
    use = []
    for kind, g in a.groupby("vehicle_type"):
        fs = fixtures.loc[fixtures.vehicle_type.eq(kind)]
        lo = ((pd.to_datetime(fs.availability_start_time)-start).dt.total_seconds()).clip(0, 86400)
        hi = ((pd.to_datetime(fs.availability_end_time)-start).dt.total_seconds()).clip(0, 86400)
        hours = float((hi-lo).clip(lower=0).sum()/3600)
        vid = g.native_vehicle_id.to_numpy()
        begin = np.maximum(g.simulation_time_s.to_numpy(), lo.reindex(vid).to_numpy())
        end = np.minimum((pd.to_datetime(g.service_end_time)-start).dt.total_seconds().to_numpy(), hi.reindex(vid).to_numpy())
        occupied = float(np.maximum(0, end-begin).sum()/3600)
        use.append(dict(vehicle_type=kind, available_hours_clipped=hours,
            occupied_pickup_service_hours_clipped=occupied, occupied_share=occupied/hours))
    return dict(cohort_groups=groups, candidate_visits=totals,
        available_vehicles=distribution(epochs.available_vehicles), waiting_orders=distribution(epochs.waiting_orders),
        occupied_hours=use, assigned_beta=distribution(a.beta),
        expiry_is_terminal_state_not_disjoint_root_cause=True)


def diagnostic(root, finalize_existing=False):
    spec, config_path = load_calibration(root, CONFIG)
    out, docs = root / spec["diagnostic_output"], root / spec["doc_output"]
    inputs = {str(p): sha(root/p) for p in (CONFIG, FULL_ORDERS_REL, CALIBRATION,
        REFERENCE/"summary.json", ASSETS/"orders.npy")}
    if (out / "summary.json").is_file():
        done = json.loads((out / "summary.json").read_text())
        if done["status"] != "COMPLETE" or done["inputs_sha256"] != inputs:
            raise ValueError("existing diagnostic is incomplete or from different inputs; no automatic retry")
        print(json.dumps(dict(status="REUSED_COMPLETE_DIAGNOSTIC")), flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    full, raw_diagnostics = load_full_test31_orders(root, spec["test_date"])
    pairs = chain_pairs(full, spec["chain_max_gap_s"])
    del full
    outcomes = pd.read_parquet(root / REFERENCE / "cohort_outcomes.parquet")
    core = pairs.merge(outcomes, on="order_id", validate="one_to_one")
    arr = np.load(root / ASSETS / "orders.npy", mmap_mode="r", allow_pickle=False)
    clocks = pd.DataFrame({"order_id": arr["order_id"], "canonical_request_time": pd.to_datetime(arr["request_ns"], utc=True)})
    core = core.merge(clocks, on="order_id", validate="one_to_one")
    core["request_proxy_shift_s"] = (core.canonical_request_time-core.request_time).dt.total_seconds()
    core["previous_in_common_cohort"] = core.previous_order_id.isin(outcomes.loc[outcomes.common_eligible, "order_id"])
    mixed_fixtures = json.loads((root / ASSETS / "fleet.json").read_text())["fixtures"]
    represented = {f["source_session_id"] for f in mixed_fixtures if f["vehicle_type"] == "HV"}
    core["historical_session_in_mixed_hv_fleet"] = core.source_session_id.isin(represented)
    write_parquet(out / "common_order_chains.parquet", core)
    candidates = core.loc[core.common_eligible & core.chain_routing_eligible].copy()
    sample = stable_sample(candidates, spec["chain_sample_size"], spec["chain_sample_seed"])
    sample["route_status"] = "PENDING"
    for flag in witness_flags(1., 0., 0.):
        sample[flag] = pd.Series(pd.NA, index=sample.index, dtype="boolean")
    for key in ("raw_empty_eta_s", "corrected_empty_eta_s", "beta", "empty_route_distance_m"):
        sample[key] = np.nan
    sample_hash = hashlib.sha256("\n".join(sample.order_id).encode()).hexdigest()
    previous_route_runtime = None
    previous_route_observed_rss = None
    router = None
    if finalize_existing:
        saved = pd.read_parquet(out / "routed_chain_sample.parquet")
        if (saved.order_id.tolist() != sample.order_id.tolist()
                or saved.route_status.eq("PENDING").any()):
            raise ValueError("only the fully completed identical routed sample can be finalized")
        check = ["order_id", "previous_order_id", "previous_lon_wgs84", "previous_lat_wgs84",
            "start_lon_wgs84", "start_lat_wgs84", "canonical_request_time", "inter_trip_gap_s"]
        pd.testing.assert_frame_equal(sample[check], saved[check], check_dtype=False)
        sample = saved
        previous_progress = json.loads((out/"progress.json").read_text())
        previous_route_runtime = previous_progress["runtime_s"]
        previous_route_observed_rss = previous_progress["rss_mib"]
    else:
        router = StaticRawRoutingAdapter(root, routing_mode=SINGLE_SOURCE_MATRIX, route_workers=1,
            persistent_cache_size=2000, raw_od_cache_size=2000, grouped_sources=True,
            certified_eta_pruning=False, route_queue_chunk_size=64)
    try:
        for offset in range(0, 0 if finalize_existing else len(sample), 64):
            chunk = sample.iloc[offset:offset+64]
            batches = [([SpatialVehicle(f"CHAIN_{offset+i}", offset+i, "HV", float(r.previous_lon_wgs84), float(r.previous_lat_wgs84))],
                float(r.start_lon_wgs84), float(r.start_lat_wgs84), pd.Timestamp(r.canonical_request_time))
                for i, r in enumerate(chunk.itertuples(index=False))]
            answers = router.estimate_epoch(batches)
            for i, (r, answer) in enumerate(zip(chunk.itertuples(index=False), answers)):
                index = offset+i
                estimate = answer.get(index)
                if estimate is None:
                    sample.loc[index, "route_status"] = "FAILED"
                    continue
                sample.loc[index, "route_status"] = "OK"
                for key, value in dict(raw_empty_eta_s=estimate.valhalla_time_s,
                    corrected_empty_eta_s=estimate.corrected_pickup_eta_s, beta=estimate.beta,
                    empty_route_distance_m=estimate.route_distance_m,
                    **witness_flags(r.inter_trip_gap_s, estimate.valhalla_time_s, estimate.corrected_pickup_eta_s)).items():
                    sample.loc[index, key] = value
            write_parquet(out / "routed_chain_sample.parquet", sample)
            progress = dict(status="RUNNING", routed=min(offset+64,len(sample)), total=len(sample),
                runtime_s=time.perf_counter()-started, rss_mib=psutil.Process().memory_info().rss/2**20)
            write_json(out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    finally:
        if router is not None:
            router.close()
    # Keep raw order/driver identities and coordinates local, under ignored
    # output products. Only aggregate reports accompany the code push.
    sample.to_csv(out / "routed_chain_sample.csv", index=False)
    ok = sample.loc[sample.route_status.eq("OK")].copy()
    flags = list(witness_flags(1., 0., 0.))
    sample_groups = []
    for name, g in (("ALL_SAMPLE_OK", ok), ("MIXED_SERVED", ok.loc[ok.matched]), ("MIXED_EXPIRED", ok.loc[ok.expired])):
        sample_groups.append(dict(group=name, n=len(g), **{k:int(g[k].fillna(False).astype(bool).sum()) for k in flags}))
    result = dict(status="COMPLETE", created_at=datetime.now(timezone.utc).isoformat(),
        inputs_sha256=inputs, raw_diagnostics=raw_diagnostics,
        full_history_valid_orders=len(pairs), full_history_predecessor_count=int(pairs.has_predecessor.sum()),
        full_history_overlap_count=int(pairs.history_overlap.sum()),
        full_history_session_break_count=int(pairs.session_break.sum()),
        common_order_count=int(core.common_eligible.sum()),
        common_no_predecessor_count=int((core.common_eligible & ~core.has_predecessor).sum()),
        common_overlap_count=int((core.common_eligible & core.history_overlap).sum()),
        common_session_break_count=int((core.common_eligible & core.session_break).sum()),
        common_chain_candidate_count=len(candidates),
        predecessor_outside_common_share=float((~candidates.previous_in_common_cohort).mean()),
        historical_session_retained_mixed_hv_share=float(core.loc[core.common_eligible,"historical_session_in_mixed_hv_fleet"].mean()),
        request_proxy_shift_s=distribution(core.request_proxy_shift_s),
        sampled=len(sample), sample_seed=spec["chain_sample_seed"], sample_order_ids_sha256=sample_hash,
        route_success_count=len(ok), route_failure_count=int(sample.route_status.eq("FAILED").sum()),
        sample_groups=sample_groups, gap_s=distribution(ok.inter_trip_gap_s),
        empty_distance_m=distribution(ok.empty_route_distance_m), raw_eta_s=distribution(ok.raw_empty_eta_s),
        corrected_eta_s=distribution(ok.corrected_empty_eta_s),
        beta_identified_from_actual_empty_pickup_labels=False,
        gap_is_idle_plus_empty_upper_budget_not_observed_pickup_time=True,
        source_beta_definition="median(realized_loaded_OD_time / Valhalla_loaded_OD_time) by 15-minute bin",
        canonical_mixed=dispatch_diagnostics(root, root/REFERENCE),
        runtime_s=(previous_route_runtime or 0.)+time.perf_counter()-started,
        completed_route_stage_runtime_s=previous_route_runtime,
        summary_finalization_runtime_s=time.perf_counter()-started if finalize_existing else None,
        summary_finalized_from_completed_routing=finalize_existing,
        runtime_definition="sum of measured completed route stage and aggregation-only finalization" if finalize_existing else "single process monotonic",
        peak_rss_mib=max(previous_route_observed_rss or 0., psutil.Process().memory_info().peak_wset/2**20),
        rss_measurement="max completed-route checkpoint observation and finalizer process lifetime peak" if finalize_existing else "process lifetime peak",
        gpu_used=False, dense_matrix=False, frozen_beta_changed=False)
    if inputs != {p:sha(root/p) for p in inputs}:
        raise ValueError("diagnostic source changed during execution")
    write_json(out / "summary.json", result)
    write_json(docs / "diagnostic_summary.json", result)
    print(json.dumps(result), flush=True)


def additional_offline_checks(root, diagnostic_directory):
    """Read existing products only; no inference, routes or simulation."""
    sample = pd.read_parquet(diagnostic_directory/"routed_chain_sample.parquet")
    subset = sample.loc[sample.previous_in_common_cohort & sample.route_status.eq("OK")]
    qualified = {"n":len(subset), **{k:int(subset[k].fillna(False).sum()) for k in witness_flags(1.,0.,0.)}}
    totals = []
    for path in sorted(root.glob("stage1/input_v1/split=test/date=20161031/bucket=*/link_traversals.parquet")):
        frame = pd.read_parquet(path, columns=["order_id","allocated_distance_m"])
        totals.append(frame.groupby("order_id").allocated_distance_m.sum())
    distance = pd.concat(totals).groupby(level=0).sum().rename("historical_route_distance_m")
    auto = pd.read_parquet(root/"stage4/input/replay_foundation/historical_valhalla_auto_eta.parquet",
        columns=["order_id","valhalla_route_distance_m","valhalla_route_time_s"])
    actual = pd.read_parquet(root/"stage4/input/replay_foundation/stage4_order_replay_base.parquet",
        columns=["order_id","realized_service_time_s"]).drop_duplicates("order_id")
    loaded = auto.merge(distance.reset_index(),on="order_id",validate="one_to_one").merge(actual,on="order_id",validate="one_to_one")
    ratio = loaded.historical_route_distance_m/loaded.valhalla_route_distance_m
    sessions = pd.read_parquet(root/"stage4/input/replay_foundation/full_test31_driver_sessions.parquet",
        columns=["session_id","session_end_time","last_order_id"])
    core = pd.read_parquet(diagnostic_directory/"common_order_chains.parquet")
    core = core.loc[core.common_eligible].copy()
    core = core.merge(sessions,left_on="source_session_id",right_on="session_id",validate="many_to_one")
    arr = np.load(root/ASSETS/"orders.npy",mmap_mode="r",allow_pickle=False)
    pred = pd.DataFrame({"order_id":arr["order_id"],"m3_p50_s":arr["predicted"]})
    core = core.merge(pred,on="order_id",validate="one_to_one")
    core["zero_pickup_own_session_forbidden"] = core.canonical_request_time + pd.to_timedelta(core.m3_p50_s,unit="s") > core.session_end_time
    template = pd.read_parquet(root/"stage4/input/replay_foundation/replay_fleet_template.parquet",columns=["source_session_id"])
    own = core.loc[core.source_session_id.isin(template.source_session_id)]
    last = core.loc[core.order_id.eq(core.last_order_id)]
    return dict(both_common_quality_predecessor_sample=qualified,
        loaded_route_distance_ratio=distribution(ratio), loaded_distance_order_count=len(loaded),
        loaded_distance_ratio_gt15_share=float(ratio.gt(1.5).mean()),
        loaded_time_ratio=distribution(loaded.realized_service_time_s/loaded.valhalla_route_time_s),
        route_length_adjusted_time_ratio=distribution((loaded.realized_service_time_s/loaded.valhalla_route_time_s)/ratio),
        original_session_represented_all_hv_count=len(own),
        original_session_represented_all_hv_share=float(len(own)/len(core)),
        zero_pickup_predicted_end_forbidden_all_common=int(core.zero_pickup_own_session_forbidden.sum()),
        zero_pickup_predicted_end_forbidden_retained_session=int(own.zero_pickup_own_session_forbidden.sum()),
        common_historical_last_order_count=len(last),
        zero_pickup_predicted_end_forbidden_last_order=int(last.zero_pickup_own_session_forbidden.sum()),
        historical_last_order_prediction_test_not_dispatch_loss_attribution=True)


def report(root):
    spec, _ = load_calibration(root, CONFIG)
    docs = root/spec["doc_output"]
    diag = json.loads((docs/"diagnostic_summary.json").read_text())
    mixed = json.loads((root/REFERENCE/"summary.json").read_text())
    hv_dir = root/spec["all_hv_output"]/spec["reference_policy"]
    hv = json.loads((hv_dir/"summary.json").read_text())
    if hv["status"] != "COMPLETE" or diag["status"] != "COMPLETE":
        raise ValueError("reference or diagnostic did not complete")
    shared_inputs = {str(p).replace('\\','/'):v for p,v in mixed["inputs_sha256"].items()}
    hv_inputs = {str(p).replace('\\','/'):v for p,v in hv["inputs_sha256"].items()}
    if any(hv_inputs.get(p) != digest for p,digest in shared_inputs.items()):
        raise ValueError("shared frozen inputs differ between all-HV and mixed reference")
    extra = additional_offline_checks(root, root/spec["diagnostic_output"])
    write_json(docs/"additional_checks.json",extra)
    a = pd.read_parquet(root/REFERENCE/"cohort_outcomes.parquet")
    b = pd.read_parquet(hv_dir/"cohort_outcomes.parquet")
    paired = a.merge(b,on="order_id",suffixes=("_mixed","_hv"),validate="one_to_one")
    if len(paired) != 30000 or not paired.common_eligible_mixed.equals(paired.common_eligible_hv):
        raise ValueError("all-HV reference population differs")
    common = paired.matched_mixed & paired.matched_hv
    supply = []
    origin = pd.Timestamp("2016-10-31T00:00:00+08:00")
    mixed_fleet = json.loads((root/ASSETS/"fleet.json").read_text())["fixtures"]
    all_hv_fleet = json.loads((hv_dir/"fleet_fixtures.json").read_text())["fixtures"]
    for label, fixtures in (("MIXED", mixed_fleet), ("ALL_HV", all_hv_fleet)):
        frame = pd.DataFrame(fixtures)
        begin = (pd.to_datetime(frame.availability_start_time)-origin).dt.total_seconds().to_numpy()
        end = (pd.to_datetime(frame.availability_end_time)-origin).dt.total_seconds().to_numpy()
        for hour in range(24):
            h = np.maximum(0., np.minimum(end,(hour+1)*3600)-np.maximum(begin,hour*3600))
            supply.append(dict(scenario=label, hour=hour, available_vehicle_hours=float(h.sum()/3600),
                hv_vehicle_hours=float(h[frame.vehicle_type.eq("HV")].sum()/3600),
                av_vehicle_hours=float(h[frame.vehicle_type.eq("AV")].sum()/3600)))
    pd.DataFrame(supply).to_csv(docs/"hourly_supply.csv", index=False)
    supply_frame = pd.DataFrame(supply).pivot(index="hour",columns="scenario",values="available_vehicle_hours")
    daytime_hv = float(supply_frame.loc[8:22,"ALL_HV"].sum())
    daytime_mixed = float(supply_frame.loc[8:22,"MIXED"].sum())
    reference = dict(status="COMPLETE", mixed_reference=str(REFERENCE), all_hv_reference=str(hv_dir.relative_to(root)),
        controlled_change="fleet composition/layout/availability plus AV compatibility/acceptance; not pure ODD or policy causal attribution",
        gained_all_hv=int((~paired.matched_mixed & paired.matched_hv).sum()),
        lost_all_hv=int((paired.matched_mixed & ~paired.matched_hv).sum()),
        common_served=int(common.sum()), matched_difference=int(hv["matched"]-mixed["matched"]),
        service_rate_difference_pp=100*(hv["service_rate_common_population"]-mixed["service_rate_common_population"]),
        paired_served_mean_wait_delta_s=float((paired.loc[common,"wait_s_hv"]-paired.loc[common,"wait_s_mixed"]).mean()),
        all_hv=hv, mixed=mixed, hourly_supply=supply, additional_offline_checks=extra,
        shared_input_hashes_identical=True, shared_input_count=len(shared_inputs),
        daytime_08_to_23_hv_vehicle_hours=daytime_hv,
        daytime_08_to_23_mixed_vehicle_hours=daytime_mixed,
        daytime_mixed_supply_deficit_share=1-daytime_mixed/daytime_hv,
        all_hv_diagnostics=dispatch_diagnostics(root,hv_dir),
        historical_service_rate_calibrated=False, no_new_timing_or_reposition_scenarios=True)
    write_json(docs/"comparison.json",reference)
    all_group = diag["sample_groups"][0]
    n = all_group["n"]
    lines = ["# 历史订单回放绝对校准诊断 v1", "", "## Material Passport", "",
        "- Origin Skill: academic-research-suite / experiment-agent + directed literature lookup",
        "- Origin Date: 2026-10-06", "- Verification Status: ANALYZED; native reference executed, not a reproduction of real dispatch",
        "- Version Label: replay_calibration_v1", "", "## 1. 当前同口径纯HV结果", "",
        "| 指标 | 冻结M/q=.5/P=.7 LOOKAHEAD | 当前纯HV LOOKAHEAD |", "|---|---:|---:|"]
    for name,key in (("共同订单","common_eligible"),("服务订单","matched"),("超时订单","expired")):
        lines.append(f"| {name} | {mixed[key]:,} | {hv[key]:,} |")
    lines += [f"| 共同人口服务率 | {mixed['service_rate_common_population']:.2%} | {hv['service_rate_common_population']:.2%} |",
        f"| 原始30,000单口径 | {mixed['service_rate_original_30000']:.2%} | {hv['service_rate_original_30000']:.2%} |",
        f"| 已服务者平均等待（秒） | {mixed['mean_wait_s']:.2f} | {hv['mean_wait_s']:.2f} |", "",
        f"纯HV新增服务{reference['gained_all_hv']:,}单、失去{reference['lost_all_hv']:,}单，净变化{reference['matched_difference']:+,}单，共同口径变化{reference['service_rate_difference_pp']:+.3f}个百分点。",
        "这是车队组成与可替代性组合参考；不是单独ODD、接受率或派单算法的因果效应。纯HV仍不是原历史车队逐司机订单链回放。", "",
        f"两组共同被服务{reference['common_served']:,}单的平均等待差（纯HV−混合）为{reference['paired_served_mean_wait_delta_s']:+.2f}秒；它仍是条件于两组均服务的描述性比较。",
        "全天总车时对齐不等于逐小时供给对齐：混合场景将一部分日间HV班次车时换成256辆全天AV车时。", "",
        "| 时段 | 纯HV可用车时 | 混合可用车时 | 混合相对变化 |", "|---|---:|---:|---:|"]
    for hour in (5,15,18):
        h,m=float(supply_frame.loc[hour,"ALL_HV"]),float(supply_frame.loc[hour,"MIXED"])
        lines.append(f"| {hour:02d}:00–{hour+1:02d}:00 | {h:.2f} | {m:.2f} | {m/h-1:+.2%} |")
    lines += ["",f"08:00–23:00纯HV可用车时{daytime_hv:.2f}，混合{daytime_mixed:.2f}，混合少{1-daytime_mixed/daytime_hv:.2%}；全部24小时见hourly_supply.csv。它是q实验内生的时段供给重分配，不是纯能力约束效应，也不是小于0.2%的全天归一化误差。",
        "## 2. 历史链与接驾倍率", "",
        f"前序构建使用{diag['full_history_valid_orders']:,}条有效原始订单，而不是只用30,000条研究订单。共同人口中{diag['common_no_predecessor_count']:,}单没有当日已观测前序，{diag['common_overlap_count']:,}单历史时间重叠，{diag['common_session_break_count']:,}单跨90分钟session边界。",
        f"同session且无重叠候选{diag['common_chain_candidate_count']:,}对，其中前序不在研究共同人口的比例{diag['predecessor_outside_common_share']:.2%}。",
        f"固定种子、与结果无关的哈希抽样{diag['sampled']:,}对；成功路由{n:,}对，失败{diag['route_failure_count']:,}对。", "",
        "| 诊断见证（可重叠，非因果分摊） | 成功路由样本数 | 比例 |", "|---|---:|---:|"]
    for name,key in (("校准ETA可放进历史间隔，但超过300秒","timing_idle_hold_conflict"),
        ("原始ETA可放进历史间隔，但校准后不行","beta_only_gap_conflict"),
        ("原始ETA≤300秒，校准后超过300秒","beta_only_patience_conflict"),
        ("原始ETA已超过历史间隔","raw_fits_historical_gap")):
        count = n-all_group[key] if key == "raw_fits_historical_gap" else all_group[key]
        lines.append(f"| {name} | {count:,} | {count/n:.2%} |")
    lines += ["", "历史两单间隔包含未知空驶、停车或其他未观测活动，是可用时间上界，不是真实接驾标签。以上只证明当前时间/空间重构与一部分历史连接存在冲突，不能直接声称这些订单会被新策略救回。",
        f"前后两单均在高质量共同人口的子集为{extra['both_common_quality_predecessor_sample']['n']}对，其中{extra['both_common_quality_predecessor_sample']['timing_idle_hold_conflict']}对仍出现校准ETA在历史间隔内、却超过300秒的冲突；该子集不重新路由。",
        "冻结beta从载客OD真实耗时与Valhalla OD耗时比值拟合；它同时可能包含拥堵、实际与最短路线差异及路网误差。当前没有独立空驶接驾标签，不能据此宣布beta过高或将其改为1。", "",
        f"30,000单匹配物理距离/Valhalla载客OD距离的中位数为{extra['loaded_route_distance_ratio']['p50']:.4f}；距离比超过1.5的比例为{extra['loaded_distance_ratio_gt15_share']:.2%}。整体约2.4的时长倍率不能简单解释为整体约2.4倍路线绕行。",
        f"即使假设原司机在该单历史起点、接驾时间为0，共同人口中仍有{extra['zero_pickup_predicted_end_forbidden_all_common']:,}单因M3 P50完成时刻超过推断session结束而被规则判不可接。其中历史session末单{extra['common_historical_last_order_count']:,}单中有{extra['zero_pickup_predicted_end_forbidden_last_order']:,}单出现此现象。它是预测与推断班次硬边界的冲突见证，不是实际超时订单的因果分摊。", "",
        "## 3. 供给与候选流失", ""]
    for name,d in (("混合",diag['canonical_mixed']),("纯HV",reference['all_hv_diagnostics'])):
        visits=d['candidate_visits']
        lines.append(f"{name}：Top-K候选访问{visits['candidate_topk_pairs']:,}次；耐心排除{visits['patience_arc_exclusions']:,}次；零接驾session预排除{visits['zero_eta_session_certified_prunes']:,}次；可行边{visits['valid_or_arcs']:,}次。分区对账={visits['candidate_partition_reconciled']}。这些是重复候选×epoch访问，不是独立订单拒绝原因。")
        for u in d['occupied_hours']:
            lines.append(f"- {name}/{u['vehicle_type']}：日内可用车时{u['available_hours_clipped']:.2f}，接驾+载客占用{u['occupied_pickup_service_hours_clipped']:.2f}，占比{u['occupied_share']:.2%}。")
        lines.append("")
    lines += ["## 4. 执行与约束", "",
        f"离线诊断实际计算耗时{diag['runtime_s']:.2f}秒（完整路由阶段与聚合修复后finalization的测量值之和），观测最大RSS {diag['peak_rss_mib']:.2f} MiB（{diag['rss_measurement']}）。纯HV实际耗时{hv['runtime_s']/60:.2f}分钟，进程组峰值RSS {hv['peak_process_group_rss_mib']:.2f} MiB；private committed {hv.get('peak_process_group_private_committed_mib',0):.2f} MiB。GPU未用、无稠密订单×车辆矩阵。",
        f"纯HV执行代码SHA `{hv['execution_code_sha']}`；路由失败={hv['routing_failures']}；资源降级={hv['resource_fallback_epochs']}；当前服务面违规={hv['current_face_violation_epochs']}；物理对账={hv['physical_reconciliation']}。",
        f"两组{len(shared_inputs)}个共享输入SHA一致。纯HV标签M仅复用已有路线/预测产品，HV不受M适配或AV接受率限制。",
        "请求时刻、300秒耐心、ETA倍率、M3、C/M/A定义、重定位与旧产物未更改。仅新增一组纯HV参考，不自动重试、不参数搜索。", "",
        "## 5. 文献与后续设计", "", "参见 [定向文献来源](literature_sources.md)。根据本轮实测与一手来源形成的建议在单独的 conclusions.md 中列出，不将建议误写为已经执行的场景。", "",
        "## 6. 解释风险（11/11检查）", "",
        "选择/幸存者风险：样本仅含历史完成订单；聚合风险：车时不等于局部匹配能力，重复arc计数不等于订单数；条件等待样本风险：只服务者的平均等待不是全体请求福利；因果风险：同一天/seed与组合车队改变不支持现实因果分解。Simpson/生态/Berkson/碰撞点/基率/均值回归/幸存者/look-elsewhere/forking paths/相关因果/反向因果均已检查；本报告不做显著性或因果效应宣称。", "",
        "## 文件", "", "- diagnostic_summary.json：完整离线计数、输入来源与分组。",
        "- 本地样本CSV：`stage4/output/replay_calibration_v1/diagnostic/routed_chain_sample.csv`（含原始身份/坐标，不推送Git）。", "- all_hv_summary.json：真实纯HV执行结果。",
        "- comparison.json：同口径订单对比及候选/占用诊断。", "", "本报告由AI辅助代码分析与来源检索生成，最终科研判断仍由作者作出。", ""]
    (docs/"report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k:reference[k] for k in ('status','matched_difference','service_rate_difference_pp','gained_all_hv','lost_all_hv')}),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("diagnostic","report"),required=True)
    parser.add_argument("--finalize-existing-sample", action="store_true",
        help="Aggregation only from an identical fully routed sample; never routes or retries an arc")
    args=parser.parse_args()
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    if args.phase == "diagnostic":
        diagnostic(Path.cwd(), args.finalize_existing_sample)
    else:
        report(Path.cwd())
