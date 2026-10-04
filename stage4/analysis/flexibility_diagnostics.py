"""One-pass existing-product diagnosis and Train-only remaining-time adapter.

No native replay, M3 inference, demand fitting, clustering, or HTTP. Full Test31
is read only for attribution and retrospective reporting, never for model fit.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from stage3.odd_tod.research_compatibility import evaluate_research_compatibility
from stage3.scripts.traffic_state_batch1 import sha, write_json, write_parquet
from stage4.analysis.flexibility_prepare import CONFIG, OUT as INPUT, S4, MovementClassifier
from stage4.analysis.flexibility_windows import OUTPUT as OLD_WINDOWS

DOC = Path("stage4/docs/flexibility_dispatch/service_preserving")
OUTPUT = Path("stage4/output/flexibility_service_preserving_v1")
NEW_CONFIG = Path("stage4/config/flexibility_service_preserving_v1.json")
MODEL = DOC / "train_remaining_time_model.json"
NC = dict(NC=1.0, ML=0.0, MR=0.0, CG=0.0, SC=0.0, U=0.0)


def attribution(root, cfg):
    routes = pd.read_parquet(root / INPUT / "test31_research_routes.parquet").set_index("order_id")
    descriptors = pd.read_parquet(root / S4 / "test31_original_route_descriptors.parquet",
        columns=["order_id", "reverse_overlay_token_count", "unresolved_token_count"]).set_index("order_id")
    overlay = pd.read_parquet(root / INPUT / "complex_control_overlay.parquet")
    classifier = MovementClassifier(root, overlay)
    encounters = pd.read_parquet(root / S4 / "test31_route_complex_encounters.parquet")
    classified, _ = classifier.summarize(encounters)
    del classifier, encounters, overlay
    gc.collect()
    direction = descriptors.reverse_overlay_token_count.eq(0) & descriptors.unresolved_token_count.eq(0)
    gates = pd.DataFrame(index=routes.index)
    gates["input"] = routes.research_data_ready
    gates["direction"] = direction.reindex(routes.index)
    for profile in "CMA":
        gates[f"movement_{profile}"] = [
            valid and evaluate_research_compatibility(profile, movements, NC,
                conservative_control_policy=cfg["conservative_control_policy"]).compatible
            for order in routes.index for valid, movements in [classified.get(str(order), (True, ()))]]
        outside_columns = {"C": ["relative_mixed_share", "congested_share", "severe_share"],
                           "M": ["severe_share"], "A": []}[profile]
        outside = routes[outside_columns].sum(axis=1, min_count=len(outside_columns)) if outside_columns else pd.Series(0., index=routes.index)
        gates[f"traffic_{profile}"] = outside.le(cfg["traffic_outside_budget"] + 1e-6)
        reconstructed = gates.input & gates.direction & gates[f"movement_{profile}"] & gates[f"traffic_{profile}"]
        if not reconstructed.equals(routes[f"compatible_{profile}"]):
            raise ValueError("attribution does not reconcile to the existing research policy")
    cohorts = {"all_test31": list(routes.index)}
    for cut in cfg["cuts_s"]:
        cohorts[str(cut)] = list(pd.read_parquet(root / OLD_WINDOWS / f"M_{cut}_MYOPIC/cohort_outcomes.parquet").order_id)
    rows = []
    for cohort, ids in cohorts.items():
        g = gates.loc[ids]
        for profile in "CMA":
            ready = g.input
            directed = ready & g.direction
            maneuver = directed & g[f"movement_{profile}"]
            compatible = maneuver & g[f"traffic_{profile}"]
            rows.append(dict(cohort=cohort, profile=profile, orders=len(g), input_ready=int(ready.sum()),
                after_direction=int(directed.sum()), after_movement_control=int(maneuver.sum()),
                after_traffic=int(compatible.sum()),
                direction_fail_among_ready=int((ready & ~g.direction).sum()),
                movement_control_fail_among_ready=int((ready & ~g[f"movement_{profile}"]).sum()),
                traffic_fail_among_ready=int((ready & ~g[f"traffic_{profile}"]).sum())))
    return dict(rows=rows, sequential_order="input -> direction -> movement/control -> traffic",
                independent_failure_counts_overlap=True,
                reverse_overlay_orders=int(descriptors.reverse_overlay_token_count.gt(0).sum()),
                unresolved_identity_orders=int(descriptors.unresolved_token_count.gt(0).sum()),
                reconstruction_exact=True)


def fit_train_remaining(root, cfg):
    templates = pd.read_parquet(root / INPUT / "train_request_templates.parquet")
    if set(templates.date.astype(str)) != set(cfg["remaining_time_train_dates"]):
        raise ValueError("remaining-time fit requires the same three frozen Train Mondays")
    parts, sources = [], []
    for date, day in templates.groupby("date", sort=True):
        columns = ["order_id", "departure_time", "arrival_time"]
        paths = sorted((root / f"stage1/input_v1/split=train/date={date}").glob("bucket=*/order_base.parquet"))
        if not paths:
            raise FileNotFoundError(f"Train observations unavailable: {date}")
        observed = pd.concat([pd.read_parquet(p, columns=columns) for p in paths], ignore_index=True)
        observed["order_id"] = observed.order_id.astype(str)
        observed["observed_gps_window_duration_s"] = observed.arrival_time - observed.departure_time
        observed = observed[["order_id", "observed_gps_window_duration_s"]]
        joined = day[["date", "order_id", "release_second", "predicted_route_time_p50_s"]].merge(
            observed, on="order_id", how="left", validate="one_to_one")
        valid = (np.isfinite(joined.predicted_route_time_p50_s) & joined.predicted_route_time_p50_s.gt(0)
                 & np.isfinite(joined.observed_gps_window_duration_s) & joined.observed_gps_window_duration_s.gt(0))
        joined = joined.loc[valid].copy()
        joined["duration_prediction_ratio"] = joined.observed_gps_window_duration_s / joined.predicted_route_time_p50_s
        parts.append(joined)
        sources.append(dict(date=str(date), template_count=len(day), valid_timing_pairs=len(joined),
            order_base_directory=f"stage1/input_v1/split=train/date={date}",
            frozen_prediction_path=f"stage3/output/odd_tod/s3/cache/m3/date={date}.parquet"))
        print(json.dumps(dict(train_remaining_prepared=str(date), rows=len(joined))), flush=True)
        del observed, joined
        gc.collect()
    paired = pd.concat(parts, ignore_index=True)
    if paired.empty:
        raise ValueError("no positive Train timing pairs")
    write_parquet(root / OUTPUT / "train_timing_pairs.parquet", paired)
    quantiles = {f"p{int(q*100)}": float(paired.duration_prediction_ratio.quantile(q)) for q in (.1, .5, .9, .99)}
    model = dict(method=cfg["remaining_time_method"], train_dates=cfg["remaining_time_train_dates"],
        training_pair_count=len(paired), test31_used_for_fit=False,
        observed_target="Stage1 arrival_time - departure_time (GPS observation window)",
        prediction_proxy="sum of frozen M3 traversal travel-time medians, NOT joint route P50",
        conditioning="R > elapsed_loaded_service_s / booked_prediction_s",
        aggregation="pooled Train empirical conditional median; no clipping, bins, or tuned multiplier",
        unsupported_tail=cfg["unsupported_remaining_tail"],
        ratio_quantiles=quantiles, sources=sources,
        train_timing_pairs_sha256=sha(root / OUTPUT / "train_timing_pairs.parquet"),
        sorted_duration_prediction_ratios=np.sort(paired.duration_prediction_ratio.to_numpy(float)).tolist())
    write_json(root / MODEL, model)
    return {k: v for k, v in model.items() if k != "sorted_duration_prediction_ratios"}


def retrospective_timing(root, cfg):
    columns = ["assignment_time", "simulation_time_s", "pickup_eta_s", "predicted_service_time_s", "service_end_time"]
    # Both shared M checkpoint builders use this exact canonical prehistory.
    path = root / "stage4/output/final_experiments/ODD_Q50_M_P70_REFERENCE/assignment_log.parquet"
    source = pd.read_parquet(path, columns=columns)
    source["actual_end_s"] = source.simulation_time_s + (
        pd.to_datetime(source.service_end_time) - pd.to_datetime(source.assignment_time)).dt.total_seconds()
    source["predicted_end_s"] = source.simulation_time_s + source.pickup_eta_s + source.predicted_service_time_s
    templates = pd.read_parquet(root / INPUT / "train_request_templates.parquet")
    rows = []
    for cut in cfg["cuts_s"]:
        active = source.loc[source.simulation_time_s.lt(cut) & source.actual_end_s.gt(cut) & source.predicted_service_time_s.gt(0)]
        overdue = active.loc[active.predicted_end_s.le(cut)]
        counts = [int(((g.release_second >= cut) & (g.release_second < cut + 900)).sum())
                  for _, g in templates.groupby("date", sort=True)]
        outcome = pd.read_parquet(root / OLD_WINDOWS / f"M_{cut}_MYOPIC/cohort_outcomes.parquet")
        rows.append(dict(cut=cut, active_booked_valid_predictions=len(active), overdue_still_busy=len(overdue),
            overdue_actual_remaining_median_s=float((overdue.actual_end_s - cut).median()),
            overdue_actual_remaining_p90_s=float((overdue.actual_end_s - cut).quantile(.9)),
            train_15min_template_counts=counts, scaled_train_mean_arrivals=float(np.mean(counts) * 3),
            test31_actual_cohort=len(outcome), test31_outcomes_used_for_fit=False))
    return dict(rows=rows, retrospective_only=True, actual_end_never_passed_to_optimizer=True)


def run(root):
    cfg = json.loads((root / NEW_CONFIG).read_text())
    base = json.loads((root / CONFIG).read_text())
    (root / DOC).mkdir(parents=True, exist_ok=True)
    (root / OUTPUT).mkdir(parents=True, exist_ok=True)
    if (root / OUTPUT / "summary.json").exists():
        raise ValueError("do not refit the remaining-time model after native execution has started")
    report = dict(status="COMPLETE", input_attribution=attribution(root, base),
                  train_remaining_time=fit_train_remaining(root, cfg),
                  retrospective_timing=retrospective_timing(root, base),
                  new_clustering=False, new_m3_inference=False, new_native_replays=0,
                  peak_rss_mib=psutil.Process().memory_info().peak_wset / 2**20)
    write_json(root / DOC / "diagnostics.json", report)
    print(json.dumps({**report, "train_remaining_time": {k: v for k, v in report["train_remaining_time"].items() if k != "sources"}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run(Path.cwd())
