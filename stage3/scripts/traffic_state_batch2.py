"""Recover existing measurement support and interpret frozen dynamic envelopes.

This analysis never fits a predictor, modifies a profile, or feeds retrospective
states into dispatch. Original interval products are read bucket by bucket.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

from stage3.scripts.traffic_state_batch1 import EDGE, STATES, sha, write_json, write_parquet

DIMS = ("crawl", "stop", "speed_cv", "acceleration_rms")
LABELS = dict(zip(DIMS, ("crawl_time_share", "stop_time_share", "speed_cv_bounded", "acceleration_rms_bounded")))
ID = ["order_id", "traversal_id"]
METRICS = [f"{d}_{m}" for d in DIMS for m in ("E", "Q", "C")]


def recover_measurements(intervals: pd.DataFrame, traversals: pd.DataFrame) -> pd.DataFrame:
    x = intervals.loc[intervals.measurement_source.eq("direct_observed") & intervals.label_valid.eq(True)].copy()
    assert not x.duplicated(["order_id", "gps_interval_id"]).any()
    assert x.observed_travel_time_s.gt(0).all() and x.observed_distance_m.ge(0).all()
    assert np.allclose(x.interval_end_time-x.interval_start_time, x.observed_travel_time_s, atol=1e-6, rtol=1e-12)
    x = x.sort_values([*ID, "interval_start_time", "interval_end_time"])
    previous = x.groupby(ID).interval_end_time.shift()
    x["gap"] = (x.interval_start_time-previous).clip(lower=0).fillna(0)
    result = x.groupby(ID, sort=False).agg(
        direct_interval_count=("gps_interval_id", "size"),
        observed_time_s=("observed_travel_time_s", "sum"),
        observed_distance_m=("observed_distance_m", "sum"),
        window_start=("interval_start_time", "min"), window_stop=("interval_end_time", "max"),
        maximum_internal_gap_s=("gap", "max"), canonical_edge_uid=("canonical_edge_uid", "first"),
        canonical_identity_count=("canonical_edge_uid", "nunique")).reset_index()
    assert result.canonical_identity_count.eq(1).all()
    physical = traversals[[*ID, "canonical_edge_uid", "allocated_distance_m"]]
    result = result.merge(physical, on=ID, how="left", validate="one_to_one", suffixes=("", "_physical"), indicator=True)
    assert result._merge.eq("both").all()
    assert result.canonical_edge_uid.eq(result.canonical_edge_uid_physical).all()
    result = result.drop(columns=["_merge", "canonical_edge_uid_physical", "canonical_identity_count"])
    result["traversal_distance_coverage"] = result.observed_distance_m / result.allocated_distance_m.where(result.allocated_distance_m.gt(0))
    result["window_span_s"] = result.window_stop-result.window_start
    result["window_time_coverage"] = result.observed_time_s / result.window_span_s.where(result.window_span_s.gt(0))
    return result


def prediction_proxy(frame: pd.DataFrame, cfg: dict) -> pd.Series:
    """Same descriptive cutpoints, different evidence source; no invented counts."""
    ratio = frame.pred_pace_p50 / frame.reference_pace_q20
    low = frame.pred_crawl + frame.pred_stop
    valid = (frame.reference_order_days.ge(cfg["reference_min_order_days"])
             & frame.reference_days.ge(cfg["reference_min_days"])
             & np.isfinite(ratio) & frame.pred_pace_p50.gt(0)
             & frame.pred_crawl.between(0, 1) & frame.pred_stop.between(0, 1)
             & low.le(1 + 1e-6))
    state = pd.Series("UNKNOWN", index=frame.index)
    state.loc[valid] = "MIXED_EVIDENCE"
    state.loc[valid & ratio.lt(cfg["pace_ratio_congested"]) & low.lt(cfg["low_speed_share_congested"])] = STATES[0]
    state.loc[valid & ratio.ge(cfg["pace_ratio_congested"]) & low.ge(cfg["low_speed_share_congested"])] = STATES[1]
    state.loc[valid & ratio.ge(cfg["pace_ratio_severe"]) & low.ge(cfg["low_speed_share_severe"])] = STATES[2]
    return state


def exposure_by_route(frame: pd.DataFrame, weight: str, prefix: str) -> pd.DataFrame:
    """Unknown is in the denominator; missing sequence tokens break continuity."""
    frame = frame.sort_values(["order_id", "route_sequence"]).copy()
    assert frame[weight].gt(0).all() and np.isfinite(frame[weight]).all()
    assert not frame.duplicated(["order_id", "route_sequence"]).any()
    total = frame.groupby("order_id")[weight].sum()
    out = total.to_frame(prefix + "total_time_s")
    for state in STATES:
        amounts = frame[weight].where(frame.state.eq(state), 0).groupby(frame.order_id).sum()
        out[prefix + state.lower() + "_share"] = amounts/total
    hit = frame.state.isin(STATES[1:3])
    out[prefix + "congested_or_severe_share"] = frame[weight].where(hit, 0).groupby(frame.order_id).sum()/total
    same_order = frame.order_id.eq(frame.order_id.shift())
    adjacent = frame.route_sequence.eq(frame.route_sequence.shift()+1)
    new_run = ~(hit & hit.shift(fill_value=False) & same_order & adjacent)
    frame["run"] = new_run.cumsum()
    runs = frame.loc[hit].groupby(["order_id", "run"])[weight].sum().groupby(level=0).max()
    out[prefix + "longest_congested_run_s"] = runs.reindex(total.index).fillna(0)
    assert np.allclose(out[[prefix+s.lower()+"_share" for s in STATES]].sum(axis=1), 1)
    return out.reset_index()


def dynamic_ratios(descriptors: pd.DataFrame, profiles: dict) -> pd.DataFrame:
    out = descriptors.copy()
    for p in profiles["profiles"]:
        ratios = pd.DataFrame({m: out[m]/p["dynamic_caps"][m.rsplit("_", 1)[0]][m.rsplit("_", 1)[1]] for m in METRICS})
        name = p["profile_id"]
        out[f"rho_dynamic_{name}"] = ratios.max(axis=1)
        out[f"crawl_stop_exceeds_{name}"] = ratios[[m for m in METRICS if m.startswith(("crawl_", "stop_"))]].gt(1).any(axis=1)
        out[f"variability_exceeds_{name}"] = ratios[[m for m in METRICS if m.startswith(("speed_cv_", "acceleration_rms_"))]].gt(1).any(axis=1)
    return out


def summarize_measurement(frame: pd.DataFrame, date: str) -> dict:
    out = {"date": date, "recovered_traversals": len(frame),
           "direct_intervals": int(frame.direct_interval_count.sum()),
           "traversal_coverage_gt_one": int(frame.traversal_distance_coverage.gt(1+1e-6).sum()),
           "discontinuous_window_count": int(frame.maximum_internal_gap_s.gt(6).sum())}
    for col in ["direct_interval_count", "observed_time_s", "traversal_distance_coverage", "canonical_distance_coverage", "window_time_coverage"]:
        v = frame[col].dropna()
        out[col+"_p10"] = float(v.quantile(.1))
        out[col+"_p50"] = float(v.median())
        out[col+"_p90"] = float(v.quantile(.9))
    for target, label in LABELS.items():
        out[target+"_history_nonnull"] = int(frame[label].notna().sum())
        out[target+"_target_valid"] = int(frame[target+"_target_valid"].sum())
        out[target+"_nonnull_but_invalid"] = int((frame[label].notna() & ~frame[target+"_target_valid"]).sum())
    return out


def run(config_path: Path) -> None:
    cfg = json.loads(config_path.read_text())
    b1 = json.loads(Path(cfg["batch1_config"]).read_text())
    assert cfg["measurement_dates"] == b1["train_dates"] + b1["evaluation_dates"]
    assert cfg["overlap_dates"] == b1["evaluation_dates"]
    out, reports = Path(cfg["output_root"]), Path(cfg["report_root"])
    out.mkdir(parents=True, exist_ok=False)
    (out/"cell_support").mkdir()
    reports.mkdir(parents=True, exist_ok=True)
    process, start = psutil.Process(), time.monotonic()
    resource = {"peak_rss_mb": 0.0}
    done = threading.Event()
    def monitor():
        while not done.wait(.2):
            resource["peak_rss_mb"] = max(resource["peak_rss_mb"], process.memory_info().rss/2**20)
            if time.monotonic()-start > cfg["hard_timeout_s"]:
                print("HARD_TIMEOUT stopping analysis", flush=True)
                os._exit(124)
    threading.Thread(target=monitor, daemon=True).start()
    protected = [Path("stage3/config/stage3_av_capability_profiles.json"),
        Path("stage2/output_v5_2/development/M3/epoch_004.pt"),
        Path(cfg["batch1_config"]), Path("stage3/output/traffic_state_batch1/train_reference.parquet"),
        Path("stage3/output/odd_tod/s3/validation_dynamic_route_descriptors.parquet")]
    before = {str(p): sha(p) for p in protected}
    profiles = json.loads(protected[0].read_text())
    ref = pd.read_parquet(protected[3])
    eqc = dynamic_ratios(pd.read_parquet(protected[4]), profiles)
    sources, measurements, cell_summaries, routes, overlap_tables = [], [], [], [], []
    for date in cfg["measurement_dates"]:
        split = "train" if date in b1["train_dates"] else "validation"
        root = Path("stage1/input_v1") / f"split={split}" / f"date={date}"
        day_parts = []
        buckets = sorted(root.glob("bucket=*"))
        assert buckets, f"no source buckets {date}"
        for bucket in buckets:
            manifest = json.loads((bucket/"manifest.json").read_text())
            assert manifest["status"] == "PASS" and manifest["date"] == date
            ip, tp = bucket/"link_interval_observations.parquet", bucket/"link_traversals.parquet"
            interval = pd.read_parquet(ip)
            assert len(interval) == manifest["product_row_counts"]["link_interval_observations"]
            physical = pd.read_parquet(tp, columns=[*ID, "canonical_edge_uid", "allocated_distance_m"])
            assert len(physical) == manifest["product_row_counts"]["link_traversals"]
            day_parts.append(recover_measurements(interval, physical))
            sources.append({"date": date, "bucket": bucket.name, "interval_sha256": sha(ip), "traversal_sha256": sha(tp)})
            del interval, physical
        measured = pd.concat(day_parts, ignore_index=True)
        del day_parts
        assert not measured.duplicated(ID).any()
        route_path = Path("stage2/output_v4/route_conditioned_dataset/revealed_route_proxy")/f"day={date}.parquet"
        route_meta = json.loads((Path("stage2/output_v4/route_conditioned_dataset/manifests/revealed_route_proxy")/f"day={date}.json").read_text())
        assert sha(route_path) == route_meta["file_sha256"]
        cols = [*ID, "route_sequence", EDGE, "canonical_edge_uid", "canonical_length_m", "canonical_highway", *[d+"_target_valid" for d in DIMS]]
        route = pd.read_parquet(route_path, columns=cols)
        assert not route.duplicated(ID).any()
        measured = measured.merge(route, on=[*ID, "canonical_edge_uid"], how="left", validate="one_to_one", indicator=True)
        assert measured._merge.eq("both").all()
        measured = measured.drop(columns="_merge")
        measured["canonical_distance_coverage"] = measured.observed_distance_m/measured.canonical_length_m.where(measured.canonical_length_m.gt(0))
        hp = Path(b1["history_root"])/"events"/f"day={date}.parquet"
        hist = pd.read_parquet(hp, columns=["history_order_id", "traversal_id", "availability_timestamp", "observed_sec_per_m", *LABELS.values()]).rename(columns={"history_order_id": "order_id"})
        expected = json.loads((Path(b1["history_root"])/"history_store_manifest.json").read_text())["event_files"][date]
        assert sha(hp) == expected["file_sha256"]
        measured = measured.merge(hist, on=ID, how="outer", validate="one_to_one", indicator=True)
        assert measured._merge.eq("both").all()
        measured = measured.drop(columns="_merge")
        end_error = float((measured.window_stop-measured.availability_timestamp).abs().max())
        assert end_error <= 1e-6
        pace_valid = measured.observed_sec_per_m.notna()
        pace_error = float((measured.loc[pace_valid, "observed_sec_per_m"] - measured.loc[pace_valid, "observed_time_s"]/measured.loc[pace_valid, "observed_distance_m"]).abs().max())
        assert pace_error < 1e-6
        row = summarize_measurement(measured, date)
        row.update(window_stop_max_error_s=end_error, pace_max_error_s_per_m=pace_error)
        measurements.append(row)
        window_end = (np.floor(measured.window_stop/1800).astype(np.int64)+1)*1800
        measured["window_end"] = window_end
        cells_path = Path(b1["output_root"])/"cells"/f"date={date}.parquet"
        cell = pd.read_parquet(cells_path, columns=[EDGE, "window_end", "state", "joint_order_count"])
        measured = measured.merge(cell, on=[EDGE, "window_end"], how="left", validate="many_to_one")
        measured["state"] = measured.state.fillna("OUTSIDE_BATCH1_DATE_WINDOW")
        in_window = measured.state.ne("OUTSIDE_BATCH1_DATE_WINDOW")
        grouped = measured.loc[in_window].groupby([EDGE, "window_end", "state"], observed=True)
        support = grouped.agg(events=("traversal_id", "size"),
            direct_intervals=("direct_interval_count", "sum"), observed_time_s=("observed_time_s", "sum"),
            traversal_coverage_median=("traversal_distance_coverage", "median"),
            canonical_coverage_median=("canonical_distance_coverage", "median"),
            cv_target_valid_events=("speed_cv_target_valid", "sum"),
            acceleration_target_valid_events=("acceleration_rms_target_valid", "sum")).reset_index()
        write_parquet(out/"cell_support"/f"date={date}.parquet", support)
        for state, g in measured.loc[in_window].groupby("state", observed=True):
            cell_summaries.append({"date": date, "state": state, "events": len(g),
                "traversal_coverage_ge_08": int(g.traversal_distance_coverage.ge(.8).sum()),
                "canonical_coverage_ge_08": int(g.canonical_distance_coverage.ge(.8).sum()),
                "traversal_coverage_p50": float(g.traversal_distance_coverage.median()),
                "canonical_coverage_p50": float(g.canonical_distance_coverage.median()),
                "cv_target_valid_events": int(g.speed_cv_target_valid.sum()),
                "acceleration_target_valid_events": int(g.acceleration_rms_target_valid.sum())})
        if date in cfg["overlap_dates"]:
            pp = Path("stage3/output/odd_tod/s3/cache/m3_validation")/f"date={date}.parquet"
            meta = json.loads(pp.with_suffix(".json").read_text())
            assert sha(pp) == meta["prediction_sha256"]
            pred = pd.read_parquet(pp, columns=[*ID, "pred_pace_p50", "pred_crawl", "pred_stop", "travel_time_p50_s"])
            valid_ids = eqc.loc[eqc.date.eq(date), "order_id"]
            pred = pred.loc[pred.order_id.isin(valid_ids)].merge(route[[*ID, EDGE, "route_sequence"]], on=ID, validate="one_to_one")
            pred = pred.merge(ref, on=EDGE, how="left", validate="many_to_one")
            pred["state"] = prediction_proxy(pred, b1)
            pred_route = exposure_by_route(pred, "travel_time_p50_s", "pred_")
            observed = measured.loc[in_window & measured.order_id.isin(valid_ids)].copy()
            obs_route = exposure_by_route(observed, "observed_time_s", "obs_")
            day = eqc.loc[eqc.date.eq(date)].merge(pred_route, on="order_id", validate="one_to_one").merge(obs_route, on="order_id", how="left", validate="one_to_one")
            assert len(day) == len(valid_ids)
            assert np.allclose(day.pred_total_time_s, day.predicted_route_time_p50_s, atol=1e-3, rtol=1e-6)
            # Direct observed time and predicted full-route time are unlike units
            # of scope; never call their ratio a measurement coverage fraction.
            coverage = measured.groupby("order_id").agg(direct_distance=("observed_distance_m", "sum"))
            rp = pd.read_parquet(route_path, columns=["order_id", "route_part_length_m"])
            distance = rp.groupby("order_id").route_part_length_m.sum()
            coverage["direct_route_distance_coverage"] = coverage.direct_distance/distance
            day = day.merge(coverage[["direct_route_distance_coverage"]], on="order_id", how="left", validate="one_to_one")
            write_parquet(out/f"route_overlap_date={date}.parquet", day)
            routes.append(day)
            states = pred.groupby("state").agg(tokens=("traversal_id", "size"), predicted_time_s=("travel_time_p50_s", "sum")).reset_index()
            states["date"] = date
            overlap_tables.append(states)
            del pred, pred_route, observed, obs_route, rp, day
        print(f"{date}: recovered={len(measured)} intervals={row['direct_intervals']} pace_error={pace_error:.3g} rss={process.memory_info().rss/2**20:.1f} MiB", flush=True)
        del measured, route, hist, cell, support
        gc.collect()
    write_json(reports/"measurement_recovery.json", {"rows": measurements})
    write_json(reports/"state_measurement_support.json", {"rows": cell_summaries})
    joined = pd.concat(routes, ignore_index=True)
    write_json(reports/"predicted_state_distribution.json", {"rows": pd.concat(overlap_tables).to_dict("records")})
    xcols = ["pred_congested_or_severe_share", "pred_severe_congestion_proxy_share", "pred_mixed_evidence_share", "pred_unknown_share", "pred_longest_congested_run_s", "obs_congested_or_severe_share"]
    correlations = []
    for date, subset in [("ALL_VALIDATION", joined), *list(joined.groupby("date"))]:
        for x in xcols:
            for y in [*METRICS, "rho_dynamic_C", "rho_dynamic_M", "rho_dynamic_A"]:
                usable = subset[[x, y]].dropna()
                r = usable[x].corr(usable[y], method="spearman") if usable[x].nunique()>1 and usable[y].nunique()>1 else np.nan
                correlations.append({"date": date, "x": x, "y": y, "n": len(usable), "spearman": float(r) if np.isfinite(r) else None})
    write_json(reports/"descriptive_correlations.json", {"rows": correlations, "p_values": "not estimated; shared-input descriptive comparison"})
    cross, family = [], []
    for date, subset in [("ALL_VALIDATION", joined), *list(joined.groupby("date"))]:
        for p in ("C", "M", "A"):
            for passed, g in subset.groupby(subset[f"rho_dynamic_{p}"].le(1)):
                rec = {"date": date, "profile": p, "dynamic_all12_pass": bool(passed), "routes": len(g)}
                for x in xcols:
                    rec[x+"_mean"] = float(g[x].mean())
                    rec[x+"_p50"] = float(g[x].median())
                rec["direct_route_distance_coverage_p50"] = float(g.direct_route_distance_coverage.median())
                cross.append(rec)
            for (cs, var), g in subset.groupby([f"crawl_stop_exceeds_{p}", f"variability_exceeds_{p}"]):
                family.append({"date": date, "profile": p, "crawl_stop_exceeds": bool(cs), "variability_exceeds": bool(var), "routes": len(g)})
    write_json(reports/"profile_cross_table.json", {"rows": cross})
    write_json(reports/"dimension_family_cross_table.json", {"rows": family})
    plot_overlap(reports, joined)
    assert before == {str(p): sha(p) for p in protected}
    done.set()
    summary = {"status": "COMPLETED_DESCRIPTIVE_PROFILE_DRAFT_ONLY", "config_sha256": sha(config_path),
        "baseline_commit": "4062ea4", "measurement_days": len(measurements), "overlap_routes": len(joined),
        "recovered_traversals": sum(r["recovered_traversals"] for r in measurements),
        "direct_intervals": sum(r["direct_intervals"] for r in measurements),
        "protected_inputs_unchanged": before, "sources": sources,
        "runtime_s": time.monotonic()-start, **resource,
        "gpu_used": False, "training": False, "test31_read": False, "dispatch": False,
        "profile_modified": False,
        "interpretation": "Common-input definition overlap plus retrospective partial-observation description; no traffic ground-truth validation"}
    write_json(reports/"summary.json", summary)
    write_json(out/"summary.json", summary)
    print(json.dumps({k: summary[k] for k in ["status", "runtime_s", "peak_rss_mb", "overlap_routes"]}), flush=True)


def plot_overlap(reports: Path, routes: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, p in zip(axes, ("C", "M", "A")):
        groups = [routes.loc[routes[f"rho_dynamic_{p}"].le(1), "pred_congested_or_severe_share"],
                  routes.loc[routes[f"rho_dynamic_{p}"].gt(1), "pred_congested_or_severe_share"]]
        ax.boxplot(groups, tick_labels=["All12 pass", "Any exceed"], showfliers=False)
        ax.set_title(f"Profile {p}")
        for i, g in enumerate(groups, 1):
            ax.text(i, .98, f"n={len(g):,}", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=8)
    axes[0].set_ylabel("Predicted congested/severe time share")
    fig.suptitle("Shared-prediction definition overlap; not independent validation", fontsize=10)
    fig.tight_layout()
    fig.savefig(reports/"eqc_state_overlap.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("stage3/config/traffic_state_batch2.json"))
    run(parser.parse_args().config)
