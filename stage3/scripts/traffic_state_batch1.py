"""Bounded-memory descriptive traffic-state inventory; never dispatch or train.

Only existing completed-traversal events are read. Per-day aggregation and a
compact numeric Train reference pool avoid an edge x time x day dense matrix.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

EDGE = "observed_directed_edge_uid"
ORDER = "history_order_id"
VALUES = ["observed_sec_per_m", "crawl_time_share", "stop_time_share",
          "speed_cv_bounded", "acceleration_rms_bounded"]
KEY = [EDGE, "window_end"]
STATES = ["NON_CONGESTED_PROXY", "CONGESTED_PROXY", "SEVERE_CONGESTION_PROXY",
          "MIXED_EVIDENCE", "UNKNOWN"]
COLORS = ["#2e8b57", "#e69500", "#c43b3b", "#8551a1", "#b2b2b2"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def classify(cells: pd.DataFrame, cfg: dict, minimum: int | None = None) -> pd.Series:
    n = cfg["cell_min_orders"] if minimum is None else minimum
    supported = (cells["joint_order_count"].ge(n)
                 & cells["reference_order_days"].ge(cfg["reference_min_order_days"])
                 & cells["reference_days"].ge(cfg["reference_min_days"])
                 & cells["joint_event_span_s"].ge(cfg["cell_min_event_span_s"])
                 & cells["latest_joint_age_s"].le(cfg["cell_max_latest_age_s"])
                 & np.isfinite(cells["pace_ratio"]))
    ratio = cells["pace_ratio"]
    low = cells["joint_low_speed_share"]
    result = pd.Series("UNKNOWN", index=cells.index)
    result.loc[supported] = "MIXED_EVIDENCE"
    result.loc[supported & ratio.lt(cfg["pace_ratio_congested"])
               & low.lt(cfg["low_speed_share_congested"])] = "NON_CONGESTED_PROXY"
    result.loc[supported & ratio.ge(cfg["pace_ratio_congested"])
               & low.ge(cfg["low_speed_share_congested"])] = "CONGESTED_PROXY"
    result.loc[supported & ratio.ge(cfg["pace_ratio_severe"])
               & low.ge(cfg["low_speed_share_severe"])] = "SEVERE_CONGESTION_PROXY"
    return result


def prepare_events(frame: pd.DataFrame, seconds: int) -> pd.DataFrame:
    """An event at an exact boundary belongs to the next half-open window."""
    frame = frame.copy()
    assert not frame.duplicated(["date", ORDER, "traversal_id"]).any()
    assert frame[EDGE].notna().all()
    ts = frame["availability_timestamp"].to_numpy(float)
    assert np.isfinite(ts).all()
    frame["window_end"] = (np.floor(ts / seconds).astype(np.int64) + 1) * seconds
    pace = frame["observed_sec_per_m"]
    valid = np.isfinite(pace) & pace.gt(0)
    for col in VALUES[1:3]:
        valid &= frame[col].between(0, 1)
    valid &= (frame["crawl_time_share"] + frame["stop_time_share"]).le(1 + 1e-9)
    frame["joint_valid"] = valid
    # LCS aggregate validity is stricter than component presence; do not conflate.
    for col in VALUES[3:]:
        frame[col] = frame[col].where(frame[col].between(0, 1))
    return frame


def aggregate_day(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.groupby(KEY, observed=True, sort=True)
    cells = g.agg(event_count=(ORDER, "size"), order_count=(ORDER, "nunique"),
                  canonical_highway=("canonical_highway", "first"),
                  road_class_count=("canonical_highway", "nunique"),
                  first_event_ts=("availability_timestamp", "min"),
                  last_event_ts=("availability_timestamp", "max"))
    # First average repeated visits within order-edge-window, then across orders.
    orders = frame.groupby([*KEY, ORDER], observed=True)[VALUES].mean()
    means = orders.groupby(level=[0, 1]).mean().add_suffix("_order_mean")
    counts = orders.groupby(level=[0, 1]).count().add_suffix("_order_count")
    cells = cells.join(means).join(counts)
    valid = frame.loc[frame["joint_valid"]].copy()
    valid["low_speed_share"] = valid["crawl_time_share"] + valid["stop_time_share"]
    joint = valid.groupby([*KEY, ORDER], observed=True).agg(
        pace=("observed_sec_per_m", "mean"), low=("low_speed_share", "mean"),
        stop=("stop_time_share", "mean"), first=("availability_timestamp", "min"),
        last=("availability_timestamp", "max"))
    joint = joint.groupby(level=[0, 1]).agg(
        joint_order_count=("pace", "size"), joint_pace=("pace", "mean"),
        joint_low_speed_share=("low", "mean"), joint_stop_share=("stop", "mean"),
        joint_first=("first", "min"), joint_last=("last", "max"))
    cells = cells.join(joint).reset_index()
    cells["joint_order_count"] = cells["joint_order_count"].fillna(0).astype(int)
    cells["joint_event_span_s"] = cells["joint_last"] - cells["joint_first"]
    cells["latest_joint_age_s"] = cells["window_end"] - cells["joint_last"]
    return cells


def inventory_predictions(dates: list[str]) -> list[dict]:
    rows = []
    for date in dates:
        folder = "m3" if date <= "20161024" else "m3_validation"
        p = Path("stage3/output/odd_tod/s3/cache") / folder / f"date={date}.parquet"
        m = p.with_suffix(".json")
        if not p.is_file() or not m.is_file():
            rows.append({"date": date, "path": str(p), "status": "MISSING_CACHE_OR_MANIFEST"})
            continue
        meta = json.loads(m.read_text())
        rows.append({"date": date, "path": str(p), "rows": pq.ParquetFile(p).metadata.num_rows,
            "columns": pq.ParquetFile(p).schema_arrow.names,
            "manifest_prediction_hash_matches": sha(p) == meta.get("prediction_sha256"),
            "checkpoint_sha256": meta.get("checkpoint_sha256"),
            "decision_time_only": meta.get("decision_time_only"),
            "predicted_progression_only": meta.get("predicted_progression_only"),
            "scope": "schema_and_provenance_inventory_only_no_reinference"})
    return rows


def run(config_path: Path) -> None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    dates = cfg["train_dates"] + cfg["evaluation_dates"]
    assert set(dates) == {str(d) for d in range(20161009, 20161028)}
    out, reports = Path(cfg["output_root"]), Path(cfg["report_root"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "cells").mkdir()
    reports.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    process = psutil.Process()
    stats = {"peak_rss_mb": 0.0}
    stop_monitor = threading.Event()

    def monitor():
        while not stop_monitor.wait(0.2):
            stats["peak_rss_mb"] = max(stats["peak_rss_mb"], process.memory_info().rss / 2**20)
            if time.monotonic() - start > cfg["hard_timeout_s"]:
                print("HARD_TIMEOUT: stopping analysis", flush=True)
                os._exit(124)
    threading.Thread(target=monitor, daemon=True).start()
    protected = [Path("stage3/config/stage3_av_capability_profiles.json"),
                 Path("stage2/output_v5_2/development/M3/epoch_004.pt")]
    protected_hashes = {str(p): sha(p) for p in protected}
    history = Path(cfg["history_root"])
    store = json.loads((history / "history_store_manifest.json").read_text())
    pool, edge_codes, sources, inventory = [], {}, [], []
    observed_edges: set[str] = set()
    observed_cells = 0
    for date in dates:
        path = history / "events" / f"day={date}.parquet"
        digest = sha(path)
        assert digest == store["event_files"][date]["file_sha256"]
        frame = pd.read_parquet(path)
        assert len(frame) == store["event_files"][date]["row_count"]
        assert frame["date"].eq(date).all()
        completion_date = pd.to_datetime(frame["availability_timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Shanghai").dt.strftime("%Y%m%d")
        same_date = completion_date.eq(date)
        # Do not silently combine source-day partitions into different calendar
        # days, or let Train order spillovers bring 25th-day events into reference.
        prepared = prepare_events(frame.loc[same_date], cfg["window_s"])
        inv = {"date": date, "events": len(frame), "orders": int(frame[ORDER].nunique()),
               "edges": int(frame[EDGE].nunique()),
               "cross_midnight_events_excluded": int((~same_date).sum()),
               "analysis_events": len(prepared),
               "joint_valid": int(prepared["joint_valid"].sum()),
               "lcs_available": int(frame["lcs_available"].sum())}
        for col in VALUES:
            inv[col + "_finite"] = int(np.isfinite(frame[col]).sum())
        end_bin = ((frame["availability_timestamp"] + 8 * 3600) % 86400 // 1800).astype(int)
        inv["start_bin_differs_from_completion_bin"] = int(end_bin.ne(frame["time_bin_30m"]).sum())
        inventory.append(inv)
        sources.append({"date": date, "path": str(path), "sha256": digest, "rows": len(frame)})
        cells = aggregate_day(prepared)
        observed_edges.update(cells[EDGE].unique())
        observed_cells += len(cells)
        cells["date"] = date
        write_parquet(out / "cells" / f"date={date}.parquet", cells)
        if date in cfg["train_dates"]:
            usable = prepared.loc[prepared["joint_valid"]]
            by_order = usable.groupby([EDGE, ORDER], observed=True)["observed_sec_per_m"].mean().reset_index()
            for edge in by_order[EDGE].unique():
                if edge not in edge_codes:
                    edge_codes[edge] = len(edge_codes)
            pool.append(pd.DataFrame({"edge_code": by_order[EDGE].map(edge_codes).to_numpy(np.int32),
                                      "date": np.full(len(by_order), int(date), dtype=np.int32),
                                      "pace": by_order["observed_sec_per_m"].to_numpy(float)}))
        print(f"inventory {date}: events={len(frame)}, cells={len(cells)}, rss={process.memory_info().rss/2**20:.1f} MiB", flush=True)
        del frame, prepared, cells
        gc.collect()
    numeric = pd.concat(pool, ignore_index=True)
    pool.clear()
    grouped = numeric.groupby("edge_code", sort=True)
    reference = grouped.agg(reference_order_days=("pace", "size"), reference_days=("date", "nunique"))
    reference["reference_pace_q20"] = grouped["pace"].quantile(cfg["reference_quantile"])
    reference["reference_pace_q50"] = grouped["pace"].median()
    names = {v: k for k, v in edge_codes.items()}
    reference[EDGE] = reference.index.map(names)
    reference = reference.reset_index(drop=True)
    write_parquet(out / "train_reference.parquet", reference)
    del numeric, grouped
    gc.collect()
    all_tables: dict[str, list[pd.DataFrame]] = {k: [] for k in ["state_by_day", "state_by_class", "state_by_hour", "support_sensitivity", "unknown_reasons", "adjacent_transitions", "reference_by_class"]}
    case_candidates = []
    for date in dates:
        path = out / "cells" / f"date={date}.parquet"
        cells = pd.read_parquet(path).merge(reference, on=EDGE, how="left", validate="many_to_one")
        cells["pace_ratio"] = cells["joint_pace"] / cells["reference_pace_q20"]
        cells["state"] = classify(cells, cfg)
        cells["role"] = "train_descriptive_in_sample" if date in cfg["train_dates"] else "validation_descriptive"
        cells["hour"] = ((cells["window_end"] - cfg["window_s"] + 8*3600) % 86400 // 3600).astype(int)
        cells["source"] = "COMPLETED_DIRECT_TRAVERSAL_EVENTS"
        cells["decision_time_deployable"] = False
        for name, keys in [("state_by_day", ["date", "role", "state"]),
                           ("state_by_class", ["role", "canonical_highway", "state"]),
                           ("state_by_hour", ["role", "hour", "state"])]:
            table = cells.groupby(keys, dropna=False).agg(cells=("state", "size"),
                    order_cell_exposures=("order_count", "sum"), events=("event_count", "sum"))
            all_tables[name].append(table.reset_index())
        for minimum in cfg["support_sensitivity_min_orders"]:
            alt = classify(cells, cfg, minimum)
            all_tables["support_sensitivity"].append(pd.DataFrame([{
                "date": date, "minimum_orders": minimum, "cells": len(cells),
                "non_unknown_cells": int(alt.ne("UNKNOWN").sum()),
                "count_only_eligible_cells": int(cells["joint_order_count"].ge(minimum).sum())}]))
        reasons = {
            "insufficient_joint_orders": cells["joint_order_count"].lt(cfg["cell_min_orders"]),
            "insufficient_reference": ~(cells["reference_order_days"].ge(cfg["reference_min_order_days"])
                                         & cells["reference_days"].ge(cfg["reference_min_days"])),
            "insufficient_event_span": ~cells["joint_event_span_s"].ge(cfg["cell_min_event_span_s"]),
            "stale_or_no_joint_evidence": ~cells["latest_joint_age_s"].le(cfg["cell_max_latest_age_s"]),
        }
        all_tables["unknown_reasons"].append(pd.DataFrame([{"date": date, "reason": k,
            "cells": int(v.sum()), "denominator": len(cells)} for k, v in reasons.items()]))
        cells["reference_supported"] = (cells.reference_order_days.ge(cfg["reference_min_order_days"])
                                         & cells.reference_days.ge(cfg["reference_min_days"]))
        all_tables["reference_by_class"].append(cells.drop_duplicates(EDGE).groupby("canonical_highway").agg(
            observed_edge_days=(EDGE, "size"), reference_supported_edge_days=("reference_supported", "sum")).reset_index())
        ordered = cells.sort_values([EDGE, "window_end"])
        previous = ordered.groupby(EDGE).shift(1)
        adjacent = (ordered["window_end"] - previous["window_end"]).eq(cfg["window_s"])
        transitions = pd.DataFrame({"from_state": previous.loc[adjacent, "state"],
                                    "to_state": ordered.loc[adjacent, "state"]})
        transitions = transitions.groupby(["from_state", "to_state"]).size().rename("pairs").reset_index()
        transitions["date"] = date
        all_tables["adjacent_transitions"].append(transitions)
        if date in cfg["evaluation_dates"]:
            # Deterministic illustrative selection, not evidence of typicality.
            ranked = cells.groupby(EDGE).agg(windows=("state", "size"),
                known=("state", lambda x: int(x.ne("UNKNOWN").sum())),
                types=("state", "nunique"))
            for edge in ranked.sort_values(["known", "windows"], ascending=False).head(2).index:
                case_candidates.append(cells.loc[cells[EDGE].eq(edge)].copy())
        write_parquet(path, cells)
        print(f"states {date}: supported={int(cells.state.ne('UNKNOWN').sum())}/{len(cells)}", flush=True)
        del cells, ordered, previous
        gc.collect()
    for name, parts in all_tables.items():
        table = pd.concat(parts, ignore_index=True)
        if name in {"state_by_class", "state_by_hour"}:
            keys = [c for c in table if c not in {"cells", "order_cell_exposures", "events"}]
            table = table.groupby(keys, dropna=False, as_index=False)[["cells", "order_cell_exposures", "events"]].sum()
        if name == "reference_by_class":
            table = table.groupby("canonical_highway", as_index=False).sum()
        # Compact aggregates only; no order/driver IDs leave ignored output.
        write_json(reports / f"{name}.json", {"rows": table.to_dict("records")})
    write_json(reports / "signal_inventory_counts.json", {"rows": inventory})
    cases = pd.concat(case_candidates, ignore_index=True)
    write_parquet(out / "illustrative_cases.parquet", cases)
    plot_results(reports, cases)
    write_json(reports / "prediction_inventory.json", {"rows": inventory_predictions(dates)})
    assert protected_hashes == {str(p): sha(p) for p in protected}
    stop_monitor.set()
    summary = {"status": "COMPLETED_DESCRIPTIVE_ONLY", "schema_version": cfg["schema_version"],
               "config_sha256": sha(config_path), "sources": sources,
               "protected_inputs_unchanged": protected_hashes,
               "dates": dates, "events": sum(r["events"] for r in inventory),
               "reference_edges": len(reference),
               "reference_supported_edges": int((reference.reference_order_days.ge(cfg["reference_min_order_days"]) & reference.reference_days.ge(cfg["reference_min_days"])).sum()),
               "study_observed_edges": len(observed_edges), "observed_cells": observed_cells,
               "no_event_cells_within_study_edge_union": len(observed_edges)*48*len(dates)-observed_cells,
               "no_event_denominator_scope": "19 days x 48 bins x union of observed directed edges; not full network",
               "test31_read": False, "training": False, "gpu_used": False,
               "runtime_s": time.monotonic()-start, **stats,
               "limitations": ["Stage1 bucket access denied; hash-verified existing history store reused",
                    "No direct interval count, duration, distance coverage or original window start in history events",
                    "Train states use full-Train retrospective reference, not online backtest",
                    "Unknown no-event cells not materialized; denominator is observed cells only",
                    "Candidate thresholds are unvalidated descriptive choices, not AV policies"]}
    write_json(out / "summary.json", summary)
    write_json(reports / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ["status", "events", "runtime_s", "peak_rss_mb"]}), flush=True)


def plot_results(reports: Path, cases: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    day = pd.DataFrame(json.loads((reports / "state_by_day.json").read_text())["rows"])
    pivot = day.pivot_table(index="date", columns="state", values="cells", aggfunc="sum", fill_value=0).reindex(columns=STATES, fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.div(pivot.sum(axis=1), axis=0).plot.bar(stacked=True, ax=ax, width=.85, color=COLORS)
    ax.set(ylabel="Share of observed cells")
    fig.suptitle("Exploratory traffic proxies (09-27); no-event cells excluded", y=.98)
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    fig.legend(handles, labels, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(.5, .92))
    fig.subplots_adjust(left=.07, right=.99, bottom=.23, top=.76)
    fig.savefig(reports / "state_support_overview.png", dpi=140)
    plt.close(fig)
    pairs = list(cases.groupby(["date", EDGE], sort=True))
    fig, axes = plt.subplots(len(pairs), 1, figsize=(12, 2.7*len(pairs)), squeeze=False)
    for ax, ((date, edge), frame) in zip(axes[:, 0], pairs):
        frame = frame.sort_values("window_end")
        hours = ((frame.window_end - 1800 + 8*3600) % 86400) / 3600 + .5
        # No connecting line across missing bins; missing != free flow.
        ax.scatter(hours, frame.pace_ratio, c=frame.state.map(dict(zip(STATES, COLORS))), s=18)
        ax.axhline(1.5, color="orange", linestyle="--", linewidth=.7)
        ax.axhline(2, color="red", linestyle="--", linewidth=.7)
        ax.set(title=f"{date} | {edge} | illustrative highest-support edge", ylabel="Pace / Train q20", xlim=(0, 24))
        twin = ax.twinx()
        twin.plot(hours, frame.joint_order_count, color="steelblue", alpha=.35, linestyle="none", marker="+")
        twin.set_ylabel("Joint orders (+)")
    axes[-1, 0].set_xlabel("Window end hour (Asia/Shanghai)")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                              label=state, markersize=6) for state, color in zip(STATES, COLORS)],
               loc="upper center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(reports / "illustrative_temporal_cases.png", dpi=120)
    plt.close(fig)


def describe_outputs(cfg: dict) -> None:
    """Read existing cells only; no inference, fitting, or threshold changes."""
    out, reports = Path(cfg["output_root"]), Path(cfg["report_root"])
    rows = []
    for date in cfg["train_dates"] + cfg["evaluation_dates"]:
        cells = pd.read_parquet(out / "cells" / f"date={date}.parquet")
        assert not cells.duplicated(KEY).any()
        assert (cells["joint_order_count"] <= cells["order_count"]).all()
        assert np.array_equal(classify(cells, cfg).to_numpy(), cells.state.to_numpy())
        known = cells.state.ne("UNKNOWN")
        ordered = cells.sort_values([EDGE, "window_end"])
        previous = ordered.groupby(EDGE).shift()
        adjacent_known = ((ordered.window_end - previous.window_end).eq(1800)
                          & ordered.state.ne("UNKNOWN") & previous.state.ne("UNKNOWN")
                          & previous.state.notna())
        row = {"date": date, "cells": len(cells), "non_unknown": int(known.sum()),
            "road_class_conflict_cells": int(cells.road_class_count.gt(1).sum()),
            "high_pace_without_low_speed": int((known & cells.pace_ratio.ge(1.5) & cells.joint_low_speed_share.lt(.5)).sum()),
            "low_speed_without_high_pace": int((known & cells.pace_ratio.lt(1.5) & cells.joint_low_speed_share.ge(.5)).sum()),
            "adjacent_known_pairs": int(adjacent_known.sum()),
            "adjacent_known_state_changes": int((adjacent_known & ordered.state.ne(previous.state)).sum()),
            "speed_cv_nonmissing_orders_not_validated": int(cells.speed_cv_bounded_order_count.sum()),
            "acceleration_nonmissing_orders_not_validated": int(cells.acceleration_rms_bounded_order_count.sum())}
        for name in ["joint_order_count", "joint_event_span_s", "latest_joint_age_s", "pace_ratio"]:
            series = cells[name].dropna()
            row[name + "_p50"] = float(series.median())
            row[name + "_p90"] = float(series.quantile(.9))
        rows.append(row)
    write_json(reports / "descriptive_checks.json", {"rows": rows,
        "verification": "cell identities, count nesting, and stored classification recomputed exactly",
        "prediction_inventory_correction": "Validation uses m3_validation directory; no scientific results recomputed"})
    write_json(reports / "prediction_inventory.json", {"rows": inventory_predictions(cfg["train_dates"]+cfg["evaluation_dates"])})
    plot_results(reports, pd.read_parquet(out / "illustrative_cases.parquet"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("stage3/config/traffic_state_batch1.json"))
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        describe_outputs(json.loads(args.config.read_text()))
    else:
        run(args.config)
