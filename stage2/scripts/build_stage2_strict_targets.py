"""Build raw, tail, percentile and rolling-uncertainty Stage2 targets.

Uncertainty is the strictly historical standard deviation for the same
link/time bin, with link and global fallbacks. Target thresholds are frozen from
the configured measurement-fit dates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["lcs", "iis", "rts", "pmis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primitive-root", type=Path, default=Path("stage1/output/prediction_split/primitives"))
    parser.add_argument("--label-root", type=Path, default=Path("stage1/output/prediction_split/link_labels"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/strict_targets"))
    parser.add_argument("--fit-dates", default="20161009,20161010,20161011,20161012,20161013,20161014,20161015")
    parser.add_argument("--target-dates", default="20161009,20161010,20161011,20161012,20161013,20161014,20161015,20161016,20161017,20161018,20161019")
    parser.add_argument("--history-days", type=int, default=7)
    parser.add_argument("--min-history-count", type=int, default=20)
    parser.add_argument("--histogram-bins", type=int, default=2000)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def parse_dates(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def daily_stats(root: Path, date: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if all((output / f"{target}.parquet").exists() for target in TARGETS):
        return
    accumulators = {target: [] for target in TARGETS}
    for path in sorted((root / f"day={date}").glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["link_id", "time_bin"] + [f"{target}_raw" for target in TARGETS])
        for target in TARGETS:
            value = f"{target}_raw"
            clean = frame[["link_id", "time_bin", value]].dropna(subset=[value]).copy()
            clean["sum"] = clean[value].astype(float)
            clean["sum_sq"] = clean[value].astype(float) ** 2
            clean["count"] = 1
            accumulators[target].append(clean.groupby(["link_id", "time_bin"], as_index=False)[["sum", "sum_sq", "count"]].sum())
    for target, parts in accumulators.items():
        combined = pd.concat(parts, ignore_index=True).groupby(["link_id", "time_bin"], as_index=False)[["sum", "sum_sq", "count"]].sum()
        combined.to_parquet(output / f"{target}.parquet", index=False, compression="zstd")


def combine_history(model_root: Path, dates: list[str], target: str) -> pd.DataFrame:
    parts = [pd.read_parquet(model_root / f"day={date}" / f"{target}.parquet") for date in dates]
    if not parts:
        return pd.DataFrame(columns=["link_id", "time_bin", "sum", "sum_sq", "count"])
    return pd.concat(parts, ignore_index=True).groupby(["link_id", "time_bin"], as_index=False)[["sum", "sum_sq", "count"]].sum()


def standard_deviation(stats: pd.DataFrame) -> pd.Series:
    numerator = stats["sum_sq"] - stats["sum"] ** 2 / stats["count"].clip(lower=1)
    return np.sqrt((numerator / (stats["count"] - 1).clip(lower=1)).clip(lower=0))


def uncertainty(frame: pd.DataFrame, history: pd.DataFrame, minimum: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = np.full(len(frame), np.nan)
    count_result = np.zeros(len(frame), dtype="int32")
    level = np.full(len(frame), "missing", dtype=object)
    if history.empty:
        return result, count_result, level
    history = history.copy()
    history["std"] = standard_deviation(history)
    lookup = history.set_index(["link_id", "time_bin"])
    index = pd.MultiIndex.from_frame(frame[["link_id", "time_bin"]])
    counts = lookup["count"].reindex(index).fillna(0).to_numpy(dtype=int)
    values = lookup["std"].reindex(index).to_numpy(dtype=float)
    eligible = (counts >= minimum) & np.isfinite(values)
    result[eligible] = values[eligible]
    count_result[eligible] = counts[eligible]
    level[eligible] = "link_time"

    unresolved = ~eligible
    link = history.groupby("link_id", as_index=True)[["sum", "sum_sq", "count"]].sum()
    link["std"] = standard_deviation(link)
    link_counts = frame["link_id"].map(link["count"]).fillna(0).to_numpy(dtype=int)
    link_values = frame["link_id"].map(link["std"]).to_numpy(dtype=float)
    eligible = unresolved & (link_counts >= minimum) & np.isfinite(link_values)
    result[eligible] = link_values[eligible]
    count_result[eligible] = link_counts[eligible]
    level[eligible] = "link"

    unresolved = ~np.isfinite(result)
    total_count = int(history["count"].sum())
    total_sum = float(history["sum"].sum())
    total_sq = float(history["sum_sq"].sum())
    if total_count > 1:
        global_std = np.sqrt(max(0.0, (total_sq - total_sum**2 / total_count) / (total_count - 1)))
        result[unresolved] = global_std
        count_result[unresolved] = total_count
        level[unresolved] = "global"
    return result, count_result, level


def frozen_thresholds(root: Path, fit_dates: list[str], bins: int) -> dict[str, float]:
    counts = {target: np.zeros(bins, dtype="int64") for target in TARGETS}
    for date in fit_dates:
        for path in sorted((root / f"day={date}").glob("*.parquet")):
            frame = pd.read_parquet(path, columns=[f"{target}_raw" for target in TARGETS])
            for target in TARGETS:
                values = frame[f"{target}_raw"].dropna().clip(0, 1).to_numpy(dtype=float)
                index = np.minimum((values * bins).astype(int), bins - 1)
                counts[target] += np.bincount(index, minlength=bins)
    thresholds = {}
    for target, histogram in counts.items():
        cumulative = np.cumsum(histogram)
        position = np.searchsorted(cumulative, cumulative[-1] * 0.90, side="left")
        thresholds[target] = float((position + 0.5) / bins)
    return thresholds


def main() -> None:
    args = parse_args()
    fit_dates = parse_dates(args.fit_dates)
    target_dates = parse_dates(args.target_dates)
    args.output_root.mkdir(parents=True, exist_ok=True)
    model_root = args.output_root / "models" / "daily_stats"
    for date in target_dates:
        daily_stats(args.primitive_root, date, model_root / f"day={date}")
    thresholds = frozen_thresholds(args.primitive_root, fit_dates, args.histogram_bins)
    day_manifest = {}
    for date in target_dates:
        output_day = args.output_root / f"day={date}"
        marker = output_day / "day_manifest.json"
        if args.skip_existing and marker.exists():
            day_manifest[date] = json.loads(marker.read_text(encoding="utf-8"))
            print(f"strict targets day={date} already complete", flush=True)
            continue
        previous = [value for value in target_dates if value < date][-args.history_days:]
        histories = {target: combine_history(model_root, previous, target) for target in TARGETS}
        output_day.mkdir(parents=True, exist_ok=True)
        rows = 0
        for primitive_path in sorted((args.primitive_root / f"day={date}").glob("*.parquet")):
            label_path = args.label_root / f"day={date}" / primitive_path.name
            primitive = pd.read_parquet(primitive_path, columns=[
                "order_id", "driver_id", "date", "link_id", "link_seq", "time_bin", "traversal_quality",
            ] + [f"{target}_raw" for target in TARGETS])
            labels = pd.read_parquet(label_path, columns=[
                "order_id", "link_id", "link_seq",
            ] + [f"{target}_pct_link" for target in TARGETS])
            frame = primitive.merge(labels, on=["order_id", "link_id", "link_seq"], how="left", validate="one_to_one")
            for target in TARGETS:
                raw = f"{target}_raw"
                pct = f"{target}_pct_link"
                frame[f"target_{target}_raw"] = frame[raw]
                frame[f"target_{target}_pct"] = frame[pct]
                frame[f"target_{target}_tail90_raw"] = frame[raw].ge(thresholds[target]).where(frame[raw].notna())
                frame[f"target_{target}_tail90_pct"] = frame[pct].ge(0.90).where(frame[pct].notna())
                std, count, level = uncertainty(frame, histories[target], args.min_history_count)
                frame[f"target_{target}_uncertainty"] = std
                frame[f"target_{target}_history_count"] = count
                frame[f"target_{target}_uncertainty_level"] = level
                frame[f"target_{target}_valid"] = frame[raw].notna() & frame[pct].notna()
            keep = ["order_id", "driver_id", "date", "link_id", "link_seq", "time_bin", "traversal_quality"]
            for target in TARGETS:
                keep.extend([
                    f"target_{target}_raw", f"target_{target}_pct", f"target_{target}_tail90_raw",
                    f"target_{target}_tail90_pct", f"target_{target}_uncertainty",
                    f"target_{target}_history_count", f"target_{target}_uncertainty_level", f"target_{target}_valid",
                ])
            frame[keep].to_parquet(output_day / primitive_path.name, index=False, compression="zstd")
            rows += len(frame)
        day_manifest[date] = {"rows": rows, "history_dates": previous, "complete": True}
        marker.write_text(json.dumps(day_manifest[date], indent=2), encoding="utf-8")
        print(f"strict targets day={date} rows={rows:,} history={previous}", flush=True)
    manifest = {
        "fit_dates": fit_dates, "target_dates": target_dates, "history_days": args.history_days,
        "min_history_count": args.min_history_count, "raw_tail90_thresholds": thresholds,
        "percentile_tail_threshold": 0.90, "uncertainty": "strictly historical std with link-time/link/global fallback",
        "days": day_manifest,
    }
    (args.output_root / "strict_target_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
