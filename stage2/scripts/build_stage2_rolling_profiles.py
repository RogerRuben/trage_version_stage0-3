"""Build leakage-safe rolling/OOF historical stress profiles.

The row-level output contains one resolved hierarchical profile per target,
rather than hundreds of duplicated level-specific columns. Lookup models for
every requested granularity are retained by day under ``models/``. This compact
layout is intentional for the 23M-row Stage2 dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from estimate_stage2_predictability_ceiling import load_split_sample, split_dates  # noqa: E402


TARGETS = ["lcs", "iis", "rts", "pmis"]
LEVELS = {
    "link_time_road": ["link_id", "time_bin", "road_class"],
    "link_time": ["link_id", "time_bin"],
    "link": ["link_id"],
    "area_time": ["area_grid", "time_bin"],
    "endpoint_time": ["endpoint_degree", "time_bin"],
    "road_time": ["road_class", "time_bin"],
}
STATS = ["mean", "std", "q50", "q75", "q90", "q95", "tail_rate_85", "tail_rate_90", "tail_rate_95", "count"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("stage2/output/link_dataset"))
    parser.add_argument("--split-config", type=Path, default=Path("split_config.json"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/rolling_profiles"))
    parser.add_argument("--mode", choices=["rolling_previous_k_days", "expanding_window", "leave_one_day_out"], default="rolling_previous_k_days")
    parser.add_argument("--previous-k-days", type=int, default=7)
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--max-rows-per-split", type=int, default=1_000_000, help="0 means all rows; use only with sufficient memory")
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def summarize(frame: pd.DataFrame, keys: list[str], target: str) -> pd.DataFrame:
    clean = frame[keys + [target]].dropna(subset=[target])
    if clean.empty:
        return pd.DataFrame(columns=keys + STATS)
    grouped = clean.groupby(keys, dropna=False, observed=True)[target]
    result = grouped.agg(["mean", "std", "count"])
    quantiles = grouped.quantile([0.50, 0.75, 0.90, 0.95]).unstack(-1)
    quantiles.columns = ["q50", "q75", "q90", "q95"]
    result = result.join(quantiles)
    for cutoff in [0.85, 0.90, 0.95]:
        rate = clean.assign(_tail=clean[target].ge(cutoff)).groupby(keys, dropna=False, observed=True)["_tail"].mean()
        result[f"tail_rate_{int(cutoff * 100)}"] = rate
    return result.reset_index()[keys + STATS]


def apply_level(rows: pd.DataFrame, model: pd.DataFrame, keys: list[str], minimum: int, unresolved: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    if model.empty or not unresolved.any():
        return pd.DataFrame(index=rows.index), np.zeros(len(rows), dtype=bool)
    candidates = rows.loc[unresolved, keys].merge(model, on=keys, how="left", sort=False)
    eligible_local = candidates["count"].fillna(0).ge(minimum).to_numpy()
    eligible = np.zeros(len(rows), dtype=bool)
    positions = np.flatnonzero(unresolved)
    eligible[positions[eligible_local]] = True
    values = pd.DataFrame(index=rows.index, columns=STATS, dtype=float)
    if eligible_local.any():
        values.loc[eligible, STATS] = candidates.loc[eligible_local, STATS].to_numpy()
    return values, eligible


def resolve_profiles(rows: pd.DataFrame, history: pd.DataFrame, target: str, minimum: int, model_dir: Path) -> pd.DataFrame:
    result = pd.DataFrame(index=rows.index)
    unresolved = np.ones(len(rows), dtype=bool)
    result[f"profile_{target}_level_used"] = "missing"
    for stat in STATS:
        result[f"profile_{target}_{stat}"] = np.nan
    for level, keys in LEVELS.items():
        model = summarize(history, keys, f"target_{target}_pct")
        model.to_parquet(model_dir / f"{target}_{level}.parquet", index=False, compression="zstd")
        values, eligible = apply_level(rows, model, keys, minimum, unresolved)
        if eligible.any():
            for stat in STATS:
                result.loc[eligible, f"profile_{target}_{stat}"] = values.loc[eligible, stat].to_numpy()
            result.loc[eligible, f"profile_{target}_level_used"] = level
            unresolved[eligible] = False
    global_values = history[f"target_{target}_pct"].dropna()
    if unresolved.any() and not global_values.empty:
        global_stats = {
            "mean": global_values.mean(), "std": global_values.std(),
            "q50": global_values.quantile(0.50), "q75": global_values.quantile(0.75),
            "q90": global_values.quantile(0.90), "q95": global_values.quantile(0.95),
            "tail_rate_85": global_values.ge(0.85).mean(), "tail_rate_90": global_values.ge(0.90).mean(),
            "tail_rate_95": global_values.ge(0.95).mean(), "count": len(global_values),
        }
        for stat, value in global_stats.items():
            result.loc[unresolved, f"profile_{target}_{stat}"] = value
        result.loc[unresolved, f"profile_{target}_level_used"] = "global"
    result[f"profile_{target}_coverage"] = result[f"profile_{target}_mean"].notna()
    result[f"profile_{target}_missing"] = ~result[f"profile_{target}_coverage"]
    return result


def history_dates(all_dates: list[str], current: str, train_dates: list[str], mode: str, k: int) -> list[str]:
    if mode == "leave_one_day_out":
        return [date for date in train_dates if date != current]
    previous = [date for date in all_dates if date < current]
    return previous[-k:] if mode == "rolling_previous_k_days" else previous


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    train_dates, validation_dates, test_dates = split_dates(args.split_config)
    split_map = {"train": train_dates, "validation": validation_dates, "test": test_dates}
    columns = [
        "order_id", "date", "link_id", "link_seq", "time_bin", "road_class", "area_grid", "endpoint_degree",
    ]
    for target in TARGETS:
        columns.extend([f"target_{target}_pct", f"{target}_valid"])
    frames: dict[str, pd.DataFrame] = {}
    for number, split in enumerate(split_map):
        frame = load_split_sample(
            args.dataset_root / f"{split}.parquet", columns, args.max_rows_per_split,
            args.batch_size, args.seed + number,
        )
        frame["split"] = split
        frames[split] = frame
    all_rows = pd.concat(frames.values(), ignore_index=True)
    all_dates = sorted(all_rows["date"].astype(str).unique())
    outputs: dict[str, list[pd.DataFrame]] = {key: [] for key in split_map}
    day_manifest: dict[str, dict] = {}
    for date in all_dates:
        rows = all_rows[all_rows["date"].astype(str).eq(date)].copy()
        historical_dates = history_dates(all_dates, date, train_dates, args.mode, args.previous_k_days)
        history = all_rows[all_rows["date"].astype(str).isin(historical_dates)]
        base = rows[["order_id", "date", "link_id", "link_seq"]].reset_index(drop=True)
        model_dir = args.output_root / "models" / f"day={date}"
        model_dir.mkdir(parents=True, exist_ok=True)
        for target in TARGETS:
            base = pd.concat([base, resolve_profiles(rows.reset_index(drop=True), history, target, args.min_count, model_dir)], axis=1)
        base["profile_mode"] = args.mode
        base["profile_history_start"] = min(historical_dates) if historical_dates else None
        base["profile_history_end"] = max(historical_dates) if historical_dates else None
        split = str(rows["split"].iloc[0])
        outputs[split].append(base)
        day_manifest[date] = {"split": split, "rows": len(rows), "history_dates": historical_dates}
        print(f"rolling profiles day={date} rows={len(rows):,} history={historical_dates}", flush=True)
    names = {"train": "train_rolling_profiles.parquet", "validation": "validation_rolling_profiles.parquet", "test": "test_rolling_profiles.parquet"}
    for split, parts in outputs.items():
        output = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        output.to_parquet(args.output_root / names[split], index=False, compression="zstd")
    manifest = {
        "mode": args.mode, "previous_k_days": args.previous_k_days, "min_count": args.min_count,
        "max_rows_per_split": args.max_rows_per_split, "compact_resolved_profile": True,
        "level_lookup_models_retained": True, "self_inclusion": False if args.mode != "leave_one_day_out" else "day_excluded",
        "future_day_leakage": False, "days": day_manifest,
    }
    (args.output_root / "rolling_profile_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

