"""Estimate how much Stage2 stress is repeatable under existing conditions.

The default run uses an order-hash sample from every split so that all dates are
represented without loading the 23M-row table into memory. Pass
``--max-rows-per-split 0`` for a census run. Raw-label analysis reads the Stage1
primitive partitions with the same deterministic order sampling rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score


TARGETS = ["lcs", "iis", "rts", "pmis"]
BASE_COLUMNS = ["order_id", "date", "link_id", "time_bin", "road_class", "area_grid"]
GROUPINGS = {
    "link": ["link_id"],
    "time_bin": ["time_bin"],
    "road_class": ["road_class"],
    "area_grid": ["area_grid"],
    "day": ["date"],
    "link_time": ["link_id", "time_bin"],
    "link_time_day": ["link_id", "time_bin", "date"],
    "area_time": ["area_grid", "time_bin"],
    "road_time": ["road_class", "time_bin"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("stage2/output/link_dataset"))
    parser.add_argument("--primitive-root", type=Path, default=Path("stage1/output/prediction_split/primitives"))
    parser.add_argument("--split-config", type=Path, default=Path("split_config.json"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/predictability_ceiling"))
    parser.add_argument("--max-rows-per-split", type=int, default=1_000_000, help="0 means all rows")
    parser.add_argument("--raw-order-sample-rate", type=float, default=0.08)
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def split_dates(path: Path) -> tuple[list[str], list[str], list[str]]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    train = [str(value) for value in cfg["train_dates"]]
    validation = [str(cfg["validation_date"])]
    test = [str(cfg["test_date"])]
    return train, validation, test


def order_hash_mask(values: pd.Series, fraction: float, seed: int) -> np.ndarray:
    if fraction >= 1:
        return np.ones(len(values), dtype=bool)
    hashed = pd.util.hash_pandas_object(values.astype(str), index=False).to_numpy(dtype="uint64")
    hashed ^= np.uint64(seed)
    return (hashed % np.uint64(1_000_000)) < np.uint64(max(1, int(fraction * 1_000_000)))


def load_split_sample(path: Path, columns: list[str], maximum: int, batch_size: int, seed: int) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema_arrow.names)
    selected = [column for column in columns if column in available]
    fraction = 1.0 if maximum <= 0 else min(1.0, maximum / max(parquet.metadata.num_rows, 1))
    parts: list[pd.DataFrame] = []
    for batch in parquet.iter_batches(columns=selected, batch_size=batch_size):
        frame = batch.to_pandas()
        if fraction < 1 and "order_id" in frame:
            frame = frame.loc[order_hash_mask(frame["order_id"], fraction, seed)]
        if not frame.empty:
            parts.append(frame)
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=selected)
    if maximum > 0 and len(result) > maximum:
        result = result.sample(n=maximum, random_state=seed)
    return result


def load_percentile_frames(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    columns = BASE_COLUMNS[:]
    for target in TARGETS:
        columns.extend([f"target_{target}_pct", f"{target}_valid"])
    result = {}
    for number, split in enumerate(["train", "validation", "test"]):
        frame = load_split_sample(
            args.dataset_root / f"{split}.parquet", columns, args.max_rows_per_split,
            args.batch_size, args.seed + number,
        )
        frame["split"] = split
        result[split] = frame
    return result


def load_raw_frames(args: argparse.Namespace, dates_by_split: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    columns = BASE_COLUMNS + [f"{target}_raw" for target in TARGETS]
    result: dict[str, pd.DataFrame] = {}
    for split, dates in dates_by_split.items():
        parts: list[pd.DataFrame] = []
        for date in dates:
            for path in sorted((args.primitive_root / f"day={date}").glob("*.parquet")):
                parquet = pq.ParquetFile(path)
                available = set(parquet.schema_arrow.names)
                selected = [column for column in columns if column in available]
                for batch in parquet.iter_batches(columns=selected, batch_size=args.batch_size):
                    frame = batch.to_pandas()
                    frame = frame.loc[order_hash_mask(frame["order_id"], args.raw_order_sample_rate, args.seed)]
                    if not frame.empty:
                        parts.append(frame)
        result[split] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
        result[split]["split"] = split
    return result


def group_stats(frame: pd.DataFrame, keys: list[str], value: str) -> pd.DataFrame:
    clean = frame[keys + [value]].dropna(subset=[value]).copy()
    if clean.empty:
        return pd.DataFrame()
    clean["_square"] = clean[value].astype(float) ** 2
    return clean.groupby(keys, dropna=False, observed=True).agg(
        n=(value, "size"), total=(value, "sum"), total_sq=("_square", "sum")
    ).reset_index()


def variance_row(frame: pd.DataFrame, target: str, grouping: str, keys: list[str], scale: str) -> dict:
    clean = frame.dropna(subset=[target])
    stats = group_stats(clean, keys, target)
    n = int(stats["n"].sum()) if not stats.empty else 0
    if n < 2 or len(stats) < 2:
        return {"label_scale": scale, "target": target, "grouping": grouping, "rows": n}
    grand = float(stats["total"].sum() / n)
    total_ss = float((clean[target].astype(float) - grand).pow(2).sum())
    means = stats["total"] / stats["n"]
    between_ss = float((stats["n"] * (means - grand) ** 2).sum())
    within_ss = float((stats["total_sq"] - stats["total"] ** 2 / stats["n"]).sum())
    k = len(stats)
    ms_between = between_ss / max(k - 1, 1)
    ms_within = within_ss / max(n - k, 1)
    n0 = (n - float((stats["n"] ** 2).sum()) / n) / max(k - 1, 1)
    denominator = ms_between + max(n0 - 1, 0) * ms_within
    icc = (ms_between - ms_within) / denominator if denominator > 0 else np.nan
    return {
        "label_scale": scale,
        "target": target,
        "grouping": grouping,
        "rows": n,
        "groups": int(k),
        "total_variance": total_ss / max(n - 1, 1),
        "between_variance_share": between_ss / total_ss if total_ss > 0 else np.nan,
        "within_residual_share": within_ss / total_ss if total_ss > 0 else np.nan,
        "icc_1": float(icc),
    }


def keyed_mean(train: pd.DataFrame, keys: list[str], target: str) -> pd.Series:
    return train.dropna(subset=[target]).groupby(keys, dropna=False, observed=True)[target].mean()


def map_profile(frame: pd.DataFrame, keys: list[str], profile: pd.Series, fallback: float) -> np.ndarray:
    if len(keys) == 1:
        pred = frame[keys[0]].map(profile)
    else:
        index = pd.MultiIndex.from_frame(frame[keys])
        pred = pd.Series(profile.reindex(index).to_numpy(), index=frame.index)
    return pred.fillna(fallback).to_numpy(dtype=float)


def top_metrics(y: np.ndarray, pred: np.ndarray, threshold: float) -> dict[str, float]:
    valid = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[valid], pred[valid]
    high = y >= threshold
    result = {
        "rows": int(len(y)),
        "tail_threshold": float(threshold),
        "tail_rate": float(high.mean()) if len(high) else np.nan,
        "mae": float(mean_absolute_error(y, pred)) if len(y) else np.nan,
        "rmse": float(mean_squared_error(y, pred, squared=False)) if len(y) else np.nan,
        "spearman": float(spearmanr(y, pred).statistic) if len(y) > 1 else np.nan,
    }
    if high.any() and (~high).any():
        result["auc"] = float(roc_auc_score(high, pred))
        result["ap"] = float(average_precision_score(high, pred))
    else:
        result["auc"] = result["ap"] = np.nan
    for share in [0.05, 0.10]:
        count = max(1, int(len(y) * share))
        top = high[np.argsort(pred)[-count:]] if len(y) else np.array([], dtype=bool)
        precision = float(top.mean()) if len(top) else np.nan
        result[f"lift_top{int(share * 100)}"] = precision / high.mean() if len(high) and high.mean() > 0 else np.nan
    return result


def profile_oracle_rows(frames: dict[str, pd.DataFrame], target: str, scale: str) -> list[dict]:
    train = frames["train"].dropna(subset=[target]).copy()
    evaluation = pd.concat([frames["validation"], frames["test"]], ignore_index=True).dropna(subset=[target])
    if train.empty or evaluation.empty:
        return []
    fallback = float(train[target].mean())
    threshold = 0.90 if scale == "percentile" else float(train[target].quantile(0.90))
    definitions = {
        "global_mean": [],
        "link_historical_mean": ["link_id"],
        "link_time_historical_mean": ["link_id", "time_bin"],
        "link_time_road_profile": ["link_id", "time_bin", "road_class"],
        "area_time_profile": ["area_grid", "time_bin"],
    }
    rows: list[dict] = []
    y = evaluation[target].to_numpy(dtype=float)
    for name, keys in definitions.items():
        pred = np.full(len(evaluation), fallback) if not keys else map_profile(evaluation, keys, keyed_mean(train, keys, target), fallback)
        row = {"label_scale": scale, "target": target, "oracle": name, **top_metrics(y, pred, threshold)}
        rows.append(row)

    # Current-day leave-one-out is deliberately non-deployable and estimates an
    # upper bound when contemporaneous same-day outcomes are available.
    keys = ["date", "link_id", "time_bin"]
    same_day = evaluation[keys + [target]].copy()
    totals = same_day.groupby(keys, dropna=False)[target].transform("sum")
    counts = same_day.groupby(keys, dropna=False)[target].transform("count")
    pred = ((totals - same_day[target]) / (counts - 1).replace(0, np.nan)).fillna(fallback).to_numpy(dtype=float)
    rows.append({"label_scale": scale, "target": target, "oracle": "current_day_leave_one_out", **top_metrics(y, pred, threshold)})

    # Expanding rolling link-time profile: each day uses strictly earlier days.
    all_days = pd.concat(frames.values(), ignore_index=True).dropna(subset=[target]).sort_values("date")
    rolling_y: list[np.ndarray] = []
    rolling_pred: list[np.ndarray] = []
    history = pd.DataFrame(columns=all_days.columns)
    for date in sorted(all_days["date"].astype(str).unique()):
        day = all_days[all_days["date"].astype(str).eq(date)]
        if not history.empty and date in set(evaluation["date"].astype(str)):
            hist_mean = float(history[target].mean())
            profile = keyed_mean(history, ["link_id", "time_bin"], target)
            rolling_y.append(day[target].to_numpy(dtype=float))
            rolling_pred.append(map_profile(day, ["link_id", "time_bin"], profile, hist_mean))
        history = pd.concat([history, day], ignore_index=True)
    if rolling_y:
        rows.append({
            "label_scale": scale,
            "target": target,
            "oracle": "expanding_rolling_link_time",
            **top_metrics(np.concatenate(rolling_y), np.concatenate(rolling_pred), threshold),
        })
    return rows


def write_summary(output: Path, variance: pd.DataFrame, oracles: pd.DataFrame, scope: dict) -> None:
    lines = [
        "# Stage2 predictability ceiling summary", "",
        "This is a predictability diagnostic, not a deployable model result.", "",
        f"Analysis scope: `{json.dumps(scope, ensure_ascii=False)}`", "",
        "## Main variance signal", "",
    ]
    focus = variance[variance["grouping"].eq("link_time")].copy()
    if not focus.empty:
        lines.append(focus[["label_scale", "target", "between_variance_share", "within_residual_share", "icc_1"]].to_markdown(index=False, floatfmt=".4f"))
    lines.extend(["", "## Profile-oracle comparison", ""])
    if not oracles.empty:
        best = oracles.sort_values(["label_scale", "target", "ap"], ascending=[True, True, False]).groupby(["label_scale", "target"], as_index=False).first()
        lines.append(best[["label_scale", "target", "oracle", "auc", "ap", "spearman", "lift_top5", "rmse"]].to_markdown(index=False, floatfmt=".4f"))
    lines.extend([
        "", "## Interpretation guardrails", "",
        "- High within-link-time residual share indicates that static road/time inputs cannot explain most realized variation.",
        "- A strong current-day leave-one-out oracle but weak historical profile indicates missing contemporaneous traffic state.",
        "- Raw and percentile results must be compared before choosing Stage2 targets.",
        "- Dynamic oracle results remain pending until strictly lagged features are built.",
    ])
    (output / "predictability_ceiling_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    train_dates, validation_dates, test_dates = split_dates(args.split_config)
    date_map = {"train": train_dates, "validation": validation_dates, "test": test_dates}
    percentile = load_percentile_frames(args)
    raw = load_raw_frames(args, date_map) if args.primitive_root.exists() else {}

    variance_rows: list[dict] = []
    oracle_rows: list[dict] = []
    combined_pct = pd.concat(percentile.values(), ignore_index=True)
    for target in TARGETS:
        pct_col = f"target_{target}_pct"
        for grouping, keys in GROUPINGS.items():
            variance_rows.append(variance_row(combined_pct, pct_col, grouping, keys, "percentile"))
        oracle_rows.extend(profile_oracle_rows(percentile, pct_col, "percentile"))
        if raw:
            raw_col = f"{target}_raw"
            combined_raw = pd.concat(raw.values(), ignore_index=True)
            for grouping, keys in GROUPINGS.items():
                variance_rows.append(variance_row(combined_raw, raw_col, grouping, keys, "raw"))
            oracle_rows.extend(profile_oracle_rows(raw, raw_col, "raw"))

    variance = pd.DataFrame(variance_rows)
    oracles = pd.DataFrame(oracle_rows)
    variance.to_csv(args.output_root / "variance_decomposition_by_target.csv", index=False)
    variance[[column for column in ["label_scale", "target", "grouping", "rows", "groups", "icc_1", "between_variance_share", "within_residual_share"] if column in variance]].to_csv(
        args.output_root / "icc_repeatability_by_target.csv", index=False
    )
    oracles.to_csv(args.output_root / "profile_oracle_metrics.csv", index=False)
    oracles.to_csv(args.output_root / "raw_vs_percentile_metrics.csv", index=False)
    pd.DataFrame([{
        "status": "pending_lagged_feature_build",
        "reason": "run build_stage2_lagged_traffic_features.py before dynamic oracle evaluation",
    }]).to_csv(args.output_root / "dynamic_oracle_metrics.csv", index=False)
    scope = {
        "max_rows_per_split": args.max_rows_per_split,
        "raw_order_sample_rate": args.raw_order_sample_rate,
        "percentile_rows": {key: len(value) for key, value in percentile.items()},
        "raw_rows": {key: len(value) for key, value in raw.items()},
        "train_dates": train_dates,
        "validation_dates": validation_dates,
        "test_dates": test_dates,
    }
    (args.output_root / "predictability_ceiling_manifest.json").write_text(json.dumps(scope, indent=2), encoding="utf-8")
    write_summary(args.output_root, variance, oracles, scope)
    print(json.dumps(scope, indent=2))


if __name__ == "__main__":
    main()

