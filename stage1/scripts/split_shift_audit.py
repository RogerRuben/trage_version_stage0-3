"""Audit train-to-validation/test distribution shift for the temporal split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DIMENSIONS = ["lcs", "iis", "gns", "rts", "pmis"]
NUMERIC_COLUMNS = ["activity_intensity_index"] + [f"{dimension}_pct_link" for dimension in DIMENSIONS]
CATEGORICAL_COLUMNS = ["road_class", "time_bin", "traversal_quality"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=20)
    return parser.parse_args()


def date_split(config: dict) -> dict[str, str]:
    result = {date: "train" for date in config["train_dates"]}
    result[config["validation_date"]] = "validation"
    result[config["test_date"]] = "test"
    return result


def safe_read_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        return pd.read_parquet(path)


def load_link_sample(root: Path, date: str) -> pd.DataFrame:
    primitive_dir = root / "primitives" / f"day={date}"
    label_dir = root / "link_labels" / f"day={date}"
    parts = []
    for primitive_path in sorted(primitive_dir.glob("*.parquet")):
        part = primitive_path.stem.split("=")[-1]
        label_path = label_dir / f"part={part}.parquet"
        primitive = safe_read_columns(
            primitive_path,
            ["order_id", "link_id", "link_seq", "road_class", "time_bin", "traversal_quality", "activity_intensity_index"],
        )
        if label_path.exists():
            labels = safe_read_columns(
                label_path,
                ["order_id", "link_id", "link_seq"] + [f"{dimension}_pct_link" for dimension in DIMENSIONS],
            )
            primitive = primitive.merge(labels, on=["order_id", "link_id", "link_seq"], how="left")
        for column in NUMERIC_COLUMNS:
            if column not in primitive.columns:
                primitive[column] = np.nan
        parts.append(primitive[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=CATEGORICAL_COLUMNS + NUMERIC_COLUMNS)


def load_order_rates(root: Path, dates: list[str]) -> dict[str, float]:
    frames = []
    for date in dates:
        path = root / "order_labels" / f"day={date}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path, columns=[
                "high_odd_exceedance_85", "high_odd_exceedance_90", "high_odd_exceedance_95",
                "composite_mean", "composite_tail",
            ]))
    if not frames:
        return {}
    frame = pd.concat(frames, ignore_index=True)
    result = {column: float(frame[column].mean()) for column in frame.columns if column.startswith("high_odd")}
    for column in ["composite_mean", "composite_tail"]:
        result[f"{column}_mean"] = float(frame[column].mean())
    return result


def total_variation(train: pd.Series, target: pd.Series) -> float:
    train_dist = train.value_counts(normalize=True, dropna=False)
    target_dist = target.value_counts(normalize=True, dropna=False)
    keys = train_dist.index.union(target_dist.index)
    return float(0.5 * np.abs(train_dist.reindex(keys, fill_value=0) - target_dist.reindex(keys, fill_value=0)).sum())


def histogram_distance(train: pd.Series, target: pd.Series, bins: int) -> float:
    clean_train = train.dropna().astype(float)
    clean_target = target.dropna().astype(float)
    if clean_train.empty or clean_target.empty:
        return float("nan")
    low = float(min(clean_train.min(), clean_target.min()))
    high = float(max(clean_train.max(), clean_target.max()))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        return 0.0
    edges = np.linspace(low, high, bins + 1)
    train_hist = np.histogram(clean_train, bins=edges)[0].astype(float)
    target_hist = np.histogram(clean_target, bins=edges)[0].astype(float)
    train_hist = train_hist / train_hist.sum()
    target_hist = target_hist / target_hist.sum()
    return float(0.5 * np.abs(train_hist - target_hist).sum())


def compare_frames(train: pd.DataFrame, target: pd.DataFrame, target_split: str, bins: int) -> list[dict]:
    rows: list[dict] = []
    for column in CATEGORICAL_COLUMNS:
        rows.append({
            "target_split": target_split,
            "feature_group": "categorical",
            "feature": column,
            "metric": "total_variation_distance",
            "value": total_variation(train[column], target[column]),
            "train_nonnull": int(train[column].notna().sum()),
            "target_nonnull": int(target[column].notna().sum()),
        })
    for column in NUMERIC_COLUMNS:
        rows.append({
            "target_split": target_split,
            "feature_group": "numeric",
            "feature": column,
            "metric": "histogram_total_variation_distance",
            "value": histogram_distance(train[column], target[column], bins),
            "train_nonnull": int(train[column].notna().sum()),
            "target_nonnull": int(target[column].notna().sum()),
        })
    return rows


def verdict(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if value < 0.10:
        return "low"
    if value < 0.20:
        return "moderate"
    return "high"


def main() -> None:
    args = parse_args()
    config = json.loads(args.split_config.read_text(encoding="utf-8"))
    splits = date_split(config)
    train_dates = config["train_dates"]
    target_groups = {
        "validation": [config["validation_date"]],
        "test": [config["test_date"]],
    }
    train = pd.concat([load_link_sample(args.output_root, date) for date in train_dates], ignore_index=True)
    rows = []
    for split, dates in target_groups.items():
        target = pd.concat([load_link_sample(args.output_root, date) for date in dates], ignore_index=True)
        rows.extend(compare_frames(train, target, split, args.bins))
        train_rates = load_order_rates(args.output_root, train_dates)
        target_rates = load_order_rates(args.output_root, dates)
        for key, train_value in train_rates.items():
            target_value = target_rates.get(key, float("nan"))
            rows.append({
                "target_split": split,
                "feature_group": "order_high_stress_rate",
                "feature": key,
                "metric": "absolute_rate_difference",
                "value": abs(float(target_value) - float(train_value)) if np.isfinite(target_value) else float("nan"),
                "train_nonnull": len(train),
                "target_nonnull": len(target),
            })
    report = pd.DataFrame(rows)
    report["shift_level"] = report["value"].map(verdict)
    reports = args.output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report.to_csv(reports / "split_shift_audit.csv", index=False)

    lines = ["# Split shift audit", "", f"Experiment: `{config.get('experiment_name', 'unknown')}`", ""]
    lines.append(f"Train dates: {', '.join(train_dates)}")
    lines.append(f"Validation date: {config['validation_date']}")
    lines.append(f"Test date: {config['test_date']}")
    lines += ["", "## Summary", ""]
    for split in ["validation", "test"]:
        subset = report[report.target_split == split]
        high = int(subset.shift_level.eq("high").sum())
        moderate = int(subset.shift_level.eq("moderate").sum())
        lines.append(f"- {split}: {high} high-shift metrics, {moderate} moderate-shift metrics.")
    lines += [
        "",
        "Distance metrics use total variation distance. Values below 0.10 are low, 0.10-0.20 are moderate, and values at or above 0.20 require review.",
    ]
    (reports / "split_shift_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
