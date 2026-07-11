"""Aggregate Stage2 link-level scores into order-level Stage3 candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DIMENSIONS = ["lcs", "iis", "rts", "pmis"]
SPLITS = ["train", "validation", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("stage2/output/link_dataset"))
    parser.add_argument("--prediction-root", type=Path, default=Path("stage2/output/baselines"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/order_features"))
    parser.add_argument("--splits", default="train,validation,test")
    return parser.parse_args()


def max_consecutive(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values)
    if not valid.any():
        return np.nan
    valid_weights = weights[valid]
    if np.nansum(valid_weights) <= 0:
        valid_weights = np.ones(valid.sum(), dtype=float)
    return float(np.average(values[valid], weights=valid_weights))


def tail_share(values: np.ndarray, weights: np.ndarray, cutoff: float) -> float:
    valid = np.isfinite(values)
    if not valid.any():
        return np.nan
    valid_weights = weights[valid]
    if np.nansum(valid_weights) <= 0:
        valid_weights = np.ones(valid.sum(), dtype=float)
    return float(valid_weights[values[valid] >= cutoff].sum() / valid_weights.sum())


def segment_mean(values: np.ndarray, position: np.ndarray, low: float, high: float) -> float:
    mask = np.isfinite(values) & (position >= low) & (position < high)
    return float(np.nanmean(values[mask])) if mask.any() else np.nan


def aggregate_group(group: pd.DataFrame, score_source: str) -> dict:
    group = group.sort_values("link_seq", kind="mergesort")
    row: dict[str, object] = {
        "order_id": group.order_id.iloc[0],
        "driver_id": group.driver_id.iloc[0] if "driver_id" in group.columns else None,
        "date": group.date.iloc[0] if "date" in group.columns else None,
        "link_count": int(len(group)),
        "score_source": score_source,
    }
    weights = pd.to_numeric(group.get("link_length_m", pd.Series(1.0, index=group.index)), errors="coerce").fillna(0).clip(lower=0).to_numpy(dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(group), dtype=float)
    position = pd.to_numeric(group.get("position_ratio", pd.Series(np.nan, index=group.index)), errors="coerce").fillna(0).to_numpy(dtype=float)
    endpoint_degree = pd.to_numeric(group.get("endpoint_degree", pd.Series(0, index=group.index)), errors="coerce").fillna(0).to_numpy(dtype=float)
    intersection_mask = endpoint_degree >= 3
    candidates = []
    valid_shares = []

    for dimension in DIMENSIONS:
        score_col = f"pred_{dimension}" if f"pred_{dimension}" in group.columns else f"target_{dimension}_pct"
        valid_col = f"{dimension}_valid"
        scores = pd.to_numeric(group[score_col], errors="coerce").to_numpy(dtype=float)
        scores = np.clip(scores, 0, 1)
        if valid_col in group.columns:
            valid_mask = group[valid_col].fillna(False).to_numpy(dtype=bool)
            scores = np.where(valid_mask, scores, np.nan)
            valid_shares.append(float(valid_mask.mean()))
        else:
            valid_mask = np.isfinite(scores)
            valid_shares.append(float(valid_mask.mean()))

        finite = scores[np.isfinite(scores)]
        prefix = dimension
        row[f"{prefix}_mean"] = float(np.nanmean(scores)) if len(finite) else np.nan
        row[f"{prefix}_weighted_mean_by_length"] = weighted_mean(scores, weights)
        row[f"{prefix}_sum"] = float(np.nansum(scores)) if len(finite) else np.nan
        row[f"{prefix}_max"] = float(np.nanmax(scores)) if len(finite) else np.nan
        for q in [0.75, 0.90, 0.95]:
            row[f"{prefix}_q{int(q * 100)}"] = float(np.nanquantile(scores, q)) if len(finite) else np.nan
        row[f"{prefix}_std"] = float(np.nanstd(scores)) if len(finite) else np.nan
        for cutoff in [0.85, 0.90, 0.95]:
            row[f"{prefix}_tail_share_{int(cutoff * 100)}"] = tail_share(scores, weights, cutoff)
        high = np.isfinite(scores) & (scores >= 0.90)
        row[f"{prefix}_consecutive_high_stress_links"] = max_consecutive(high)
        row[f"{prefix}_high_stress_persistence"] = tail_share(scores, weights, 0.90)
        row[f"{prefix}_early_route_stress"] = segment_mean(scores, position, 0.10, 0.33)
        row[f"{prefix}_middle_route_stress"] = segment_mean(scores, position, 0.33, 0.66)
        row[f"{prefix}_late_route_stress"] = segment_mean(scores, position, 0.66, 0.90)
        row[f"{prefix}_pickup_side_stress"] = segment_mean(scores, position, -0.001, 0.10)
        row[f"{prefix}_dropoff_side_stress"] = segment_mean(scores, position, 0.90, 1.001)
        intersection_scores = scores[intersection_mask]
        row[f"{prefix}_intersection_cluster_exposure"] = (
            float(np.nanmean(intersection_scores)) if np.isfinite(intersection_scores).any() else np.nan
        )
        if dimension == "iis":
            row["high_IIS_segment_count"] = int(np.diff(np.r_[False, high, False]).clip(min=0).sum())
            row["iis_valid_link_count"] = int(valid_mask.sum())
            row["iis_valid_route_share"] = float(valid_mask.mean())
        candidates.append(row[f"{prefix}_weighted_mean_by_length"])

    row["LCS_score_candidate"] = row["lcs_weighted_mean_by_length"]
    row["IIS_score_candidate"] = row["iis_weighted_mean_by_length"]
    row["RTS_score_candidate"] = row["rts_weighted_mean_by_length"]
    row["PMIS_score_candidate"] = row["pmis_weighted_mean_by_length"]
    candidate_values = np.array(candidates, dtype=float)
    row["composite_ODD_stress_candidate"] = float(np.nanmean(candidate_values)) if np.isfinite(candidate_values).any() else np.nan
    row["uncertainty_candidate"] = (
        float(np.nanstd(candidate_values) + np.nanmean([1 - value for value in valid_shares]))
        if np.isfinite(candidate_values).any()
        else np.nan
    )
    return row


def load_predictions(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    columns = ["order_id", "link_id", "link_seq"] + [f"pred_{dimension}" for dimension in DIMENSIONS]
    available = set(pq.ParquetFile(path).schema_arrow.names)
    return pd.read_parquet(path, columns=[column for column in columns if column in available])


def aggregate_split(split: str, args: argparse.Namespace) -> dict:
    dataset_path = args.dataset_root / f"{split}.parquet"
    prediction_path = args.prediction_root / f"predictions_{split}.parquet"
    predictions = load_predictions(prediction_path)
    score_source = "baseline_prediction" if predictions is not None else "stage2_label"
    parquet = pq.ParquetFile(dataset_path)
    columns = [
        "order_id", "driver_id", "date", "link_id", "link_seq", "link_length_m", "position_ratio",
        "endpoint_degree",
    ] + [f"target_{dimension}_pct" for dimension in DIMENSIONS] + [f"{dimension}_valid" for dimension in DIMENSIONS]
    available = set(parquet.schema_arrow.names)
    columns = [column for column in columns if column in available]
    parts = []
    for row_group in range(parquet.metadata.num_row_groups):
        frame = parquet.read_row_group(row_group, columns=columns).to_pandas()
        if predictions is not None:
            frame = frame.merge(predictions, on=["order_id", "link_id", "link_seq"], how="left")
        rows = [aggregate_group(group, score_source) for _, group in frame.groupby("order_id", sort=False)]
        parts.append(pd.DataFrame(rows))
        print(f"{split} row_group={row_group + 1}/{parquet.metadata.num_row_groups} orders={len(rows):,}", flush=True)
    orders = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    # Safety consolidation in case an order ever spans row groups.
    if orders.order_id.duplicated().any():
        orders = orders.sort_values(["order_id", "link_count"]).drop_duplicates("order_id", keep="last")
    output_path = args.output_root / f"{split}_order_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    orders.to_parquet(output_path, index=False, compression="zstd")
    return {
        "split": split,
        "orders": int(len(orders)),
        "score_source": score_source,
        "output": str(output_path),
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    requested_splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    summaries = [aggregate_split(split, args) for split in requested_splits]
    completed = {summary["split"] for summary in summaries}
    for split in SPLITS:
        if split in completed:
            continue
        output_path = args.output_root / f"{split}_order_features.parquet"
        if output_path.exists():
            summaries.append({
                "split": split,
                "orders": int(pq.ParquetFile(output_path).metadata.num_rows),
                "score_source": "stage2_label" if split == "train" else "existing_output",
                "output": str(output_path),
            })
    summaries = sorted(summaries, key=lambda item: SPLITS.index(item["split"]) if item["split"] in SPLITS else 99)
    manifest = {
        "dataset_root": str(args.dataset_root),
        "prediction_root": str(args.prediction_root),
        "splits": summaries,
        "stage3_candidate_columns": [
            "LCS_score_candidate", "IIS_score_candidate", "RTS_score_candidate",
            "PMIS_score_candidate", "composite_ODD_stress_candidate", "uncertainty_candidate",
        ],
    }
    (args.output_root / "order_feature_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
