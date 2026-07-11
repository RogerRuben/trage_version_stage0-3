"""Layered evaluation for Stage2 deep modeling v2 predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stage2_model_utils import TARGETS, TARGET_ORDER, evaluate_predictions, unique_existing_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("stage2/output/link_dataset"))
    parser.add_argument("--prediction-root", type=Path, default=Path("stage2/output/deep_predictions_v2"))
    parser.add_argument("--baseline-prediction-root", type=Path, default=Path("stage2/output/deep_predictions"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/deep_baselines_v2"))
    parser.add_argument("--rare-threshold", type=int, default=100)
    return parser.parse_args()


def train_link_counts(path: Path) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    parts = []
    for group_no in range(parquet.metadata.num_row_groups):
        frame = parquet.read_row_group(group_no, columns=["link_id"]).to_pandas()
        parts.append(frame.link_id.value_counts().rename("train_link_count").reset_index().rename(columns={"index": "link_id"}))
    return pd.concat(parts).groupby("link_id", as_index=False).train_link_count.sum()


def read_dataset_context(path: Path) -> pd.DataFrame:
    columns = [
        "order_id", "link_id", "link_seq", "endpoint_degree", "peak_offpeak", "route_link_count",
        "position_ratio", "road_class", "time_bin",
    ]
    for target, mask, high in TARGETS.values():
        columns.extend([target, mask, high])
    return pd.read_parquet(path, columns=unique_existing_columns(path, columns))


def prediction_files(prediction_root: Path, baseline_root: Path) -> list[tuple[str, str, Path]]:
    items = [
        ("full_tabular_lgbm_3m_tail", "validation", baseline_root / "full_tabular_validation.parquet"),
        ("full_tabular_lgbm_3m_tail", "test", baseline_root / "full_tabular_test.parquet"),
        ("route_local_transformer", "validation", prediction_root / "route_local_transformer_validation.parquet"),
        ("route_local_transformer", "test", prediction_root / "route_local_transformer_test.parquet"),
        ("dual_graph_route_transformer", "validation", prediction_root / "dual_graph_route_transformer_validation.parquet"),
        ("dual_graph_route_transformer", "test", prediction_root / "dual_graph_route_transformer_test.parquet"),
    ]
    return [(model, split, path) for model, split, path in items if path.exists()]


def slice_masks(frame: pd.DataFrame, rare_threshold: int) -> dict[str, pd.Series]:
    route_count = pd.to_numeric(frame.get("route_link_count"), errors="coerce")
    endpoint = pd.to_numeric(frame.get("endpoint_degree"), errors="coerce")
    position = pd.to_numeric(frame.get("position_ratio"), errors="coerce")
    link_count = pd.to_numeric(frame.get("train_link_count"), errors="coerce").fillna(0)
    return {
        "all": pd.Series(True, index=frame.index),
        "seen_common_links": link_count >= rare_threshold,
        "rare_links": link_count < rare_threshold,
        "high_endpoint_degree": endpoint >= 3,
        "peak": frame.get("peak_offpeak", pd.Series("", index=frame.index)).astype(str).eq("peak"),
        "offpeak": frame.get("peak_offpeak", pd.Series("", index=frame.index)).astype(str).eq("offpeak"),
        "short_route": route_count <= 20,
        "long_route": route_count >= 50,
        "pickup_side": position <= 0.10,
        "dropoff_side": position >= 0.90,
    }


def link_slice_metrics(frame: pd.DataFrame, model: str, split: str, rare_threshold: int) -> pd.DataFrame:
    rows = []
    masks = slice_masks(frame, rare_threshold)
    for target_name in TARGET_ORDER:
        target_col, valid_col, high_col = TARGETS[target_name]
        pred_col = f"pred_{target_name.lower()}"
        target_masks = dict(masks)
        target_masks[f"{target_name.lower()}_valid_subset"] = frame[valid_col].fillna(False)
        if target_name == "IIS":
            target_masks["iis_valid_intersection_heavy"] = frame[valid_col].fillna(False) & (pd.to_numeric(frame.endpoint_degree, errors="coerce") >= 3)
        for slice_name, slice_mask in target_masks.items():
            valid = slice_mask.fillna(False) & frame[valid_col].fillna(False) & frame[target_col].notna() & frame[pred_col].notna()
            if valid.sum() < 100:
                continue
            metrics = evaluate_predictions(
                frame.loc[valid, target_col].to_numpy(dtype=float),
                frame.loc[valid, pred_col].to_numpy(dtype=float),
                frame.loc[valid, high_col].to_numpy(dtype=bool),
            )
            rows.append({"model": model, "split": split, "target": target_name, "slice": slice_name, **metrics})
    return pd.DataFrame(rows)


def order_tail(frame: pd.DataFrame, model: str, split: str) -> pd.DataFrame:
    rows = []
    for target_name in TARGET_ORDER:
        target_col, valid_col, _ = TARGETS[target_name]
        pred_col = f"pred_{target_name.lower()}"
        valid = frame[valid_col].fillna(False) & frame[target_col].notna()
        if not valid.any():
            continue
        per_order = frame.loc[valid, ["order_id", pred_col, target_col]].groupby("order_id").agg(
            pred_mean=(pred_col, "mean"),
            pred_max=(pred_col, "max"),
            actual_tail=(target_col, lambda values: float(np.nanmax(values) >= 0.90)),
        ).reset_index()
        base = float(per_order.actual_tail.mean())
        for score_col in ["pred_mean", "pred_max"]:
            row = {"model": model, "split": split, "target": target_name, "score": score_col, "orders": len(per_order), "base_tail_rate": base}
            for share, label in [(0.10, "top10"), (0.05, "top5")]:
                n = max(1, int(len(per_order) * share))
                top = per_order.nlargest(n, score_col)
                precision = float(top.actual_tail.mean())
                recall = float(top.actual_tail.sum() / max(per_order.actual_tail.sum(), 1))
                row[f"precision_at_{label}"] = precision
                row[f"recall_at_{label}"] = recall
                row[f"lift_at_{label}"] = precision / base if base > 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    link_counts = train_link_counts(args.dataset_root / "train.parquet")
    link_counts.to_parquet(args.output_root / "train_link_counts.parquet", index=False, compression="zstd")
    metrics = []
    order_rows = []
    contexts = {}
    for model, split, path in prediction_files(args.prediction_root, args.baseline_prediction_root):
        if split not in contexts:
            contexts[split] = read_dataset_context(args.dataset_root / f"{split}.parquet").merge(link_counts, on="link_id", how="left")
        pred = pd.read_parquet(path)
        frame = pred.merge(contexts[split], on=["order_id", "link_id", "link_seq"], how="left", suffixes=("", "_data"))
        for target_name in TARGET_ORDER:
            data_target, data_mask, data_high = TARGETS[target_name]
            for base in [data_target, data_mask, data_high]:
                data_col = f"{base}_data"
                if data_col in frame.columns:
                    frame[base] = frame[data_col]
        metrics.append(link_slice_metrics(frame, model, split, args.rare_threshold))
        order_rows.append(order_tail(frame, model, split))
    link_metrics = pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame()
    order_metrics = pd.concat(order_rows, ignore_index=True) if order_rows else pd.DataFrame()
    link_metrics.to_csv(args.output_root / "v2_slice_metrics.csv", index=False)
    order_metrics.to_csv(args.output_root / "v2_order_tail_separation.csv", index=False)
    report = ["# Stage2 deep modeling v2 layered evaluation", ""]
    test_all = link_metrics[(link_metrics.split == "test") & (link_metrics.slice == "all")]
    if not test_all.empty:
        report += ["## Test all-link metrics", "", "| model | target | AUC | Spearman | top10 lift | top5 lift |", "|---|---:|---:|---:|---:|---:|"]
        for row in test_all.sort_values(["target", "auc"], ascending=[True, False]).itertuples(index=False):
            report.append(f"| {row.model} | {row.target} | {row.auc:.3f} | {row.spearman:.3f} | {row.top10_lift:.2f} | {row.top5_lift:.2f} |")
    rare = link_metrics[(link_metrics.split == "test") & (link_metrics.slice.isin(["rare_links", "high_endpoint_degree", "peak", "long_route"]))]
    if not rare.empty:
        report += ["", "## Key test slices", "", "| model | target | slice | AUC | Spearman | top10 lift |", "|---|---:|---|---:|---:|---:|"]
        for row in rare.sort_values(["slice", "target", "auc"], ascending=[True, True, False]).itertuples(index=False):
            report.append(f"| {row.model} | {row.target} | {row.slice} | {row.auc:.3f} | {row.spearman:.3f} | {row.top10_lift:.2f} |")
    if not order_metrics.empty:
        order_test = order_metrics[(order_metrics.split == "test") & (order_metrics.score == "pred_max")]
        report += ["", "## Test order-level tail separation", "", "| model | target | base tail | lift@top10 | lift@top5 |", "|---|---:|---:|---:|---:|"]
        for row in order_test.sort_values(["target", "lift_at_top10"], ascending=[True, False]).itertuples(index=False):
            report.append(f"| {row.model} | {row.target} | {row.base_tail_rate:.3f} | {row.lift_at_top10:.2f} | {row.lift_at_top5:.2f} |")
    (args.output_root / "v2_layered_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(args.output_root / "v2_layered_report.md")


if __name__ == "__main__":
    main()
