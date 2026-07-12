"""Build Stage3 order targets from realized Stage1 link/movement measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["lcs", "pmis", "rts"]
SPLITS = {"train": "20161017", "validation": "20161018", "test": "20161019"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-dataset-root", type=Path, required=True)
    parser.add_argument("--movement-dataset-root", type=Path, required=True)
    parser.add_argument("--warehouse-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("stage3/output/order_targets"))
    return parser.parse_args()


def weighted_mean(value: pd.Series, weight: pd.Series) -> float:
    mask = value.notna() & weight.notna() & weight.gt(0)
    return float(np.average(value[mask], weights=weight[mask])) if mask.any() else np.nan


def max_run(frame: pd.DataFrame, flag_column: str) -> tuple[int, float]:
    flags = frame[flag_column].fillna(False).to_numpy(bool)
    lengths = pd.to_numeric(frame["route_link_length_m"], errors="coerce").fillna(0).to_numpy(float)
    best_count = count = 0
    best_m = current_m = 0.0
    for flag, length in zip(flags, lengths):
        if flag:
            count += 1
            current_m += length
            best_count = max(best_count, count)
            best_m = max(best_m, current_m)
        else:
            count = 0
            current_m = 0.0
    return best_count, best_m


def aggregate_links(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for order_id, group in frame.sort_values(["order_id", "route_link_seq"]).groupby("order_id", sort=False):
        row = {"order_id": order_id}
        for target in TARGETS:
            valid = group[f"target_{target}_valid"].fillna(False)
            data = group.loc[valid]
            values = pd.to_numeric(data[f"target_{target}_raw"], errors="coerce")
            row.update({
                f"order_{target}_mean": float(values.mean()) if len(values) else np.nan,
                f"order_{target}_weighted_mean": weighted_mean(values, data["route_link_length_m"]) if len(values) else np.nan,
                f"order_{target}_q90": float(values.quantile(0.90)) if len(values) else np.nan,
                f"order_{target}_q95": float(values.quantile(0.95)) if len(values) else np.nan,
                f"order_{target}_max": float(values.max()) if len(values) else np.nan,
                f"order_{target}_tail_share": float(data[f"target_{target}_tail90_raw"].fillna(False).mean()) if len(data) else np.nan,
                f"order_{target}_valid_link_share": float(valid.mean()),
            })
            for label, low, high in [("early", 0.0, 1 / 3), ("middle", 1 / 3, 2 / 3), ("late", 2 / 3, 1.01)]:
                section = data[data["position_ratio"].ge(low) & data["position_ratio"].lt(high)]
                row[f"order_{target}_{label}_mean"] = float(section[f"target_{target}_raw"].mean()) if len(section) else np.nan
            temp = group.copy()
            temp[f"_{target}_tail"] = temp[f"target_{target}_tail90_raw"].fillna(False) & valid
            count, distance = max_run(temp, f"_{target}_tail")
            row[f"order_{target}_consecutive_tail_links"] = count
            row[f"order_{target}_consecutive_tail_m"] = distance
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_iis(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for order_id, group in frame.groupby("order_id", sort=False):
        applicable = group["iis_applicable"].fillna(False)
        observed = group["iis_observed"].fillna(False) & group["target_iis_valid"].fillna(False)
        severity = pd.to_numeric(group.loc[observed, "target_iis_raw"], errors="coerce")
        rows.append({
            "order_id": order_id,
            "order_iis_applicable_share": float(applicable.mean()),
            "order_iis_observed_count": int(observed.sum()),
            "order_iis_severity_mean": float(severity.mean()) if len(severity) else np.nan,
            "order_iis_severity_q90": float(severity.quantile(0.90)) if len(severity) else np.nan,
            "order_iis_severity_max": float(severity.max()) if len(severity) else np.nan,
            "order_iis_tail_share": float(group.loc[observed, "target_iis_tail90_raw"].fillna(False).mean()) if observed.any() else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split, date in SPLITS.items():
        order_index = pd.read_parquet(args.warehouse_root / "order_index" / "orders.parquet")
        order_index = order_index[order_index["split"].eq(split)][["order_id", "fold_id", "split", "date", "link_count", "route_length_m", "movement_count"]]
        route = pd.read_parquet(args.route_dataset_root / f"day={date}.parquet")
        route = route[route["order_id"].isin(order_index["order_id"])]
        target = order_index.merge(aggregate_links(route), on="order_id", how="left", validate="one_to_one")
        movement_path = args.movement_dataset_root / f"day={date}.parquet"
        if movement_path.exists():
            movement = pd.read_parquet(movement_path)
            movement = movement[movement["order_id"].isin(order_index["order_id"])]
            target = target.merge(aggregate_iis(movement), on="order_id", how="left", validate="one_to_one")
        outputs[split] = target
    thresholds = {target: float(outputs["train"][f"order_{target}_q90"].quantile(0.90)) for target in TARGETS}
    iis_threshold = float(outputs["train"]["order_iis_severity_q90"].quantile(0.90))
    manifest = {"tail_threshold_fit_split": "train", "tail_thresholds": thresholds, "iis_severity_threshold": iis_threshold, "realized_columns_are_targets_only": True, "splits": {}}
    for split, frame in outputs.items():
        for target in TARGETS:
            frame[f"order_{target}_raw"] = frame[f"order_{target}_q90"]
            frame[f"order_{target}_tail"] = frame[f"order_{target}_raw"].ge(thresholds[target]) & frame[f"order_{target}_raw"].notna()
        frame["order_iis_tail"] = frame["order_iis_severity_q90"].ge(iis_threshold) & frame["order_iis_severity_q90"].notna()
        frame["order_overall_high_stress"] = frame[[f"order_{target}_tail" for target in TARGETS] + ["order_iis_tail"]].any(axis=1)
        split_root = args.output_root / f"split={split}"
        split_root.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(split_root / "order_targets.parquet", index=False, compression="zstd")
        manifest["splits"][split] = {"orders": len(frame), "overall_high_rate": float(frame["order_overall_high_stress"].mean()), **{f"{target}_valid_ratio": float(frame[f"order_{target}_raw"].notna().mean()) for target in TARGETS}}
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
