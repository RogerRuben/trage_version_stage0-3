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
    parser.add_argument("--fold", type=int, default=None, help="Optional Stage3 rolling fold id.")
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
    base = pd.DataFrame({"order_id": frame["order_id"].drop_duplicates()})
    for target in TARGETS:
        valid = frame[f"target_{target}_valid"].fillna(False)
        data = frame.loc[valid, ["order_id", "route_link_length_m", "position_ratio", f"target_{target}_raw", f"target_{target}_tail90_raw"]].copy()
        data[f"target_{target}_raw"] = pd.to_numeric(data[f"target_{target}_raw"], errors="coerce")
        grouped = data.groupby("order_id", sort=False)
        agg = grouped[f"target_{target}_raw"].agg(
            **{
                f"order_{target}_mean": "mean",
                f"order_{target}_q90": lambda value: value.quantile(0.90),
                f"order_{target}_q95": lambda value: value.quantile(0.95),
                f"order_{target}_max": "max",
            }
        ).reset_index()
        weighted = data.assign(_wx=data[f"target_{target}_raw"] * data["route_link_length_m"].fillna(0))
        weighted = weighted.groupby("order_id", sort=False).agg(_wx=("_wx", "sum"), _w=("route_link_length_m", "sum")).reset_index()
        weighted[f"order_{target}_weighted_mean"] = weighted["_wx"] / weighted["_w"].replace(0, np.nan)
        tail = grouped[f"target_{target}_tail90_raw"].mean().rename(f"order_{target}_tail_share").reset_index()
        valid_share = valid.groupby(frame["order_id"]).mean().rename(f"order_{target}_valid_link_share").reset_index()
        agg = agg.merge(weighted[["order_id", f"order_{target}_weighted_mean"]], on="order_id", how="left")
        agg = agg.merge(tail, on="order_id", how="left").merge(valid_share, on="order_id", how="left")
        for label, low, high in [("early", 0.0, 1 / 3), ("middle", 1 / 3, 2 / 3), ("late", 2 / 3, 1.01)]:
            section = data[data["position_ratio"].ge(low) & data["position_ratio"].lt(high)]
            section_mean = section.groupby("order_id", sort=False)[f"target_{target}_raw"].mean().rename(f"order_{target}_{label}_mean").reset_index()
            agg = agg.merge(section_mean, on="order_id", how="left")
        agg[f"order_{target}_consecutive_tail_links"] = np.nan
        agg[f"order_{target}_consecutive_tail_m"] = np.nan
        base = base.merge(agg, on="order_id", how="left")
    return base


def aggregate_iis(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["_observed"] = frame["iis_observed"].fillna(False) & frame["target_iis_valid"].fillna(False)
    observed = frame.loc[frame["_observed"]].copy()
    observed["target_iis_raw"] = pd.to_numeric(observed["target_iis_raw"], errors="coerce")
    base = frame.groupby("order_id", as_index=False).agg(
        order_iis_applicable_share=("iis_applicable", "mean"),
        order_iis_observed_count=("_observed", "sum"),
    )
    if observed.empty:
        return base
    severity = observed.groupby("order_id", as_index=False).agg(
        order_iis_severity_mean=("target_iis_raw", "mean"),
        order_iis_severity_q90=("target_iis_raw", lambda value: value.quantile(0.90)),
        order_iis_severity_max=("target_iis_raw", "max"),
        order_iis_tail_share=("target_iis_tail90_raw", "mean"),
    )
    return base.merge(severity, on="order_id", how="left")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split, default_date in SPLITS.items():
        if args.fold is None:
            order_index = pd.read_parquet(args.warehouse_root / "order_index" / "orders.parquet")
            order_index = order_index[order_index["split"].eq(split)][["order_id", "fold_id", "split", "date", "link_count", "route_length_m", "movement_count"]]
            dates = [default_date]
        else:
            link_paths = sorted((args.warehouse_root / "link_predictions" / f"fold={args.fold}" / f"split={split}").glob("day=*.parquet"))
            link_parts = [pd.read_parquet(path, columns=["order_id", "fold_id_stage3", "stage3_split", "date", "route_link_seq", "route_link_length_m"]) for path in link_paths]
            if not link_parts:
                raise FileNotFoundError(f"No rolling link predictions for fold={args.fold} split={split}")
            link_index = pd.concat(link_parts, ignore_index=True)
            order_index = link_index.groupby(["order_id", "fold_id_stage3", "stage3_split", "date"], as_index=False).agg(
                link_count=("route_link_seq", "size"),
                route_length_m=("route_link_length_m", "sum"),
            ).rename(columns={"fold_id_stage3": "fold_id", "stage3_split": "split"})
            movement_paths = sorted((args.warehouse_root / "movement_predictions" / f"fold={args.fold}" / f"split={split}").glob("day=*.parquet"))
            if movement_paths:
                movement_index = pd.concat([pd.read_parquet(path, columns=["order_id", "movement_seq"]) for path in movement_paths], ignore_index=True)
                movement_count = movement_index.groupby("order_id", as_index=False).size().rename(columns={"size": "movement_count"})
                order_index = order_index.merge(movement_count, on="order_id", how="left")
            else:
                order_index["movement_count"] = 0
            order_index["movement_count"] = order_index["movement_count"].fillna(0)
            dates = sorted(order_index["date"].astype(str).unique())
        route = pd.concat([pd.read_parquet(args.route_dataset_root / f"day={date}.parquet") for date in dates], ignore_index=True)
        route = route[route["order_id"].isin(order_index["order_id"])]
        target = order_index.merge(aggregate_links(route), on="order_id", how="left", validate="one_to_one")
        movement_parts = []
        for date in dates:
            movement_path = args.movement_dataset_root / f"day={date}.parquet"
            if movement_path.exists():
                movement_parts.append(pd.read_parquet(movement_path))
        if movement_parts:
            movement = pd.concat(movement_parts, ignore_index=True)
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
        frame["core_overall_high_stress"] = frame[[f"order_{target}_tail" for target in TARGETS]].any(axis=1)
        frame["extended_overall_high_stress"] = frame["core_overall_high_stress"] | frame["order_iis_tail"].fillna(False)
        frame["order_overall_high_stress"] = frame["extended_overall_high_stress"]
        split_root = args.output_root / f"split={split}"
        split_root.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(split_root / "order_targets.parquet", index=False, compression="zstd")
        manifest["splits"][split] = {"orders": len(frame), "overall_high_rate": float(frame["order_overall_high_stress"].mean()), **{f"{target}_valid_ratio": float(frame[f"order_{target}_raw"].notna().mean()) for target in TARGETS}}
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
