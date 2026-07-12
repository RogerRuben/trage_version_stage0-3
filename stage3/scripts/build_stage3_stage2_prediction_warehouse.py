"""Build a leakage-audited Stage3 warehouse from rolling test predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["lcs", "pmis", "rts"]
KEYS = ["order_id", "date", "route_link_id", "route_link_seq"]
CANONICAL = {
    1: {"date": "20161017", "stage3_split": "train"},
    2: {"date": "20161018", "stage3_split": "validation"},
    3: {"date": "20161019", "stage3_split": "test"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep-prediction-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--uncertainty-root", type=Path, required=True)
    parser.add_argument("--lightgbm-oof", type=Path, required=True)
    parser.add_argument("--iis-prediction-root", type=Path, required=True)
    parser.add_argument("--route-dataset-root", type=Path, required=True)
    parser.add_argument("--fold-config", type=Path, default=Path("rolling_threefold_config.json"))
    parser.add_argument("--output-root", type=Path, default=Path("stage3/output/stage2_prediction_warehouse"))
    return parser.parse_args()


def lgbm_wide(frame: pd.DataFrame, fold: int) -> pd.DataFrame:
    data = frame[frame["fold"].eq(fold) & frame["ablation"].eq("static_rolling_dynamic_topology_route")].copy()
    keys = ["order_id", "date", "planned_link_id", "planned_link_seq"]
    raw = data.pivot(index=keys, columns="target", values="pred_raw").add_prefix("lgbm_").add_suffix("_raw_pred")
    tail = data.pivot(index=keys, columns="target", values="pred_tail_probability").add_prefix("lgbm_").add_suffix("_tail_prob")
    output = raw.join(tail).reset_index().rename(columns={"planned_link_id": "route_link_id", "planned_link_seq": "route_link_seq"})
    return output


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    folds = {int(fold["fold"]): fold for fold in json.loads(args.fold_config.read_text(encoding="utf-8"))["folds"]}
    lgbm = pd.read_parquet(args.lightgbm_oof)
    leakage_errors = []
    order_rows = []
    link_manifest = []
    movement_manifest = []
    seen_link_keys = set()
    seen_movement_keys = set()
    for fold, spec in CANONICAL.items():
        expected_date = spec["date"]
        split = spec["stage3_split"]
        deep = pd.read_parquet(args.deep_prediction_root / f"fold={fold}" / "test_predictions.parquet")
        calibration = pd.read_parquet(args.calibration_root / "calibrated_predictions" / f"fold={fold}" / "test_predictions.parquet")
        uncertainty = pd.read_parquet(args.uncertainty_root / "predictions" / f"fold={fold}" / "test_uncertainty.parquet")
        route = pd.read_parquet(
            args.route_dataset_root / f"day={expected_date}.parquet",
            columns=["order_id", "date", "route_link_id", "route_link_seq", "position_ratio", "route_link_count", "route_link_length_m", "estimated_link_entry_time", "route_conditioned_time_check"],
        )
        for frame, name in [(deep, "deep"), (calibration, "calibration"), (uncertainty, "uncertainty")]:
            if set(frame["date"].astype(str).unique()) != {expected_date}:
                leakage_errors.append(f"fold={fold} {name} date mismatch")
            if frame.duplicated(KEYS).any():
                leakage_errors.append(f"fold={fold} {name} duplicate prediction key")
        link = deep[KEYS + [f"pred_{target}_raw" for target in TARGETS] + [f"pred_{target}_tail_prob" for target in TARGETS]].copy()
        link = link.rename(columns={f"pred_{target}_raw": f"{target}_raw_pred" for target in TARGETS} | {f"pred_{target}_tail_prob": f"{target}_tail_prob_raw" for target in TARGETS})
        calibrated_columns = KEYS + [f"{target}_tail_prob_calibrated" for target in TARGETS]
        uncertainty_columns = KEYS + sum(([f"{target}_uncertainty", f"{target}_lower", f"{target}_upper", f"{target}_ensemble_variance"] for target in TARGETS), [])
        link = link.merge(calibration[calibrated_columns], on=KEYS, validate="one_to_one")
        link = link.merge(uncertainty[uncertainty_columns], on=KEYS, validate="one_to_one")
        link = link.merge(route, on=KEYS, validate="one_to_one")
        link = link.merge(lgbm_wide(lgbm, fold), on=KEYS, how="left", validate="one_to_one")
        link.insert(0, "fold_id", fold)
        link.insert(1, "split", split)
        link["lightgbm_prediction_available"] = link.filter(regex=r"^lgbm_.*_raw_pred$").notna().any(axis=1)
        if not link["route_conditioned_time_check"].fillna(False).all():
            leakage_errors.append(f"fold={fold} failed route conditioned time check")
        training_dates = set(folds[fold]["train_dates"])
        if expected_date in training_dates:
            leakage_errors.append(f"fold={fold} prediction date included in Stage2 training dates")
        link["stage2_model_train_end_date"] = max(training_dates)
        link["prediction_date_after_train_end"] = link["date"].astype(str) > max(training_dates)
        if not link["prediction_date_after_train_end"].all():
            leakage_errors.append(f"fold={fold} non-forward prediction date")
        forbidden = [column for column in link if column.startswith("target_") or "actual_" in column or "travel_time_sec" in column]
        if forbidden:
            leakage_errors.append(f"fold={fold} forbidden link columns {forbidden}")
        key_tuples = set(map(tuple, link[KEYS].astype(str).to_numpy()))
        if seen_link_keys & key_tuples:
            leakage_errors.append(f"fold={fold} cross-split link key overlap")
        seen_link_keys |= key_tuples
        link_root = args.output_root / "link_predictions" / f"split={split}"
        link_root.mkdir(parents=True, exist_ok=True)
        link.to_parquet(link_root / f"day={expected_date}.parquet", index=False, compression="zstd")
        link_manifest.append({"fold": fold, "split": split, "date": expected_date, "rows": len(link), "orders": link["order_id"].nunique(), "lightgbm_coverage": float(link["lightgbm_prediction_available"].mean())})

        movement = pd.read_parquet(args.iis_prediction_root / f"fold={fold}" / "test_movement_predictions.parquet")
        movement_id = movement[["from_link_id", "node_id", "to_link_id"]].astype(str).agg("|".join, axis=1)
        movement_out = pd.DataFrame({
            "fold_id": fold, "split": split, "date": movement["date"].astype(str), "order_id": movement["order_id"],
            "movement_id": movement_id, "movement_seq": movement["planned_link_seq"], "from_link": movement["from_link_id"],
            "node_id": movement["node_id"], "to_link": movement["to_link_id"],
            "iis_applicability_prob": movement["pred_iis_applicability"], "iis_severity_pred": movement["pred_iis_severity"],
            "iis_tail_prob": movement["pred_iis_tail_prob"], "iis_uncertainty": np.nan,
            "iis_valid_prediction_flag": movement["pred_iis_applicability"].ge(0.5),
            "stage2_model_train_end_date": max(training_dates),
        })
        movement_key = ["date", "order_id", "movement_seq", "movement_id"]
        key_tuples = set(map(tuple, movement_out[movement_key].astype(str).to_numpy()))
        if seen_movement_keys & key_tuples:
            leakage_errors.append(f"fold={fold} cross-split movement key overlap")
        seen_movement_keys |= key_tuples
        movement_root = args.output_root / "movement_predictions" / f"split={split}"
        movement_root.mkdir(parents=True, exist_ok=True)
        movement_out.to_parquet(movement_root / f"day={expected_date}.parquet", index=False, compression="zstd")
        movement_manifest.append({"fold": fold, "split": split, "date": expected_date, "rows": len(movement_out), "orders": movement_out["order_id"].nunique()})
        link_orders = link.groupby(["fold_id", "split", "date", "order_id"], as_index=False).agg(link_count=("route_link_seq", "size"), route_length_m=("route_link_length_m", "sum"))
        movement_counts = movement_out.groupby(["fold_id", "split", "date", "order_id"], as_index=False).size().rename(columns={"size": "movement_count"})
        order_rows.append(link_orders.merge(movement_counts, on=["fold_id", "split", "date", "order_id"], how="left").fillna({"movement_count": 0}))
    order_index = pd.concat(order_rows, ignore_index=True)
    order_root = args.output_root / "order_index"
    order_root.mkdir(parents=True, exist_ok=True)
    order_index.to_parquet(order_root / "orders.parquet", index=False, compression="zstd")
    leakage = {"status": "PASS" if not leakage_errors else "FAIL", "errors": leakage_errors, "canonical_test_only_mapping": CANONICAL, "forbidden_realized_inputs": True, "future_profile_check": "route_conditioned_time_check and train_end_date audit"}
    (args.output_root / "leakage_audit.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    manifest = {"link_predictions": link_manifest, "movement_predictions": movement_manifest, "order_rows": len(order_index), "leakage_status": leakage["status"]}
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if leakage_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
