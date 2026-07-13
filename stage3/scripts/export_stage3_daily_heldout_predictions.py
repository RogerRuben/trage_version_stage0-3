"""Export Stage2 held-out predictions into day-level Stage3 rolling inputs.

This script intentionally writes a small, deployment-facing schema:

```
heldout_root/
  link_predictions/day=YYYYMMDD.parquet
  movement_predictions/day=YYYYMMDD.parquet
  manifest.json
  leakage_audit.json
```

The inputs are Stage2 rolling test predictions. Each date must be produced by
exactly one fold whose training dates end before that date.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["lcs", "pmis", "rts"]
KEYS = ["order_id", "driver_id", "date", "route_link_id", "route_link_seq"]
ROUTE_COLUMNS = [
    "order_id",
    "date",
    "route_link_id",
    "route_link_seq",
    "position_ratio",
    "route_link_count",
    "route_link_length_m",
    "estimated_link_entry_time",
    "route_conditioned_time_check",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-config", type=Path, required=True)
    parser.add_argument("--date-fold-map", required=True, help="Comma list such as 20161017:1,20161018:2")
    parser.add_argument("--route-dataset-root", type=Path, default=Path("stage2/output/route_conditioned_dataset_15k/estimated_time_daily"))
    parser.add_argument("--old-deep-root", type=Path, default=Path("stage2/output/deep_v3_100k/formal_rolling_predictions/rc_mstnet"))
    parser.add_argument("--old-calibration-root", type=Path, default=Path("stage2/output/deep_v3/calibration"))
    parser.add_argument("--old-uncertainty-root", type=Path, default=Path("stage2/output/deep_v3/uncertainty"))
    parser.add_argument("--old-iis-root", type=Path, default=Path("stage2/output/deep_v3/iis_movement_15k_fast/predictions"))
    parser.add_argument("--new-deep-root", type=Path, default=Path("stage2/output/deep_v3_stage3_rolling_100k/predictions/rc_mstnet"))
    parser.add_argument("--new-calibration-root", type=Path, default=Path("stage2/output/deep_v3_stage3_rolling_100k/calibration"))
    parser.add_argument("--new-uncertainty-root", type=Path, default=Path("stage2/output/deep_v3_stage3_rolling_100k/uncertainty"))
    parser.add_argument("--new-iis-root", type=Path, default=Path("stage2/output/deep_v3_stage3_rolling_100k/iis_movement_15k_fast/predictions"))
    parser.add_argument("--new-fold-min", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=Path("stage3/output/stage2_heldout_daily_20161017_23"))
    return parser.parse_args()


def parse_date_fold_map(spec: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        date, fold = item.split(":", 1)
        mapping[date.strip()] = int(fold)
    if not mapping:
        raise ValueError("Empty --date-fold-map")
    return mapping


def fold_roots(args: argparse.Namespace, fold: int) -> tuple[Path, Path, Path, Path]:
    if fold >= args.new_fold_min:
        return args.new_deep_root, args.new_calibration_root, args.new_uncertainty_root, args.new_iis_root
    return args.old_deep_root, args.old_calibration_root, args.old_uncertainty_root, args.old_iis_root


def read_link_predictions(args: argparse.Namespace, date: str, fold: int, train_end: str) -> tuple[pd.DataFrame, list[str]]:
    deep_root, calibration_root, uncertainty_root, _ = fold_roots(args, fold)
    errors: list[str] = []
    deep = pd.read_parquet(deep_root / f"fold={fold}" / "test_predictions.parquet")
    calibration = pd.read_parquet(calibration_root / "calibrated_predictions" / f"fold={fold}" / "test_predictions.parquet")
    uncertainty = pd.read_parquet(uncertainty_root / "predictions" / f"fold={fold}" / "test_uncertainty.parquet")
    route = pd.read_parquet(args.route_dataset_root / f"day={date}.parquet", columns=ROUTE_COLUMNS)
    for name, frame in [("deep", deep), ("calibration", calibration), ("uncertainty", uncertainty), ("route", route)]:
        dates = set(frame["date"].astype(str).unique())
        if dates != {date}:
            errors.append(f"day={date} fold={fold} {name} dates={sorted(dates)}")
        key_columns = KEYS if "driver_id" in frame.columns else [column for column in KEYS if column != "driver_id"]
        if frame.duplicated(key_columns).any():
            errors.append(f"day={date} fold={fold} {name} duplicate keys")

    link = deep[KEYS + [f"pred_{target}_raw" for target in TARGETS] + [f"pred_{target}_tail_prob" for target in TARGETS]].copy()
    rename = {f"pred_{target}_raw": f"{target}_raw_pred" for target in TARGETS}
    rename.update({f"pred_{target}_tail_prob": f"{target}_tail_prob_raw" for target in TARGETS})
    link = link.rename(columns=rename)
    calibration_columns = [column for column in KEYS if column in calibration.columns]
    calibration_columns += [f"{target}_tail_prob_calibrated" for target in TARGETS]
    uncertainty_columns = [column for column in KEYS if column in uncertainty.columns]
    for target in TARGETS:
        uncertainty_columns += [f"{target}_uncertainty", f"{target}_lower", f"{target}_upper", f"{target}_ensemble_variance"]
    link = link.merge(calibration[calibration_columns], on=[column for column in KEYS if column in calibration.columns], validate="one_to_one")
    link = link.merge(uncertainty[uncertainty_columns], on=[column for column in KEYS if column in uncertainty.columns], validate="one_to_one")
    link = link.merge(route, on=["order_id", "date", "route_link_id", "route_link_seq"], validate="one_to_one")
    link.insert(0, "fold_id", fold)
    link["stage2_model_train_end_date"] = train_end
    link["prediction_date_after_train_end"] = link["date"].astype(str).gt(train_end)
    link["model_version"] = "RC-MSTNet-100k"
    link["lightgbm_prediction_available"] = False
    if not link["route_conditioned_time_check"].fillna(False).all():
        errors.append(f"day={date} failed route_conditioned_time_check")
    if not link["prediction_date_after_train_end"].all():
        errors.append(f"day={date} prediction date is not after train_end={train_end}")
    return link, errors


def read_movement_predictions(args: argparse.Namespace, date: str, fold: int, train_end: str) -> tuple[pd.DataFrame, list[str]]:
    _, _, _, iis_root = fold_roots(args, fold)
    errors: list[str] = []
    movement = pd.read_parquet(iis_root / f"fold={fold}" / "test_movement_predictions.parquet")
    dates = set(movement["date"].astype(str).unique())
    if dates != {date}:
        errors.append(f"day={date} fold={fold} movement dates={sorted(dates)}")
    movement_id = movement[["from_link_id", "node_id", "to_link_id"]].astype(str).agg("|".join, axis=1)
    output = pd.DataFrame({
        "fold_id": fold,
        "date": movement["date"].astype(str),
        "order_id": movement["order_id"],
        "movement_id": movement_id,
        "movement_seq": movement["planned_link_seq"],
        "from_link": movement["from_link_id"],
        "node_id": movement["node_id"],
        "to_link": movement["to_link_id"],
        "iis_applicability_prob": movement["pred_iis_applicability"].astype("float32"),
        "iis_severity_pred": movement["pred_iis_severity"].astype("float32"),
        "iis_tail_prob": movement["pred_iis_tail_prob"].astype("float32"),
        "iis_uncertainty": np.nan,
        "iis_valid_prediction_flag": movement["iis_prediction_available"].fillna(True).astype(bool),
        "iis_severity_prediction_available": movement["iis_severity_prediction_available"].fillna(True).astype(bool),
        "stage2_model_train_end_date": train_end,
    })
    key = ["date", "order_id", "movement_seq", "from_link", "node_id", "to_link"]
    if output.duplicated(key).any():
        errors.append(f"day={date} duplicate movement keys")
    return output, errors


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    link_root = args.output_root / "link_predictions"
    movement_root = args.output_root / "movement_predictions"
    link_root.mkdir(parents=True, exist_ok=True)
    movement_root.mkdir(parents=True, exist_ok=True)
    fold_config = {int(item["fold"]): item for item in json.loads(args.fold_config.read_text(encoding="utf-8"))["folds"]}
    date_fold = parse_date_fold_map(args.date_fold_map)
    manifest_rows = []
    errors: list[str] = []
    seen_link_keys: set[tuple[str, ...]] = set()
    seen_movement_keys: set[tuple[str, ...]] = set()
    for date, fold in sorted(date_fold.items()):
        train_end = max(fold_config[fold]["train_dates"])
        link, link_errors = read_link_predictions(args, date, fold, train_end)
        movement, movement_errors = read_movement_predictions(args, date, fold, train_end)
        errors.extend(link_errors)
        errors.extend(movement_errors)
        link_keys = set(map(tuple, link[["date", "order_id", "route_link_id", "route_link_seq"]].astype(str).to_numpy()))
        movement_keys = set(map(tuple, movement[["date", "order_id", "movement_seq", "from_link", "node_id", "to_link"]].astype(str).to_numpy()))
        if seen_link_keys & link_keys:
            errors.append(f"day={date} cross-day duplicate link keys")
        if seen_movement_keys & movement_keys:
            errors.append(f"day={date} cross-day duplicate movement keys")
        seen_link_keys |= link_keys
        seen_movement_keys |= movement_keys
        link.to_parquet(link_root / f"day={date}.parquet", index=False, compression="zstd")
        movement.to_parquet(movement_root / f"day={date}.parquet", index=False, compression="zstd")
        manifest_rows.append({
            "date": date,
            "fold": fold,
            "train_end": train_end,
            "link_rows": len(link),
            "link_orders": int(link["order_id"].nunique()),
            "movement_rows": len(movement),
            "movement_orders": int(movement["order_id"].nunique()),
            "iis_order_coverage": float(movement["order_id"].nunique() / max(1, link["order_id"].nunique())),
        })
    manifest = {"status": "PASS" if not errors else "FAIL", "rows": manifest_rows}
    leakage = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "heldout_only": True,
        "forbidden_realized_columns_excluded": True,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output_root / "leakage_audit.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    pd.DataFrame(manifest_rows).to_csv(args.output_root / "daily_export_summary.csv", index=False)
    print(json.dumps({"status": manifest["status"], "rows": manifest_rows, "errors": errors[:10]}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
