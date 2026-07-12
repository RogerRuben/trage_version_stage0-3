"""Audit a Stage3 rolling warehouse for temporal leakage and modality coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FORBIDDEN_PREFIXES = ("target_", "actual_")
FORBIDDEN_SUBSTRINGS = ("travel_time_sec", "mean_speed_mps_current_order", "low_speed_ratio_on_poi_link", "stop_time_on_poi_link")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warehouse-root", type=Path, default=Path("stage3/output/rolling_stage2_prediction_warehouse"))
    parser.add_argument("--fold-config", type=Path, default=Path("stage3/config/stage3_rolling_fold_config.json"))
    parser.add_argument("--output-root", type=Path, default=Path("stage3/output/rolling_stage2_prediction_warehouse/audit"))
    return parser.parse_args()


def read_parts(root: Path, family: str, fold: int, split: str) -> pd.DataFrame:
    base = root / family / f"fold={fold}" / f"split={split}"
    if not base.exists():
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in sorted(base.glob("day=*.parquet"))], ignore_index=True)


def forbidden_columns(columns: list[str]) -> list[str]:
    return [
        column for column in columns
        if column.startswith(FORBIDDEN_PREFIXES) or any(bad in column for bad in FORBIDDEN_SUBSTRINGS)
    ]


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.fold_config.read_text(encoding="utf-8"))
    rows = []
    errors = []
    for fold in config["folds"]:
        fold_id = int(fold["fold"])
        split_dates = {
            "train": set(fold["train_dates"]),
            "validation": {fold["validation_date"]},
            "test": {fold["test_date"]},
        }
        if not (max(fold["train_dates"]) < fold["validation_date"] < fold["test_date"]):
            errors.append(f"fold={fold_id} violates train < validation < test")
        for split, expected_dates in split_dates.items():
            link = read_parts(args.warehouse_root, "link_predictions", fold_id, split)
            movement = read_parts(args.warehouse_root, "movement_predictions", fold_id, split)
            for family, frame in [("link", link), ("movement", movement)]:
                dates = set(frame["date"].astype(str).unique()) if not frame.empty and "date" in frame else set()
                if dates != expected_dates:
                    errors.append(f"fold={fold_id} split={split} {family} dates {sorted(dates)} expected {sorted(expected_dates)}")
                forbidden = forbidden_columns(frame.columns.tolist()) if not frame.empty else []
                if forbidden:
                    errors.append(f"fold={fold_id} split={split} {family} forbidden columns {forbidden}")
            link_orders = set(link["order_id"].astype(str)) if not link.empty else set()
            movement_orders = set(movement["order_id"].astype(str)) if not movement.empty else set()
            rows.append({
                "fold": fold_id,
                "split": split,
                "link_rows": len(link),
                "link_orders": len(link_orders),
                "movement_rows": len(movement),
                "movement_orders": len(movement_orders),
                "iis_order_coverage": len(link_orders & movement_orders) / max(1, len(link_orders)),
            })
    table = pd.DataFrame(rows)
    table.to_csv(args.output_root / "rolling_warehouse_coverage.csv", index=False)
    result = {"status": "PASS" if not errors else "FAIL", "errors": errors, "rows": rows}
    (args.output_root / "leakage_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "errors": errors[:10]}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
