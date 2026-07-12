"""Build a multi-date Stage3 rolling warehouse from Stage2 held-out stores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-config", type=Path, default=Path("stage3/config/stage3_rolling_fold_config.json"))
    parser.add_argument("--heldout-root", type=Path, required=True, help="Directory containing link_predictions/day=YYYYMMDD.parquet and movement_predictions/day=YYYYMMDD.parquet, or split=*/day files.")
    parser.add_argument("--output-root", type=Path, default=Path("stage3/output/rolling_stage2_prediction_warehouse"))
    parser.add_argument("--skip-missing", action="store_true")
    return parser.parse_args()


def find_day(root: Path, family: str, date: str) -> Path | None:
    candidates = [
        root / family / f"day={date}.parquet",
        root / family / f"date={date}.parquet",
    ]
    candidates += list((root / family).glob(f"split=*/day={date}.parquet")) if (root / family).exists() else []
    for path in candidates:
        if path.exists():
            return path
    return None


def load_day(root: Path, family: str, date: str, skip_missing: bool) -> pd.DataFrame:
    path = find_day(root, family, date)
    if path is None:
        if skip_missing:
            return pd.DataFrame()
        raise FileNotFoundError(f"Missing {family} held-out predictions for day={date} under {root}")
    return pd.read_parquet(path)


def write_family(frame: pd.DataFrame, output_root: Path, family: str, fold: int, split: str) -> dict:
    root = output_root / family / f"fold={fold}" / f"split={split}"
    root.mkdir(parents=True, exist_ok=True)
    rows = 0
    orders = 0
    for date, part in frame.groupby("date", sort=True):
        part.to_parquet(root / f"day={date}.parquet", index=False, compression="zstd")
        rows += len(part)
        orders += part["order_id"].nunique() if "order_id" in part.columns else 0
    return {"fold": fold, "split": split, "family": family, "rows": rows, "orders_sum_by_day": orders, "dates": sorted(frame["date"].astype(str).unique()) if len(frame) else []}


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.fold_config.read_text(encoding="utf-8"))
    manifest = {"folds": [], "missing": []}
    for fold in config["folds"]:
        fold_id = int(fold["fold"])
        split_dates = {
            "train": fold["train_dates"],
            "validation": [fold["validation_date"]],
            "test": [fold["test_date"]],
        }
        for split, dates in split_dates.items():
            link_parts = []
            movement_parts = []
            for date in dates:
                link = load_day(args.heldout_root, "link_predictions", date, args.skip_missing)
                movement = load_day(args.heldout_root, "movement_predictions", date, args.skip_missing)
                if link.empty:
                    manifest["missing"].append({"fold": fold_id, "split": split, "date": date, "family": "link_predictions"})
                else:
                    link = link.copy()
                    link["fold_id_stage3"] = fold_id
                    link["stage3_split"] = split
                    link_parts.append(link)
                if movement.empty:
                    manifest["missing"].append({"fold": fold_id, "split": split, "date": date, "family": "movement_predictions"})
                else:
                    movement = movement.copy()
                    movement["fold_id_stage3"] = fold_id
                    movement["stage3_split"] = split
                    movement_parts.append(movement)
            if link_parts:
                manifest["folds"].append(write_family(pd.concat(link_parts, ignore_index=True), args.output_root, "link_predictions", fold_id, split))
            if movement_parts:
                manifest["folds"].append(write_family(pd.concat(movement_parts, ignore_index=True), args.output_root, "movement_predictions", fold_id, split))
    manifest["status"] = "PASS" if not manifest["missing"] else ("PARTIAL" if args.skip_missing else "FAIL")
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "missing": manifest["missing"][:10]}, indent=2))
    if manifest["missing"] and not args.skip_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
