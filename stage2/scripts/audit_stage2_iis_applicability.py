"""Audit the two-part IIS contract: applicability first, severity second."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("stage2/output/iis_movement_causal_dataset"))
    parser.add_argument("--fold-config", type=Path, default=Path("rolling_threefold_config.json"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/iis_movement_audit"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.fold_config.read_text(encoding="utf-8"))
    dates = sorted({date for fold in config["folds"] for date in fold["train_dates"] + [fold["validation_date"], fold["test_date"]]})
    rows = []
    for date in dates:
        frame = pd.read_parquet(args.dataset_root / f"day={date}.parquet")
        applicable = frame["iis_applicable"].fillna(False).astype(bool)
        observed = frame["iis_observed"].fillna(False).astype(bool)
        severity = frame["target_iis_raw"].notna()
        rows.append({
            "date": date, "planned_movements": len(frame),
            "iis_applicable_ratio": float(applicable.mean()),
            "iis_observed_ratio": float(observed.mean()),
            "severity_valid_ratio": float(severity.mean()),
            "severity_valid_given_applicable": float(severity[applicable].mean()) if applicable.any() else None,
            "severity_missing_given_applicable": float((~severity[applicable]).mean()) if applicable.any() else None,
            "nonapplicable_nonnull_severity_ratio": float(severity[~applicable].mean()) if (~applicable).any() else None,
            "unobserved_rows_filled_zero_count": int(((~observed) & frame["target_iis_raw"].eq(0)).sum()),
        })
    report = pd.DataFrame(rows)
    report.to_csv(args.output_root / "iis_applicability_severity_by_day.csv", index=False)
    manifest = {
        "contract": "IIS applicability is a pre-dispatch planned-movement descriptor; IIS severity is evaluated only where a matched realized movement supplies a non-null label.",
        "missing_is_zero": False,
        "days": rows,
    }
    (args.output_root / "iis_applicability_severity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = ["# IIS applicability and severity audit", "", report.to_markdown(index=False, floatfmt=".4f"), "",
             "Missing IIS severity is structural missingness and is never filled with zero."]
    (args.output_root / "iis_applicability_severity_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
