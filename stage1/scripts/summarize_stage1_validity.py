"""Aggregate daily Stage1 validity audits into a temporal-split report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("stage1/output/prediction_split"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    day_dirs = sorted((args.output_root / "validity").rglob("day=*"))
    if not day_dirs:
        raise FileNotFoundError("no daily Stage1 validity audits found")
    monotonicity = []
    sensitivities = []
    for directory in day_dirs:
        date = directory.name.split("=")[1]
        table = pd.read_csv(directory / "link_decile_monotonicity.csv")
        table["date"] = date
        table["split"] = directory.parent.name
        monotonicity.append(table)
        sensitivity_path = directory / "sensitivity.json"
        if sensitivity_path.exists():
            value = json.loads(sensitivity_path.read_text(encoding="utf-8")); value["date"] = date; sensitivities.append(value)
    mono = pd.concat(monotonicity, ignore_index=True)
    summary = mono.groupby(["split", "dimension", "indicator"], as_index=False).agg(
        median_spearman=("spearman_decile_mean", "median"), min_spearman=("spearman_decile_mean", "min"), days=("date", "nunique")
    )
    reports = args.output_root / "reports"; reports.mkdir(parents=True, exist_ok=True)
    summary.to_csv(reports / "stage1_split_monotonicity.csv", index=False)
    report = f"""# Stage1 label validity report

Daily audits completed: **{len(day_dirs)}**.

Median link-level indicator/decile Spearman correlation across all dimensions and days: **{summary.median_spearman.median():.3f}**.

The split summary is stored in `reports/stage1_split_monotonicity.csv`. Daily link/order monotonicity, spatial maps, POI-by-hour checks, and threshold sensitivity results are under `validity/{{train,validation,test}}/day=YYYYMMDD/`.

This report audits label measurement validity; it is not prediction calibration.
"""
    (reports / "stage1_label_validity_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
