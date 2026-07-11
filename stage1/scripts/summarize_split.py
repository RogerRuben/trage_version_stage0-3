"""Summarize split-aware Stage1 coverage and validity readiness for Stage2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coverage_path = args.output_root / "reports" / "stage1_label_coverage.csv"
    coverage = pd.read_csv(coverage_path)
    split_manifest = args.output_root / "manifests" / "split_config.json"
    expected_days = 9
    if split_manifest.exists():
        config = json.loads(split_manifest.read_text(encoding="utf-8"))
        expected_days = len(config["train_dates"]) + 2
    complete = coverage.get("status", pd.Series(index=coverage.index, dtype="object")).eq("complete")
    test = coverage[coverage.split == "test"]

    def test_metric(name: str, default: float = float("nan")) -> float:
        if test.empty or name not in test.columns:
            return default
        return float(test.iloc[0][name])

    monotonicity_path = args.output_root / "reports" / "stage1_split_monotonicity.csv"
    positive_monotonicity_ratio = float("nan")
    if monotonicity_path.exists():
        monotonicity = pd.read_csv(monotonicity_path)
        test_monotonicity = monotonicity[monotonicity.split == "test"]
        if len(test_monotonicity):
            positive_monotonicity_ratio = float(test_monotonicity.median_spearman.gt(0).mean())

    checks = {
        "all_split_days_complete": bool(complete.all() and len(coverage) == expected_days),
        "test_gns_coverage_ge_98pct": bool(test_metric("gns_nonnull_ratio") >= 0.98),
        "test_core_realized_labels_ge_55pct": bool(
            min(
                test_metric("lcs_nonnull_ratio"),
                test_metric("rts_nonnull_ratio"),
                test_metric("pmis_nonnull_ratio"),
            )
            >= 0.55
        ),
        "test_iis_coverage_ge_30pct": bool(test_metric("iis_nonnull_ratio") >= 0.30),
        "test_global_fallback_below_30pct": bool(
            max(
                test_metric("lcs_cohort_level_6_ratio", 1.0),
                test_metric("iis_cohort_level_6_ratio", 1.0),
                test_metric("gns_cohort_level_6_ratio", 1.0),
                test_metric("rts_cohort_level_6_ratio", 1.0),
                test_metric("pmis_cohort_level_6_ratio", 1.0),
            )
            < 0.30
        ),
        "test_high_odd_90_non_degenerate": bool(
            0.01 <= test_metric("high_odd_exceedance_90_ratio", -1.0) <= 0.80
        ),
        "test_monotonicity_positive_ge_70pct": bool(positive_monotonicity_ratio >= 0.70),
    }
    verdict = "READY FOR STAGE2" if all(checks.values()) else "NOT READY / REVIEW CHECKS"

    train_tables = []
    train_root = args.output_root / "validity" / "train"
    for path in sorted(train_root.glob("day=*/link_decile_monotonicity.csv")):
        table = pd.read_csv(path)
        table["date"] = path.parent.name.split("=")[1]
        train_tables.append(table)
    if train_tables:
        train = pd.concat(train_tables, ignore_index=True)
        train_summary = train.groupby(["dimension", "indicator", "decile"], as_index=False).agg(
            mean=("mean", "mean"), days=("date", "nunique")
        )
        train_dir = args.output_root / "validity" / "train_summary"
        train_dir.mkdir(parents=True, exist_ok=True)
        train_summary.to_csv(train_dir / "link_decile_monotonicity.csv", index=False)

    lines = ["# Stage1 temporal-split summary", "", f"**{verdict}**", "", "## Readiness checks", ""]
    for name, passed in checks.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} - `{name}`")
    lines += [
        "",
        "The prediction-label models were fitted on train dates only. "
        "Validation and test dates were transform-only targets.",
    ]
    report = "\n".join(lines) + "\n"
    reports = args.output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "stage1_split_summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
