"""Write a focused readiness summary for the configured test day."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REALIZED_DIMS = ["lcs", "iis", "rts", "pmis"]
ALL_DIMS = ["lcs", "iis", "gns", "rts", "pmis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0-output-root", type=Path, required=True)
    parser.add_argument("--stage1-output-root", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    return parser.parse_args()


def yes_no(value: bool) -> str:
    return "PASS" if value else "REVIEW"


def first_row(path: Path, **filters) -> dict:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    for column, value in filters.items():
        if column in frame.columns:
            frame = frame[frame[column].astype(str) == str(value)]
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def finite_float(row: dict, key: str, default: float = float("nan")) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except Exception:
        return default


def finite_first(row: dict, keys: list[str], default: float = float("nan")) -> float:
    for key in keys:
        value = finite_float(row, key, default)
        if np.isfinite(value):
            return value
    return default


def main() -> None:
    args = parse_args()
    config = json.loads(args.split_config.read_text(encoding="utf-8"))
    test_date = config["test_date"]
    reports = args.stage1_output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    stage0 = first_row(args.stage0_output_root / "reports" / "stage0_readiness.csv", date=test_date)
    coverage = first_row(args.stage1_output_root / "reports" / "stage1_label_coverage.csv", date=test_date)
    shift_path = args.stage1_output_root / "reports" / "split_shift_audit.csv"
    shift = pd.read_csv(shift_path) if shift_path.exists() else pd.DataFrame()
    test_shift = shift[shift.target_split.eq("test")] if not shift.empty and "target_split" in shift.columns else pd.DataFrame()

    checks = {}
    checks["stage0_gate_pass"] = stage0.get("stage0_gate") == "PASS"
    matcher_success = finite_first(stage0, ["matcher_success_ratio", "hmm_success_ratio"])
    checks["matcher_success_ge_85pct"] = matcher_success >= 0.85
    checks["fallback_le_20pct"] = finite_float(stage0, "fallback_ratio") <= 0.20
    checks["median_p90_match_le_30m"] = finite_float(stage0, "median_order_p90_match_dist_m") <= 30.0
    checks["observed_route_length_ge_70pct"] = finite_float(stage0, "observed_link_length_ratio") >= 0.70
    checks["stage1_coverage_complete"] = coverage.get("status") == "complete"
    checks["gns_coverage_ge_98pct"] = finite_float(coverage, "gns_nonnull_ratio") >= 0.98
    checks["core_realized_label_coverage_ge_55pct"] = min(
        finite_float(coverage, f"{dimension}_nonnull_ratio") for dimension in ["lcs", "rts", "pmis"]
    ) >= 0.55
    checks["iis_coverage_ge_30pct"] = finite_float(coverage, "iis_nonnull_ratio") >= 0.30
    checks["global_fallback_below_30pct"] = max(
        finite_float(coverage, f"{dimension}_cohort_level_6_ratio", 1.0) for dimension in ALL_DIMS
    ) < 0.30
    checks["high_odd_90_non_degenerate"] = 0.01 <= finite_float(coverage, "high_odd_exceedance_90_ratio", -1.0) <= 0.80
    if test_shift.empty:
        checks["split_shift_not_high"] = False
        high_shift_metrics = []
    else:
        high_shift_metrics = test_shift[test_shift.shift_level.eq("high")]["feature"].tolist()
        checks["split_shift_not_high"] = len(high_shift_metrics) == 0

    verdict = "READY FOR STAGE2/3/4 TEST-DAY USE" if all(checks.values()) else "REVIEW BEFORE STAGE2/3/4"

    lines = [
        "# Test-day readiness summary",
        "",
        f"Configured test date: `{test_date}`",
        "",
        f"**{verdict}**",
        "",
        "## Checks",
        "",
    ]
    for key, passed in checks.items():
        lines.append(f"- {yes_no(passed)} - `{key}`")
    lines += ["", "## Key metrics", ""]
    if stage0:
        lines += [
            f"- Matcher fallback ratio: {finite_float(stage0, 'fallback_ratio'):.2%}",
            f"- Matcher success ratio: {matcher_success:.2%}",
            f"- Median order P90 match distance: {finite_float(stage0, 'median_order_p90_match_dist_m'):.2f} m",
            f"- Observed traversal ratio: {finite_float(stage0, 'observed_traversal_ratio'):.2%}",
            f"- Observed route length ratio: {finite_float(stage0, 'observed_link_length_ratio'):.2%}",
            f"- Retained compact Stage0 size: {finite_float(stage0, 'retained_mb'):.1f} MB",
        ]
    else:
        lines.append("- Stage0 readiness row is missing.")
    if coverage:
        lines += ["", "## Stage1 coverage", ""]
        for dimension in ALL_DIMS:
            lines.append(f"- {dimension.upper()} non-null ratio: {finite_float(coverage, f'{dimension}_nonnull_ratio'):.2%}")
        lines.append(f"- Inferred path ratio: {finite_float(coverage, 'inferred_path_ratio'):.2%}")
        lines.append(f"- Low-quality traversal ratio: {finite_float(coverage, 'low_quality_ratio'):.2%}")
        lines.append(f"- High ODD exceedance 90 ratio: {finite_float(coverage, 'high_odd_exceedance_90_ratio'):.2%}")
    if high_shift_metrics:
        lines += ["", "## High-shift metrics to review", ""]
        for feature in high_shift_metrics:
            lines.append(f"- {feature}")
    lines += [
        "",
        "Note: this summary always follows the currently configured `test_date` in `split_config.json`.",
    ]
    target = reports / "test_day_readiness_summary.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
