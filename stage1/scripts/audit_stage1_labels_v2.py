"""Compute Stage 1 v2 acceptance evidence from produced labels and models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage1.canonical.labels import CORE_DIMENSIONS, DIMENSIONS
from stage1.canonical.quantiles import MergeableHistogram, empirical_cdf_interpolated


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fit-traversal", type=Path, required=True)
    parser.add_argument("--dates", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    return parser.parse_args()


def partition_audit(path: Path) -> dict:
    frame = pd.read_parquet(path, columns=["travel_time_sec", "observed_distance_m", "traversal_quality"])
    values = (
        frame.loc[frame.traversal_quality.ne("low") & frame.observed_distance_m.ge(10), "travel_time_sec"]
        / frame.loc[frame.traversal_quality.ne("low") & frame.observed_distance_m.ge(10), "observed_distance_m"]
    ).clip(0.01, 10.0).dropna().to_numpy()
    edges = np.geomspace(0.01, 10.0, 4097)
    medians = []
    for partition_count in (1, 5, 20):
        merged = MergeableHistogram.empty(edges)
        for part in np.array_split(values, partition_count):
            local = MergeableHistogram.empty(edges); local.update(part); merged.merge(local)
        medians.append(merged.quantile(0.5))
    return {
        "partition_counts": [1, 5, 20],
        "median_estimates": medians,
        "maximum_difference": float(max(medians) - min(medians)),
        "pass": len(set(medians)) == 1,
    }


def cdf_model_audit(root: Path) -> dict:
    failures = 0; models = 0; lower_failures = 0; upper_failures = 0
    for path in (root / "models" / "cohort_cdf_v2").glob("*.parquet"):
        model = pd.read_parquet(path)
        for _, group in model.groupby("key", sort=False):
            values = np.linspace(-0.01, 1.01, 1003)
            cdf = empirical_cdf_interpolated(values, group.support_value.to_numpy(), group["count"].to_numpy())
            failures += int(np.any(np.diff(cdf) < -1e-12))
            lower_failures += int(cdf[0] != 0.0)
            upper_failures += int(cdf[-1] != 1.0)
            models += 1
    return {
        "cohort_models": models,
        "monotonicity_failures": failures,
        "lower_tail_failures": lower_failures,
        "upper_tail_failures": upper_failures,
        "empty_bin_policy": "ordered_support_interpolation_not_fill_0.5",
        "pass": failures == 0 and lower_failures == 0 and upper_failures == 0,
    }


def main():
    args = arguments()
    coverage_rows = []; day_results = []
    for date in [item.strip() for item in args.dates.split(",") if item.strip()]:
        links = pd.read_parquet(args.output_root / "link_labels" / f"day={date}.parquet")
        orders = pd.read_parquet(args.output_root / "order_labels" / f"day={date}.parquet")
        range_failures = 0; missingness_failures = 0
        coverage = {"date": date, "orders": int(len(orders)), "link_rows": int(len(links))}
        for dimension in DIMENSIONS:
            pct = links[f"{dimension}_pct_link"]
            raw = links[f"{dimension}_raw"]
            range_failures += int((pct.dropna().lt(0) | pct.dropna().gt(1)).sum())
            missingness_failures += int((raw.isna() & pct.notna()).sum())
            coverage[f"{dimension}_link_coverage"] = float(pct.notna().mean())
            coverage[f"{dimension}_order_coverage"] = float(orders[f"{dimension}_available"].mean())
        expected = orders[[f"{dimension}_mean" for dimension in CORE_DIMENSIONS]].mean(axis=1, skipna=False)
        composite_failures = int((orders.core_composite_mean - expected).abs().gt(1e-12).sum())
        coverage_rows.append(coverage)
        day_results.append({
            "date": date,
            "orders": int(len(orders)),
            "range_failures": range_failures,
            "missing_raw_but_normalized_nonmissing": missingness_failures,
            "core_composite_recalculation_failures": composite_failures,
            "distinct_composition_signatures": int(orders.composition_signature.nunique()),
            "status": "PASS" if range_failures == 0 and missingness_failures == 0 and composite_failures == 0 else "FAIL",
        })
    partition = partition_audit(args.fit_traversal)
    cdf = cdf_model_audit(args.output_root)
    comparison_path = args.output_root / "v1_v2_comparison.csv"
    comparison = pd.read_csv(comparison_path).to_dict("records")
    status = "PASS" if partition["pass"] and cdf["pass"] and all(day["status"] == "PASS" for day in day_results) else "FAIL"
    result = {
        "status": status,
        "schema_version": "stage1_label_schema_v2",
        "partition_invariance": partition,
        "cdf": cdf,
        "core_composite_dimensions": list(CORE_DIMENSIONS),
        "pmis_role": "interaction_output_excluded_from_core_composite",
        "iis_missing_policy": "NA_not_zero",
        "days": day_results,
        "v1_v2_comparison": comparison,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(coverage_rows).to_csv(args.coverage, index=False)
    print(json.dumps(result, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
