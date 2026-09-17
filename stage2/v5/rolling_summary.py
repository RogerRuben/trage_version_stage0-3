"""Aggregate preregistered rolling-origin fold evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def summarize(*, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    docs = root / "stage2/docs/v5"
    metric_parts: list[pd.DataFrame] = []
    bootstrap_parts: list[pd.DataFrame] = []
    scenario_parts: list[pd.DataFrame] = []
    protocol_summaries: list[dict[str, Any]] = []
    for fold in ("fold_1", "fold_2", "fold_3"):
        report = docs / "protocols" / fold
        summary = json.loads((report / "protocol_summary.json").read_text(encoding="utf-8"))
        protocol_summaries.append(summary)
        metrics = pd.read_csv(report / "service_time_metrics.csv")
        metrics = metrics[metrics["split"].eq("evaluation")].copy()
        metrics.insert(0, "fold", fold)
        metric_parts.append(metrics)
        bootstrap = pd.read_csv(report / "deep_paired_error_bootstrap.csv")
        bootstrap = bootstrap[bootstrap["split"].eq("evaluation")].copy()
        bootstrap.insert(0, "fold", fold)
        bootstrap_parts.append(bootstrap)
        scenarios = pd.read_csv(report / "scenario_coverage.csv")
        scenarios = scenarios[
            scenarios["split"].eq("evaluation")
            & scenarios["scenario_model"].astype(str).str.endswith("_frozen_calibrated")
        ].copy()
        scenarios.insert(0, "fold", fold)
        scenario_parts.append(scenarios)
    metrics = pd.concat(metric_parts, ignore_index=True)
    bootstraps = pd.concat(bootstrap_parts, ignore_index=True)
    scenarios = pd.concat(scenario_parts, ignore_index=True)
    metrics.to_csv(docs / "rolling_fold_metrics.csv", index=False)
    bootstraps.to_csv(docs / "rolling_fold_paired_error_bootstrap.csv", index=False)
    scenarios.to_csv(docs / "rolling_fold_scenario_coverage.csv", index=False)

    pace = metrics[metrics["model"].isin(["rc_mstnet_v5_mean", "hist_gradient_boosting"])].copy()
    pace["weighted_absolute_error"] = pace["mae"] * pace["count"]
    aggregate = pace.groupby("model", observed=True)[["weighted_absolute_error", "count"]].sum()
    aggregate_mae = aggregate["weighted_absolute_error"] / aggregate["count"]
    fold_daily = pace.pivot_table(index=["fold", "date"], columns="model", values="mae", aggfunc="first")
    fold_daily["v5_win"] = fold_daily["rc_mstnet_v5_mean"] < fold_daily["hist_gradient_boosting"]
    fold_wins = fold_daily.groupby(level="fold")["v5_win"].all()
    scenario_weights = scenarios["route_count"].to_numpy(float)
    route_coverage = {
        name: float(np.average(scenarios[name], weights=scenario_weights))
        for name in ("p50_coverage", "p90_coverage", "p95_coverage")
    }
    aggregate_relative_change = float(
        aggregate_mae["rc_mstnet_v5_mean"] / aggregate_mae["hist_gradient_boosting"] - 1.0
    )
    aggregate_baseline_result = "WIN" if aggregate_relative_change < 0.0 else "LOSS_OR_TIE"
    result = {
        "schema_version": "stage2_v5_rolling_origin_summary.1",
        # Completion and scientific acceptance are deliberately separate: a
        # successfully executed preregistered protocol can still reject the
        # candidate model.
        "execution_status": "COMPLETED",
        "scientific_status": (
            "BASELINE_VALIDATED"
            if aggregate_baseline_result == "WIN"
            else "BASELINE_NOT_BEATEN"
        ),
        "aggregate_baseline_result": aggregate_baseline_result,
        "fold_count": 3,
        "evaluation_dates": sorted(metrics["date"].astype(str).unique().tolist()),
        "v5_aggregate_mae": float(aggregate_mae["rc_mstnet_v5_mean"]),
        "tree_aggregate_mae": float(aggregate_mae["hist_gradient_boosting"]),
        "aggregate_relative_mae_change": aggregate_relative_change,
        "daily_win_count": int(fold_daily["v5_win"].sum()),
        "daily_comparison_count": int(len(fold_daily)),
        "fold_win_count": int(fold_wins.sum()),
        "fold_results": {fold: ("WIN" if bool(value) else "LOSS_OR_MIXED") for fold, value in fold_wins.items()},
        "route_scenario_coverage": route_coverage,
        "percentile_targets_used_for_model_selection": False,
        "upstream_rebuild_performed": False,
        "protocol_model_ids": {item["protocol"]: item["model_id"] for item in protocol_summaries},
    }
    (docs / "stage2_v5_rolling_origin_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    print(json.dumps(summarize(repo_root=args.repo_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
