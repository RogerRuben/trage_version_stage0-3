"""Two-stage stable checkpoint admission on merged unique traversals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .distribution_stability import summarize_traversals


def evaluate_candidate(
    frame: pd.DataFrame,
    *,
    checkpoint_id: str,
    validation_distribution_nll: float,
    thresholds: dict[str, float],
    route_scenario_smoke_pass: bool,
) -> dict[str, Any]:
    if frame.duplicated(["order_id", "traversal_id"]).any():
        raise ValueError("checkpoint selection requires one merged row per physical traversal")
    summary = summarize_traversals(frame)
    truth = frame["pace_sec_per_m"].to_numpy(float)
    valid = frame["pace_target_valid"].to_numpy(bool) & np.isfinite(truth) & (truth > 0)
    p50 = frame["pace_pred_p50"].to_numpy(float)
    mean = frame["pace_pred_mean"].to_numpy(float)
    p90 = frame["pace_pred_p90"].to_numpy(float)
    p95 = frame["pace_pred_p95"].to_numpy(float)
    coverage_error = max(abs(float((truth[valid] <= p90[valid]).mean()) - 0.9), abs(float((truth[valid] <= p95[valid]).mean()) - 0.95))
    hard_checks = {
        "all_outputs_finite": sum(summary["nonfinite_counts"].values()) == 0,
        "quantiles_monotonic": summary["non_monotonic_or_nonpositive_quantile_count"] == 0,
        "pace_mean_stable": summary["pace_mean"]["max"] <= thresholds["maximum_pace_mean_s_per_m"] and summary["pace_mean"]["p99_9"] <= thresholds["maximum_p99_9_pace_mean_s_per_m"],
        "mean_to_p50_stable": summary["mean_to_p50_ratio"]["max"] <= thresholds["maximum_mean_to_p50_ratio"],
        "route_scenario_smoke_stable": bool(route_scenario_smoke_pass),
    }
    return {
        "checkpoint_id": checkpoint_id,
        "hard_gate_status": "PASS" if all(hard_checks.values()) else "FAIL",
        "hard_checks": hard_checks,
        "diagnostics": {
            "maximum_row_mae_contribution_share": summary["mean_error"]["maximum_row_mae_contribution_share"],
            "maximum_row_rmse_contribution_share": summary["mean_error"]["maximum_row_rmse_contribution_share"],
            "single_row_contribution_threshold_pass": summary["mean_error"]["maximum_row_mae_contribution_share"] <= thresholds["maximum_single_row_mae_contribution_share"] and summary["mean_error"]["maximum_row_rmse_contribution_share"] <= thresholds["maximum_single_row_rmse_contribution_share"],
        },
        "validation_p50_mae": float(np.mean(np.abs(p50[valid] - truth[valid]))),
        "validation_distribution_nll": float(validation_distribution_nll),
        "validation_distribution_loss": float(validation_distribution_nll),
        "validation_mean_mae": float(np.mean(np.abs(mean[valid] - truth[valid]))),
        "validation_quantile_coverage_error": float(coverage_error),
        "unique_traversal_count": int(len(frame)),
    }


def select_checkpoint(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(candidates)
    eligible = [row for row in rows if row["hard_gate_status"] == "PASS"]
    if not eligible:
        return {"status": "NO_STABLE_CHECKPOINT", "selected_checkpoint_id": None, "candidate_count": len(rows), "stable_candidate_count": 0}
    selected = min(
        eligible,
        key=lambda row: (
            row["validation_p50_mae"],
            row.get("validation_distribution_loss", row["validation_distribution_nll"]),
            row["validation_mean_mae"],
            row["validation_quantile_coverage_error"],
            row["checkpoint_id"],
        ),
    )
    return {
        "status": "STABLE_CHECKPOINT_SELECTED",
        "selected_checkpoint_id": selected["checkpoint_id"],
        "candidate_count": len(rows),
        "stable_candidate_count": len(eligible),
        "selection_order": ["validation_p50_mae", "validation_distribution_loss", "validation_mean_mae", "validation_quantile_coverage_error", "checkpoint_id"],
        "selected_metrics": selected,
    }


def write_policy_report(*, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config = json.loads((root / "stage2/config/stage2_v5_1.json").read_text(encoding="utf-8"))
    result = {
        "schema_version": "stage2_v5_1_checkpoint_selection.1",
        "status": "FROZEN_FOR_NEXT_TRAINING_RUN",
        **config["checkpoint_selection"],
        "current_v5_checkpoint_selection_compatible": False,
        "current_v5_reason": "v5 selected checkpoints by aggregate multi-task validation loss before merged-output stability admission",
    }
    output = root / "stage2/docs/v5_1/stage2_v5_1_checkpoint_selection.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# Stage 2 v5.1 checkpoint selection\n\n"
        "Every candidate must first pass finite-output, monotonic-quantile, pace mean/ratio, and route-scenario smoke gates on merged unique traversals. Single-row error contribution is not used to choose among checkpoints because it can be driven by one extreme validation label, but it remains a fail-closed final Stage 3 admission check.\n\n"
        "Stable candidates are ordered by validation P50 MAE, family-appropriate distribution loss (pinball for M3; NLL for parametric baselines), mean MAE, P90/P95 coverage error, then checkpoint ID. A lower multi-task batch loss cannot override a failed hard gate.\n\n"
        "The frozen v5 checkpoint does not satisfy this selection contract and remains evidence only; the policy applies to the next v5.1 training run.\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    print(json.dumps(write_policy_report(repo_root=args.repo_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
