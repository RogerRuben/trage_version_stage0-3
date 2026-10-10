"""Read-only diagnostics for the preregistered Fold 3 service-time tail.

This module deliberately does not modify predictions, thresholds, checkpoints,
or admission decisions.  It explains sensitivity of the log-normal mean after
the frozen evaluation has completed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def diagnose(*, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    prediction_root = root / "stage2/output_v5/protocols/fold_3/predictions/split=evaluation"
    date_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    for date in ("20161026", "20161027"):
        frame = pd.read_parquet(prediction_root / f"date={date}" / "traversal_predictions.parquet")
        valid = frame["pace_target_valid"].astype(bool).to_numpy(copy=True)
        valid &= np.isfinite(frame["pace_sec_per_m"].to_numpy(float))
        local = frame.loc[valid].copy()
        truth = local["pace_sec_per_m"].to_numpy(float)
        prediction = local["pace_pred_mean"].to_numpy(float)
        absolute_error = np.abs(prediction - truth)
        below_one = prediction <= 1.0
        largest = int(np.argmax(prediction))
        largest_error_contribution = float(absolute_error[largest] / len(local))
        date_rows.append(
            {
                "date": date,
                "valid_count": int(len(local)),
                "reported_mean_mae": float(np.mean(absolute_error)),
                "diagnostic_mae_excluding_predictions_above_1_sec_per_m": (
                    float(np.mean(absolute_error[below_one])) if below_one.any() else None
                ),
                "prediction_count_above_1_sec_per_m": int((prediction > 1.0).sum()),
                "prediction_count_above_10_sec_per_m": int((prediction > 10.0).sum()),
                "maximum_prediction_sec_per_m": float(np.max(prediction)),
                "largest_row_mae_contribution": largest_error_contribution,
                "largest_row_share_of_reported_mae": float(
                    largest_error_contribution / np.mean(absolute_error)
                ),
            }
        )
        columns = [
            "order_id",
            "traversal_id",
            "route_sequence",
            "allocated_distance_m",
            "pace_sec_per_m",
            "pace_pred_mean",
            "pace_pred_p50",
            "pace_log_mu",
            "pace_log_scale",
            "service_time_availability_probability",
        ]
        for row in local.nlargest(3, "pace_pred_mean")[columns].to_dict("records"):
            top_rows.append(
                {
                    "date": date,
                    **{
                        column: (str(row[column]) if column == "order_id" else _finite(float(row[column])))
                        for column in columns
                    },
                    "implied_log_normal_sigma": _finite(float(np.exp(row["pace_log_scale"]))),
                }
            )

    result = {
        "schema_version": "stage2_v5_frozen_fold3_tail_diagnostic.1",
        "scope": "read_only_post_evaluation_diagnostic",
        "protocol_modified": False,
        "predictions_modified": False,
        "admission_metrics_modified": False,
        "interpretation": (
            "The frozen log-normal mean is highly sensitive to a single large predicted scale on "
            "20161026; this diagnostic is explanatory only and the preregistered MAE is unchanged."
        ),
        "dates": date_rows,
        "largest_predictions": top_rows,
    }
    docs = root / "stage2/docs/v5"
    (docs / "stage2_v5_fold3_tail_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Stage 2 v5 Fold 3 frozen-tail diagnostic",
        "",
        "This is a read-only post-evaluation diagnostic. It does not alter predictions, metrics, or admission.",
        "",
        "| Date | Valid rows | Frozen mean MAE | Diagnostic MAE with prediction <= 1 s/m | >1 s/m | >10 s/m | Max prediction | Largest-row share of MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in date_rows:
        lines.append(
            f"| {item['date']} | {item['valid_count']} | {item['reported_mean_mae']:.9f} | "
            f"{item['diagnostic_mae_excluding_predictions_above_1_sec_per_m']:.9f} | "
            f"{item['prediction_count_above_1_sec_per_m']} | {item['prediction_count_above_10_sec_per_m']} | "
            f"{item['maximum_prediction_sec_per_m']:.6f} | {item['largest_row_share_of_reported_mae']:.2%} |"
        )
    lines += [
        "",
        "The reported Fold 3 metrics remain the frozen values. The diagnostic shows whether a log-normal mean tail, rather than broad daily degradation, explains the instability.",
    ]
    (docs / "stage2_v5_fold3_tail_diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    print(json.dumps(diagnose(repo_root=args.repo_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
