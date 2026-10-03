"""Summarize the frozen 20161031 legacy benchmark without model selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage2.v4.metrics import binary_metrics
from stage2.v4.prediction_io import merge_chunk_predictions

from .baselines import continuous_metrics
from .state_evaluation import CONTINUOUS, TAILS


def _json_number(value: Any) -> float | int | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return value.item() if hasattr(value, "item") else value


def summarize(*, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    report = root / "stage2/docs/v5/protocols/legacy"
    metrics = pd.read_csv(report / "service_time_metrics.csv")
    pace = metrics[
        metrics["split"].eq("legacy")
        & metrics["date"].astype(str).eq("20161031")
        & metrics["model"].isin(
            [
                "strict_historical_profile",
                "v4_static_entry_time",
                "hist_gradient_boosting",
                "rc_mstnet_v5_mean",
                "rc_mstnet_v5_p50",
            ]
        )
    ].copy()
    pace.to_csv(report / "legacy_service_time_comparison.csv", index=False)

    v4 = merge_chunk_predictions(
        root / "stage2/output_v4/predictions/uncalibrated",
        split="test",
        date="20161031",
    ).sort_values(["order_id", "route_sequence"], kind="stable", ignore_index=True)
    v5 = pd.read_parquet(
        root
        / "stage2/output_v5/protocols/legacy/predictions/split=legacy/date=20161031/traversal_predictions.parquet"
    ).sort_values(["order_id", "route_sequence"], kind="stable", ignore_index=True)
    probability_columns = [
        column for column in v5.columns if column.endswith("_availability_probability")
    ]
    nonfinite_probability_count = int(
        sum((~np.isfinite(v5[column].to_numpy(float))).sum() for column in probability_columns)
    )
    if nonfinite_probability_count:
        raise ValueError(
            f"legacy benchmark has {nonfinite_probability_count} non-finite availability probabilities"
        )
    prediction_manifest = json.loads(
        (
            root
            / "stage2/output_v5/protocols/legacy/deep_predictions/legacy_prediction_manifest.json"
        ).read_text(encoding="utf-8")
    )
    identity_columns = ["order_id", "traversal_id"]
    if len(v4) != len(v5) or not np.array_equal(
        v4[identity_columns].astype(str).to_numpy(),
        v5[identity_columns].astype(str).to_numpy(),
    ):
        raise ValueError("v4/v5 traversal identity mismatch on legacy benchmark")

    continuous_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    for target, mask_name, prediction_name in CONTINUOUS:
        truth = v5[target].to_numpy(float)
        valid = v5[mask_name].to_numpy(bool) & np.isfinite(truth)
        if not np.array_equal(valid, v4[mask_name].to_numpy(bool) & np.isfinite(v4[target].to_numpy(float))):
            raise ValueError(f"v4/v5 {target} validity mismatch on legacy benchmark")
        if not np.allclose(truth[valid], v4[target].to_numpy(float)[valid]):
            raise ValueError(f"v4/v5 {target} target mismatch on legacy benchmark")
        for model, prediction in (
            ("rc_mstnet_v4_frozen", v4[prediction_name].to_numpy(float)),
            ("rc_mstnet_v5_frozen", v5[prediction_name].to_numpy(float)),
        ):
            continuous_rows.append(
                {
                    "date": "20161031",
                    "target": target,
                    "model": model,
                    **continuous_metrics(np.where(valid, truth, np.nan), prediction),
                }
            )
    for target, mask_name, score_name in TAILS:
        truth = v5[target].to_numpy(float)
        valid = v5[mask_name].to_numpy(bool) & np.isfinite(truth)
        v4_valid = v4[mask_name.replace("_target_valid", "_tail_valid")].to_numpy(bool)
        if not np.array_equal(valid, v4_valid & np.isfinite(v4[target].to_numpy(float))):
            raise ValueError(f"v4/v5 {target} validity mismatch on legacy benchmark")
        for model, frame in (("rc_mstnet_v4_frozen", v4), ("rc_mstnet_v5_frozen", v5)):
            tail_rows.append(
                {
                    "date": "20161031",
                    "target": target,
                    "model": model,
                    **binary_metrics(truth, frame[score_name].to_numpy(float), valid),
                }
            )
    continuous = pd.DataFrame(continuous_rows)
    tails = pd.DataFrame(tail_rows)
    continuous.to_csv(report / "legacy_v4_v5_state_metrics.csv", index=False)
    tails.to_csv(report / "legacy_v4_v5_tail_metrics.csv", index=False)

    scenario = pd.read_csv(report / "scenario_coverage.csv")
    frozen = scenario[
        scenario["split"].eq("legacy")
        & scenario["scenario_model"].astype(str).str.endswith("_frozen_calibrated")
    ].iloc[0]
    pace_index = pace.set_index("model")
    v5_mae = float(pace_index.loc["rc_mstnet_v5_mean", "mae"])
    tree_mae = float(pace_index.loc["hist_gradient_boosting", "mae"])
    result = {
        "schema_version": "stage2_v5_legacy_frozen_benchmark.1",
        "status": "COMPLETED",
        "scientific_role": "legacy_frozen_benchmark_for_version_comparability",
        "used_for_model_or_hyperparameter_selection": False,
        "date": "20161031",
        "v5_model_id": json.loads((report / "protocol_summary.json").read_text(encoding="utf-8"))["model_id"],
        "service_time": {
            "direct_pace_row_count": int(pace_index.loc["rc_mstnet_v5_mean", "count"]),
            "v5_mean_mae": v5_mae,
            "v5_p50_mae": float(pace_index.loc["rc_mstnet_v5_p50", "mae"]),
            "tree_mae": tree_mae,
            "strict_historical_profile_mae": float(pace_index.loc["strict_historical_profile", "mae"]),
            "v4_static_entry_time_proxy_mae": float(pace_index.loc["v4_static_entry_time", "mae"]),
            "v5_mean_relative_to_tree": v5_mae / tree_mae - 1.0,
        },
        "route_scenarios": {
            column: _json_number(frozen[column])
            for column in (
                "route_count",
                "mean_mae_s",
                "mean_rmse_s",
                "p50_coverage",
                "p90_coverage",
                "p95_coverage",
            )
        },
        "prediction_numeric_stability": {
            "nonfinite_availability_probability_count": nonfinite_probability_count,
            "mixed_precision_fallback_batch_count": int(
                prediction_manifest.get("mixed_precision_fallback_batch_count", 0)
            ),
        },
        "actual_v4_v5_auxiliary_comparison_path": "protocols/legacy/legacy_v4_v5_state_metrics.csv",
        "percentiles_are_descriptive_only": True,
        "new_unseen_test_claimed": False,
    }
    destination = root / "stage2/docs/v5/stage2_v5_legacy_benchmark_summary.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = [
        "# Stage 2 v5 legacy frozen benchmark",
        "",
        "20161031 is a legacy frozen benchmark for v4/v5 comparability. It was not used for model selection and is not claimed as an untouched final test.",
        "",
        "| Model | Direct pace MAE (s/m) |",
        "|---|---:|",
    ]
    for model in (
        "strict_historical_profile",
        "v4_static_entry_time",
        "hist_gradient_boosting",
        "rc_mstnet_v5_mean",
        "rc_mstnet_v5_p50",
    ):
        markdown.append(f"| {model} | {float(pace_index.loc[model, 'mae']):.9f} |")
    markdown += [
        "",
        f"The v5 mean improves on the tree baseline by {-100.0 * (v5_mae / tree_mae - 1.0):.2f}% on 717,805 direct-pace traversals.",
        "",
        f"Frozen calibrated route coverage: P50 {float(frozen['p50_coverage']):.4f}, P90 {float(frozen['p90_coverage']):.4f}, P95 {float(frozen['p95_coverage']):.4f} across {int(frozen['route_count']):,} routes.",
        "",
        "Actual frozen RC-MSTNet v4/v5 auxiliary-target metrics are recorded separately on identical traversal rows; percentile-tail results are descriptive only.",
    ]
    (root / "stage2/docs/v5/stage2_v5_legacy_benchmark_summary.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
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
