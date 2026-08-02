"""Aggregate v5.1 rolling diagnostics without claiming a new holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .baselines import _predict as predict_baselines
from .config import load_inherited_payload
from .data import load_v5_day


PROTOCOLS = {
    "development": "stage2/config/stage2_v5_1_training.json",
    "fold_1": "stage2/config/stage2_v5_1_fold_1.json",
    "fold_2": "stage2/config/stage2_v5_1_fold_2.json",
    "fold_3": "stage2/config/stage2_v5_1_fold_3.json",
    "legacy": "stage2/config/stage2_v5_1_legacy.json",
}


def _scenario_tail_audit(product_root: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Recompute bounded mean/CVaR diagnostics from frozen formal samples."""

    audits: dict[tuple[str, str], dict[str, float]] = {}
    for manifest_path in sorted(product_root.glob("split=*/date=*/manifest.json")):
        split = manifest_path.parent.parent.name.removeprefix("split=")
        date = manifest_path.parent.name.removeprefix("date=")
        sample_path = manifest_path.parent / "route_scenario_samples.npz"
        with np.load(sample_path, allow_pickle=True) as payload:
            samples = np.asarray(payload["route_time_s"], dtype=np.float64)
        route_mean = samples.mean(axis=1)
        tail_count = max(1, int(np.ceil(samples.shape[1] * 0.05)))
        tail = np.partition(samples, samples.shape[1] - tail_count, axis=1)[:, -tail_count:]
        route_cvar95 = tail.mean(axis=1)
        audits[(split, date)] = {
            "maximum_route_mean_s": float(route_mean.max()),
            "maximum_route_cvar95_s": float(route_cvar95.max()),
        }
    return audits


def _paired_bootstrap(differences: pd.DataFrame, *, seed: int, replicates: int = 1000) -> dict[str, Any]:
    grouped = differences.groupby("order_id", sort=False, observed=True)["difference"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(replicates)
    for index in range(replicates):
        sample = rng.integers(0, len(sums), size=len(sums))
        draws[index] = sums[sample].sum() / counts[sample].sum()
    return {
        "p50_minus_tree_absolute_error": float(differences["difference"].mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "order_count": int(len(grouped)),
        "replicates": int(replicates),
    }


def _day_metrics(
    *,
    root: Path,
    protocol: str,
    split: str,
    date: str,
    prediction_root: Path,
    baseline_bundle: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    columns = [
        "order_id", "traversal_id", "pace_sec_per_m", "pace_target_valid",
        "pace_pred_p50", "pace_pred_p90", "pace_pred_p95",
    ]
    prediction = pd.read_parquet(
        prediction_root / f"split={split}" / f"date={date}" / "traversal_predictions.parquet",
        columns=columns,
    )
    source = load_v5_day(date, split=split, repo_root=root)
    baseline = predict_baselines(source, baseline_bundle)["hist_gradient_boosting"]
    tree = source[["order_id", "traversal_id"]].copy()
    tree["tree"] = baseline
    comparison = prediction.merge(tree, on=["order_id", "traversal_id"], how="inner", validate="one_to_one")
    truth = comparison["pace_sec_per_m"].to_numpy(float)
    valid = comparison["pace_target_valid"].to_numpy(bool) & np.isfinite(truth)
    comparison = comparison.loc[valid].copy()
    truth = comparison["pace_sec_per_m"].to_numpy(float)
    p50 = comparison["pace_pred_p50"].to_numpy(float)
    tree_values = comparison["tree"].to_numpy(float)
    metrics = {
        "protocol": protocol,
        "split": split,
        "date": date,
        "count": int(len(truth)),
        "p50_mae": float(np.mean(np.abs(p50 - truth))),
        "p50_rmse": float(np.sqrt(np.mean(np.square(p50 - truth)))),
        "tree_mae": float(np.mean(np.abs(tree_values - truth))),
        "tree_rmse": float(np.sqrt(np.mean(np.square(tree_values - truth)))),
        "p90_coverage": float(np.mean(truth <= comparison["pace_pred_p90"].to_numpy(float))),
        "p95_coverage": float(np.mean(truth <= comparison["pace_pred_p95"].to_numpy(float))),
    }
    differences = pd.DataFrame(
        {
            "order_id": comparison["order_id"].astype(str).to_numpy(),
            "difference": np.abs(p50 - truth) - np.abs(tree_values - truth),
        }
    )
    bootstrap = {"protocol": protocol, "split": split, "date": date, **_paired_bootstrap(differences, seed=seed)}
    return metrics, bootstrap


def run(*, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    metric_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    route_frames: list[pd.DataFrame] = []
    tree_route_rows: list[dict[str, Any]] = []
    cross_order_rows: list[dict[str, Any]] = []
    for protocol, config_name in PROTOCOLS.items():
        protocol_root = root / "stage2/output_v5_1" / protocol
        model_manifest = protocol_root / "deep_model/model_manifest.json"
        if not model_manifest.is_file():
            continue
        config = load_inherited_payload(root / config_name)
        baseline_path = root / "stage2/output_v5/protocols" / protocol / "baselines/service_time_baselines.joblib"
        bundle = joblib.load(baseline_path)
        prediction_root = protocol_root / "predictions"
        evaluation_split = "legacy" if protocol == "legacy" else "evaluation"
        evaluation_dates = config["split"]["legacy_test_dates" if protocol == "legacy" else "evaluation_dates"]
        for offset, date in enumerate(evaluation_dates):
            metrics, bootstrap = _day_metrics(
                root=root,
                protocol=protocol,
                split=evaluation_split,
                date=date,
                prediction_root=prediction_root,
                baseline_bundle=bundle,
                seed=20261009 + offset,
            )
            metric_rows.append(metrics)
            bootstrap_rows.append(bootstrap)
            manifest_path = protocol_root / "formal_calibrated" / f"split={evaluation_split}" / f"date={date}" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            cross_order_rows.append(
                {
                    "protocol": protocol,
                    "date": date,
                    "selected_family": manifest["correlation_model_id"],
                    "cross_order_scenario_coherent": manifest["cross_order_scenario_coherent"],
                    "fixed_seed": manifest["scenario_seed"],
                    "status": "PASS" if manifest["cross_order_scenario_coherent"] else "FAIL",
                }
            )
        route_path = protocol_root / "reports/route_scoring.csv"
        if route_path.is_file():
            route_frame = pd.read_csv(route_path)
            tail_audit = _scenario_tail_audit(protocol_root / "formal_calibrated")
            audit_rows = [
                {"split": split, "date": date, **values}
                for (split, date), values in tail_audit.items()
            ]
            route_frame["split"] = route_frame["split"].astype(str)
            route_frame["date"] = route_frame["date"].astype(str)
            route_frame = route_frame.drop(
                columns=["maximum_route_mean_s", "maximum_route_cvar95_s"], errors="ignore"
            ).merge(pd.DataFrame(audit_rows), on=["split", "date"], how="left", validate="one_to_one")
            route_frames.append(route_frame)
        tree_manifest_path = protocol_root / "reports/tree_scenario_manifest.json"
        if tree_manifest_path.is_file():
            tree_manifest = json.loads(tree_manifest_path.read_text(encoding="utf-8"))
            tree_route_rows.extend(
                {
                    "protocol": protocol,
                    "split": product["split"],
                    "date": product["date"],
                    "prediction_source": "tree",
                    "stability_status": product["stability_status"],
                    **product["quality"],
                }
                for product in tree_manifest.get("products", [])
            )
    metrics = pd.DataFrame(metric_rows)
    bootstraps = pd.DataFrame(bootstrap_rows)
    rolling = metrics[metrics["protocol"].isin(["fold_1", "fold_2", "fold_3"])].copy()
    rolling_count = float(rolling["count"].sum())
    p50_mae = float((rolling["p50_mae"] * rolling["count"]).sum() / rolling_count)
    tree_mae = float((rolling["tree_mae"] * rolling["count"]).sum() / rolling_count)
    daily_wins = int((rolling["p50_mae"] < rolling["tree_mae"]).sum())
    all_ci_below_zero = bool(
        (bootstraps.loc[bootstraps["protocol"].isin(["fold_1", "fold_2", "fold_3"]), "ci95_high"] < 0).all()
    )
    summary = {
        "schema_version": "stage2_v5_1_rolling_diagnostic.1",
        "scientific_role": "rolling_temporal_backtesting_evidence_not_new_holdout_confirmation",
        "fold_count": int(rolling["protocol"].nunique()),
        "evaluation_day_count": int(len(rolling)),
        "aggregate_p50_mae": p50_mae,
        "aggregate_tree_mae": tree_mae,
        "relative_difference": p50_mae / tree_mae - 1.0,
        "daily_p50_wins": daily_wins,
        "all_paired_ci_below_zero": all_ci_below_zero,
        "status": "PASS" if p50_mae < tree_mae and all_ci_below_zero else "FAIL",
    }
    docs = root / "stage2/docs/v5_1"
    metrics.to_csv(docs / "stage2_v5_1_model_comparison.csv", index=False)
    metrics[["protocol", "split", "date", "count", "p50_mae", "p50_rmse", "p90_coverage", "p95_coverage"]].to_csv(
        docs / "stage2_v5_1_quantile_metrics.csv", index=False
    )
    bootstraps.to_csv(docs / "stage2_v5_1_paired_bootstrap.csv", index=False)
    if route_frames:
        route_all = pd.concat(route_frames, ignore_index=True)
        route_all["prediction_source"] = "deep_scenario"
        route_all.to_csv(docs / "stage2_v5_1_route_scoring.csv", index=False)
        route_summary = route_all.groupby("protocol", observed=True)[
            [
                "route_p50_mae_s", "route_mean_mae_s", "route_mean_rmse_s",
                "sample_crps_s", "weighted_interval_score_s", "p90_coverage",
                "p95_coverage", "p90_p50_width_s", "p95_p50_width_s",
                "maximum_scenario_s", "maximum_route_mean_s", "maximum_route_cvar95_s",
                "extreme_scenario_share",
            ]
        ].mean(numeric_only=True).reset_index()
        headers = route_summary.columns.tolist()
        lines = [
            "# Stage 2 v5.1 scenario quality",
            "",
            "All model selection used the frozen validation proper-score rule; scale, dispersion, and offset were fitted only on each protocol's calibration date.",
            "",
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
        ]
        for values in route_summary.to_numpy():
            lines.append("| " + " | ".join(str(value) for value in values) + " |")
        lines += [
            "",
            "Formal scenarios pass a cross-order-coherence hard gate before the frozen proper-score ordering. Mean/CVaR remain blocked unless the final rolling verification admits them.",
        ]
        (docs / "stage2_v5_1_scenario_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if tree_route_rows:
        pd.DataFrame(tree_route_rows).to_csv(
            docs / "stage2_v5_1_tree_scenario_comparison.csv", index=False
        )
    (docs / "stage2_v5_1_rolling_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    cross_order = {
        "schema_version": "stage2_v5_1_cross_order_scenario_audit.1",
        "status": "PASS" if cross_order_rows and all(row["status"] == "PASS" for row in cross_order_rows) else "FAIL",
        "same_scenario_index_semantics": "one system traffic state shared across the complete day batch",
        "rows": cross_order_rows,
    }
    (docs / "stage2_v5_1_cross_order_scenario_audit.json").write_text(
        json.dumps(cross_order, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"rolling": summary, "cross_order": cross_order}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    print(json.dumps(run(repo_root=args.repo_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
