"""Same-row evaluation for RC-MSTNet v5 and the frozen strong baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .baselines import _paired_bootstrap, _predict, continuous_metrics
from .availability import stabilized_ipw
from .config import load_config
from .data import load_v5_day
from .metrics import evaluate_stop_two_part


def _binary(truth: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(truth, dtype=np.int8)
    p = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    classes = len(np.unique(y)) == 2
    return {
        "count": len(y), "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, p)) if classes else None,
        "roc_auc": float(roc_auc_score(y, p)) if classes else None,
        "brier": float(brier_score_loss(y, p)),
    }


def _quantiles(truth: np.ndarray, prediction: pd.DataFrame, valid: np.ndarray, split: str, date: str) -> list[dict[str, Any]]:
    y = truth[valid]
    rows: list[dict[str, Any]] = []
    for quantile in (0.5, 0.9, 0.95):
        name = f"pace_pred_p{int(quantile * 100)}"
        pred = prediction.loc[valid, name].to_numpy(float)
        error = y - pred
        pinball = np.maximum(quantile * error, (quantile - 1.0) * error)
        rows.append({"split": split, "date": date, "target": "pace_sec_per_m", "quantile": quantile, "empirical_coverage": float((y <= pred).mean()), "coverage_error": float((y <= pred).mean() - quantile), "pinball_loss": float(pinball.mean()), "count": len(y)})
    return rows


def evaluate(*, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config = load_config(root / "stage2/config/stage2_v5.json")
    baseline_bundle = joblib.load(root / "stage2/output_v5/baselines/service_time_baselines.joblib")
    dates = [("validation_model", date) for date in config.section("split")["validation_model_dates"]]
    dates += [("calibration", date) for date in config.section("split")["calibration_dates"]]
    metric_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []
    stop_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    subgroup_parts: list[pd.DataFrame] = []
    seed = int(config.section("runtime")["random_seed"])
    for split, date in dates:
        prediction_path = root / "stage2/output_v5/predictions" / f"split={split}" / f"date={date}" / "traversal_predictions.parquet"
        prediction = pd.read_parquet(prediction_path)
        source = load_v5_day(date, split=split, repo_root=root)
        source = source.sort_values(["order_id", "route_sequence"], kind="stable", ignore_index=True)
        if not np.array_equal(prediction[["order_id", "traversal_id"]].astype(str).to_numpy(), source[["order_id", "traversal_id"]].astype(str).to_numpy()):
            raise ValueError(f"prediction/source identity mismatch on {date}")
        baseline = _predict(source, baseline_bundle)
        truth = pd.to_numeric(source["pace_sec_per_m"], errors="coerce").to_numpy(float)
        valid = source["pace_target_valid"].to_numpy(bool) & np.isfinite(truth)
        models = {**baseline, "rc_mstnet_v5_mean": prediction["pace_pred_mean"].to_numpy(float), "rc_mstnet_v5_p50": prediction["pace_pred_p50"].to_numpy(float)}
        for model, values in models.items():
            metric_rows.append({"split": split, "date": date, "target": "pace_sec_per_m", "model": model, **continuous_metrics(np.where(valid, truth, np.nan), values)})
        comparison = pd.DataFrame({"order_id": source["order_id"].astype(str), "truth": np.where(valid, truth, np.nan), "rc_mstnet_v5_mean": models["rc_mstnet_v5_mean"], "hist_gradient_boosting": models["hist_gradient_boosting"]})
        bootstrap_rows.append({"split": split, "date": date, "left_model": "rc_mstnet_v5_mean", "right_model": "hist_gradient_boosting", **_paired_bootstrap(comparison, "rc_mstnet_v5_mean", "hist_gradient_boosting", seed=seed)})
        quantile_rows.extend(_quantiles(truth, prediction, valid, split, date))
        service_available = source["pace_target_valid"].to_numpy(bool)
        service_probability = prediction["service_time_availability_probability"].to_numpy(float)
        ipw, ipw_diagnostics = stabilized_ipw(
            service_available,
            service_probability,
            epsilon=float(config.section("selection")["epsilon"]),
            maximum_weight=float(config.section("selection")["maximum_weight"]),
        )
        deep_absolute_error = np.abs(models["rc_mstnet_v5_mean"] - truth)
        tree_absolute_error = np.abs(models["hist_gradient_boosting"] - truth)
        availability_rows.append({
            "split": split, "date": date, "target": "service_time",
            **_binary(service_available, service_probability),
            "effective_sample_size_complete_case": int(service_available.sum()),
            "effective_sample_size_ipw": ipw_diagnostics.effective_sample_size,
            "ipw_weight_p50": ipw_diagnostics.weight_p50,
            "ipw_weight_p90": ipw_diagnostics.weight_p90,
            "ipw_weight_p99": ipw_diagnostics.weight_p99,
            "ipw_weight_max": ipw_diagnostics.weight_max,
            "ipw_clipped_count": ipw_diagnostics.clipped_count,
            "complete_case_deep_mae": float(deep_absolute_error[service_available].mean()),
            "stabilized_ipw_deep_mae": float(np.sum(ipw * np.nan_to_num(deep_absolute_error)) / np.sum(ipw)),
            "complete_case_tree_mae": float(tree_absolute_error[service_available].mean()),
            "stabilized_ipw_tree_mae": float(np.sum(ipw * np.nan_to_num(tree_absolute_error)) / np.sum(ipw)),
            "available_mean_forecast_horizon_s": float(pd.to_numeric(source.loc[service_available, "forecast_horizon_s"], errors="coerce").mean()),
            "unavailable_mean_forecast_horizon_s": float(pd.to_numeric(source.loc[~service_available, "forecast_horizon_s"], errors="coerce").mean()),
            "available_mean_history_support": float(pd.to_numeric(source.loc[service_available, "observed_sec_per_m_profile_count"], errors="coerce").mean()),
            "unavailable_mean_history_support": float(pd.to_numeric(source.loc[~service_available, "observed_sec_per_m_profile_count"], errors="coerce").mean()),
        })
        stop_result = evaluate_stop_two_part(
            pd.to_numeric(source["stop_time_share"], errors="coerce").to_numpy(float),
            prediction["stop_occurrence_probability"].to_numpy(float),
            prediction["stop_positive_share"].to_numpy(float),
            valid_mask=source["stop_target_valid"].to_numpy(bool),
        )
        stop_rows.append({"split": split, "date": date, **stop_result})
        highway = source["canonical_highway"].astype("string").fillna("unknown").astype(str)
        local = pd.DataFrame({"split": split, "date": date, "subgroup_type": "highway", "subgroup": highway, "valid": valid, "absolute_error": np.abs(models["rc_mstnet_v5_mean"] - truth), "squared_error": np.square(models["rc_mstnet_v5_mean"] - truth)})
        local.loc[~local["valid"], ["absolute_error", "squared_error"]] = np.nan
        grouped = local.groupby(["split", "date", "subgroup_type", "subgroup"], sort=False, observed=True)[["valid", "absolute_error", "squared_error"]].agg({"valid": "sum", "absolute_error": "mean", "squared_error": "mean"}).reset_index()
        grouped = grouped.rename(columns={"valid": "count", "absolute_error": "mae"})
        grouped["rmse"] = np.sqrt(grouped.pop("squared_error"))
        subgroup_parts.append(grouped)
    metrics = pd.DataFrame(metric_rows)
    quantiles = pd.DataFrame(quantile_rows)
    availability = pd.DataFrame(availability_rows)
    bootstraps = pd.DataFrame(bootstrap_rows)
    stops = pd.DataFrame(stop_rows)
    subgroups = pd.concat(subgroup_parts, ignore_index=True)
    report_root = root / "stage2/docs/v5"
    metrics.to_csv(report_root / "service_time_metrics.csv", index=False)
    quantiles.to_csv(report_root / "quantile_calibration.csv", index=False)
    availability.to_csv(report_root / "availability_audit.csv", index=False)
    bootstraps.to_csv(report_root / "deep_paired_error_bootstrap.csv", index=False)
    stops.to_json(report_root / "stop_two_part_metrics.json", orient="records", indent=2)
    subgroups.to_csv(report_root / "subgroup_metrics.csv", index=False)
    validation = metrics[metrics["split"].eq("validation_model")].copy()
    validation["weighted_error"] = validation["mae"] * validation["count"]
    totals = validation.groupby("model", sort=False, observed=True)[["weighted_error", "count"]].sum()
    aggregate_mae = totals["weighted_error"] / totals["count"]
    deep_mae = float(aggregate_mae["rc_mstnet_v5_mean"])
    baseline_mae = float(aggregate_mae["hist_gradient_boosting"])
    validation_bootstrap = bootstraps[bootstraps["split"].eq("validation_model")]
    all_ci_below_zero = all((value if isinstance(value, list) else json.loads(value))[1] < 0 for value in validation_bootstrap["ci95"])
    if deep_mae < baseline_mae and all_ci_below_zero:
        status = "PREDICTIVE_BASELINE_VALIDATED"
    elif deep_mae <= baseline_mae * 1.02:
        status = "BASELINE_COMPETITIVE"
    else:
        status = "BASELINE_NOT_BEATEN"
    summary = {
        "schema_version": "stage2_v5_model_evaluation.1",
        "status": status,
        "validation_deep_mae": deep_mae,
        "validation_strong_baseline_mae": baseline_mae,
        "relative_mae_change": deep_mae / baseline_mae - 1.0,
        "validation_paired_ci_all_below_zero": all_ci_below_zero,
        "new_final_test_consumed": False,
        "legacy_evaluated_in_this_step": False,
    }
    (report_root / "stage2_v5_model_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    summary = evaluate(repo_root=args.repo_root)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
