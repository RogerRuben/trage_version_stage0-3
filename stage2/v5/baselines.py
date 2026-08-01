"""Train-only fitting and same-row service-time baseline evaluation."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from stage2.v4.models.baselines import _feature_candidates

from .config import Stage2V5Config, load_config
from .data import load_v5_day


def continuous_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    y = np.asarray(truth, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    valid = np.isfinite(y) & np.isfinite(pred)
    if not valid.any():
        return {"count": 0, "mae": None, "rmse": None, "pearson": None, "spearman": None}
    y = y[valid]
    pred = pred[valid]
    error = pred - y
    y_series = pd.Series(y)
    pred_series = pd.Series(pred)
    varying = y_series.nunique() > 1 and pred_series.nunique() > 1
    return {
        "count": int(len(y)),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "pearson": float(y_series.corr(pred_series)) if varying else None,
        "spearman": float(y_series.rank().corr(pred_series.rank())) if varying else None,
    }


def _stable_cap(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    hashed = pd.util.hash_pandas_object(frame[["order_id", "traversal_id"]], index=False, hash_key=f"{seed:016d}"[-16:]).to_numpy(dtype=np.uint64)
    selected = np.argpartition(hashed, limit - 1)[:limit]
    return frame.iloc[selected].copy()


def _feature_matrix(frame: pd.DataFrame, features: tuple[str, ...], highway: dict[str, int]) -> np.ndarray:
    numeric = frame.loc[:, features].copy()
    for column in features:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    highway_code = frame["canonical_highway"].astype("string").fillna("unknown").astype(str).map(highway).fillna(-1).to_numpy(np.float32)
    return np.column_stack((numeric.to_numpy(dtype=np.float32, na_value=np.nan), highway_code))


def fit_service_baselines(config: Stage2V5Config, *, repo_root: str | Path = ".") -> dict[str, Any]:
    started = time.perf_counter()
    dates = tuple(config.section("split")["train_dates"])
    baseline = config.section("baseline")
    maximum = int(baseline["tree_max_train_rows"])
    per_day = int(math.ceil(maximum / len(dates)))
    seed = int(config.section("runtime")["random_seed"])
    samples: list[pd.DataFrame] = []
    group_parts: list[pd.DataFrame] = []
    global_sum = 0.0
    global_count = 0
    highways: set[str] = set()
    candidate_features = _feature_candidates()
    for offset, date in enumerate(dates):
        frame = load_v5_day(date, split="train", repo_root=repo_root)
        usable = frame["pace_target_valid"].to_numpy(bool) & np.isfinite(pd.to_numeric(frame["pace_sec_per_m"], errors="coerce").to_numpy(float))
        supervised = frame.loc[usable].copy()
        pace = supervised["pace_sec_per_m"].to_numpy(float)
        global_sum += float(pace.sum())
        global_count += len(pace)
        grouped = supervised.assign(_pace=pace).groupby(["canonical_highway", "estimated_time_bin"], sort=False, observed=True, dropna=False)["_pace"].agg(["sum", "count"]).reset_index()
        group_parts.append(grouped)
        highways.update(frame["canonical_highway"].astype("string").dropna().astype(str).unique())
        samples.append(_stable_cap(supervised, per_day, seed + offset))
    training = _stable_cap(pd.concat(samples, ignore_index=True), maximum, seed)
    feature_columns = tuple(column for column in candidate_features if column in training.columns)
    highway_vocabulary = {value: index for index, value in enumerate(sorted(highways))}
    tree = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=int(baseline["tree_max_iter"]),
        learning_rate=float(baseline["tree_learning_rate"]),
        max_leaf_nodes=int(baseline["tree_max_leaf_nodes"]),
        min_samples_leaf=int(baseline["tree_min_samples_leaf"]),
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
    )
    tree.fit(_feature_matrix(training, feature_columns, highway_vocabulary), training["pace_sec_per_m"].to_numpy(float))
    combined = pd.concat(group_parts, ignore_index=True).groupby(["canonical_highway", "estimated_time_bin"], sort=False, observed=True, dropna=False)[["sum", "count"]].sum().reset_index()
    combined["mean"] = combined["sum"] / combined["count"]
    return {
        "schema_version": "stage2_v5_service_baselines.1",
        "fit_dates": list(dates),
        "fit_label_count": global_count,
        "tree_fit_row_count": len(training),
        "global_mean": global_sum / global_count,
        "highway_time": combined,
        "feature_columns": feature_columns,
        "highway_vocabulary": highway_vocabulary,
        "tree_model": tree,
        "runtime_s": time.perf_counter() - started,
    }


def _predict(frame: pd.DataFrame, bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    global_mean = float(bundle["global_mean"])
    lookup = bundle["highway_time"]
    keyed = frame[["canonical_highway", "estimated_time_bin"]].merge(lookup[["canonical_highway", "estimated_time_bin", "mean"]], on=["canonical_highway", "estimated_time_bin"], how="left", sort=False)
    highway_time = keyed["mean"].fillna(global_mean).to_numpy(float)
    strict = pd.to_numeric(frame["observed_sec_per_m_profile_mean"], errors="coerce").to_numpy(float)
    strict = np.where(np.isfinite(strict), strict, highway_time)
    recent15 = pd.to_numeric(frame["edge_15m_observed_sec_per_m_mean"], errors="coerce").to_numpy(float)
    recent60 = pd.to_numeric(frame["edge_60m_observed_sec_per_m_mean"], errors="coerce").to_numpy(float)
    edge_rolling = np.where(np.isfinite(recent15), recent15, np.where(np.isfinite(recent60), recent60, strict))
    distance = pd.to_numeric(frame["route_part_length_m"], errors="coerce").to_numpy(float)
    estimated_time = pd.to_numeric(frame["estimated_travel_time_s"], errors="coerce").to_numpy(float)
    v4_static = np.divide(estimated_time, distance, out=np.full(len(frame), np.nan), where=np.isfinite(distance) & (distance > 0))
    tree = bundle["tree_model"].predict(_feature_matrix(frame, tuple(bundle["feature_columns"]), bundle["highway_vocabulary"]))
    return {
        "global_mean": np.full(len(frame), global_mean),
        "highway_time_bin_mean": highway_time,
        "strict_historical_profile": strict,
        "edge_rolling_mean": edge_rolling,
        "v4_static_entry_time": v4_static,
        "hist_gradient_boosting": np.maximum(tree, np.finfo(float).tiny),
    }


def _paired_bootstrap(frame: pd.DataFrame, left: str, right: str, *, seed: int, replicates: int = 500) -> dict[str, Any]:
    valid = frame[["order_id", "truth", left, right]].dropna().copy()
    valid["difference"] = np.abs(valid[left] - valid["truth"]) - np.abs(valid[right] - valid["truth"])
    order_stats = valid.groupby("order_id", sort=False, observed=True)["difference"].agg(["sum", "count"])
    sums = order_stats["sum"].to_numpy(float)
    counts = order_stats["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for index in range(replicates):
        sample = rng.integers(0, len(sums), size=len(sums))
        draws[index] = sums[sample].sum() / counts[sample].sum()
    return {
        "left_minus_right_absolute_error": float(valid["difference"].mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "replicates": replicates,
        "order_count": len(order_stats),
    }


def evaluate_service_baselines(bundle: dict[str, Any], config: Stage2V5Config, *, repo_root: str | Path = ".") -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    split = config.section("split")
    dates = [("validation_model", date) for date in split["validation_model_dates"]]
    dates += [("calibration", date) for date in split["calibration_dates"]]
    dates += [("legacy", date) for date in split["legacy_test_dates"]]
    metric_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    seed = int(config.section("runtime")["random_seed"])
    for split_name, date in dates:
        input_split = "test" if split_name == "legacy" else split_name
        frame = load_v5_day(date, split=input_split, repo_root=repo_root)
        predictions = _predict(frame, bundle)
        truth = pd.to_numeric(frame["pace_sec_per_m"], errors="coerce").to_numpy(float)
        valid = frame["pace_target_valid"].to_numpy(bool)
        compare = pd.DataFrame({"order_id": frame["order_id"].astype(str), "truth": np.where(valid, truth, np.nan), **predictions})
        day_metrics: dict[str, dict[str, Any]] = {}
        for model, prediction in predictions.items():
            metrics = continuous_metrics(np.where(valid, truth, np.nan), prediction)
            day_metrics[model] = metrics
            metric_rows.append({"split": split_name, "date": date, "target": "pace_sec_per_m", "model": model, **metrics})
        candidates = {name: values for name, values in day_metrics.items() if name != "hist_gradient_boosting" and values["mae"] is not None}
        best = min(candidates, key=lambda name: candidates[name]["mae"])
        paired = _paired_bootstrap(compare, "hist_gradient_boosting", best, seed=seed)
        bootstrap_rows.append({"split": split_name, "date": date, "left_model": "hist_gradient_boosting", "right_model": best, **paired})
    metrics_frame = pd.DataFrame(metric_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    validation = metrics_frame[metrics_frame["split"].eq("validation_model")].copy()
    validation["_weighted_error"] = validation["mae"] * validation["count"]
    totals = validation.groupby("model", sort=False, observed=True)[["_weighted_error", "count"]].sum()
    by_model = totals["_weighted_error"] / totals["count"]
    best_model = str(by_model.idxmin())
    summary = {
        "schema_version": "stage2_v5_baseline_evaluation.1",
        "evaluation_rule": "all models use identical direct-pace rows per date",
        "best_validation_model": best_model,
        "best_validation_mae": float(by_model.min()),
        "legacy_is_untouched_test": False,
        "new_final_test_consumed": False,
    }
    return metrics_frame, bootstrap_frame, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage2/config/stage2_v5.json")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-root", default="stage2/output_v5/baselines")
    parser.add_argument("--report-root", default="stage2/docs/v5")
    args = parser.parse_args()
    config = load_config(args.config)
    bundle = fit_service_baselines(config, repo_root=args.repo_root)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output / "service_time_baselines.joblib")
    metrics, bootstrap, summary = evaluate_service_baselines(bundle, config, repo_root=args.repo_root)
    report = Path(args.report_root)
    metrics.to_csv(report / "baseline_comparison.csv", index=False)
    bootstrap.to_csv(report / "paired_error_bootstrap.csv", index=False)
    (report / "stage2_v5_baseline_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
