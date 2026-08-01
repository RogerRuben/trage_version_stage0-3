"""One-shot final evaluation on the preregistered 20161028-30 dates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .baselines import _paired_bootstrap, _predict, continuous_metrics
from .config import load_config
from .data import load_v5_day
from .scenario_pipeline import _calibrate_values, _metrics, _route_product, generate_day


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_splits(input_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in input_root.glob("split=*/date=*/bucket=*/manifest.json"):
        date = path.parents[1].name.split("=", 1)[1]
        split = path.parents[2].name.split("=", 1)[1]
        previous = result.setdefault(date, split)
        if previous != split:
            raise ValueError(f"date {date} spans multiple Stage 0 splits")
    return result


def _order_truth(input_root: Path, split: str, date: str) -> pd.Series:
    parts = [
        pd.read_parquet(path, columns=["order_id", "departure_time", "arrival_time"])
        for path in sorted((input_root / f"split={split}" / f"date={date}").glob("bucket=*/order_base.parquet"))
    ]
    if not parts:
        raise ValueError(f"missing final order truth for {date}")
    frame = pd.concat(parts, ignore_index=True).drop_duplicates("order_id")
    duration = pd.to_numeric(frame["arrival_time"], errors="coerce") - pd.to_numeric(frame["departure_time"], errors="coerce")
    return pd.Series(duration.to_numpy(float), index=frame["order_id"].astype(str), name="actual_route_time_s")


def run(*, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config = load_config(root / "stage2/config/stage2_v5.json")
    dates = list(config.section("split")["final_test_dates"])
    final_root = root / "stage2/output_v5/final_upstream"
    input_root = final_root / "stage1/input_v1"
    label_root = final_root / "stage1/output_v3"
    feature_root = final_root / "stage2/route_conditioned_dataset/revealed_route_proxy"
    prediction_root = final_root / "stage2/predictions"
    source_splits = _source_splits(input_root)
    baseline_bundle = joblib.load(root / "stage2/output_v5/baselines/service_time_baselines.joblib")
    selection = json.loads((root / "stage2/docs/v5/stage2_v5_scenario_selection.json").read_text(encoding="utf-8"))
    model_manifest = json.loads((root / "stage2/output_v5/deep_model/model_manifest.json").read_text(encoding="utf-8"))
    scenario_config = config.section("scenario")
    admission = config.section("admission")
    metric_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    combined_parts: list[pd.DataFrame] = []
    seed = int(config.section("runtime")["random_seed"])

    for date in dates:
        prediction_path = prediction_root / "split=final_test" / f"date={date}" / "traversal_predictions.parquet"
        prediction = pd.read_parquet(prediction_path)
        source = load_v5_day(
            date,
            split=source_splits[date],
            repo_root=root,
            stage1_input_root=input_root,
            stage1_output_root=label_root,
            route_feature_root=feature_root,
        ).sort_values(["order_id", "route_sequence"], kind="stable", ignore_index=True)
        if not np.array_equal(
            prediction[["order_id", "traversal_id"]].astype(str).to_numpy(),
            source[["order_id", "traversal_id"]].astype(str).to_numpy(),
        ):
            raise ValueError(f"final prediction/source identity mismatch on {date}")
        baseline = _predict(source, baseline_bundle)["hist_gradient_boosting"]
        deep = prediction["pace_pred_mean"].to_numpy(float)
        truth = pd.to_numeric(source["pace_sec_per_m"], errors="coerce").to_numpy(float)
        valid = source["pace_target_valid"].to_numpy(bool) & np.isfinite(truth)
        for model, values in (("hist_gradient_boosting", baseline), ("rc_mstnet_v5", deep)):
            metric_rows.append({"split": "final_test", "date": date, "target": "pace_sec_per_m", "model": model, **continuous_metrics(np.where(valid, truth, np.nan), values)})
        comparison = pd.DataFrame({
            "order_id": date + ":" + source["order_id"].astype(str),
            "truth": np.where(valid, truth, np.nan),
            "rc_mstnet_v5": deep,
            "hist_gradient_boosting": baseline,
        })
        bootstrap_rows.append({
            "split": "final_test",
            "date": date,
            "left_model": "rc_mstnet_v5",
            "right_model": "hist_gradient_boosting",
            **_paired_bootstrap(comparison, "rc_mstnet_v5", "hist_gradient_boosting", seed=seed),
        })
        combined_parts.append(comparison)
        for quantile in (0.5, 0.9, 0.95):
            values = prediction[f"pace_pred_p{int(quantile * 100)}"].to_numpy(float)[valid]
            actual = truth[valid]
            error = actual - values
            quantile_rows.append({
                "split": "final_test",
                "date": date,
                "quantile": quantile,
                "empirical_coverage": float((actual <= values).mean()),
                "coverage_error": float((actual <= values).mean() - quantile),
                "pinball_loss": float(np.maximum(quantile * error, (quantile - 1.0) * error).mean()),
                "count": len(actual),
            })

        route_id, raw_scenarios = generate_day(
            prediction,
            model=str(selection["selected_model"]),
            scenario_count=int(selection["scenario_count"]),
            seed=int(selection["scenario_seed"]),
            rho=float(scenario_config["shared_route_rho"]),
            block_size=int(scenario_config["residual_block_size"]),
        )
        calibrated = _calibrate_values(
            raw_scenarios,
            scale=float(selection["route_time_scale"]),
            dispersion=float(selection["route_dispersion_multiplier"]),
            offset_s=float(selection["route_offset_s"]),
        ).astype(np.float32)
        truth_by_order = _order_truth(input_root, source_splits[date], date)
        scenario_rows.append(_metrics(
            route_id,
            calibrated,
            truth_by_order,
            model=f"{selection['selected_model']}_frozen_calibrated",
            split="final_test",
            date=date,
        ))
        formal_root = final_root / "stage2/route_scenarios" / "split=final_test" / f"date={date}"
        formal_root.mkdir(parents=True, exist_ok=True)
        sample_path = formal_root / "route_scenario_samples.npz"
        temporary = sample_path.with_name(f".{sample_path.name}.tmp.npz")
        np.savez_compressed(temporary, route_id=route_id, route_time_s=calibrated)
        os.replace(temporary, sample_path)
        product = _route_product(
            route_id,
            calibrated,
            thresholds_s=[float(value) for value in scenario_config["evaluation_timeout_thresholds_s"]],
            model_id=model_manifest["model_id"],
            seed=int(selection["scenario_seed"]),
            generator_id=str(selection["generator_id"]),
            input_hash=_sha256(prediction_path),
        )
        product_path = formal_root / "route_service_predictions.parquet"
        temporary_product = product_path.with_name(f".{product_path.name}.tmp")
        product.to_parquet(temporary_product, index=False, compression="zstd")
        os.replace(temporary_product, product_path)

    metrics = pd.DataFrame(metric_rows)
    quantiles = pd.DataFrame(quantile_rows)
    bootstraps = pd.DataFrame(bootstrap_rows)
    scenarios = pd.DataFrame(scenario_rows)
    combined = pd.concat(combined_parts, ignore_index=True)
    aggregate_bootstrap = _paired_bootstrap(combined, "rc_mstnet_v5", "hist_gradient_boosting", seed=seed)
    totals = metrics.assign(weighted=lambda value: value["mae"] * value["count"]).groupby("model", observed=True)[["weighted", "count"]].sum()
    aggregate_mae = totals["weighted"] / totals["count"]
    daily = metrics.pivot(index="date", columns="model", values="mae")
    daily_wins = int((daily["rc_mstnet_v5"] < daily["hist_gradient_boosting"]).sum())
    route_count = scenarios["route_count"].to_numpy(float)
    pooled_p90 = float(np.average(scenarios["p90_coverage"], weights=route_count))
    pooled_p95 = float(np.average(scenarios["p95_coverage"], weights=route_count))
    scenario_acceptable = (
        float(admission["route_p90_coverage_minimum"]) <= pooled_p90 <= float(admission["route_p90_coverage_maximum"])
        and float(admission["route_p95_coverage_minimum"]) <= pooled_p95 <= float(admission["route_p95_coverage_maximum"])
    )
    report_root = root / "stage2/docs/v5"
    metrics.to_csv(report_root / "final_service_time_metrics.csv", index=False)
    quantiles.to_csv(report_root / "final_quantile_calibration.csv", index=False)
    bootstraps.to_csv(report_root / "final_paired_error_bootstrap.csv", index=False)
    scenarios.to_csv(report_root / "final_scenario_coverage.csv", index=False)
    result = {
        "schema_version": "stage2_v5_final_test_results.1",
        "protocol": "one_shot_preregistered",
        "dates": dates,
        "day_count": len(dates),
        "selection_model_id": model_manifest["model_id"],
        "scenario_generator_id": selection["generator_id"],
        "post_test_tuning_count": 0,
        "deep_aggregate_mae": float(aggregate_mae["rc_mstnet_v5"]),
        "strong_baseline_aggregate_mae": float(aggregate_mae["hist_gradient_boosting"]),
        "aggregate_relative_mae_change": float(aggregate_mae["rc_mstnet_v5"] / aggregate_mae["hist_gradient_boosting"] - 1.0),
        "aggregate_mae_better_than_strong_baseline": bool(aggregate_mae["rc_mstnet_v5"] < aggregate_mae["hist_gradient_boosting"]),
        "daily_mae_wins": daily_wins,
        "paired_bootstrap_ci95": aggregate_bootstrap["ci95"],
        "paired_bootstrap_ci_below_zero": bool(aggregate_bootstrap["ci95"][1] < 0),
        "maximum_pace_quantile_coverage_error": float(quantiles["coverage_error"].abs().max()),
        "route_scenario_coverage": {"p90": pooled_p90, "p95": pooled_p95},
        "route_scenario_coverage_acceptable": bool(scenario_acceptable),
        "new_final_test_consumed": True,
    }
    (report_root / "stage2_v5_final_test_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    print(json.dumps(run(repo_root=args.repo_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
