"""Batched correlated route scenarios, validation selection, and calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .scenario import generate_route_scenarios


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _route_product(
    route_id: np.ndarray,
    scenarios: np.ndarray,
    *,
    thresholds_s: list[float],
    model_id: str,
    seed: int,
    generator_id: str,
    input_hash: str,
) -> pd.DataFrame:
    values = np.asarray(scenarios, dtype=np.float64)
    quantiles = np.quantile(values, [0.5, 0.9, 0.95], axis=1)
    cvar90 = np.where(values >= quantiles[1, :, None], values, np.nan)
    cvar95 = np.where(values >= quantiles[2, :, None], values, np.nan)
    result = pd.DataFrame(
        {
            "order_id": route_id.astype(str),
            "route_service_time_mean_s": values.mean(axis=1),
            "route_service_time_std_s": values.std(axis=1),
            "route_service_time_p50_s": quantiles[0],
            "route_service_time_p90_s": quantiles[1],
            "route_service_time_p95_s": quantiles[2],
            "route_service_time_cvar90_s": np.nanmean(cvar90, axis=1),
            "route_service_time_cvar95_s": np.nanmean(cvar95, axis=1),
            "scenario_count": values.shape[1],
            "scenario_model_id": model_id,
            "scenario_generator_id": generator_id,
            "scenario_seed": seed,
            "scenario_input_hash": input_hash,
        }
    )
    for threshold in thresholds_s:
        result[f"timeout_probability_threshold_{int(threshold)}s"] = (
            values > float(threshold)
        ).mean(axis=1)
    return result


def _order_truth(root: Path, date: str) -> pd.Series:
    day = root / "stage1/input_v1/split=validation" / f"date={date}"
    parts: list[pd.DataFrame] = []
    for path in sorted(day.glob("bucket=*/order_base.parquet")):
        parts.append(pd.read_parquet(path, columns=["order_id", "departure_time", "arrival_time"]))
    frame = pd.concat(parts, ignore_index=True).drop_duplicates("order_id")
    duration = pd.to_numeric(frame["arrival_time"], errors="coerce") - pd.to_numeric(frame["departure_time"], errors="coerce")
    return pd.Series(duration.to_numpy(float), index=frame["order_id"].astype(str), name="actual_route_time_s")


def generate_day(
    prediction: pd.DataFrame,
    *,
    model: str,
    scenario_count: int,
    seed: int,
    rho: float,
    block_size: int,
    route_batch_size: int = 250,
) -> tuple[np.ndarray, np.ndarray]:
    frame = prediction.sort_values(["order_id", "route_sequence"], kind="stable", ignore_index=True)
    order = frame["order_id"].astype(str).to_numpy()
    route_start = np.concatenate((np.array([True]), order[1:] != order[:-1]))
    starts = np.flatnonzero(route_start)
    ends = np.concatenate((starts[1:], np.array([len(frame)])))
    route_parts: list[np.ndarray] = []
    scenario_parts: list[np.ndarray] = []
    for first_route in range(0, len(starts), route_batch_size):
        last_route = min(first_route + route_batch_size, len(starts))
        left = starts[first_route]
        right = ends[last_route - 1]
        local = frame.iloc[left:right]
        result = generate_route_scenarios(
            local["order_id"].astype(str).to_numpy(),
            local["pace_log_mu"].to_numpy(float),
            local["pace_log_scale"].to_numpy(float),
            local["allocated_distance_m"].to_numpy(float),
            scenario_count=scenario_count,
            seed=seed + first_route,
            model=model,
            shared_route_rho=rho,
            residual_block_id=(local["route_sequence"].to_numpy(np.int64) // block_size).astype(str) if model == "residual_block" else None,
        )
        route_parts.append(result.route_codes.astype(str))
        scenario_parts.append(result.route_time_s.astype(np.float32))
    return np.concatenate(route_parts), np.concatenate(scenario_parts, axis=0)


def _calibrate_values(scenarios: np.ndarray, *, scale: float, dispersion: float = 1.0, offset_s: float = 0.0) -> np.ndarray:
    scaled = scenarios.astype(np.float64) * float(scale)
    mean = scaled.mean(axis=1, keepdims=True)
    return np.maximum(mean + float(dispersion) * (scaled - mean) + float(offset_s), 1e-6)


def _metrics(route_id: np.ndarray, scenarios: np.ndarray, truth: pd.Series, *, model: str, split: str, date: str, scale: float = 1.0, dispersion: float = 1.0, offset_s: float = 0.0) -> dict[str, Any]:
    values = _calibrate_values(scenarios, scale=scale, dispersion=dispersion, offset_s=offset_s)
    actual = truth.reindex(route_id).to_numpy(float)
    valid = np.isfinite(actual)
    values = values[valid]
    actual = actual[valid]
    q = np.quantile(values, [0.5, 0.9, 0.95], axis=1)
    mean = values.mean(axis=1)
    error = mean - actual
    return {
        "split": split, "date": date, "scenario_model": model,
        "scenario_count": values.shape[1], "route_count": len(actual), "scale": float(scale),
        "dispersion_multiplier": float(dispersion), "offset_s": float(offset_s),
        "mean_mae_s": float(np.abs(error).mean()), "mean_rmse_s": float(np.sqrt(np.square(error).mean())),
        "p50_coverage": float((actual <= q[0]).mean()), "p90_coverage": float((actual <= q[1]).mean()),
        "p95_coverage": float((actual <= q[2]).mean()),
        "p50_mae_s": float(np.abs(q[0] - actual).mean()),
        "average_interval_width_p90_minus_p50_s": float((q[1] - q[0]).mean()),
        "average_interval_width_p95_minus_p50_s": float((q[2] - q[0]).mean()),
    }


def run(*, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config = load_config(root / "stage2/config/stage2_v5.json")
    scenario_config = config.section("scenario")
    models = tuple(scenario_config["models"])
    scenario_count = int(scenario_config["count"])
    output = root / "stage2/output_v5/route_scenarios"
    report_rows: list[dict[str, Any]] = []
    generated: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    dates = [("validation_model", date) for date in config.section("split")["validation_model_dates"]]
    dates += [("calibration", date) for date in config.section("split")["calibration_dates"]]
    for split, date in dates:
        prediction = pd.read_parquet(root / "stage2/output_v5/predictions" / f"split={split}" / f"date={date}" / "traversal_predictions.parquet")
        truth = _order_truth(root, date)
        for model_index, model in enumerate(models):
            path = output / f"split={split}" / f"date={date}" / f"model={model}" / "route_scenarios.npz"
            if path.is_file():
                with np.load(path, allow_pickle=False) as existing:
                    route_id = existing["route_id"].astype(str)
                    scenarios = existing["route_time_s"].astype(np.float32)
            else:
                route_id, scenarios = generate_day(
                    prediction,
                    model=model,
                    scenario_count=scenario_count,
                    seed=int(scenario_config["seed"]) + model_index * 1_000_000,
                    rho=float(scenario_config["shared_route_rho"]),
                    block_size=int(scenario_config["residual_block_size"]),
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(f".{path.name}.tmp.npz")
                np.savez_compressed(temporary, route_id=route_id, route_time_s=scenarios)
                os.replace(temporary, path)
            generated[(date, model)] = (route_id, scenarios)
            report_rows.append(_metrics(route_id, scenarios, truth, model=model, split=split, date=date))
    metrics = pd.DataFrame(report_rows)
    validation = metrics[metrics["split"].eq("validation_model")].copy()
    validation["selection_score"] = (
        np.abs(validation["p90_coverage"] - 0.9)
        + np.abs(validation["p95_coverage"] - 0.95)
        + validation["mean_mae_s"] / validation["mean_mae_s"].median()
    )
    score = validation.groupby("scenario_model", sort=False, observed=True)["selection_score"].mean()
    selected = str(score.idxmin())
    calibration_date = config.section("split")["calibration_dates"][0]
    calibration_routes, calibration_scenarios = generated[(calibration_date, selected)]
    calibration_truth = _order_truth(root, calibration_date).reindex(calibration_routes).to_numpy(float)
    calibration_mean = calibration_scenarios.mean(axis=1)
    valid = np.isfinite(calibration_truth) & np.isfinite(calibration_mean) & (calibration_mean > 0)
    route_scale = float(np.median(calibration_truth[valid] / calibration_mean[valid]))
    scaled = calibration_scenarios.astype(np.float64) * route_scale
    scaled_mean = scaled.mean(axis=1, keepdims=True)
    scaled_quantiles = np.quantile(scaled, [0.5, 0.9, 0.95], axis=1)
    mean_vector = scaled_mean[:, 0]
    dispersion_candidates = np.arange(0.5, 6.01, 0.1)
    calibration_scores = np.empty(len(dispersion_candidates))
    offsets = np.empty(len(dispersion_candidates))
    for index, dispersion in enumerate(dispersion_candidates):
        calibrated_quantiles = mean_vector + dispersion * (scaled_quantiles - mean_vector)
        p50 = calibrated_quantiles[0]
        offset = float(np.median(calibration_truth[valid] - p50[valid]))
        quantiles = np.maximum(calibrated_quantiles + offset, 1e-6)
        coverage = np.asarray([(calibration_truth[valid] <= quantiles[row, valid]).mean() for row in range(3)])
        calibration_scores[index] = np.abs(coverage - np.asarray([0.5, 0.9, 0.95])).sum()
        offsets[index] = offset
    chosen_index = int(np.argmin(calibration_scores))
    dispersion_multiplier = float(dispersion_candidates[chosen_index])
    offset_s = float(offsets[chosen_index])
    calibrated = _metrics(calibration_routes, calibration_scenarios, _order_truth(root, calibration_date), model=selected, split="calibration", date=calibration_date, scale=route_scale, dispersion=dispersion_multiplier, offset_s=offset_s)
    calibrated["scenario_model"] = f"{selected}_calibrated"
    metrics = pd.concat((metrics, pd.DataFrame([calibrated])), ignore_index=True)
    report_root = root / "stage2/docs/v5"
    metrics.to_csv(report_root / "scenario_coverage.csv", index=False)
    selection = {
        "schema_version": "stage2_v5_scenario_selection.1",
        "selected_model": selected,
        "route_time_scale": route_scale,
        "route_dispersion_multiplier": dispersion_multiplier,
        "route_offset_s": offset_s,
        "calibration_status": "FIT_ON_CALIBRATION_DATE_NOT_AN_UNBIASED_EVALUATION",
        "selection_dates": config.section("split")["validation_model_dates"],
        "calibration_dates": config.section("split")["calibration_dates"],
        "scenario_count": scenario_count,
        "scenario_seed": int(scenario_config["seed"]),
        "generator_id": scenario_config["generator_id"],
        "new_final_test_consumed": False,
        "scores": {name: float(value) for name, value in score.items()},
    }
    (report_root / "stage2_v5_scenario_selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raw_path = output / "split=calibration" / f"date={calibration_date}" / f"model={selected}" / "route_scenarios.npz"
    calibrated_values = _calibrate_values(
        calibration_scenarios,
        scale=route_scale,
        dispersion=dispersion_multiplier,
        offset_s=offset_s,
    ).astype(np.float32)
    formal_root = output / "formal_calibrated" / "split=calibration" / f"date={calibration_date}"
    formal_root.mkdir(parents=True, exist_ok=True)
    scenario_path = formal_root / "route_scenario_samples.npz"
    temporary_scenario = scenario_path.with_name(f".{scenario_path.name}.tmp.npz")
    np.savez_compressed(temporary_scenario, route_id=calibration_routes, route_time_s=calibrated_values)
    os.replace(temporary_scenario, scenario_path)
    model_manifest = json.loads((root / "stage2/output_v5/deep_model/model_manifest.json").read_text(encoding="utf-8"))
    product = _route_product(
        calibration_routes,
        calibrated_values,
        thresholds_s=[float(value) for value in scenario_config["evaluation_timeout_thresholds_s"]],
        model_id=model_manifest["model_id"],
        seed=int(scenario_config["seed"]),
        generator_id=scenario_config["generator_id"],
        input_hash=_sha256(raw_path),
    )
    product_path = formal_root / "route_service_predictions.parquet"
    temporary_product = product_path.with_name(f".{product_path.name}.tmp")
    product.to_parquet(temporary_product, index=False, compression="zstd")
    os.replace(temporary_product, product_path)
    return selection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    report = run(repo_root=args.repo_root)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
