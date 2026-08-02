"""v5.1 bounded quantile scenarios, calibration, and formal product export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_inherited_payload
from .scenario import _bounded_quantile_inverse, _normal_cdf_approximation
from .scenario_pipeline import _order_truth


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _day_scenarios(
    frame: pd.DataFrame,
    *,
    family: str,
    scenario_count: int,
    seed: int,
    route_batch_size: int,
    correlation_weights: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source = frame.sort_values(["order_id", "route_sequence"], kind="stable", ignore_index=True)
    order = source["order_id"].astype(str).to_numpy()
    route_codes, route_inverse = np.unique(order, return_inverse=True)
    route_start = np.concatenate((np.array([True]), order[1:] != order[:-1]))
    starts = np.flatnonzero(route_start)
    ends = np.concatenate((starts[1:], np.array([len(source)])))
    p50 = source["pace_pred_p50"].to_numpy(float)
    p90 = source["pace_pred_p90"].to_numpy(float)
    p95 = source["pace_pred_p95"].to_numpy(float)
    distance = source["allocated_distance_m"].to_numpy(float)
    route_sequence = source["route_sequence"].to_numpy(np.int64)
    highway = (
        source["canonical_highway_index"].to_numpy(np.int64)
        if "canonical_highway_index" in source
        else np.zeros(len(source), dtype=np.int64)
    )
    time_bin = (
        source["estimated_time_bin_index"].to_numpy(np.int64)
        if "estimated_time_bin_index" in source
        else np.zeros(len(source), dtype=np.int64)
    )
    if np.any(~np.isfinite(np.column_stack((p50, p90, p95, distance)))):
        raise ValueError("scenario inputs must be finite")
    if np.any(p50 <= 0) or np.any(p50 > p90) or np.any(p90 > p95) or np.any(distance < 0):
        raise ValueError("scenario quantiles or distances are invalid")
    _, network_inverse = np.unique(time_bin, return_inverse=True)
    _, region_inverse = np.unique(time_bin, return_inverse=True)
    highway_key = np.rec.fromarrays((highway, time_bin), names=("highway", "time"))
    _, highway_inverse = np.unique(highway_key, return_inverse=True)
    rng = np.random.default_rng(int(seed))
    route_common = rng.standard_normal((len(route_codes), scenario_count))
    network_common = rng.standard_normal((int(network_inverse.max() + 1), scenario_count))
    region_common = rng.standard_normal((int(region_inverse.max() + 1), scenario_count))
    highway_common = rng.standard_normal((int(highway_inverse.max() + 1), scenario_count))
    weights = {
        "network": float(correlation_weights["network_time"]),
        "region": float(correlation_weights["region_time"]),
        "highway": float(correlation_weights["highway_time"]),
        "route": float(correlation_weights["route"]),
    }
    output_parts: list[np.ndarray] = []
    for first_route in range(0, len(starts), route_batch_size):
        last_route = min(first_route + route_batch_size, len(starts))
        left = int(starts[first_route])
        right = int(ends[last_route - 1])
        local_route = route_inverse[left:right]
        residual = rng.standard_normal((right - left, scenario_count))
        if family == "independent_quantile":
            latent = residual
        elif family == "shared_route_quantile":
            latent = np.sqrt(0.35) * route_common[local_route] + np.sqrt(0.65) * residual
        elif family == "residual_block_quantile":
            block_key = local_route.astype(np.int64) * 1_000_000 + route_sequence[left:right] // 5
            _, block_inverse = np.unique(block_key, return_inverse=True)
            block_common = rng.standard_normal((int(block_inverse.max() + 1), scenario_count))
            latent = np.sqrt(0.35) * block_common[block_inverse] + np.sqrt(0.65) * residual
        elif family == "hierarchical_cross_order_quantile":
            total = sum(weights.values())
            if total >= 1.0 or min(weights.values()) < 0:
                raise ValueError("correlation weights must be non-negative and sum below one")
            latent = (
                np.sqrt(weights["network"]) * network_common[network_inverse[left:right]]
                + np.sqrt(weights["region"]) * region_common[region_inverse[left:right]]
                + np.sqrt(weights["highway"]) * highway_common[highway_inverse[left:right]]
                + np.sqrt(weights["route"]) * route_common[local_route]
                + np.sqrt(1.0 - total) * residual
            )
        else:
            raise ValueError(f"unknown scenario family: {family}")
        traversal = _bounded_quantile_inverse(
            _normal_cdf_approximation(latent), p50[left:right], p90[left:right], p95[left:right]
        ) * distance[left:right, None]
        local_inverse = local_route - first_route
        local_route_time = np.zeros((last_route - first_route, scenario_count), dtype=np.float64)
        np.add.at(local_route_time, local_inverse, traversal)
        output_parts.append(local_route_time.astype(np.float32))
    route_time = np.concatenate(output_parts, axis=0)
    provenance = {
        "system_scenario_id": np.arange(scenario_count, dtype=np.int64),
        "network_shock_id": np.arange(scenario_count, dtype=np.int64),
        "route_shock_id": np.arange(scenario_count, dtype=np.int64),
        "correlation_model_id": (
            "hierarchical_quantile_copula.network_region_highway_route_residual.1"
            if family == "hierarchical_cross_order_quantile"
            else f"stage2_v5_1.{family}.1"
        ),
        "cross_order_coherent": family == "hierarchical_cross_order_quantile",
    }
    return route_codes, route_time, provenance


def _sample_crps(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    values = np.sort(np.asarray(samples, dtype=np.float64), axis=1)
    actual = np.asarray(truth, dtype=np.float64)
    count = values.shape[1]
    coefficients = 2.0 * np.arange(1, count + 1) - count - 1.0
    pair_term = (values * coefficients[None, :]).sum(axis=1) / float(count * count)
    return np.abs(values - actual[:, None]).mean(axis=1) - pair_term


def _quality(route_id: np.ndarray, scenarios: np.ndarray, truth: pd.Series) -> dict[str, float | int]:
    actual = truth.reindex(route_id).to_numpy(float)
    valid = np.isfinite(actual)
    values = scenarios[valid].astype(np.float64)
    actual = actual[valid]
    quantiles = np.quantile(values, [0.05, 0.10, 0.50, 0.90, 0.95], axis=1)
    p05, p10, p50, p90, p95 = quantiles
    mean = values.mean(axis=1)
    cvar95 = np.nanmean(np.where(values >= p95[:, None], values, np.nan), axis=1)
    interval_score_80 = (p90 - p10) + 10.0 * (p10 - actual) * (actual < p10) + 10.0 * (actual - p90) * (actual > p90)
    interval_score_90 = (p95 - p05) + 20.0 * (p05 - actual) * (actual < p05) + 20.0 * (actual - p95) * (actual > p95)
    wis = (0.5 * np.abs(p50 - actual) + 0.1 * interval_score_80 + 0.05 * interval_score_90) / 0.65
    maximum = values.max(axis=1)
    return {
        "route_count": int(len(actual)),
        "route_p50_mae_s": float(np.abs(p50 - actual).mean()),
        "route_mean_mae_s": float(np.abs(mean - actual).mean()),
        "route_mean_rmse_s": float(np.sqrt(np.square(mean - actual).mean())),
        "sample_crps_s": float(_sample_crps(values, actual).mean()),
        "weighted_interval_score_s": float(wis.mean()),
        "p90_coverage": float((actual <= p90).mean()),
        "p95_coverage": float((actual <= p95).mean()),
        "p90_p50_width_s": float((p90 - p50).mean()),
        "p95_p50_width_s": float((p95 - p50).mean()),
        "maximum_scenario_s": float(maximum.max()),
        "maximum_route_mean_s": float(mean.max()),
        "maximum_route_cvar95_s": float(cvar95.max()),
        "extreme_scenario_share": float((values > 14400.0).mean()),
    }


def _calibrate(scenarios: np.ndarray, *, scale: float, dispersion: float, offset_s: float) -> np.ndarray:
    scaled = scenarios.astype(np.float64) * scale
    p50 = np.quantile(scaled, 0.5, axis=1, keepdims=True)
    return np.maximum(p50 + dispersion * (scaled - p50) + offset_s, 1.0e-6)


def _fit_calibration(
    route_id: np.ndarray,
    scenarios: np.ndarray,
    truth: pd.Series,
    dispersion_grid: list[float],
) -> dict[str, float]:
    actual = truth.reindex(route_id).to_numpy(float)
    valid = np.isfinite(actual)
    raw_p50 = np.quantile(scenarios, 0.5, axis=1)
    scale = float(np.median(actual[valid] / np.maximum(raw_p50[valid], 1.0e-6)))
    scaled_p50 = raw_p50 * scale
    offset = float(np.median(actual[valid] - scaled_p50[valid]))
    candidates: list[tuple[float, float]] = []
    for dispersion in dispersion_grid:
        calibrated = _calibrate(scenarios, scale=scale, dispersion=float(dispersion), offset_s=offset)
        quality = _quality(route_id, calibrated, truth)
        coverage_error = abs(float(quality["p90_coverage"]) - 0.9) + abs(float(quality["p95_coverage"]) - 0.95)
        score = coverage_error + float(quality["weighted_interval_score_s"]) / max(float(np.median(actual[valid])), 1.0)
        candidates.append((score, float(dispersion)))
    _, dispersion = min(candidates)
    return {"scale": scale, "dispersion": dispersion, "offset_s": offset}


def run_protocol(
    *,
    repo_root: str | Path,
    protocol: str,
    config_path: str | Path,
    prediction_root: str | Path,
    model_root: str | Path,
    output_root: str | Path,
    frozen_family: str | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config = load_inherited_payload(Path(config_path))
    v51 = json.loads((root / "stage2/config/stage2_v5_1.json").read_text(encoding="utf-8"))
    scenario_config = v51["scenario_selection"]
    predictions = Path(prediction_root)
    model_manifest = json.loads((Path(model_root) / "model_manifest.json").read_text(encoding="utf-8"))
    output = Path(output_root)
    split_config = config["split"]
    days = [("validation_model", date) for date in split_config["validation_model_dates"]]
    days += [("calibration", date) for date in split_config["calibration_dates"]]
    days += [("evaluation", date) for date in split_config.get("evaluation_dates", [])]
    days += [("legacy", date) for date in split_config.get("legacy_test_dates", [])]
    validation_scores: list[dict[str, Any]] = []
    candidate_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for date in split_config["validation_model_dates"]:
        frame = pd.read_parquet(predictions / "split=validation_model" / f"date={date}" / "traversal_predictions.parquet")
        truth = _order_truth(root, date)
        median_truth = float(truth.median())
        families = [frozen_family] if frozen_family else scenario_config["families"]
        for family in families:
            route_id, values, provenance = _day_scenarios(
                frame,
                family=family,
                scenario_count=int(scenario_config["scenario_count"]),
                seed=int(scenario_config["seed"]),
                route_batch_size=int(scenario_config["route_batch_size"]),
                correlation_weights=scenario_config["correlation_weights"],
            )
            quality = _quality(route_id, values, truth)
            score = (
                scenario_config["score_weights"]["sample_crps"] * float(quality["sample_crps_s"]) / max(median_truth, 1.0)
                + scenario_config["score_weights"]["calibration_error"]
                * (abs(float(quality["p90_coverage"]) - 0.9) + abs(float(quality["p95_coverage"]) - 0.95))
                + scenario_config["score_weights"]["sharpness"] * float(quality["p95_p50_width_s"]) / max(median_truth, 1.0)
                + scenario_config["score_weights"]["extreme_tail"] * float(quality["extreme_scenario_share"])
            )
            validation_scores.append({"date": date, "family": family, "selection_score": float(score), **quality})
            candidate_cache[(date, family)] = (route_id, values, provenance)
    score_frame = pd.DataFrame(validation_scores)
    mean_score = score_frame.groupby("family", observed=True)["selection_score"].mean()
    diagnostic_best = str(mean_score.idxmin())
    if frozen_family:
        selected = frozen_family
    elif scenario_config.get("require_cross_order_coherence_for_formal", False):
        eligible_families = [
            name for name in mean_score.index if name == "hierarchical_cross_order_quantile"
        ]
        if not eligible_families:
            raise RuntimeError("no cross-order coherent scenario family is available")
        selected = str(mean_score.loc[eligible_families].idxmin())
    else:
        selected = diagnostic_best
    calibration_date = split_config["calibration_dates"][0]
    calibration_frame = pd.read_parquet(predictions / "split=calibration" / f"date={calibration_date}" / "traversal_predictions.parquet")
    calibration_tuple = _day_scenarios(
        calibration_frame,
        family=selected,
        scenario_count=int(scenario_config["scenario_count"]),
        seed=int(scenario_config["seed"]),
        route_batch_size=int(scenario_config["route_batch_size"]),
        correlation_weights=scenario_config["correlation_weights"],
    )
    calibration = _fit_calibration(
        calibration_tuple[0],
        calibration_tuple[1],
        _order_truth(root, calibration_date),
        scenario_config["calibration_dispersion_grid"],
    )
    quality_rows: list[dict[str, Any]] = []
    products: list[dict[str, Any]] = []
    for split, date in days:
        cached = candidate_cache.get((date, selected))
        if cached is None:
            frame = pd.read_parquet(predictions / f"split={split}" / f"date={date}" / "traversal_predictions.parquet")
            cached = _day_scenarios(
                frame,
                family=selected,
                scenario_count=int(scenario_config["scenario_count"]),
                seed=int(scenario_config["seed"]),
                route_batch_size=int(scenario_config["route_batch_size"]),
                correlation_weights=scenario_config["correlation_weights"],
            )
        route_id, raw_values, provenance = cached
        values = _calibrate(raw_values, **calibration)
        quality = _quality(route_id, values, _order_truth(root, date))
        quantiles = np.quantile(values, [0.5, 0.9, 0.95], axis=1)
        availability_source = pd.read_parquet(
            predictions / f"split={split}" / f"date={date}" / "traversal_predictions.parquet",
            columns=["order_id", "allocated_distance_m", "service_time_availability_probability"],
        )
        availability_source["weighted"] = (
            availability_source["allocated_distance_m"]
            * availability_source["service_time_availability_probability"]
        )
        availability_group = availability_source.groupby("order_id", sort=False, observed=True)[["weighted", "allocated_distance_m"]].sum()
        route_availability = (
            availability_group["weighted"] / availability_group["allocated_distance_m"].clip(lower=1.0e-6)
        ).reindex(route_id).to_numpy(float)
        finite = bool(np.isfinite(values).all())
        monotonic = bool(np.all(quantiles[0] <= quantiles[1]) and np.all(quantiles[1] <= quantiles[2]))
        quantile_pass = finite and monotonic and float(quality["extreme_scenario_share"]) <= v51["stability_thresholds"]["maximum_extreme_scenario_share"]
        full_pass = (
            quantile_pass
            and float(quality["maximum_scenario_s"]) <= scenario_config["maximum_route_scenario_s"]
            and float(quality["maximum_route_mean_s"]) <= v51["stability_thresholds"]["maximum_route_mean_s"]
            and float(quality["maximum_route_cvar95_s"]) <= v51["stability_thresholds"]["maximum_route_cvar95_s"]
            and float(quality["route_mean_rmse_s"]) <= v51["stability_thresholds"]["maximum_route_mean_rmse_s"]
        )
        formal = output / "formal_calibrated" / f"split={split}" / f"date={date}"
        formal.mkdir(parents=True, exist_ok=True)
        scenario_path = formal / "route_scenario_samples.npz"
        temporary_scenario = scenario_path.with_name(f".{scenario_path.name}.tmp.npz")
        np.savez_compressed(
            temporary_scenario,
            route_id=route_id,
            route_time_s=values.astype(np.float32),
            system_scenario_id=provenance["system_scenario_id"],
            network_shock_id=provenance["network_shock_id"],
            route_shock_id=provenance["route_shock_id"],
            correlation_model_id=np.array(provenance["correlation_model_id"]),
        )
        os.replace(temporary_scenario, scenario_path)
        product = pd.DataFrame(
            {
                "order_id": route_id,
                "route_service_time_mean_s": values.mean(axis=1),
                "route_service_time_std_s": values.std(axis=1),
                "route_service_time_p50_s": quantiles[0],
                "route_service_time_p90_s": quantiles[1],
                "route_service_time_p95_s": quantiles[2],
                "route_service_availability_probability": route_availability,
                "route_quantile_source": "calibrated_monotonic_quantile_scenarios",
                "route_scenario_model": selected,
                "calibration_identity": f"{protocol}:{calibration_date}",
                "stability_status": "PASS" if quantile_pass else "FAIL",
                "eligible_point_forecast": quantile_pass,
                "eligible_distribution_forecast": quantile_pass,
                "quantile_field_status": "ELIGIBLE" if quantile_pass else "BLOCKED",
                "mean_std_cvar_field_status": "EXPERIMENTAL" if full_pass else "BLOCKED",
            }
        )
        product_path = formal / "route_service_predictions.parquet"
        temporary_product = product_path.with_name(f".{product_path.name}.tmp")
        product.to_parquet(temporary_product, index=False, compression="zstd")
        os.replace(temporary_product, product_path)
        prediction_path = predictions / f"split={split}" / f"date={date}" / "traversal_predictions.parquet"
        route_field_eligibility = dict(v51["field_status"])
        route_field_eligibility["availability_probability"] = "BLOCKED"
        manifest = {
            "schema_version": "stage2_v5_1_quantile_formal_product.1",
            "protocol": protocol,
            "split": split,
            "date": date,
            "prediction_source": "deep_scenario",
            "route_count": int(len(route_id)),
            "scenario_count": int(values.shape[1]),
            "model_id": model_manifest["model_id"],
            "checkpoint_sha256": model_manifest["checkpoint_sha256"],
            "scenario_generator_id": "stage2_v5_1_bounded_quantile_copula.1",
            "scenario_seed": int(scenario_config["seed"]),
            "input_prediction_hash": _sha256(prediction_path),
            "scale": calibration["scale"],
            "dispersion": calibration["dispersion"],
            "offset": calibration["offset_s"],
            "calibration_date": calibration_date,
            "calibration_identity": f"{protocol}:{calibration_date}",
            "field_eligibility": route_field_eligibility,
            "stability_check_id": f"stage2_v5_1_quantile_stability:{protocol}:{date}",
            "stability_check_status": "PASS" if quantile_pass else "FAIL",
            "stability_status": "PASS" if quantile_pass else "FAIL",
            "full_distribution_stability_status": "PASS" if full_pass else "FAIL",
            "eligible_for_stage3": bool(quantile_pass),
            "eligible_for_formal_stage3": False,
            "cross_order_scenario_coherent": bool(provenance["cross_order_coherent"]),
            "correlation_model_id": provenance["correlation_model_id"],
            "files": {
                scenario_path.name: _sha256(scenario_path),
                product_path.name: _sha256(product_path),
            },
            "quality": quality,
        }
        _atomic_json(formal / "manifest.json", manifest)
        point_formal = output / "deep_p50_formal" / f"split={split}" / f"date={date}"
        point_formal.mkdir(parents=True, exist_ok=True)
        point_product = product[
            ["order_id", "route_service_time_p50_s", "route_service_availability_probability"]
        ].copy()
        point_path = point_formal / "route_service_predictions.parquet"
        temporary_point = point_path.with_name(f".{point_path.name}.tmp")
        point_product.to_parquet(temporary_point, index=False, compression="zstd")
        os.replace(temporary_point, point_path)
        point_eligibility = {
            field: (
                "ELIGIBLE"
                if field in {
                    "route_service_time_p50_s",
                    "route_service_availability_probability",
                }
                else "BLOCKED"
            )
            for field in v51["field_status"]
        }
        point_manifest = {
            **manifest,
            "schema_version": "stage2_v5_1_deep_p50_formal_product.1",
            "prediction_source": "deep_p50",
            "scenario_count": 0,
            "scenario_generator_id": "none_point_forecast_only",
            "field_eligibility": point_eligibility,
            "eligible_for_formal_stage3": False,
            "cross_order_scenario_coherent": False,
            "correlation_model_id": "none_point_forecast_only",
            "files": {point_path.name: _sha256(point_path)},
        }
        _atomic_json(point_formal / "manifest.json", point_manifest)
        quality_rows.append({"protocol": protocol, "split": split, "date": date, **quality})
        products.append(manifest)
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    score_frame.to_csv(reports / "scenario_family_selection.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(reports / "route_scoring.csv", index=False)
    selection = {
        "schema_version": "stage2_v5_1_scenario_selection.1",
        "protocol": protocol,
        "selected_family": selected,
        "diagnostic_best_family_before_cross_order_gate": diagnostic_best,
        "selection_rule": "cross_order_coherence_hard_gate_then_frozen_proper_score",
        "family_frozen_from_development": bool(frozen_family),
        "selection_dates": split_config["validation_model_dates"],
        "calibration_date": calibration_date,
        "calibration": calibration,
        "cross_order_scenario_selected": selected == "hierarchical_cross_order_quantile",
        "products": products,
    }
    _atomic_json(reports / "scenario_selection.json", selection)
    return selection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frozen-family")
    args = parser.parse_args()
    result = run_protocol(
        repo_root=args.repo_root,
        protocol=args.protocol,
        config_path=args.config,
        prediction_root=args.prediction_root,
        model_root=args.model_root,
        output_root=args.output_root,
        frozen_family=args.frozen_family,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
