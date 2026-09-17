"""Formal tree scenario baseline with calibration residual bootstrap."""

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

from .baselines import _predict as predict_baselines
from .config import load_inherited_payload
from .data import load_v5_day
from .scenario_pipeline import _order_truth
from .scenario_v5_1 import _atomic_json, _calibrate, _fit_calibration, _quality, _sha256


def _bundle_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_day_scenarios(
    frame: pd.DataFrame,
    tree_pace: np.ndarray,
    calibration_residual: np.ndarray,
    *,
    scenario_count: int,
    seed: int,
    route_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    working = frame[["order_id", "route_sequence", "allocated_distance_m"]].copy()
    working["tree_pace"] = np.asarray(tree_pace, dtype=np.float64)
    working = working.sort_values(["order_id", "route_sequence"], kind="stable", ignore_index=True)
    order = working["order_id"].astype(str).to_numpy()
    routes, inverse = np.unique(order, return_inverse=True)
    start_mask = np.concatenate((np.array([True]), order[1:] != order[:-1]))
    starts = np.flatnonzero(start_mask)
    ends = np.concatenate((starts[1:], np.array([len(working)])))
    pace = working["tree_pace"].to_numpy(float)
    distance = working["allocated_distance_m"].to_numpy(float)
    residual = np.asarray(calibration_residual, dtype=np.float64)
    residual = residual[np.isfinite(residual)]
    if not len(residual):
        raise ValueError("tree scenario calibration has no finite residuals")
    rng = np.random.default_rng(int(seed))
    system = rng.choice(residual, size=scenario_count, replace=True)
    route_residual = rng.choice(residual, size=(len(routes), scenario_count), replace=True)
    route_parts: list[np.ndarray] = []
    for first_route in range(0, len(starts), route_batch_size):
        last_route = min(first_route + route_batch_size, len(starts))
        left = int(starts[first_route])
        right = int(ends[last_route - 1])
        local_inverse = inverse[left:right]
        individual = rng.choice(residual, size=(right - left, scenario_count), replace=True)
        scenario_pace = np.maximum(
            pace[left:right, None]
            + np.sqrt(0.20) * system[None, :]
            + np.sqrt(0.25) * route_residual[local_inverse]
            + np.sqrt(0.55) * individual,
            1.0e-4,
        )
        traversal = scenario_pace * distance[left:right, None]
        local_route = local_inverse - first_route
        route_time = np.zeros((last_route - first_route, scenario_count), dtype=np.float64)
        np.add.at(route_time, local_route, traversal)
        route_parts.append(route_time.astype(np.float32))
    return routes, np.concatenate(route_parts, axis=0)


def run_protocol(
    *,
    repo_root: str | Path,
    protocol: str,
    config_path: str | Path,
    baseline_bundle_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config = load_inherited_payload(Path(config_path))
    v51 = json.loads((root / "stage2/config/stage2_v5_1.json").read_text(encoding="utf-8"))
    scenario_config = v51["scenario_selection"]
    bundle_path = Path(baseline_bundle_path)
    bundle = joblib.load(bundle_path)
    split = config["split"]
    calibration_date = split["calibration_dates"][0]
    calibration_frame = load_v5_day(calibration_date, split="calibration", repo_root=root)
    calibration_tree = predict_baselines(calibration_frame, bundle)["hist_gradient_boosting"]
    truth = calibration_frame["pace_sec_per_m"].to_numpy(float)
    valid = calibration_frame["pace_target_valid"].to_numpy(bool) & np.isfinite(truth)
    residual = truth[valid] - calibration_tree[valid]
    calibration_routes, calibration_raw = _tree_day_scenarios(
        calibration_frame,
        calibration_tree,
        residual,
        scenario_count=int(scenario_config["scenario_count"]),
        seed=int(scenario_config["seed"]),
        route_batch_size=int(scenario_config["route_batch_size"]),
    )
    calibration = _fit_calibration(
        calibration_routes,
        calibration_raw,
        _order_truth(root, calibration_date),
        scenario_config["calibration_dispersion_grid"],
    )
    days = [("validation_model", date) for date in split["validation_model_dates"]]
    days += [("calibration", date) for date in split["calibration_dates"]]
    days += [("evaluation", date) for date in split.get("evaluation_dates", [])]
    days += [("legacy", date) for date in split.get("legacy_test_dates", [])]
    output = Path(output_root)
    products: list[dict[str, Any]] = []
    for split_name, date in days:
        frame = load_v5_day(date, split=split_name, repo_root=root)
        tree_pace = predict_baselines(frame, bundle)["hist_gradient_boosting"]
        if split_name == "calibration" and date == calibration_date:
            route_id, raw = calibration_routes, calibration_raw
        else:
            route_id, raw = _tree_day_scenarios(
                frame,
                tree_pace,
                residual,
                scenario_count=int(scenario_config["scenario_count"]),
                seed=int(scenario_config["seed"]),
                route_batch_size=int(scenario_config["route_batch_size"]),
            )
        values = _calibrate(raw, **calibration)
        quality = _quality(route_id, values, _order_truth(root, date))
        quantiles = np.quantile(values, [0.5, 0.9, 0.95], axis=1)
        stable = bool(
            np.isfinite(values).all()
            and np.all(quantiles[0] <= quantiles[1])
            and np.all(quantiles[1] <= quantiles[2])
            and float(quality["extreme_scenario_share"])
            <= v51["stability_thresholds"]["maximum_extreme_scenario_share"]
        )
        formal = output / "tree_formal_calibrated" / f"split={split_name}" / f"date={date}"
        formal.mkdir(parents=True, exist_ok=True)
        scenario_path = formal / "route_scenario_samples.npz"
        temporary = scenario_path.with_name(f".{scenario_path.name}.tmp.npz")
        scenario_ids = np.arange(values.shape[1], dtype=np.int64)
        np.savez_compressed(
            temporary,
            route_id=route_id,
            route_time_s=values.astype(np.float32),
            system_scenario_id=scenario_ids,
            network_shock_id=scenario_ids,
            route_shock_id=scenario_ids,
            correlation_model_id=np.array("tree_calibration_residual_bootstrap.system_route_residual.1"),
        )
        os.replace(temporary, scenario_path)
        product = pd.DataFrame(
            {
                "order_id": route_id,
                "route_service_time_mean_s": values.mean(axis=1),
                "route_service_time_std_s": values.std(axis=1),
                "route_service_time_p50_s": quantiles[0],
                "route_service_time_p90_s": quantiles[1],
                "route_service_time_p95_s": quantiles[2],
                "route_quantile_source": "tree_calibration_residual_bootstrap",
                "route_scenario_model": "hist_gradient_boosting_shared_system_residual",
                "calibration_identity": f"{protocol}:{calibration_date}:tree",
                "stability_status": "PASS" if stable else "FAIL",
                "eligible_point_forecast": stable,
                "eligible_distribution_forecast": stable,
            }
        )
        product_path = formal / "route_service_predictions.parquet"
        temporary_product = product_path.with_name(f".{product_path.name}.tmp")
        product.to_parquet(temporary_product, index=False, compression="zstd")
        os.replace(temporary_product, product_path)
        tree_field_eligibility = dict(v51["field_status"])
        tree_field_eligibility["route_service_availability_probability"] = "BLOCKED"
        tree_field_eligibility["availability_probability"] = "BLOCKED"
        manifest = {
            "schema_version": "stage2_v5_1_tree_formal_product.1",
            "protocol": protocol,
            "split": split_name,
            "date": date,
            "prediction_source": "tree",
            "route_count": int(len(route_id)),
            "scenario_count": int(values.shape[1]),
            "model_id": f"hist_gradient_boosting:{_bundle_hash(bundle_path)}",
            "checkpoint_sha256": _bundle_hash(bundle_path),
            "scenario_generator_id": "stage2_v5_1_tree_residual_bootstrap.1",
            "scenario_seed": int(scenario_config["seed"]),
            "input_prediction_hash": _bundle_hash(bundle_path),
            "scale": calibration["scale"],
            "dispersion": calibration["dispersion"],
            "offset": calibration["offset_s"],
            "calibration_date": calibration_date,
            "field_eligibility": tree_field_eligibility,
            "stability_check_id": f"stage2_v5_1_tree_stability:{protocol}:{date}",
            "stability_check_status": "PASS" if stable else "FAIL",
            "stability_status": "PASS" if stable else "FAIL",
            "eligible_for_stage3": stable,
            "eligible_for_formal_stage3": False,
            "cross_order_scenario_coherent": True,
            "correlation_model_id": "tree_calibration_residual_bootstrap.system_route_residual.1",
            "files": {scenario_path.name: _sha256(scenario_path), product_path.name: _sha256(product_path)},
            "quality": quality,
        }
        _atomic_json(formal / "manifest.json", manifest)
        products.append(manifest)
    result = {"schema_version": "stage2_v5_1_tree_products.1", "protocol": protocol, "products": products}
    _atomic_json(output / "reports/tree_scenario_manifest.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_protocol(
        repo_root=args.repo_root,
        protocol=args.protocol,
        config_path=args.config,
        baseline_bundle_path=args.baseline_bundle,
        output_root=args.output_root,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
