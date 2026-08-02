"""Export complete calibrated v5 products with fail-closed field eligibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .scenario_pipeline import _calibrate_values, _order_truth, _route_product


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


def _scenario_quality(route_id: np.ndarray, values: np.ndarray, truth: pd.Series) -> dict[str, float | int]:
    actual = truth.reindex(route_id.astype(str)).to_numpy(float)
    valid = np.isfinite(actual) & np.isfinite(values).all(axis=1)
    scenarios = values[valid].astype(np.float64, copy=False)
    actual = actual[valid]
    quantiles = np.quantile(scenarios, (0.05, 0.1, 0.5, 0.9, 0.95), axis=1)
    mean = scenarios.mean(axis=1)
    error = mean - actual
    ordered = np.sort(scenarios, axis=1)
    coefficient = 2.0 * np.arange(1, scenarios.shape[1] + 1) - scenarios.shape[1] - 1.0
    crps = np.abs(scenarios - actual[:, None]).mean(axis=1) - (ordered * coefficient).sum(axis=1) / (scenarios.shape[1] ** 2)
    def interval_score(lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
        return upper - lower + 2.0 / alpha * (lower - actual) * (actual < lower) + 2.0 / alpha * (actual - upper) * (actual > upper)
    wis = (
        0.5 * np.abs(actual - quantiles[2])
        + 0.1 * interval_score(quantiles[1], quantiles[3], 0.2)
        + 0.05 * interval_score(quantiles[0], quantiles[4], 0.1)
    ) / 2.5
    return {
        "route_count": int(valid.sum()),
        "route_p50_mae_s": float(np.mean(np.abs(quantiles[2] - actual))),
        "route_mean_mae_s": float(np.mean(np.abs(error))),
        "route_mean_rmse_s": float(np.sqrt(np.mean(np.square(error)))),
        "sample_crps_s": float(np.mean(crps)),
        "weighted_interval_score_s": float(np.mean(wis)),
        "p90_coverage": float(np.mean(actual <= quantiles[3])),
        "p95_coverage": float(np.mean(actual <= quantiles[4])),
        "p90_p50_width_s": float(np.mean(quantiles[3] - quantiles[2])),
        "p95_p50_width_s": float(np.mean(quantiles[4] - quantiles[2])),
    }


def export(*, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config = json.loads((root / "stage2/config/stage2_v5_1.json").read_text(encoding="utf-8"))
    stability = json.loads((root / "stage2/docs/v5_1/stage2_v5_1_distribution_stability.json").read_text(encoding="utf-8"))
    stability_index = {(row["protocol"], row["split"], row["date"]): row for row in stability["days"]}
    output_root = root / config["formal_products"]["output_root"]
    protocol_specs = {
        "development": {
            "prediction_root": root / "stage2/output_v5/protocols/development/ablations/ordinary_concatenation/predictions",
            "model_root": root / "stage2/output_v5/protocols/development/ablations/ordinary_concatenation/deep_model",
        },
        "legacy": {
            "prediction_root": root / "stage2/output_v5/protocols/legacy/predictions",
            "model_root": root / "stage2/output_v5/protocols/legacy/deep_model",
        },
    }
    products: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for protocol, spec in protocol_specs.items():
        protocol_root = root / "stage2/output_v5/protocols" / protocol
        report_root = root / "stage2/docs/v5/protocols" / protocol
        selection_path = report_root / "stage2_v5_scenario_selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        model_manifest = json.loads((spec["model_root"] / "model_manifest.json").read_text(encoding="utf-8"))
        selected = selection["selected_model"]
        scenario_root = protocol_root / "route_scenarios/ordinary_concatenation"
        for prediction_path in sorted(spec["prediction_root"].glob("split=*/date=*/traversal_predictions.parquet")):
            split = prediction_path.parent.parent.name.split("=", 1)[1]
            date = prediction_path.parent.name.split("=", 1)[1]
            raw_path = scenario_root / f"split={split}" / f"date={date}" / f"model={selected}" / "route_scenarios.npz"
            with np.load(raw_path, allow_pickle=False) as archive:
                route_id = archive["route_id"].astype(str)
                calibrated = _calibrate_values(
                    archive["route_time_s"],
                    scale=float(selection["route_time_scale"]),
                    dispersion=float(selection["route_dispersion_multiplier"]),
                    offset_s=float(selection["route_offset_s"]),
                ).astype(np.float32)
            audit = stability_index[(protocol, split, date)]
            traversal_checks = audit["stability"]["checks"]
            route_nonfinite = audit["route"]["nonfinite_route_field_counts"]
            quantile_pass = (
                traversal_checks["all_traversal_fields_finite"]
                and traversal_checks["quantiles_monotonic"]
                and audit["route"]["nonfinite_scenario_sample_count"] == 0
                and all(route_nonfinite[name] == 0 for name in ("p50", "p90", "p95"))
            )
            full_distribution_pass = audit["stability"]["status"] == "PASS"
            formal_root = output_root / f"protocol={protocol}" / f"split={split}" / f"date={date}"
            formal_root.mkdir(parents=True, exist_ok=True)
            scenario_path = formal_root / "route_scenario_samples.npz"
            temporary_scenario = scenario_path.with_name(f".{scenario_path.name}.tmp.npz")
            np.savez_compressed(temporary_scenario, route_id=route_id, route_time_s=calibrated)
            os.replace(temporary_scenario, scenario_path)
            product = _route_product(
                route_id,
                calibrated,
                thresholds_s=[600.0, 900.0, 1200.0],
                model_id=model_manifest["model_id"],
                seed=int(selection["scenario_seed"]),
                generator_id=selection["generator_id"],
                input_hash=_sha256(prediction_path),
            )
            product = product.rename(columns={
                "scenario_model_id": "route_scenario_model",
                "scenario_input_hash": "input_prediction_hash",
            })
            product["route_quantile_source"] = "frozen_calibrated_empirical_scenarios"
            product["calibration_identity"] = f"{protocol}:{selection['calibration_dates'][0]}:{_sha256(selection_path)}"
            product["stability_status"] = "PASS" if quantile_pass else "FAIL"
            product["full_distribution_stability_status"] = "PASS" if full_distribution_pass else "FAIL"
            product["eligible_point_forecast"] = bool(quantile_pass)
            product["eligible_distribution_forecast"] = bool(quantile_pass)
            product["mean_std_cvar_field_status"] = "EXPERIMENTAL" if full_distribution_pass else "BLOCKED"
            product["quantile_field_status"] = "ELIGIBLE" if quantile_pass else "BLOCKED"
            availability_source = pd.read_parquet(
                prediction_path,
                columns=["order_id", "allocated_distance_m", "service_time_availability_probability"],
            )
            availability_source["weighted"] = (
                availability_source["allocated_distance_m"]
                * availability_source["service_time_availability_probability"]
            )
            availability_group = availability_source.groupby("order_id", sort=False, observed=True)[["weighted", "allocated_distance_m"]].sum()
            product["route_service_availability_probability"] = (
                availability_group["weighted"] / availability_group["allocated_distance_m"].clip(lower=1.0e-6)
            ).reindex(route_id).to_numpy(float)
            for old in (600, 900, 1200):
                product = product.rename(columns={f"timeout_probability_threshold_{old}s": f"diagnostic_example_timeout_probability_{old}s"})
            product_path = formal_root / "route_service_predictions.parquet"
            temporary_product = product_path.with_name(f".{product_path.name}.tmp")
            product.to_parquet(temporary_product, index=False, compression="zstd")
            os.replace(temporary_product, product_path)
            quality = _scenario_quality(route_id, calibrated, _order_truth(root, date))
            quality_rows.append({"protocol": protocol, "split": split, "date": date, **quality})
            route_field_eligibility = dict(config["field_status"])
            route_field_eligibility["availability_probability"] = "BLOCKED"
            manifest = {
                "schema_version": "stage2_v5_1_formal_route_scenarios.1",
                "protocol": protocol,
                "split": split,
                "date": date,
                "route_count": int(len(route_id)),
                "scenario_count": int(calibrated.shape[1]),
                "model_id": model_manifest["model_id"],
                "checkpoint_sha256": model_manifest["checkpoint_sha256"],
                "scenario_generator_id": selection["generator_id"],
                "prediction_source": "deep_scenario",
                "scenario_seed": int(selection["scenario_seed"]),
                "input_prediction_hash": _sha256(prediction_path),
                "scale": float(selection["route_time_scale"]),
                "dispersion": float(selection["route_dispersion_multiplier"]),
                "offset": float(selection["route_offset_s"]),
                "calibration_date": selection["calibration_dates"][0],
                "calibration_identity": product["calibration_identity"].iloc[0],
                "field_eligibility": route_field_eligibility,
                "stability_check_id": f"stage2_v5_1_distribution_stability.1:{protocol}:{split}:{date}",
                "stability_check_status": "PASS" if quantile_pass else "FAIL",
                "stability_status": "PASS" if quantile_pass else "FAIL",
                "full_distribution_stability_status": "PASS" if full_distribution_pass else "FAIL",
                "eligible_for_stage3": bool(quantile_pass),
                "eligible_for_formal_stage3": False,
                "example_timeout_thresholds_are_diagnostics_only": True,
                "files": {
                    "route_scenario_samples.npz": _sha256(scenario_path),
                    "route_service_predictions.parquet": _sha256(product_path),
                },
                "quality": quality,
            }
            manifest_path = formal_root / "manifest.json"
            _atomic_json(manifest_path, manifest)
            products.append({"path": manifest_path.relative_to(root).as_posix(), "manifest_sha256": _sha256(manifest_path), **manifest})
            del calibrated, product
    quality_frame = pd.DataFrame(quality_rows)
    docs = root / "stage2/docs/v5_1"
    quality_frame.to_csv(docs / "stage2_v5_1_route_scoring.csv", index=False)
    result = {
        "schema_version": "stage2_v5_1_formal_product_manifest.1",
        "status": "PASS" if all(item["eligible_for_stage3"] for item in products) else "FAIL",
        "product_count": len(products),
        "all_formal_stage3_blocked": all(not item["eligible_for_formal_stage3"] for item in products),
        "products": products,
    }
    _atomic_json(docs / "stage2_v5_1_formal_product_manifest.json", result)
    pooled = quality_frame.groupby("protocol", observed=True)[["route_count", "route_p50_mae_s", "route_mean_mae_s", "route_mean_rmse_s", "sample_crps_s", "weighted_interval_score_s", "p90_coverage", "p95_coverage", "p90_p50_width_s", "p95_p50_width_s"]].mean(numeric_only=True).reset_index()
    columns = pooled.columns.tolist()
    table = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for values in pooled.to_numpy():
        table.append("| " + " | ".join(f"{value:.6g}" if isinstance(value, float) else str(value) for value in values) + " |")
    lines = ["# Stage 2 v5.1 scenario quality", "", "Frozen calibrated scenarios are scored without changing the v5 family or calibration.", "", *table, "", "Mean/std/CVaR remain blocked whenever the full distribution stability gate fails. Quantile fields are independently eligible only when finite and monotonic."]
    (docs / "stage2_v5_1_scenario_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    print(json.dumps(export(repo_root=args.repo_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
