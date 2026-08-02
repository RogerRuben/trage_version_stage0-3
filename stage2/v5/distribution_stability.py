"""Full traversal and route distribution stability audit for frozen v5 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .scenario_pipeline import _calibrate_values, _order_truth


PROTOCOLS = ("development", "fold_1", "fold_2", "fold_3", "legacy")
PACE_FIELDS = (
    "pace_log_mu",
    "pace_log_scale",
    "pace_pred_mean",
    "pace_pred_p50",
    "pace_pred_p90",
    "pace_pred_p95",
    "service_time_availability_probability",
    "lcs_availability_probability",
    "rts_availability_probability",
    "dynamics_availability_probability",
)


def _quantiles(values: np.ndarray, probabilities: tuple[float, ...]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    names = {0.0: "min", 0.01: "p01", 0.5: "p50", 0.9: "p90", 0.95: "p95", 0.99: "p99", 0.999: "p99_9", 1.0: "max"}
    if not finite.size:
        return {names[value]: None for value in probabilities}
    result = np.quantile(finite, probabilities)
    return {names[probability]: float(value) for probability, value in zip(probabilities, result)}


def _error_contribution(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | None]:
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if not valid.any():
        return {"count": 0, "mae": None, "rmse": None, "maximum_row_mae_contribution_share": None, "maximum_row_rmse_contribution_share": None}
    error = prediction[valid] - truth[valid]
    absolute = np.abs(error)
    squared = np.square(error)
    return {
        "count": int(valid.sum()),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(squared.mean())),
        "maximum_row_mae_contribution_share": float(absolute.max() / absolute.sum()) if absolute.sum() else 0.0,
        "maximum_row_rmse_contribution_share": float(squared.max() / squared.sum()) if squared.sum() else 0.0,
    }


def summarize_traversals(frame: pd.DataFrame) -> dict[str, Any]:
    mean = frame["pace_pred_mean"].to_numpy(float)
    p50 = frame["pace_pred_p50"].to_numpy(float)
    ratio = np.divide(mean, p50, out=np.full(len(frame), np.nan), where=np.isfinite(p50) & (p50 > 0))
    truth = frame["pace_sec_per_m"].to_numpy(float)
    valid = frame["pace_target_valid"].to_numpy(bool)
    mean_error = _error_contribution(np.where(valid, truth, np.nan), mean)
    p50_error = _error_contribution(np.where(valid, truth, np.nan), p50)
    thresholds = {}
    for threshold in (0.5, 1.0, 2.0, 5.0, 10.0):
        count = int(np.count_nonzero(np.isfinite(mean) & (mean > threshold)))
        thresholds[str(threshold)] = {"count": count, "share": float(count / len(frame)) if len(frame) else 0.0}
    finite_counts = {column: int(np.count_nonzero(~np.isfinite(frame[column].to_numpy(float)))) for column in PACE_FIELDS}
    monotonic = (
        np.isfinite(p50)
        & np.isfinite(frame["pace_pred_p90"].to_numpy(float))
        & np.isfinite(frame["pace_pred_p95"].to_numpy(float))
        & (p50 > 0)
        & (p50 <= frame["pace_pred_p90"].to_numpy(float))
        & (frame["pace_pred_p90"].to_numpy(float) <= frame["pace_pred_p95"].to_numpy(float))
    )
    return {
        "row_count": int(len(frame)),
        "pace_log_mu": _quantiles(frame["pace_log_mu"].to_numpy(float), (0.0, 0.01, 0.5, 0.95, 0.99, 1.0)),
        "pace_log_scale": _quantiles(frame["pace_log_scale"].to_numpy(float), (0.0, 0.01, 0.5, 0.95, 0.99, 1.0)),
        "sigma": _quantiles(np.exp(frame["pace_log_scale"].to_numpy(float)), (0.0, 0.5, 0.95, 0.99, 1.0)),
        "pace_mean": _quantiles(mean, (0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0)),
        "pace_p50": _quantiles(p50, (0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0)),
        "mean_to_p50_ratio": _quantiles(ratio, (0.5, 0.9, 0.95, 0.99, 1.0)),
        "pace_mean_thresholds": thresholds,
        "nonfinite_counts": finite_counts,
        "non_monotonic_or_nonpositive_quantile_count": int(len(frame) - monotonic.sum()),
        "mean_error": mean_error,
        "p50_error": p50_error,
    }


def summarize_routes(route_id: np.ndarray, scenarios: np.ndarray, truth: pd.Series, *, extreme_width_s: float) -> dict[str, Any]:
    values = np.asarray(scenarios, dtype=np.float64)
    q = np.quantile(values, (0.5, 0.9, 0.95), axis=1)
    mean = values.mean(axis=1)
    std = values.std(axis=1)
    cvar90 = np.nanmean(np.where(values >= q[1, :, None], values, np.nan), axis=1)
    cvar95 = np.nanmean(np.where(values >= q[2, :, None], values, np.nan), axis=1)
    actual = truth.reindex(route_id.astype(str)).to_numpy(float)
    valid = np.isfinite(actual) & np.isfinite(mean)
    squared = np.square(mean[valid] - actual[valid])
    total_squared = squared.sum()
    top = np.sort(squared)[::-1]
    width90 = q[1] - q[0]
    width95 = q[2] - q[0]
    route_fields = {"mean": mean, "std": std, "p50": q[0], "p90": q[1], "p95": q[2], "cvar90": cvar90, "cvar95": cvar95}
    return {
        "route_count": int(len(route_id)),
        "scenario_count": int(values.shape[1]),
        "nonfinite_scenario_sample_count": int(np.count_nonzero(~np.isfinite(values))),
        "nonfinite_route_field_counts": {name: int(np.count_nonzero(~np.isfinite(array))) for name, array in route_fields.items()},
        "route_mean_max_s": float(np.nanmax(mean)),
        "route_p50_max_s": float(np.nanmax(q[0])),
        "route_p90_max_s": float(np.nanmax(q[1])),
        "route_p95_max_s": float(np.nanmax(q[2])),
        "route_mean_to_p50_ratio_max": float(np.nanmax(np.divide(mean, q[0], out=np.full(len(mean), np.nan), where=q[0] > 0))),
        "route_std_to_p50_ratio_max": float(np.nanmax(np.divide(std, q[0], out=np.full(len(mean), np.nan), where=q[0] > 0))),
        "route_cvar90_max_s": float(np.nanmax(cvar90)),
        "route_cvar95_max_s": float(np.nanmax(cvar95)),
        "route_mean_mae_s": float(np.mean(np.abs(mean[valid] - actual[valid]))),
        "route_mean_rmse_s": float(np.sqrt(np.mean(squared))),
        "top1_route_rmse_contribution_share": float(top[:1].sum() / total_squared) if total_squared else 0.0,
        "top10_route_rmse_contribution_share": float(top[:10].sum() / total_squared) if total_squared else 0.0,
        "average_p90_p50_width_s": float(np.mean(width90)),
        "average_p95_p50_width_s": float(np.mean(width95)),
        "extreme_p95_p50_width_count": int(np.count_nonzero(width95 > extreme_width_s)),
        "extreme_p95_p50_width_share": float(np.mean(width95 > extreme_width_s)),
    }


def _gate(traversal: dict[str, Any], route: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    checks = {
        "all_traversal_fields_finite": sum(traversal["nonfinite_counts"].values()) == 0,
        "quantiles_monotonic": traversal["non_monotonic_or_nonpositive_quantile_count"] == 0,
        "maximum_pace_mean": traversal["pace_mean"]["max"] <= thresholds["maximum_pace_mean_s_per_m"],
        "maximum_mean_to_p50_ratio": traversal["mean_to_p50_ratio"]["max"] <= thresholds["maximum_mean_to_p50_ratio"],
        "p99_9_pace_mean": traversal["pace_mean"]["p99_9"] <= thresholds["maximum_p99_9_pace_mean_s_per_m"],
        "single_row_mae_contribution": traversal["mean_error"]["maximum_row_mae_contribution_share"] <= thresholds["maximum_single_row_mae_contribution_share"],
        "single_row_rmse_contribution": traversal["mean_error"]["maximum_row_rmse_contribution_share"] <= thresholds["maximum_single_row_rmse_contribution_share"],
        "all_route_fields_finite": route["nonfinite_scenario_sample_count"] == 0 and sum(route["nonfinite_route_field_counts"].values()) == 0,
        "maximum_route_mean": route["route_mean_max_s"] <= thresholds["maximum_route_mean_s"],
        "route_mean_rmse": route["route_mean_rmse_s"] <= thresholds["maximum_route_mean_rmse_s"],
        "route_p90_width": route["average_p90_p50_width_s"] <= thresholds["maximum_average_p90_p50_width_s"],
        "route_p95_width": route["average_p95_p50_width_s"] <= thresholds["maximum_average_p95_p50_width_s"],
        "extreme_route_width_share": route["extreme_p95_p50_width_share"] <= thresholds["maximum_extreme_route_interval_share"],
        "route_cvar95": route["route_cvar95_max_s"] <= thresholds["maximum_route_cvar95_s"],
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def audit(*, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config = json.loads((root / "stage2/config/stage2_v5_1.json").read_text(encoding="utf-8"))
    thresholds = config["stability_thresholds"]
    rows: list[dict[str, Any]] = []
    for protocol in PROTOCOLS:
        protocol_root = root / "stage2/output_v5/protocols" / protocol
        report_root = root / "stage2/docs/v5/protocols" / protocol
        selection = json.loads((report_root / "stage2_v5_scenario_selection.json").read_text(encoding="utf-8"))
        selected = selection["selected_model"]
        scenario_root = protocol_root / "route_scenarios" / "ordinary_concatenation"
        prediction_root = (
            protocol_root / "ablations/ordinary_concatenation/predictions"
            if protocol == "development"
            else protocol_root / "predictions"
        )
        for prediction_path in sorted(prediction_root.glob("split=*/date=*/traversal_predictions.parquet")):
            split = prediction_path.parent.parent.name.split("=", 1)[1]
            date = prediction_path.parent.name.split("=", 1)[1]
            frame = pd.read_parquet(prediction_path, columns=[*PACE_FIELDS, "pace_sec_per_m", "pace_target_valid"])
            traversal = summarize_traversals(frame)
            scenario_path = scenario_root / f"split={split}" / f"date={date}" / f"model={selected}" / "route_scenarios.npz"
            with np.load(scenario_path, allow_pickle=False) as archive:
                route_id = archive["route_id"].astype(str)
                values = _calibrate_values(
                    archive["route_time_s"],
                    scale=float(selection["route_time_scale"]),
                    dispersion=float(selection["route_dispersion_multiplier"]),
                    offset_s=float(selection["route_offset_s"]),
                )
            route = summarize_routes(route_id, values, _order_truth(root, date), extreme_width_s=float(thresholds["extreme_route_interval_width_s"]))
            gate = _gate(traversal, route, thresholds)
            rows.append({"protocol": protocol, "split": split, "date": date, "traversal": traversal, "route": route, "stability": gate})
            del frame, values
    failure_count = sum(item["stability"]["status"] == "FAIL" for item in rows)
    result = {
        "schema_version": "stage2_v5_1_distribution_stability.1",
        "status": "PASS" if failure_count == 0 else "FAIL",
        "audit_scope": "frozen_v5_outputs_read_only",
        "thresholds_frozen_before_v5_1_training": True,
        "thresholds": thresholds,
        "threshold_rationale": config["threshold_rationale"],
        "day_count": len(rows),
        "failed_day_count": failure_count,
        "days": rows,
    }
    output = root / "stage2/docs/v5_1"
    output.mkdir(parents=True, exist_ok=True)
    (output / "stage2_v5_1_distribution_stability.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Stage 2 v5.1 distribution stability audit",
        "",
        "This audit is read-only over frozen v5 predictions and scenarios. It does not trim rows or replace the frozen rolling mean metric with P50.",
        "",
        "The gates were frozen before v5.1 training. Pace limits are physical/numerical operational bounds, tail limits prevent one finite row from dominating population risk, and route limits are city taxi-order engineering bounds; none were fitted from Evaluation or legacy results.",
        "",
        "| Protocol | Split | Date | Status | Pace mean max | Mean/P50 max | P99.9 mean | Max-row MAE share | Route mean RMSE | P95 width |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in rows:
        traversal = item["traversal"]
        route = item["route"]
        lines.append(
            f"| {item['protocol']} | {item['split']} | {item['date']} | {item['stability']['status']} | "
            f"{traversal['pace_mean']['max']:.6f} | {traversal['mean_to_p50_ratio']['max']:.3f} | "
            f"{traversal['pace_mean']['p99_9']:.6f} | {traversal['mean_error']['maximum_row_mae_contribution_share']:.4%} | "
            f"{route['route_mean_rmse_s']:.2f} | {route['average_p95_p50_width_s']:.2f} |"
        )
    lines += ["", f"Overall frozen-v5 stability status: `{result['status']}`; failed protocol-days: {failure_count}/{len(rows)}."]
    (output / "stage2_v5_1_distribution_stability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    print(json.dumps(audit(repo_root=args.repo_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
