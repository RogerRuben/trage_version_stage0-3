"""Compute Stage 2 v5 engineering, temporal, scientific, and admission states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _check(name: str, condition: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "status": "PASS" if condition else "FAIL", "detail": detail}


def verify(*, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    docs = root / "stage2/docs/v5"
    config = load_config(root / "stage2/config/stage2_v5.json")
    split = _json(docs / "stage2_v5_split_freeze.json")
    service = _json(docs / "stage2_v5_service_time_target_audit.json")
    static = _json(docs / "stage2_v5_static_complexity_audit.json")
    preflight = _json(docs / "stage2_v5_preflight.json")
    tests = _json(docs / "stage2_v5_test_results.json")
    development = _json(docs / "protocols/development/protocol_summary.json")
    rolling = _json(docs / "stage2_v5_rolling_origin_summary.json")
    legacy = _json(docs / "protocols/legacy/protocol_summary.json")
    legacy_report = _json(docs / "stage2_v5_legacy_benchmark_summary.json")
    admission = config.section("admission")

    engineering_checks = [
        _check("service_time_target_audit", service.get("engineering_status") == "PASS", service.get("engineering_status")),
        _check("regression_tests", tests.get("status") == "PASS" and tests.get("pytest", {}).get("failed") == 0, tests.get("pytest")),
        _check(
            "legacy_predictions_finite",
            legacy_report.get("prediction_numeric_stability", {}).get(
                "nonfinite_availability_probability_count"
            )
            == 0,
            legacy_report.get("prediction_numeric_stability"),
        ),
    ]
    performance_checks = [
        _check("static_complexity", static.get("status") == "PASS" and static.get("blocking_finding_count") == 0, static.get("blocking_finding_count")),
        _check("layered_preflight", preflight.get("performance_gate_status") == "PASS", preflight.get("performance_gate_status")),
    ]
    expected = config.section("split")
    temporal_checks = [
        _check("split_frozen", split.get("freeze_status") == "FROZEN_BEFORE_PROTOCOL_RETRAIN", split.get("freeze_status")),
        _check("development_train", split.get("development", {}).get("train_dates") == expected["train_dates"], split.get("development", {}).get("train_dates")),
        _check("development_validation", split.get("development", {}).get("validation_model_dates") == expected["validation_model_dates"], split.get("development", {}).get("validation_model_dates")),
        _check("development_calibration", split.get("development", {}).get("calibration_dates") == expected["calibration_dates"], split.get("development", {}).get("calibration_dates")),
        _check("development_evaluation", split.get("development", {}).get("evaluation_dates") == expected["evaluation_dates"], split.get("development", {}).get("evaluation_dates")),
        _check("rolling_fold_count", len(split.get("rolling_folds", [])) == 3, len(split.get("rolling_folds", []))),
        _check("no_upstream_rebuild", split.get("upstream_rebuild_required") is False, split.get("upstream_rebuild_required")),
        _check("percentile_not_used_for_rolling_selection", split.get("leakage_rules", {}).get("rts_lcs_percentile_supervision_in_development_or_rolling") is False, split.get("leakage_rules", {}).get("rts_lcs_percentile_supervision_in_development_or_rolling")),
        _check("legacy_identity", split.get("legacy_benchmark", {}).get("benchmark_dates") == ["20161031"], split.get("legacy_benchmark", {}).get("benchmark_dates")),
    ]
    scientific_checks: list[dict[str, Any]] = []
    if rolling:
        coverage = rolling.get("route_scenario_coverage", {})
        scientific_checks = [
            _check("three_rolling_folds", rolling.get("fold_count") == int(admission["minimum_rolling_fold_count"]), rolling.get("fold_count")),
            _check("rolling_aggregate_beats_tree", float(rolling.get("aggregate_relative_mae_change", 1.0)) < 0.0, rolling.get("aggregate_relative_mae_change")),
            _check("rolling_fold_stability", int(rolling.get("fold_win_count", 0)) >= 2, rolling.get("fold_results")),
            _check("rolling_p90_coverage", float(admission["route_p90_coverage_minimum"]) <= float(coverage.get("p90_coverage", -1)) <= float(admission["route_p90_coverage_maximum"]), coverage.get("p90_coverage")),
            _check("rolling_p95_coverage", float(admission["route_p95_coverage_minimum"]) <= float(coverage.get("p95_coverage", -1)) <= float(admission["route_p95_coverage_maximum"]), coverage.get("p95_coverage")),
            _check("no_percentile_selection", rolling.get("percentile_targets_used_for_model_selection") is False, rolling.get("percentile_targets_used_for_model_selection")),
        ]
    engineering_status = "PASS" if all(item["status"] == "PASS" for item in engineering_checks) else "FAIL"
    temporal_status = "PASS" if all(item["status"] == "PASS" for item in temporal_checks) else "FAIL"
    performance_status = "PASS" if all(item["status"] == "PASS" for item in performance_checks) else "FAIL"
    rolling_status = "PASS" if scientific_checks and all(item["status"] == "PASS" for item in scientific_checks) else ("PENDING" if not scientific_checks else "FAIL")
    development_status = development.get("model_evaluation", {}).get("status", "PENDING")
    legacy_status = (
        "PASS"
        if legacy.get("status") == "PASS"
        and legacy_report.get("status") == "COMPLETED"
        and legacy_report.get("date") == "20161031"
        and legacy_report.get("used_for_model_or_hyperparameter_selection") is False
        else "PENDING"
    )
    foundation = engineering_status == temporal_status == performance_status == "PASS"
    if foundation and rolling_status == "PASS" and legacy_status == "PASS":
        admission_status = "READY_FOR_STAGE3"
    elif foundation and development_status in {"BASELINE_COMPETITIVE", "PREDICTIVE_BASELINE_VALIDATED"}:
        admission_status = "READY_FOR_ROUTE_SCENARIO_PROTOTYPE"
    else:
        admission_status = "NOT_READY"
    result = {
        "schema_version": "stage2_v5_final_verification.2",
        "engineering_status": engineering_status,
        "temporal_contract_status": temporal_status,
        "performance_gate_status": performance_status,
        "development_scientific_status": development_status,
        "rolling_origin_status": rolling_status,
        "legacy_benchmark_status": legacy_status,
        "stage3_admission_status": admission_status,
        "checks": {
            "engineering": engineering_checks,
            "performance": performance_checks,
            "temporal": temporal_checks,
            "rolling_scientific": scientific_checks,
        },
    }
    (docs / "stage2_v5_final_verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Stage 2 v5 final evaluation",
        "",
        f"- Engineering: `{engineering_status}`",
        f"- Temporal contract: `{temporal_status}`",
        f"- Performance gate: `{performance_status}`",
        f"- Development temporal evaluation: `{development_status}`",
        f"- Rolling-origin evaluation: `{rolling_status}`",
        f"- 20161031 legacy frozen benchmark: `{legacy_status}`",
        f"- Stage 3 admission: `{admission_status}`",
        "",
        "20161028–30 未生产、未读取。主要科学稳定性由三个预注册 rolling-origin folds 判定；20161031 仅用于与冻结 v4 的版本可比性。",
        "RTS/LCS percentile 在 development 与 rolling 训练中禁用，不参与模型结构选择；主结论使用 direct pace、物理时间、raw 组件与路线服务时间。",
    ]
    (docs / "stage2_v5_final_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_v5_1(*, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    docs = root / "stage2/docs/v5_1"
    config = _json(root / "stage2/config/stage2_v5_1.json")
    evidence = _json(docs / "stage2_v5_1_frozen_v5_evidence.json")
    stability = _json(docs / "stage2_v5_1_distribution_stability.json")
    products = _json(docs / "stage2_v5_1_formal_product_manifest.json")
    rolling = _json(root / "stage2/docs/v5/stage2_v5_rolling_origin_summary.json")
    static_path = docs / "stage2_v5_1_static_complexity_audit.json"
    static = _json(static_path if static_path.is_file() else root / "stage2/docs/v5/stage2_v5_static_complexity_audit.json")
    performance = _json(docs / "stage2_v5_1_performance_benchmarks.json")
    thresholds = config.get("stability_thresholds", {})
    admission = config.get("admission", {})

    evidence_checks = []
    for item in evidence.get("files", []):
        path = root / item["path"]
        evidence_checks.append(_check(f"frozen_evidence:{item['path']}", path.is_file() and _sha256(path) == item["sha256"], item["sha256"]))

    days = stability.get("days", [])
    all_finite = all(
        sum(day["traversal"]["nonfinite_counts"].values()) == 0
        and day["route"]["nonfinite_scenario_sample_count"] == 0
        and sum(day["route"]["nonfinite_route_field_counts"].values()) == 0
        for day in days
    )
    all_monotonic = all(day["traversal"]["non_monotonic_or_nonpositive_quantile_count"] == 0 for day in days)
    maximum_coverage_error = 0.0
    quantile_files = sorted((root / "stage2/docs/v5/protocols").glob("*/quantile_calibration.csv"))
    for path in quantile_files:
        frame = pd.read_csv(path)
        formal = frame[frame["quantile"].isin([0.9, 0.95])]
        if len(formal):
            maximum_coverage_error = max(maximum_coverage_error, float(formal["coverage_error"].abs().max()))

    rolling_metrics = pd.read_csv(root / "stage2/docs/v5/rolling_fold_metrics.csv")
    deep = rolling_metrics[rolling_metrics["model"].eq("rc_mstnet_v5_mean")].set_index(["fold", "date"])["mae"]
    tree = rolling_metrics[rolling_metrics["model"].eq("hist_gradient_boosting")].set_index(["fold", "date"])["mae"]
    daily_wins = int((deep < tree).sum())
    bootstrap = pd.read_csv(root / "stage2/docs/v5/rolling_fold_paired_error_bootstrap.csv")
    paired_all_below_zero = True
    for value in bootstrap["ci95"]:
        interval = value if isinstance(value, list) else json.loads(value)
        paired_all_below_zero &= float(interval[1]) < 0.0

    product_checks: list[dict[str, Any]] = []
    formal_splits = set()
    calibration_by_protocol: dict[str, set[tuple[Any, ...]]] = {}
    for item in products.get("products", []):
        manifest_path = root / item["path"]
        manifest = _json(manifest_path)
        files_valid = all((manifest_path.parent / name).is_file() and _sha256(manifest_path.parent / name) == expected for name, expected in manifest.get("files", {}).items())
        product_checks.append(_check(f"formal_product:{item.get('protocol')}:{item.get('split')}:{item.get('date')}", files_valid and manifest.get("stability_status") == "PASS" and manifest.get("eligible_for_stage3") is True, {"files_valid": files_valid, "stability_status": manifest.get("stability_status")}))
        formal_splits.add(item.get("split"))
        calibration_by_protocol.setdefault(str(item.get("protocol")), set()).add(
            (
                manifest.get("calibration_date"),
                manifest.get("scale"),
                manifest.get("dispersion"),
                manifest.get("offset"),
            )
        )

    coverage = rolling.get("route_scenario_coverage", {})
    maximum_pace_mean = max(day["traversal"]["pace_mean"]["max"] for day in days)
    maximum_mean_to_p50 = max(day["traversal"]["mean_to_p50_ratio"]["max"] for day in days)
    maximum_p99_9_pace_mean = max(day["traversal"]["pace_mean"]["p99_9"] for day in days)
    maximum_row_mae_share = max(day["traversal"]["mean_error"]["maximum_row_mae_contribution_share"] for day in days)
    maximum_row_rmse_share = max(day["traversal"]["mean_error"]["maximum_row_rmse_contribution_share"] for day in days)
    configuration_checks = [
        _check("all_formal_fields_finite", all_finite, all_finite),
        _check("all_quantiles_monotonic", all_monotonic, all_monotonic),
        _check("static_complexity_gate", static.get("status") == "PASS", static.get("status")),
        _check("performance_scaling_gate", performance.get("status") == "PASS", performance.get("status", "MISSING")),
        _check("maximum_pace_mean", maximum_pace_mean <= float(thresholds["maximum_pace_mean_s_per_m"]), maximum_pace_mean),
        _check("maximum_mean_to_p50_ratio", maximum_mean_to_p50 <= float(thresholds["maximum_mean_to_p50_ratio"]), maximum_mean_to_p50),
        _check("maximum_p99_9_pace_mean", maximum_p99_9_pace_mean <= float(thresholds["maximum_p99_9_pace_mean_s_per_m"]), maximum_p99_9_pace_mean),
        _check("maximum_single_row_mae_contribution", maximum_row_mae_share <= float(thresholds["maximum_single_row_mae_contribution_share"]), maximum_row_mae_share),
        _check("maximum_single_row_rmse_contribution", maximum_row_rmse_share <= float(thresholds["maximum_single_row_rmse_contribution_share"]), maximum_row_rmse_share),
        _check("maximum_pace_quantile_coverage_error", maximum_coverage_error <= float(admission["maximum_pace_quantile_coverage_error"]), maximum_coverage_error),
        _check("minimum_final_daily_mae_wins_vs_strong_baseline", daily_wins >= int(admission["minimum_final_daily_mae_wins_vs_strong_baseline"]), daily_wins),
        _check("require_aggregate_mae_better_than_strong_baseline", (not admission["require_aggregate_mae_better_than_strong_baseline"]) or float(rolling["aggregate_relative_mae_change"]) < 0.0, rolling["aggregate_relative_mae_change"]),
        _check("require_paired_bootstrap_ci_below_zero", (not admission["require_paired_bootstrap_ci_below_zero"]) or paired_all_below_zero, paired_all_below_zero),
        _check("route_p90_coverage", float(admission["route_p90_coverage_minimum"]) <= float(coverage["p90_coverage"]) <= float(admission["route_p90_coverage_maximum"]), coverage["p90_coverage"]),
        _check("route_p95_coverage", float(admission["route_p95_coverage_minimum"]) <= float(coverage["p95_coverage"]) <= float(admission["route_p95_coverage_maximum"]), coverage["p95_coverage"]),
        _check("route_mean_rmse_stability", all(day["route"]["route_mean_rmse_s"] <= float(thresholds["maximum_route_mean_rmse_s"]) for day in days), max(day["route"]["route_mean_rmse_s"] for day in days)),
        _check("route_mean_stability", all(day["route"]["route_mean_max_s"] <= float(thresholds["maximum_route_mean_s"]) for day in days), max(day["route"]["route_mean_max_s"] for day in days)),
        _check("route_cvar95_stability", all(day["route"]["route_cvar95_max_s"] <= float(thresholds["maximum_route_cvar95_s"]) for day in days), max(day["route"]["route_cvar95_max_s"] for day in days)),
        _check("route_p90_interval_width_stability", all(day["route"]["average_p90_p50_width_s"] <= float(thresholds["maximum_average_p90_p50_width_s"]) for day in days), max(day["route"]["average_p90_p50_width_s"] for day in days)),
        _check("route_p95_interval_width_stability", all(day["route"]["average_p95_p50_width_s"] <= float(thresholds["maximum_average_p95_p50_width_s"]) for day in days), max(day["route"]["average_p95_p50_width_s"] for day in days)),
        _check("route_extreme_interval_share", all(day["route"]["extreme_p95_p50_width_share"] <= float(thresholds["maximum_extreme_route_interval_share"]) for day in days), max(day["route"]["extreme_p95_p50_width_share"] for day in days)),
        _check("formal_evaluation_and_legacy_present", {"evaluation", "legacy"}.issubset(formal_splits), sorted(formal_splits)),
        _check(
            "calibration_parameters_frozen_across_post_calibration_dates",
            all(len(values) == 1 for values in calibration_by_protocol.values()),
            {name: len(values) for name, values in calibration_by_protocol.items()},
        ),
    ]
    foundation = (
        all(item["status"] == "PASS" for item in [*evidence_checks, *product_checks])
        and all_finite
        and all_monotonic
        and static.get("status") == "PASS"
        and performance.get("status") == "PASS"
    )
    scientific = all(item["status"] == "PASS" for item in configuration_checks)
    retrained_checks: list[dict[str, Any]] = []
    rolling_v51_path = docs / "stage2_v5_1_rolling_summary.json"
    cross_order_path = docs / "stage2_v5_1_cross_order_scenario_audit.json"
    if rolling_v51_path.is_file() and cross_order_path.is_file():
        rolling_v51 = _json(rolling_v51_path)
        cross_order = _json(cross_order_path)
        quantile_v51 = pd.read_csv(docs / "stage2_v5_1_quantile_metrics.csv")
        route_v51 = pd.read_csv(docs / "stage2_v5_1_route_scoring.csv")
        rolling_route = route_v51[route_v51["protocol"].isin(["fold_1", "fold_2", "fold_3"]) & route_v51["split"].eq("evaluation")]
        new_product_manifests = sorted(
            (root / "stage2/output_v5_1").glob("*/formal_calibrated/split=*/date=*/manifest.json")
        )
        new_products_valid = bool(new_product_manifests)
        new_calibration: dict[str, set[tuple[Any, ...]]] = {}
        for manifest_path in new_product_manifests:
            manifest = _json(manifest_path)
            new_products_valid &= manifest.get("stability_status") == "PASS"
            new_products_valid &= manifest.get("stability_check_status") == "PASS"
            new_products_valid &= manifest.get("full_distribution_stability_status") == "PASS"
            new_products_valid &= all(
                (manifest_path.parent / name).is_file()
                and _sha256(manifest_path.parent / name) == digest
                for name, digest in manifest.get("files", {}).items()
            )
            new_calibration.setdefault(str(manifest.get("protocol")), set()).add(
                (
                    manifest.get("calibration_date"),
                    manifest.get("scale"),
                    manifest.get("dispersion"),
                    manifest.get("offset"),
                )
            )
        model_manifests = sorted((root / "stage2/output_v5_1").glob("*/deep_model/model_manifest.json"))
        stable_models = len(model_manifests) == 5
        selected_single_row_diagnostics: dict[str, bool] = {}
        for model_path in model_manifests:
            model = _json(model_path)
            candidates = model.get("checkpoint_candidates", [])
            stable_models &= bool(candidates) and any(
                row.get("hard_gate_status") == "PASS" for row in candidates
            )
            selected_id = f"epoch_{int(model.get('best_epoch', -1)):03d}"
            selected = next(
                (row for row in candidates if row.get("checkpoint_id") == selected_id), {}
            )
            selected_single_row_diagnostics[model_path.parent.parent.name] = bool(
                selected.get("diagnostics", {}).get("single_row_contribution_threshold_pass")
            )
        retrained_checks = [
            _check("v5_1_three_rolling_folds", int(rolling_v51["fold_count"]) == 3, rolling_v51["fold_count"]),
            _check("v5_1_aggregate_p50_better_than_tree", float(rolling_v51["relative_difference"]) < 0, rolling_v51["relative_difference"]),
            _check("v5_1_paired_ci_below_zero", bool(rolling_v51["all_paired_ci_below_zero"]), rolling_v51["all_paired_ci_below_zero"]),
            _check("v5_1_daily_wins", int(rolling_v51["daily_p50_wins"]) >= int(admission["minimum_final_daily_mae_wins_vs_strong_baseline"]), rolling_v51["daily_p50_wins"]),
            _check("v5_1_quantile_coverage_error", max(float((quantile_v51["p90_coverage"] - 0.9).abs().max()), float((quantile_v51["p95_coverage"] - 0.95).abs().max())) <= float(admission["maximum_pace_quantile_coverage_error"]), None),
            _check("v5_1_route_p90_coverage", bool(rolling_route["p90_coverage"].between(float(admission["route_p90_coverage_minimum"]), float(admission["route_p90_coverage_maximum"])).all()), rolling_route["p90_coverage"].tolist()),
            _check("v5_1_route_p95_coverage", bool(rolling_route["p95_coverage"].between(float(admission["route_p95_coverage_minimum"]), float(admission["route_p95_coverage_maximum"])).all()), rolling_route["p95_coverage"].tolist()),
            _check("v5_1_route_mean_rmse_stability", bool((rolling_route["route_mean_rmse_s"] <= float(thresholds["maximum_route_mean_rmse_s"])).all()), float(rolling_route["route_mean_rmse_s"].max())),
            _check("v5_1_route_mean_stability", bool((rolling_route["maximum_route_mean_s"] <= float(thresholds["maximum_route_mean_s"])).all()), float(rolling_route["maximum_route_mean_s"].max())),
            _check("v5_1_route_cvar95_stability", bool((rolling_route["maximum_route_cvar95_s"] <= float(thresholds["maximum_route_cvar95_s"])).all()), float(rolling_route["maximum_route_cvar95_s"].max())),
            _check("v5_1_extreme_scenario_stability", bool((rolling_route["maximum_scenario_s"] <= float(thresholds["maximum_route_cvar95_s"])).all()), float(rolling_route["maximum_scenario_s"].max())),
            _check("v5_1_cross_order_scenario", cross_order.get("status") == "PASS", cross_order.get("status")),
            _check("v5_1_stable_checkpoint_manifests", stable_models, len(model_manifests)),
            _check("v5_1_selected_checkpoint_single_row_stability", bool(selected_single_row_diagnostics) and all(selected_single_row_diagnostics.values()), selected_single_row_diagnostics),
            _check("v5_1_formal_product_hashes_and_stability", new_products_valid, len(new_product_manifests)),
            _check("v5_1_calibration_frozen_by_protocol", bool(new_calibration) and all(len(values) == 1 for values in new_calibration.values()), {key: len(value) for key, value in new_calibration.items()}),
        ]
    retrained_scientific = bool(retrained_checks) and all(item["status"] == "PASS" for item in retrained_checks)
    status = (
        "READY_FOR_STAGE3"
        if foundation and retrained_scientific
        else ("READY_FOR_ROUTE_SCENARIO_PROTOTYPE" if foundation else "NOT_READY")
    )
    result = {
        "schema_version": "stage2_v5_1_verification.1",
        "status": status,
        "p0_status": "PASS" if foundation else "FAIL",
        "full_distribution_stability_status": stability.get("status"),
        "retraining_required": not retrained_scientific,
        "checks": {"frozen_evidence": evidence_checks, "formal_products": product_checks, "configuration_and_science": configuration_checks, "v5_1_retraining": retrained_checks},
        "availability_model_role": "availability_aware_diagnostic_and_ipw_sensitivity_only",
        "horizon_gate_role": "negative_ablation_not_selected",
    }
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "stage2_v5_1_verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--v5-1", action="store_true")
    args = parser.parse_args()
    report = verify_v5_1(repo_root=args.repo_root) if args.v5_1 else verify(repo_root=args.repo_root)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
