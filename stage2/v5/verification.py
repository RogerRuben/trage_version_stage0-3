"""Compute Stage 2 v5 verification and Stage 3 admission from artifacts.

The verifier deliberately distinguishes development readiness from a final
scientific release.  Missing preregistered final-test products can yield route
scenario prototype readiness, but can never yield READY_FOR_STAGE3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "FAIL", "detail": detail}


def verify(*, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    docs = root / "stage2/docs/v5"
    config_path = root / "stage2/config/stage2_v5.json"
    config = load_config(config_path)
    expected_split = config.section("split")

    split = _json(docs / "stage2_v5_split_freeze.json")
    service = _json(docs / "stage2_v5_service_time_target_audit.json")
    static = _json(docs / "stage2_v5_static_complexity_audit.json")
    benchmark = _json(docs / "stage2_v5_performance_benchmarks.json")
    preflight = _json(docs / "stage2_v5_preflight.json")
    baseline = _json(docs / "stage2_v5_baseline_summary.json")
    model = _json(docs / "stage2_v5_model_summary.json")
    ablation = _json(docs / "stage2_v5_horizon_ablation.json")
    scenario = _json(docs / "stage2_v5_scenario_selection.json")
    state_targets = _json(docs / "stage2_v5_state_target_summary.json")
    test_results = _json(docs / "stage2_v5_test_results.json")
    model_manifest = _json(root / "stage2/output_v5/deep_model/model_manifest.json")
    prediction_manifest = _json(root / "stage2/output_v5/predictions/manifest.json")

    split_value = split["split"]
    temporal_checks = [
        _check("split_frozen_before_fit", split.get("freeze_status") == "FROZEN_BEFORE_MODEL_FIT", split.get("freeze_status")),
        _check("train_dates_match", split_value.get("train_dates") == expected_split["train_dates"], split_value.get("train_dates")),
        _check("validation_dates_match", split_value.get("validation_model_dates") == expected_split["validation_model_dates"], split_value.get("validation_model_dates")),
        _check("calibration_dates_match", split_value.get("calibration_dates") == expected_split["calibration_dates"], split_value.get("calibration_dates")),
        _check("final_dates_match", split_value.get("final_test_dates") == expected_split["final_test_dates"], split_value.get("final_test_dates")),
        _check("legacy_not_untouched", split["rules"].get("legacy_20161031_is_untouched_test") is False, split["rules"].get("legacy_20161031_role")),
        _check("main_fit_read_no_final_rows", int(model_manifest.get("new_final_test_rows_read", -1)) == 0, model_manifest.get("new_final_test_rows_read")),
    ]
    temporal_status = "PASS" if all(item["status"] == "PASS" for item in temporal_checks) else "FAIL"

    quantiles = pd.read_csv(docs / "quantile_calibration.csv")
    validation_quantiles = quantiles[quantiles["split"].eq("validation_model")]
    maximum_quantile_error = float(validation_quantiles["coverage_error"].abs().max())
    admission = config.section("admission")
    engineering_checks = [
        _check(
            "service_time_target_audit",
            service.get("engineering_status") == "PASS",
            service.get("engineering_status"),
        ),
        _check("prediction_merge", prediction_manifest.get("status") == "PASS", prediction_manifest.get("status")),
        _check("model_fit", model_manifest.get("status") == "PASS", model_manifest.get("status")),
        _check("horizon_ablation_complete", ablation.get("status") == "PASS", ablation.get("status")),
        _check("horizon_gate_selected", ablation.get("selected_history_mode") == "horizon_gate", ablation.get("selected_history_mode")),
        _check("trained_distribution_scale", bool(model_manifest.get("checkpoint_sha256")), model_manifest.get("checkpoint_sha256")),
        _check(
            "pace_quantile_calibration_development",
            maximum_quantile_error <= float(admission["maximum_pace_quantile_coverage_error"]),
            maximum_quantile_error,
        ),
        _check("scenario_selected", scenario.get("selected_model") in {"shared_route_latent", "residual_block"}, scenario.get("selected_model")),
        _check("v4_v5_state_same_row_evaluation", state_targets.get("status") == "PASS", state_targets.get("status")),
        _check("regression_tests", test_results.get("status") == "PASS", test_results.get("pytest")),
    ]
    engineering_status = "PASS" if all(item["status"] == "PASS" for item in engineering_checks) else "FAIL"

    performance_checks = [
        _check("static_complexity", static.get("status") == "PASS" and int(static.get("blocking_finding_count", -1)) == 0, static.get("blocking_finding_count")),
        _check("micro_benchmarks", benchmark.get("status") == "PASS", benchmark.get("status")),
        _check("layered_preflight", preflight.get("performance_gate_status") == "PASS", preflight.get("performance_gate_status")),
        _check(
            "peak_rss",
            float(preflight.get("maximum_peak_rss_mb", np.inf)) <= float(config.section("performance")["maximum_peak_rss_mb"]),
            preflight.get("maximum_peak_rss_mb"),
        ),
    ]
    performance_status = "PASS" if all(item["status"] == "PASS" for item in performance_checks) else "FAIL"

    development_scientific_checks = [
        _check("same_set_baseline_complete", baseline.get("best_validation_model") == "hist_gradient_boosting", baseline.get("best_validation_model")),
        _check("strong_baseline_beaten", model.get("status") == "PREDICTIVE_BASELINE_VALIDATED", model.get("status")),
        _check("paired_bootstrap", model.get("validation_paired_ci_all_below_zero") is True, model.get("validation_paired_ci_all_below_zero")),
    ]
    development_scientific_status = (
        model.get("status", "BASELINE_NOT_BEATEN")
        if all(item["status"] == "PASS" for item in development_scientific_checks)
        else "BASELINE_NOT_BEATEN"
    )

    final_path = docs / "stage2_v5_final_test_results.json"
    final_result = _json(final_path) if final_path.is_file() else None
    final_checks: list[dict[str, Any]] = []
    if final_result is not None:
        final_checks = [
            _check("final_test_protocol", final_result.get("protocol") == "one_shot_preregistered", final_result.get("protocol")),
            _check("final_test_dates", final_result.get("dates") == expected_split["final_test_dates"], final_result.get("dates")),
            _check("final_test_day_count", int(final_result.get("day_count", 0)) >= int(admission["minimum_final_test_day_count"]), final_result.get("day_count")),
            _check("final_aggregate_mae", final_result.get("aggregate_mae_better_than_strong_baseline") is True, final_result.get("aggregate_relative_mae_change")),
            _check("final_daily_mae_wins", int(final_result.get("daily_mae_wins", 0)) >= int(admission["minimum_final_daily_mae_wins_vs_strong_baseline"]), final_result.get("daily_mae_wins")),
            _check("final_paired_bootstrap", final_result.get("paired_bootstrap_ci_below_zero") is True, final_result.get("paired_bootstrap_ci95")),
            _check("final_route_scenario_coverage", final_result.get("route_scenario_coverage_acceptable") is True, final_result.get("route_scenario_coverage")),
            _check("no_final_tuning", final_result.get("post_test_tuning_count") == 0, final_result.get("post_test_tuning_count")),
        ]
    final_test_status = (
        "PASS" if final_checks and all(item["status"] == "PASS" for item in final_checks)
        else "FAIL" if final_checks
        else "PENDING_UPSTREAM_PRODUCTS"
    )

    foundation_pass = engineering_status == temporal_status == performance_status == "PASS"
    if foundation_pass and final_test_status == "PASS":
        stage3_admission = "READY_FOR_STAGE3"
    elif foundation_pass and development_scientific_status == "PREDICTIVE_BASELINE_VALIDATED":
        stage3_admission = "READY_FOR_ROUTE_SCENARIO_PROTOTYPE"
    else:
        stage3_admission = "NOT_READY"

    report = {
        "schema_version": "stage2_v5_final_verification.1",
        "engineering_status": engineering_status,
        "temporal_contract_status": temporal_status,
        "performance_gate_status": performance_status,
        "development_scientific_status": development_scientific_status,
        "final_test_status": final_test_status,
        "stage3_admission_status": stage3_admission,
        "checks": {
            "engineering": engineering_checks,
            "temporal": temporal_checks,
            "performance": performance_checks,
            "development_scientific": development_scientific_checks,
            "final_test": final_checks,
        },
        "identities": {
            "config_sha256": _sha256(config_path),
            "split_freeze_sha256": _sha256(docs / "stage2_v5_split_freeze.json"),
            "performance_benchmark_sha256": _sha256(docs / "stage2_v5_performance_benchmarks.json"),
            "profile_sha256": _sha256(docs / "performance_profile_hotspots.txt"),
            "selection_model_id": model_manifest.get("model_id"),
            "distribution_model_id": model_manifest.get("model_id"),
            "scenario_generator_id": scenario.get("generator_id"),
            "scenario_seed": scenario.get("scenario_seed"),
        },
    }
    (docs / "stage2_v5_final_verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    final_lines = [
        "# Stage 2 v5 final evaluation",
        "",
        f"- Engineering: `{engineering_status}`",
        f"- Temporal contract: `{temporal_status}`",
        f"- Performance gate: `{performance_status}`",
        f"- Development scientific status: `{development_scientific_status}`",
        f"- Preregistered 20161028–30 final test: `{final_test_status}`",
        f"- Stage 3 admission: `{stage3_admission}`",
        "",
    ]
    if final_result is None:
        final_lines += [
            "20161028–30 的原始归档存在，但冻结的 Stage 0/1 上游产品尚未物化，因此尚未读取 final-test 标签。",
            "当前状态只允许路线情景原型验证，不构成 Stage 2 v5 最终冻结或正式 Stage 3 准入。",
        ]
    else:
        final_lines += ["最终测试结果由 `stage2_v5_final_test_results.json` 逐项计算并纳入上述准入状态。"]
    (docs / "stage2_v5_final_evaluation.md").write_text("\n".join(final_lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    report = verify(repo_root=args.repo_root)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["stage3_admission_status"] != "NOT_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
