"""Compute Stage 2 v5 engineering, temporal, scientific, and admission states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    print(json.dumps(verify(repo_root=args.repo_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
