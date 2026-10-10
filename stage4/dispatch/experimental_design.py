"""Freeze the compact Stage4-S5B experimental design without dispatch runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from stage4.fleetpy_adapter.upstream import FleetPyCompatibilityError

from .acceptance import passenger_acceptance
from .parameterization_diagnostics import fleet_vehicle_hour_scenarios

CANONICAL_COMMIT = "18d03a9df87eb518689f77346cb080ab84f2f402"
FLEETPY_COMMIT = "0379f9725a147ff33c674de4884cdf89fd787fa9"
OUTPUT_REL = Path("stage4/config/experimental_design")
DOC_REL = Path("stage4/docs/experimental_design")
ACCEPTANCE_SEED = 20260827
H_BASE_EXACT_EXPECTED = 12279.336388888889
MAIN_Q_LEVELS = (0.25, 0.50, 0.75)
PROFILE_LEVELS = ("C", "M", "A")
ACCEPTANCE_LEVELS = (0.40, 0.70, 1.00)
ETA_LEVELS = (0.50, 0.75, 1.00, 1.25)
EPSILON_LEVELS = (0.00, 0.05)
FAMILIES = ("static", "dynamic", "speed")
GAMMA_PRESETS: dict[str, dict[str, float | None]] = {
    "STRICT": {"static": 0.0, "dynamic": 0.0, "speed": 0.0},
    "REFERENCE": {
        "static": 2.145067625590382,
        "dynamic": 0.14934256810554178,
        "speed": 0.0,
    },
    "UNCONSTRAINED": {"static": None, "dynamic": None, "speed": None},
}
PATH_DIAGNOSTIC = {
    "static": 2.2008744038155803,
    "dynamic": 0.4011750995716139,
    "speed": 0.0,
    "dynamic_path_max_assignment_rank": 1,
    "dynamic_path_max_assignment_time": "2016-10-31T08:01:00+08:00",
}
REGISTRY_COLUMNS = (
    "scenario_id",
    "experiment_block",
    "requested_q_A",
    "profile_id",
    "acceptance_probability",
    "acceptance_seed",
    "gamma_policy",
    "gamma_static",
    "gamma_dynamic",
    "gamma_speed",
    "cost_enabled",
    "eta_cost_av_to_hv",
    "pickup_cost_epsilon",
    "benchmark_flag",
    "reuse_source_scenario_id",
    "scientific_role",
    "H_base_exact",
    "achieved_q_A",
    "AV_vehicle_count",
    "target_HV_vehicle_hours",
    "achieved_HV_vehicle_hours",
    "HV_vehicle_hour_error_pct",
    "selected_HV_session_count",
)
CONFIG_SIGNATURE_COLUMNS = (
    "requested_q_A",
    "profile_id",
    "acceptance_probability",
    "acceptance_seed",
    "gamma_static",
    "gamma_dynamic",
    "gamma_speed",
    "cost_enabled",
    "eta_cost_av_to_hv",
    "pickup_cost_epsilon",
)


def _q_code(value: float) -> str:
    return f"{int(round(100 * value)):02d}"


def _p_code(value: float) -> str:
    return f"{int(round(100 * value)):02d}"


def _eta_code(value: float) -> str:
    return f"{int(round(100 * value)):03d}"


def _epsilon_code(value: float) -> str:
    return f"{int(round(100 * value)):03d}"


def _fleet_accounting(root: Path) -> dict[float, dict[str, Any]]:
    frame = fleet_vehicle_hour_scenarios(root, q_levels=(0.0, 0.25, 0.50, 0.75, 1.0))
    h_exact = float(frame.iloc[0]["H_base_exact"])
    if abs(h_exact - H_BASE_EXACT_EXPECTED) > 1e-6:
        raise FleetPyCompatibilityError(
            f"S5B requires frozen H_base_exact, observed {h_exact}"
        )
    if (frame["H_base_15min_equivalent"] == frame["H_base_exact"]).any():
        raise FleetPyCompatibilityError("q_A must not use 15-minute equivalent hours")
    return {
        float(row.requested_q_A): {
            "H_base_exact": float(row.H_base_exact),
            "achieved_q_A": float(row.achieved_q_A),
            "AV_vehicle_count": int(row.AV_vehicle_count),
            "target_HV_vehicle_hours": float(row.target_HV_vehicle_hours),
            "achieved_HV_vehicle_hours": float(row.achieved_HV_vehicle_hours),
            "HV_vehicle_hour_error_pct": float(row.HV_vehicle_hour_error_pct),
            "selected_HV_session_count": int(row.selected_HV_session_count),
        }
        for row in frame.itertuples(index=False)
    }


def _scenario(
    accounting: dict[float, dict[str, Any]],
    *,
    scenario_id: str,
    block: str,
    q_a: float,
    profile: str,
    p_a: float,
    gamma_policy: str = "UNCONSTRAINED",
    cost_enabled: bool = False,
    eta: float = 1.0,
    epsilon: float = 0.0,
    benchmark: bool = False,
    reuse: str = "",
    role: str,
) -> dict[str, Any]:
    gamma = GAMMA_PRESETS[gamma_policy]
    return {
        "scenario_id": scenario_id,
        "experiment_block": block,
        "requested_q_A": q_a,
        "profile_id": profile,
        "acceptance_probability": p_a,
        "acceptance_seed": ACCEPTANCE_SEED,
        "gamma_policy": gamma_policy,
        "gamma_static": gamma["static"],
        "gamma_dynamic": gamma["dynamic"],
        "gamma_speed": gamma["speed"],
        "cost_enabled": cost_enabled,
        "eta_cost_av_to_hv": eta,
        "pickup_cost_epsilon": epsilon,
        "benchmark_flag": benchmark,
        "reuse_source_scenario_id": reuse,
        "scientific_role": role,
        **accounting[q_a],
    }


def build_scenario_registry(root: str | Path) -> list[dict[str, Any]]:
    """Return the frozen S5B registry; this function never launches FleetPy."""
    root = Path(root).resolve()
    accounting = _fleet_accounting(root)
    rows: list[dict[str, Any]] = []
    for q_a in MAIN_Q_LEVELS:
        for profile in PROFILE_LEVELS:
            for p_a in ACCEPTANCE_LEVELS:
                rows.append(
                    _scenario(
                        accounting,
                        scenario_id=f"MAIN_Q{_q_code(q_a)}_{profile}_P{_p_code(p_a)}",
                        block="MAIN_STRUCTURAL",
                        q_a=q_a,
                        profile=profile,
                        p_a=p_a,
                        role="mixed_fleet_transition",
                    )
                )
    rows.append(
        _scenario(
            accounting,
            scenario_id="BENCH_HV",
            block="BENCHMARK",
            q_a=0.0,
            profile="M",
            p_a=1.0,
            benchmark=True,
            role="all_hv_benchmark_profile_and_acceptance_inactive",
        )
    )
    for profile in PROFILE_LEVELS:
        rows.append(
            _scenario(
                accounting,
                scenario_id=f"BENCH_AV_{profile}",
                block="BENCHMARK",
                q_a=1.0,
                profile=profile,
                p_a=1.0,
                benchmark=True,
                role="all_av_capability_upper_bound",
            )
        )
    central = "MAIN_Q50_M_P70"
    for policy in ("STRICT", "REFERENCE"):
        rows.append(
            _scenario(
                accounting,
                scenario_id=f"ODD_Q50_M_P70_{policy}",
                block="ODD_POLICY",
                q_a=0.50,
                profile="M",
                p_a=0.70,
                gamma_policy=policy,
                role="joint_three_family_exposure_policy",
            )
        )
    rows.append(
        _scenario(
            accounting,
            scenario_id="ODD_Q50_M_P70_UNCONSTRAINED",
            block="ODD_POLICY",
            q_a=0.50,
            profile="M",
            p_a=0.70,
            reuse=central,
            role="unconstrained_odd_policy_reference_reused_from_main",
        )
    )
    for eta in ETA_LEVELS:
        for epsilon in EPSILON_LEVELS:
            rows.append(
                _scenario(
                    accounting,
                    scenario_id=(
                        f"COST_Q50_M_P70_ETA{_eta_code(eta)}_"
                        f"EPS{_epsilon_code(epsilon)}"
                    ),
                    block="COST_ROBUSTNESS",
                    q_a=0.50,
                    profile="M",
                    p_a=0.70,
                    cost_enabled=True,
                    eta=eta,
                    epsilon=epsilon,
                    role="normalized_operating_cost_robustness",
                )
            )
    validate_registry(rows)
    return rows


def configuration_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[column] for column in CONFIG_SIGNATURE_COLUMNS)


def validate_registry(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    rows = list(rows)
    by_id: dict[str, dict[str, Any]] = {}
    unique_signatures: dict[tuple[Any, ...], str] = {}
    unique_runs = 0
    reuse_rows = 0
    for row in rows:
        scenario_id = str(row["scenario_id"])
        if scenario_id in by_id:
            raise FleetPyCompatibilityError(f"duplicate scenario_id: {scenario_id}")
        signature = configuration_signature(row)
        reuse = str(row["reuse_source_scenario_id"] or "")
        if reuse:
            if reuse not in by_id:
                raise FleetPyCompatibilityError(
                    f"reuse source must precede scenario: {scenario_id} -> {reuse}"
                )
            if configuration_signature(by_id[reuse]) != signature:
                raise FleetPyCompatibilityError(
                    f"reuse source configuration mismatch: {scenario_id}"
                )
            reuse_rows += 1
        else:
            if signature in unique_signatures:
                raise FleetPyCompatibilityError(
                    f"duplicate full configuration without reuse: {scenario_id} and "
                    f"{unique_signatures[signature]}"
                )
            unique_signatures[signature] = scenario_id
            unique_runs += 1
        by_id[scenario_id] = row
    if unique_runs > 45:
        raise FleetPyCompatibilityError("S5B exceeds 45 unique dispatch scenarios")
    return {
        "registry_rows": len(rows),
        "unique_dispatch_scenarios": unique_runs,
        "reuse_rows": reuse_rows,
    }


def acceptance_design() -> dict[str, Any]:
    synthetic_orders = [f"S5B_CRN_{index:04d}" for index in range(2048)]
    accepted = {
        str(rate): {
            order_id
            for order_id in synthetic_orders
            if passenger_acceptance(
                order_id, rate, ACCEPTANCE_SEED
            ).passenger_accepts_av
        }
        for rate in ACCEPTANCE_LEVELS
    }
    nested = accepted["0.4"] <= accepted["0.7"] <= accepted["1.0"]
    if not nested:
        raise FleetPyCompatibilityError(
            "common-random-number acceptance nesting failed"
        )
    return {
        "schema_version": "stage4_s5b_acceptance_v1",
        "acceptance_seed": ACCEPTANCE_SEED,
        "probability_levels": list(ACCEPTANCE_LEVELS),
        "probability_parameter": "p_A",
        "realized_indicator": "a_o^A = 1(u_o <= p_A)",
        "uniform_draw": "SHA256(f'{seed}|{order_id}') first 8 bytes / 2**64",
        "common_random_numbers": True,
        "nested_acceptance_required": True,
        "synthetic_construction_check_order_count": len(synthetic_orders),
        "synthetic_construction_check_nested": nested,
        "interpretation": {
            "0.4": "conservative_low_acceptance_scenario_not_xian_estimate",
            "0.7": "literature_anchored_central_scenario_not_xian_estimate",
            "1.0": "all_accept_upper_bound",
        },
    }


def gamma_policy_design() -> dict[str, Any]:
    if any(set(vector) != set(FAMILIES) for vector in GAMMA_PRESETS.values()):
        raise FleetPyCompatibilityError("Gamma policy dropped an exposure family")
    return {
        "schema_version": "stage4_s5b_gamma_policy_v1",
        "families": list(FAMILIES),
        "near_binding_tolerance": 1e-7,
        "policies": GAMMA_PRESETS,
        "path_diagnostic_only": PATH_DIAGNOSTIC,
        "path_exclusion_reason": (
            "dynamic PATH is determined by assignment rank 1 and is an "
            "initialization envelope, not the primary long-run policy anchor"
        ),
        "interpretation": (
            "reference-envelope exposure budget, not a safety, failure, or legal threshold"
        ),
        "required_wording": (
            "The model retains all three operational-envelope dimensions, while "
            "their empirical activity and binding relevance are allowed to vary by "
            "capability profile and operating condition."
        ),
    }


def kpi_design() -> dict[str, Any]:
    return {
        "schema_version": "stage4_s5b_kpi_v1",
        "passenger_service": [
            "request_count",
            "matched",
            "completed",
            "patience_expired",
            "service_rate",
            "first_window_match_rate",
            "carry_over_entry_rate",
            "carry_over_recovery_rate",
            "critical_recovery",
        ],
        "waiting_seconds": [
            "request_to_pickup_mean",
            "request_to_pickup_p50",
            "request_to_pickup_p90",
            "request_to_pickup_p95",
        ],
        "fleet_utilization": [
            "HV_assignments",
            "AV_assignments",
            "HV_assignment_share",
            "AV_assignment_share",
            "available_HV_supply_diagnostics",
            "available_AV_supply_diagnostics",
        ],
        "matching_computation": [
            "mean_attempts",
            "expanded_radius_match_share",
            "candidate_spatial_pairs",
            "candidate_topk_pairs",
            "valid_or_arcs",
            "routing_workload",
            "solver_workload",
            "runtime_s",
        ],
        "acceptance_realization": [
            "target_p_A",
            "realized_accepted_order_share",
            "accepted_order_count",
        ],
        "exposure_for_each_family": [
            "mean_assigned_exposure",
            "positive_assigned_exposure_share",
            "final_cumulative_mean_exposure",
            "maximum_cumulative_mean_exposure",
        ],
        "gamma_enabled_for_each_family": [
            "binding_epoch_count",
            "near_binding_epoch_count",
            "minimum_slack",
            "mean_slack",
        ],
        "cost_enabled": [
            "normalized_operating_cost_total",
            "normalized_operating_cost_per_matched_order",
            "HV_equivalent_operating_seconds",
            "pickup_ETA_objective_value",
            "relative_pickup_ETA_degradation_vs_epsilon_0_reference",
        ],
        "prohibitions": [
            "no_collapsed_risk_or_safety_score",
            "no_currency_conversion",
        ],
    }


def comparison_design() -> dict[str, Any]:
    return {
        "schema_version": "stage4_s5b_comparison_v1",
        "research_questions": {
            "RQ1": "mixed_fleet_transition",
            "RQ2": "joint_odd_exposure_control",
            "RQ3": "service_cost_tradeoff_under_lexicographic_protection",
        },
        "predefined_contrasts": [
            "q_A effect at fixed profile_id and p_A",
            "profile_id effect at fixed q_A and p_A",
            "p_A effect at fixed q_A and profile_id",
        ],
        "interaction_patterns": ["q_A x profile_id", "q_A x p_A", "profile_id x p_A"],
        "primary_analysis": ["scenario contrasts", "tables", "response surfaces"],
        "no_automatic_complex_interaction_model": True,
        "profile_family_activity_required": True,
    }


def compute_budget(counts: dict[str, int]) -> dict[str, Any]:
    unique = counts["unique_dispatch_scenarios"]
    one_hour_low_s, one_hour_high_s = 110.0, 125.0
    return {
        "schema_version": "stage4_s5b_compute_budget_v1",
        **counts,
        "one_hour_runtime_per_scenario_s": {
            "low": one_hour_low_s,
            "high": one_hour_high_s,
        },
        "one_hour_matrix_sequential_hours": {
            "low": unique * one_hour_low_s / 3600.0,
            "high": unique * one_hour_high_s / 3600.0,
        },
        "full_day_linear_extrapolation_sequential_hours": {
            "low": unique * one_hour_low_s * 24.0 / 3600.0,
            "high": unique * one_hour_high_s * 24.0 / 3600.0,
        },
        "recommended_parallel_scenario_slots": 4,
        "full_day_ideal_four_slot_wall_clock_hours": {
            "low": unique * one_hour_low_s * 24.0 / 3600.0 / 4.0,
            "high": unique * one_hour_high_s * 24.0 / 3600.0 / 4.0,
        },
        "planning_caveat": (
            "linear estimate only; routing cache reuse and Valhalla contention may change runtime"
        ),
        "execution_architecture": (
            "scenario-level parallelism only; cKDTree, Top-K 20, Valhalla cache/fallback, "
            "CSR sparse MILP, CPU-only, no dense order-by-fleet matrix"
        ),
        "matrix_execution_authorized": False,
    }


def _write_json(value: Any, path: Path) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_registry(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(root: Path, counts: dict[str, int], budget: dict[str, Any]) -> None:
    lines = [
        "# Stage4 S5B Experimental Design Freeze",
        "",
        "Recommendation: `GO_STAGE4_FINAL_EXPERIMENTS`",
        "",
        "## Canonical scientific base",
        "",
        f"- Commit: `{CANONICAL_COMMIT}`.",
        f"- FleetPy: `{FLEETPY_COMMIT}`.",
        f"- `H_base_exact = {H_BASE_EXACT_EXPECTED:.6f}` vehicle-hours.",
        "- `q_A = 24 N_AV / H_base_exact`; the 15-minute supply-profile equivalent is not a penetration denominator.",
        "",
        "## Main structural experiment",
        "",
        "- `q_A in {0.25, 0.50, 0.75}`.",
        "- Capability profile `k in {C, M, A}`.",
        "- Acceptance probability `p_A in {0.40, 0.70, 1.00}`.",
        "- 27 structural scenarios, plus one all-HV and three all-AV benchmarks.",
        "",
        "## Acceptance semantics",
        "",
        f"- Common-random-number seed: `{ACCEPTANCE_SEED}`.",
        "- `p_A` is a probabilistic scenario parameter.",
        "- `a_o^A = 1(u_o <= p_A)` is the realized binary acceptance indicator.",
        "- The same SHA-256 order draw is reused across scenarios, so acceptance sets are nested.",
        "",
        "## ODD exposure policies",
        "",
        "- STRICT: `(static, dynamic, speed) = (0, 0, 0)`.",
        "- REFERENCE: `(2.145068, 0.149343, 0)`.",
        "- UNCONSTRAINED: `(null, null, null)`.",
        "- PATH `(2.200874, 0.401175, 0)` remains diagnostic only because dynamic PATH is fixed by assignment rank 1 at 08:01.",
        "",
        "## Three-family model status",
        "",
        "Static, dynamic, and speed are all retained. Family activity is data-, profile-, and operating-condition-dependent; speed is not deleted or independently swept under Profile M.",
        "",
        "## Cost robustness",
        "",
        "- `eta_cost_av_to_hv in {0.50, 0.75, 1.00, 1.25}`.",
        "- `epsilon_W in {0, 0.05}`; 5% is a platform-policy sensitivity, not passenger behavior.",
        "- Normalized operating time only; no monetary calibration is claimed.",
        "",
        "## Scenario count and compute budget",
        "",
        f"- Registry rows: {counts['registry_rows']}; reused configurations: {counts['reuse_rows']}.",
        f"- Unique dispatch scenarios: {counts['unique_dispatch_scenarios']} (hard cap 45).",
        f"- One-hour sequential estimate: {budget['one_hour_matrix_sequential_hours']['low']:.2f}-{budget['one_hour_matrix_sequential_hours']['high']:.2f} hours.",
        f"- Full-day linear sequential estimate: {budget['full_day_linear_extrapolation_sequential_hours']['low']:.2f}-{budget['full_day_linear_extrapolation_sequential_hours']['high']:.2f} hours.",
        f"- Four-slot ideal full-day wall-clock estimate: {budget['full_day_ideal_four_slot_wall_clock_hours']['low']:.2f}-{budget['full_day_ideal_four_slot_wall_clock_hours']['high']:.2f} hours, subject to Valhalla contention.",
        "- S5B launched no FleetPy run and no scenario matrix.",
        "",
    ]
    report = root / DOC_REL / "stage4_s5b_experimental_design_summary.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def build_experimental_design(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    output = root / OUTPUT_REL
    output.mkdir(parents=True, exist_ok=True)
    rows = build_scenario_registry(root)
    counts = validate_registry(rows)
    acceptance = acceptance_design()
    gamma = gamma_policy_design()
    kpis = kpi_design()
    comparisons = comparison_design()
    budget = compute_budget(counts)
    _write_registry(rows, output / "scenario_registry.csv")
    _write_json(gamma, output / "gamma_policy_presets.json")
    _write_json(acceptance, output / "acceptance_design.json")
    _write_json(kpis, output / "kpi_schema.json")
    _write_json(comparisons, output / "comparison_plan.json")
    _write_json(budget, output / "compute_budget.json")
    _write_report(root, counts, budget)
    return {
        "phase_status": "STAGE4_S5B_EXPERIMENTAL_DESIGN_FROZEN",
        "recommendation": "GO_STAGE4_FINAL_EXPERIMENTS",
        "canonical_commit": CANONICAL_COMMIT,
        "main_structural_scenarios": 27,
        "benchmark_scenarios": 4,
        "odd_policy_additional_scenarios": 2,
        "cost_scenarios": 8,
        **counts,
        "acceptance_nested": acceptance["synthetic_construction_check_nested"],
        "three_family_retained": set(gamma["families"]) == set(FAMILIES),
        "fleetpy_launched": False,
        "scenario_matrix_launched": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(build_experimental_design(args.root), indent=2))


if __name__ == "__main__":
    main()
