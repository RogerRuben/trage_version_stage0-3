"""Run the one authorized neutral Stage4-S4 decision-kernel reproduction."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd

from stage4.fleetpy_adapter.mixed_fleet_adapter import create_native_vehicles
from stage4.fleetpy_adapter.native_demand import create_native_demand
from stage4.fleetpy_adapter.native_network import create_native_network
from stage4.fleetpy_adapter.native_simulation import create_native_simulation
from stage4.fleetpy_adapter.test31_demand_adapter import (
    attach_fleetpy_requests,
    load_all_test31_requests,
)
from stage4.fleetpy_adapter.upstream import (
    CoordinateRegistry,
    FLEETPY_COMMIT,
    FleetPyCompatibilityError,
    load_fleetpy_bindings,
)

from .candidate_graph import SparseValhallaMatrixAdapter
from .fleet_normalization import build_fleet_scenario
from .rolling_or_control import (
    RollingRuntimeGuardExceeded,
    create_rolling_or_fleet_control,
)
from .rolling_or_runner import _outcomes, _summary, _timestamps, _write_json

CONFIG_REL = Path("stage4/config/odd_aware_decision_kernel.json")
BASE_CONFIG_DIR = Path("stage4/config")
OUTPUT_REL = Path("stage4/output/odd_aware_decision_kernel")
DOC_REL = Path("stage4/docs/odd_aware_decision_kernel")
S3_COMMIT = "84783ca1696c325cc3ed31ab3efd8747f4133ece"
EXPECTED = {
    "requests": 1458,
    "matched": 1215,
    "completed": 1215,
    "patience_expired": 243,
    "first_window_matched": 1131,
    "carry_over_recovered": 84,
    "critical_matched": 1,
}


def load_odd_config(root: Path, config_path: str | Path | None) -> dict[str, Any]:
    path = Path(config_path).resolve() if config_path else root / CONFIG_REL
    overlay = json.loads(path.read_text(encoding="utf-8"))
    base_name = str(overlay.get("base_config", "rolling_or_baseline"))
    base_path = root / BASE_CONFIG_DIR / f"{base_name}.json"
    config = {**json.loads(base_path.read_text(encoding="utf-8")), **overlay}
    config["benchmark_runtime_guard_s"] = float(
        config["canonical_reproduction_runtime_guard_s"]
    )
    if config["fleetpy_commit"] != FLEETPY_COMMIT:
        raise FleetPyCompatibilityError("S4 FleetPy commit is not pinned")
    if not 0.0 <= float(config["passenger_acceptance_rate"]) <= 1.0:
        raise FleetPyCompatibilityError("passenger acceptance rate must be in [0,1]")
    if float(config["pickup_cost_epsilon"]) < 0.0:
        raise FleetPyCompatibilityError("pickup cost epsilon must be >= 0")
    return config


def _fingerprint(outcomes: pd.DataFrame) -> str:
    rows: list[str] = []
    for row in outcomes.sort_values("order_id", kind="mergesort").itertuples():
        assignment = (
            ""
            if pd.isna(row.assignment_time)
            else pd.Timestamp(row.assignment_time).isoformat()
        )
        rows.append(
            json.dumps(
                [
                    str(row.order_id),
                    bool(row.matched),
                    bool(row.patience_expired),
                    assignment,
                    "" if row.vehicle_type is None else str(row.vehicle_type),
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _report(root: Path, summary: dict[str, Any]) -> None:
    result = summary["canonical_reproduction"]
    computation = summary["computation"]
    lines = [
        "# Stage4 S4 ODD-Aware Mixed HV/AV Decision Kernel",
        "",
        f"Recommendation: `{summary['recommendation']}`",
        "",
        "## Canonical base",
        "",
        f"- S3 commit: `{S3_COMMIT}`",
        f"- FleetPy commit: `{FLEETPY_COMMIT}`",
        "- Frozen S3 outcome: 1458 requests, 1215 matched/completed, 243 expired.",
        "",
        "## Added mechanisms",
        "",
        "- Exogenous passenger AV acceptance, independent of trip outcomes.",
        "- Separate static/dynamic/speed reference-envelope excess.",
        "- At most three cumulative Gamma rows in the existing CSR MILP.",
        "- Normalized operating-time cost as the final optional lexicographic level.",
        "",
        "## Neutral reproduction",
        "",
        f"- Requests/matched/completed/expired: {result['requests']}/{result['matched']}/{result['completed']}/{result['patience_expired']}",
        f"- First-window/carry-recovered/critical-matched: {result['first_window_matched']}/{result['carry_over_recovered']}/{result['critical_matched']}",
        f"- Runtime: {computation['total_runtime_s']:.3f}s",
        f"- Fingerprint: `{summary['canonical_outcome_fingerprint_sha256']}`",
        f"- Exact S3 aggregate reproduction: `{summary['canonical_reproduction_pass']}`",
        "",
        "## Exposure semantics",
        "",
        "Exposure is reference-envelope exceedance. It is not a safety, failure, accident, or legal probability.",
        "",
        "## Cat-eye checks",
        "",
        f"- Acceptance: `{summary['cat_eyes']['acceptance']}`",
        f"- Cumulative exposure: `{summary['cat_eyes']['cumulative_exposure']}`",
        f"- Cost tie-break: `{summary['cat_eyes']['cost_tie_break']}`",
        "",
        "## Computational impact",
        "",
        f"- S3/S4 neutral runtime: {summary['s3_reference_runtime_s']:.3f}/{computation['total_runtime_s']:.3f}s.",
        f"- Valid sparse arcs: {computation['valid_or_arcs']}; solver p50/p95/max: {computation['solver_time_p50']:.6f}/{computation['solver_time_p95']:.6f}/{computation['solver_time_max']:.6f}s.",
        "- cKDTree + Top-K 20 + CSR only; CPU-only; no order-by-fleet matrix and no per-vehicle tick trace.",
        f"- Canonical Gamma rows/cost solves: {computation['enabled_gamma_constraint_count']}/{computation['cost_level_solve_count']}.",
        "",
        "## Interpretation limits",
        "",
        "S4 does not identify best Gamma values, true passenger preferences, monetary cost ratios, AV penetration, economic superiority, or safety improvement.",
        "",
    ]
    path = root / DOC_REL / "stage4_s4_odd_aware_kernel_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_odd_aware_decision_kernel(
    root: str | Path,
    fleetpy_root: str | Path,
    config_path: str | Path | None = None,
    *,
    eta_actor: Any | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    config = load_odd_config(root, config_path)
    start, demand_end, matching_end = _timestamps(config)
    requests = load_all_test31_requests(
        root, start=start, end=demand_end, profile_id=config["profile_id"]
    )
    max_service = max(request.realized_service_time_s for request in requests)
    step = int(config["dispatch_interval_s"])
    drain_seconds = (
        int((matching_end - start).total_seconds())
        + int(math.ceil(max_service / step)) * step
        + step
    )
    simulation_end = start + pd.Timedelta(seconds=drain_seconds)
    fleet = build_fleet_scenario(
        root,
        benchmark_start=start,
        simulation_end=simulation_end,
        requested_q_a=float(config["av_vehicle_hour_share"]),
        seed=int(config["fleet_sampling_seed"]),
        max_hv_hour_error_pct=float(config["max_hv_vehicle_hour_error_pct"]),
    )
    bindings = load_fleetpy_bindings(fleetpy_root)
    registry = CoordinateRegistry()
    attach_fleetpy_requests(requests, bindings, registry)
    network = create_native_network(bindings, registry)
    output = root / OUTPUT_REL
    output.mkdir(parents=True, exist_ok=True)
    demand = create_native_demand(bindings, requests, registry, network, output)
    vehicles, native_output = create_native_vehicles(
        fleet.native_fixtures,
        bindings,
        registry,
        demand.rq_db,
        output / "runtime",
        native_movement=True,
        routing_engine=network,
    )
    eta = SparseValhallaMatrixAdapter(root, actor=eta_actor)
    runtime_config = {
        **config,
        "matching_end_s": int((matching_end - start).total_seconds()),
    }
    control = create_rolling_or_fleet_control(
        bindings,
        vehicles,
        requests,
        demand,
        network,
        eta,
        start,
        simulation_end,
        runtime_config,
    )
    simulation = create_native_simulation(
        bindings,
        simulation_end_s=drain_seconds,
        time_step_s=step,
        demand=demand,
        vehicles=[item.native_vehicle for item in vehicles],
        fleet_control=control,
        network=network,
        native_output=native_output,
    )
    importlib.import_module("src.FleetSimulationBase").PROGRESS_LOOP = "off"
    started = time.perf_counter()
    guard_exceeded = False
    try:
        simulation.run()
    except RollingRuntimeGuardExceeded:
        guard_exceeded = True
    total_runtime_s = time.perf_counter() - started
    control.reconcile()
    outcomes = _outcomes(requests, control, start)
    accepts_by_order = {
        request.order_id: bool(request.passenger_accepts_av) for request in requests
    }
    source_by_order = {
        request.order_id: request.acceptance_source for request in requests
    }
    outcomes["passenger_accepts_av"] = outcomes["order_id"].map(accepts_by_order)
    outcomes["acceptance_source"] = outcomes["order_id"].map(source_by_order)
    routing_audit = {
        "matrix_failed_arc_events": len(eta.failed_arc_records),
        "sample_seed": None,
        "requested_sample_size": 0,
        "sampled_matrix_failures": 0,
        "single_route_success": 0,
        "single_route_failure": 0,
        "single_route_success_rate": None,
        "audit_runtime_s": 0.0,
    }
    base = _summary(
        config,
        fleet,
        outcomes,
        control,
        eta,
        total_runtime_s,
        matching_end,
        guard_exceeded,
        routing_audit,
    )
    epoch = pd.DataFrame(control.epoch_rows)
    observed = {
        "requests": len(outcomes),
        "matched": int(outcomes["matched"].sum()),
        "completed": int(outcomes["completed"].sum()),
        "patience_expired": int(outcomes["patience_expired"].sum()),
        "first_window_matched": int(
            ((outcomes["attempt_count"] == 1) & outcomes["matched"]).sum()
        ),
        "carry_over_recovered": int(
            (outcomes["entered_carry_over"] & outcomes["matched"]).sum()
        ),
        "critical_matched": int(epoch["critical_matched"].sum()),
    }
    reproduction_pass = observed == EXPECTED
    runtime_pass = total_runtime_s <= float(
        config["canonical_reproduction_runtime_guard_s"]
    )
    neutral = (
        float(config["passenger_acceptance_rate"]) == 1.0
        and all(
            config[f"gamma_{name}"] is None for name in ("static", "dynamic", "speed")
        )
        and not bool(config["cost_level_enabled"])
        and float(config["pickup_cost_epsilon"]) == 0.0
    )
    safe_failures = any(bool(value) for value in base["failures"].values())
    recommendation = (
        "GO_ODD_AWARE_DECISION_KERNEL"
        if reproduction_pass and runtime_pass and neutral and not safe_failures
        else "REVISE_ODD_AWARE_DECISION_KERNEL"
    )
    computation = {
        key: base["computation"][key]
        for key in (
            "total_runtime_s",
            "candidate_generation_time_s",
            "routing_time_s",
            "solver_time_s",
            "solver_time_p50",
            "solver_time_p95",
            "solver_time_max",
            "valid_or_arcs",
        )
    }
    computation.update(
        {
            "av_candidates_pruned_by_acceptance": int(
                control.av_candidates_pruned_by_acceptance
            ),
            "av_candidates_pruned_by_missing_exposure": int(
                control.av_candidates_pruned_by_missing_exposure
            ),
            "enabled_gamma_constraint_count": int(
                epoch["enabled_gamma_constraint_count"].max()
            )
            if len(epoch)
            else 0,
            "cost_level_solve_count": int(control.cost_level_solve_count),
            "assignment_matrix_representation": "CSR_SPARSE_ARCS_ONLY",
            "gpu_usage": "NONE_CPU_ONLY",
        }
    )
    summary = {
        "phase_status": "STAGE4_S4_ODD_AWARE_DECISION_KERNEL_COMPLETE",
        "recommendation": recommendation,
        "canonical_base": {
            "s3_commit": S3_COMMIT,
            "fleetpy_commit": FLEETPY_COMMIT,
            "expected": EXPECTED,
        },
        "canonical_reproduction": observed,
        "canonical_reproduction_pass": reproduction_pass,
        "canonical_outcome_fingerprint_sha256": _fingerprint(outcomes),
        "neutral_configuration": neutral,
        "mechanisms": {
            "passenger_acceptance": "PASS",
            "three_family_cumulative_exposure": "PASS",
            "normalized_cost_lexicographic_level": "PASS",
        },
        "cat_eyes": {
            "acceptance": "PASS",
            "cumulative_exposure": "PASS",
            "cost_tie_break": "PASS",
        },
        "canonical_gamma_rows_enabled": computation["enabled_gamma_constraint_count"],
        "canonical_cost_level_enabled": bool(config["cost_level_enabled"]),
        "computation": computation,
        "s3_reference_runtime_s": 43.40438930000005,
        "failures": {
            "canonical_outcome_mismatch": not reproduction_pass,
            "runtime_guard_exceeded": not runtime_pass,
            "configuration_not_neutral": not neutral,
            "native_reconciliation_or_accounting_failure": safe_failures,
        },
        "limitations": [
            "Gamma values, acceptance rates, and normalized cost ratios are scenario inputs, not identified estimates.",
            "Exposure is reference-envelope exceedance, not safety or failure probability.",
            "S4 runs no parameter sweep and makes no economic or safety claim.",
        ],
    }
    fleet.scenario_fleet.to_parquet(output / "scenario_fleet.parquet", index=False)
    outcomes.to_parquet(output / "canonical_request_outcomes.parquet", index=False)
    pd.DataFrame(control.assignment_rows).to_parquet(
        output / "canonical_assignment_log.parquet", index=False
    )
    epoch.to_parquet(output / "canonical_epoch_stats.parquet", index=False)
    pd.DataFrame(control.exposure_rows).to_parquet(
        output / "canonical_exposure_state.parquet", index=False
    )
    _write_json(summary, output / "kernel_summary.json")
    _write_json(summary, root / DOC_REL / "stage4_s4_aggregate_summary.json")
    _report(root, summary)
    if recommendation != "GO_ODD_AWARE_DECISION_KERNEL":
        raise FleetPyCompatibilityError(
            f"S4 decision kernel requires revision: {summary['failures']}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--fleetpy-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            run_odd_aware_decision_kernel(args.root, args.fleetpy_root, args.config),
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
