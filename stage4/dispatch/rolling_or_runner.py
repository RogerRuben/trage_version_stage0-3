"""Run the bounded Stage4-S3 patience-aware sparse rolling OR baseline."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
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
from .matrix_equivalence_audit import MatrixFailureRouteAuditor
from .rolling_or_control import (
    RollingRuntimeGuardExceeded,
    create_rolling_or_fleet_control,
)

TIMEZONE = "Asia/Shanghai"
CONFIG_REL = Path("stage4/config/rolling_or_baseline.json")
OUTPUT_REL = Path("stage4/output/rolling_or_baseline")
DOC_REL = Path("stage4/docs/rolling_or_baseline")


def _load_config(root: Path, config_path: str | Path | None) -> dict[str, Any]:
    path = Path(config_path).resolve() if config_path else root / CONFIG_REL
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "test_date",
        "benchmark_start_time",
        "benchmark_end_time",
        "profile_id",
        "av_vehicle_hour_share",
        "passenger_acceptance_policy",
        "dispatch_interval_s",
        "max_pickup_wait_s",
        "search_radius_initial_m",
        "search_radius_step_m",
        "search_radius_cap_m",
        "candidate_top_k",
        "fleet_sampling_seed",
        "max_hv_vehicle_hour_error_pct",
        "benchmark_runtime_guard_s",
        "matrix_failure_route_audit_sample_size",
        "matrix_failure_route_audit_seed",
        "fleetpy_commit",
    }
    if required - set(config):
        raise FleetPyCompatibilityError(
            f"rolling config missing {sorted(required - set(config))}"
        )
    if config["fleetpy_commit"] != FLEETPY_COMMIT:
        raise FleetPyCompatibilityError("rolling baseline FleetPy commit is not pinned")
    if config["passenger_acceptance_policy"] != "ALL_ACCEPT_AV":
        raise FleetPyCompatibilityError("S3 permits only ALL_ACCEPT_AV")
    return config


def _timestamps(
    config: dict[str, Any]
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(
        f"{config['test_date']} {config['benchmark_start_time']}", tz=TIMEZONE
    )
    demand_end = pd.Timestamp(
        f"{config['test_date']} {config['benchmark_end_time']}", tz=TIMEZONE
    )
    matching_end = demand_end + pd.Timedelta(seconds=int(config["max_pickup_wait_s"]))
    return start, demand_end, matching_end


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def _outcomes(requests: list[Any], control: Any, start: pd.Timestamp) -> pd.DataFrame:
    assignments = {
        int(row["native_request_id"]): row for row in control.assignment_rows
    }
    rows: list[dict[str, Any]] = []
    for request in requests:
        rid = request.native_id
        meta = control.request_meta.get(rid, {})
        assignment = assignments.get(rid, {})
        native = request.native_request
        pickup_time = (
            pd.NaT
            if native.pu_time is None
            else start + pd.Timedelta(seconds=float(native.pu_time))
        )
        rows.append(
            {
                "order_id": request.order_id,
                "request_time": request.request_time,
                "pickup_deadline": request.request_time
                + pd.Timedelta(seconds=control.max_pickup_wait_s),
                "first_attempt_time": meta.get("first_attempt_time", pd.NaT),
                "attempt_count": int(meta.get("attempt_count", 0)),
                "failed_round_count": int(meta.get("failed_round_count", 0)),
                "matched": rid in assignments,
                "assignment_time": assignment.get("assignment_time", pd.NaT),
                "pickup_time": pickup_time,
                "completed": native.do_time is not None,
                "patience_expired": rid in control.expired_rids,
                "vehicle_type": assignment.get("vehicle_type"),
                "final_search_radius_m": float(
                    meta.get("final_search_radius_m", np.nan)
                ),
                "assignment_pickup_eta_s": float(
                    assignment.get("pickup_eta_s", np.nan)
                ),
                "total_request_to_pickup_wait_s": float(
                    (pickup_time - request.request_time).total_seconds()
                )
                if pd.notna(pickup_time)
                else np.nan,
                "entered_carry_over": bool(meta.get("carry_over_flag", False)),
                "entered_critical": bool(meta.get("entered_critical", False)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["request_time", "order_id"], kind="mergesort"
    )


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _summary(
    config: dict[str, Any],
    fleet: Any,
    outcomes: pd.DataFrame,
    control: Any,
    eta: Any,
    total_runtime_s: float,
    matching_end: pd.Timestamp,
    guard_exceeded: bool,
    routing_audit: dict[str, Any],
) -> dict[str, Any]:
    epoch = pd.DataFrame(control.epoch_rows)
    assignment = pd.DataFrame(control.assignment_rows)
    waits = pd.to_numeric(
        outcomes.loc[outcomes["pickup_time"].notna(), "total_request_to_pickup_wait_s"],
        errors="coerce",
    )
    requests = len(outcomes)
    matched = int(outcomes["matched"].sum())
    carry = int(outcomes["entered_carry_over"].sum())
    carry_recovered = int((outcomes["entered_carry_over"] & outcomes["matched"]).sum())
    critical = int(outcomes["entered_critical"].sum())
    critical_recovered = int((outcomes["entered_critical"] & outcomes["matched"]).sum())
    spatial_pairs = int(
        epoch.get("candidate_spatial_pairs", pd.Series(dtype=int)).sum()
    )
    topk_pairs = int(epoch.get("candidate_topk_pairs", pd.Series(dtype=int)).sum())
    valid_arcs = int(epoch.get("valid_or_arcs", pd.Series(dtype=int)).sum())
    cache_lookups = eta.routing_arc_evaluations + eta.cache_hit_count
    failures = {
        "runtime_guard_exceeded": bool(guard_exceeded),
        "activation_count_mismatch": len(control.activation_rows) != requests,
        "terminal_outcome_mismatch": matched + int(outcomes["patience_expired"].sum())
        != requests,
        "av_availability_violations": int(control.av_availability_violations),
        "position_reconciliation_failures": int(
            control.position_reconciliation_failures
        ),
        "request_state_reconciliation_failures": int(
            control.request_state_reconciliation_failures
        ),
        "vehicle_state_reconciliation_failures": int(
            control.vehicle_state_reconciliation_failures
        ),
        "matrix_failure_record_count_mismatch": int(
            routing_audit["matrix_failed_arc_events"]
        )
        != int(eta.matrix_failed_arcs),
        "matrix_fallback_accounting_mismatch": int(eta.matrix_failed_arcs)
        != int(eta.route_fallback_attempts),
    }
    recommendation = (
        "GO_ROLLING_OR_BASELINE"
        if not any(failures.values())
        else "REVISE_ROLLING_OR_BASELINE"
    )
    still_at_drain_start = 0
    if not assignment.empty:
        still_at_drain_start = int(
            (
                assignment["service_end_time"].isna()
                | (assignment["service_end_time"] > matching_end)
            ).sum()
        )
    solve = pd.to_numeric(
        epoch.get("solver_time_s", pd.Series(dtype=float)), errors="coerce"
    )
    candidate_time = float(
        epoch.get("candidate_generation_time_s", pd.Series(dtype=float)).sum()
    )
    routing_time = float(epoch.get("routing_time_s", pd.Series(dtype=float)).sum())
    solver_time = float(solve.sum())
    return {
        "phase_status": "STAGE4_S3_ROLLING_OR_BASELINE_COMPLETE",
        "recommendation": recommendation,
        "scientific_scope": "ONE_BOUNDED_QA_025_COMPUTATIONAL_BENCHMARK_NOT_POLICY_EVIDENCE",
        "interval": {
            "demand_start": outcomes["request_time"].min(),
            "demand_end": config["benchmark_end_time"],
            "matching_end": matching_end,
        },
        "fleet": fleet.accounting,
        "matching": {
            "requests": requests,
            "matched": matched,
            "completed": int(outcomes["completed"].sum()),
            "patience_expired": int(outcomes["patience_expired"].sum()),
            "still_in_progress_at_drain_start": still_at_drain_start,
        },
        "waiting": {
            "request_to_pickup_mean_s": float(waits.mean()) if len(waits) else None,
            "p50_s": float(waits.quantile(0.50)) if len(waits) else None,
            "p90_s": float(waits.quantile(0.90)) if len(waits) else None,
            "p95_s": float(waits.quantile(0.95)) if len(waits) else None,
        },
        "queue": {
            "first_window_match_rate": _safe_div(
                int(((outcomes["attempt_count"] == 1) & outcomes["matched"]).sum()),
                requests,
            ),
            "carry_over_entry_rate": _safe_div(carry, requests),
            "carry_over_recovery_rate": _safe_div(carry_recovered, carry),
            "critical_order_count": critical,
            "critical_order_recovery_rate": _safe_div(critical_recovered, critical),
            "mean_matching_attempts": float(outcomes["attempt_count"].mean()),
            "expanded_radius_match_share": _safe_div(
                int(
                    (
                        outcomes["matched"]
                        & (
                            outcomes["final_search_radius_m"]
                            > config["search_radius_initial_m"]
                        )
                    ).sum()
                ),
                matched,
            ),
        },
        "assignment": {
            "hv": int((outcomes["vehicle_type"] == "HV").sum()),
            "av": int((outcomes["vehicle_type"] == "AV").sum()),
            "solver_backend": "SCIPY_HIGHS_MILP_SEQUENTIAL_LEXICOGRAPHIC",
        },
        "computation": {
            "dispatch_epoch_count": len(epoch),
            "waiting_orders_mean": float(epoch["waiting_orders"].mean())
            if len(epoch)
            else 0.0,
            "waiting_orders_max": int(epoch["waiting_orders"].max())
            if len(epoch)
            else 0,
            "available_vehicles_mean": float(epoch["available_vehicles"].mean())
            if len(epoch)
            else 0.0,
            "available_vehicles_max": int(epoch["available_vehicles"].max())
            if len(epoch)
            else 0,
            "candidate_spatial_pairs": spatial_pairs,
            "candidate_topk_pairs": topk_pairs,
            "valid_or_arcs": valid_arcs,
            "peak_topk_pairs_per_epoch": (
                int(epoch["candidate_topk_pairs"].max()) if len(epoch) else 0
            ),
            "peak_valid_or_arcs_per_epoch": (
                int(epoch["valid_or_arcs"].max()) if len(epoch) else 0
            ),
            "assignment_matrix_representation": "CSR_SPARSE_ARCS_ONLY",
            "gpu_usage": "NONE_CPU_ONLY",
            "candidate_reduction_ratio": 1.0 - _safe_div(topk_pairs, spatial_pairs),
            "matrix_batch_queries": int(eta.routing_queries),
            "uncached_arc_evaluations": int(eta.routing_arc_evaluations),
            "arc_cache_hits": int(eta.cache_hit_count),
            "arc_cache_hit_rate": _safe_div(eta.cache_hit_count, cache_lookups),
            "matrix_failed_arcs": int(eta.matrix_failed_arcs),
            "matrix_arc_failure_rate": _safe_div(
                eta.matrix_failed_arcs, eta.routing_arc_evaluations
            ),
            "route_fallback_attempts": int(eta.route_fallback_attempts),
            "route_fallback_successes": int(eta.route_fallback_successes),
            "route_fallback_failures": int(eta.route_fallback_failures),
            "failed_routing_arcs": int(eta.routing_failures),
            "arc_failure_rate": _safe_div(
                eta.routing_failures, eta.routing_arc_evaluations
            ),
            "candidate_generation_time_s": candidate_time,
            "routing_time_s": routing_time,
            "solver_time_s": solver_time,
            "fleetpy_time_s": max(
                0.0, total_runtime_s - candidate_time - routing_time - solver_time
            ),
            "total_runtime_s": total_runtime_s,
            "solver_time_p50": float(solve.quantile(0.50)) if len(solve) else 0.0,
            "solver_time_p95": float(solve.quantile(0.95)) if len(solve) else 0.0,
            "solver_time_max": float(solve.max()) if len(solve) else 0.0,
        },
        "limitations": [
            "Stage3 suitability gates passenger service routes only; AV pickup legs are checked only for Valhalla routability.",
            "HV supply units are effective service-session templates, not a physical fleet count.",
            "Passengers use the S3 ALL_ACCEPT_AV baseline.",
        ],
        "matrix_route_closure": {
            "status": "CLOSED_WITH_NON_REPRODUCTION_AND_FAILED_CELL_ROUTE_FALLBACK",
            "original_commit": "9eed0652aaec1fb84cd5507c5a186c3937a8f0c1",
            "original_matrix_failed_arcs": 1860,
            "original_estimated_uncached_arcs": 55828,
            "original_estimated_matrix_arc_failure_rate": 1860 / 55828,
            "exact_reproduction_matrix_failed_arcs": int(eta.matrix_failed_arcs),
            "requested_failed_arc_sample_size": int(
                config["matrix_failure_route_audit_sample_size"]
            ),
            "available_failed_arc_sample_size": int(
                routing_audit["sampled_matrix_failures"]
            ),
            "empirical_100_failure_sample_completed": int(
                routing_audit["sampled_matrix_failures"]
            )
            == int(config["matrix_failure_route_audit_sample_size"]),
            "production_failed_cell_policy": "MATRIX_FAILED_THEN_SINGLE_ROUTE_FALLBACK",
            "candidate_deleted_only_if": "MATRIX_AND_SINGLE_ROUTE_BOTH_FAIL",
            "fallback_unit_cat_eye": "PASS",
        },
        "routing_equivalence_audit": routing_audit,
        "failures": failures,
    }


def _report(root: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage4 S3 Rolling OR Baseline",
        "",
        f"Recommendation: `{summary['recommendation']}`",
        "",
        "This is one bounded qA=0.25 scientific-interface and computational benchmark, not a policy comparison.",
        "",
        "## Result",
        "",
        f"- Requests/matched/completed/expired: {summary['matching']['requests']}/{summary['matching']['matched']}/{summary['matching']['completed']}/{summary['matching']['patience_expired']}",
        f"- HV/AV assignments: {summary['assignment']['hv']}/{summary['assignment']['av']}",
        f"- Requested/achieved qA: {summary['fleet']['requested_q_a']:.6f}/{summary['fleet']['achieved_q_a']:.6f}",
        f"- HV vehicle-hour error: {summary['fleet']['vehicle_hour_error_pct']:.3f}%",
        "",
        "## Queue",
        "",
        f"- First-window match rate: {summary['queue']['first_window_match_rate']:.6f}",
        f"- Carry-over entry/recovery: {summary['queue']['carry_over_entry_rate']:.6f}/{summary['queue']['carry_over_recovery_rate']:.6f}",
        f"- Critical count/recovery: {summary['queue']['critical_order_count']}/{summary['queue']['critical_order_recovery_rate']:.6f}",
        "",
        "## Computation",
        "",
        f"- Runtime: {summary['computation']['total_runtime_s']:.3f}s",
        f"- Spatial/Top-K/valid pairs: {summary['computation']['candidate_spatial_pairs']}/{summary['computation']['candidate_topk_pairs']}/{summary['computation']['valid_or_arcs']}",
        f"- Peak Top-K/valid OR arcs per epoch: {summary['computation']['peak_topk_pairs_per_epoch']}/{summary['computation']['peak_valid_or_arcs_per_epoch']}",
        "- Memory design: cKDTree + Top-K 20 + CSR sparse constraints; no order-by-fleet dense matrix and no per-vehicle tick trace.",
        "- GPU usage: none (CPU-only SciPy/HiGHS and Valhalla).",
        f"- Matrix batches/uncached arcs/cache hits: {summary['computation']['matrix_batch_queries']}/{summary['computation']['uncached_arc_evaluations']}/{summary['computation']['arc_cache_hits']}",
        f"- Matrix failed arcs/fallback success/fallback failure/final failed arcs: {summary['computation']['matrix_failed_arcs']}/{summary['computation']['route_fallback_successes']}/{summary['computation']['route_fallback_failures']}/{summary['computation']['failed_routing_arcs']}",
        f"- Matrix/final arc failure rates: {summary['computation']['matrix_arc_failure_rate']:.6f}/{summary['computation']['arc_failure_rate']:.6f}",
        f"- Solver p50/p95/max: {summary['computation']['solver_time_p50']:.6f}/{summary['computation']['solver_time_p95']:.6f}/{summary['computation']['solver_time_max']:.6f}s",
        "",
        "## Matrix failure vs single route cat-eye",
        "",
        f"- Sampled matrix failures: {summary['routing_equivalence_audit']['sampled_matrix_failures']}",
        f"- Single route success/failure: {summary['routing_equivalence_audit']['single_route_success']}/{summary['routing_equivalence_audit']['single_route_failure']}",
        f"- Single route success rate: {summary['routing_equivalence_audit']['single_route_success_rate'] if summary['routing_equivalence_audit']['single_route_success_rate'] is not None else 'N/A'}",
        "",
        "## Matrix-route closure",
        "",
        "- Original 9eed065 observation: 1,860 matrix-failed arcs from approximately 55,828 uncached arcs (3.332%).",
        f"- Exact production-adapter reproduction: {summary['matrix_route_closure']['exact_reproduction_matrix_failed_arcs']} matrix-failed arcs.",
        f"- Requested/available failed-arc sample: {summary['matrix_route_closure']['requested_failed_arc_sample_size']}/{summary['matrix_route_closure']['available_failed_arc_sample_size']}.",
        "- The original failure population was not reproducible, so no empirical 100-failure equivalence rate is claimed.",
        "- Production policy now retries only failed matrix cells with an identical single route; an arc is deleted only if both calls fail.",
        f"- Closure: `{summary['matrix_route_closure']['status']}`",
        "",
        "## Interpretation limits",
        "",
        *[f"- {item}" for item in summary["limitations"]],
        "",
    ]
    path = root / DOC_REL / "stage4_s3_rolling_or_baseline_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_rolling_or_baseline(
    root: str | Path,
    fleetpy_root: str | Path,
    config_path: str | Path | None = None,
    *,
    eta_actor: Any | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    config = _load_config(root, config_path)
    start, demand_end, matching_end = _timestamps(config)
    requests = load_all_test31_requests(
        root, start=start, end=demand_end, profile_id=config["profile_id"]
    )
    max_service = max(request.realized_service_time_s for request in requests)
    drain_seconds = (
        int((matching_end - start).total_seconds())
        + int(math.ceil(max_service / config["dispatch_interval_s"]))
        * int(config["dispatch_interval_s"])
        + int(config["dispatch_interval_s"])
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
    audit_actor = MatrixFailureRouteAuditor(eta.actor)
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
        time_step_s=int(config["dispatch_interval_s"]),
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
    audit_actor.failed_arcs = list(eta.failed_arc_records)
    try:
        routing_audit, routing_audit_detail = audit_actor.audit_with_single_route(
            sample_size=int(config["matrix_failure_route_audit_sample_size"]),
            seed=int(config["matrix_failure_route_audit_seed"]),
        )
    except Exception as exc:
        _write_json(
            {"exception_type": type(exc).__name__, "message": str(exc)},
            output / "matrix_failure_route_audit_error.json",
        )
        raise
    control.reconcile()
    outcomes = _outcomes(requests, control, start)
    summary = _summary(
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
    fleet.scenario_fleet.to_parquet(output / "scenario_fleet.parquet", index=False)
    outcomes.to_parquet(output / "request_outcomes.parquet", index=False)
    pd.DataFrame(control.assignment_rows).to_parquet(
        output / "assignment_log.parquet", index=False
    )
    pd.DataFrame(control.epoch_rows).to_parquet(
        output / "candidate_epoch_stats.parquet", index=False
    )
    routing_audit_detail.to_parquet(
        output / "matrix_failure_route_audit_sample.parquet", index=False
    )
    _write_json(routing_audit, output / "matrix_failure_route_audit.json")
    _write_json(summary["computation"], output / "runtime_stats.json")
    _write_json(summary, output / "baseline_summary.json")
    _write_json(summary, root / DOC_REL / "stage4_s3_aggregate_summary.json")
    _report(root, summary)
    if summary["recommendation"] != "GO_ROLLING_OR_BASELINE":
        raise FleetPyCompatibilityError(
            f"rolling baseline requires revision: {summary['failures']}"
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
            run_rolling_or_baseline(args.root, args.fleetpy_root, args.config),
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
