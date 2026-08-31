"""Execute the frozen Stage4 final experiment registry with safe resume."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

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
from .rolling_or_control import (
    RollingRuntimeGuardExceeded,
    create_rolling_or_fleet_control,
)
from .rolling_or_runner import _outcomes, _summary
from .repositioning_policy import (
    POLICY_NAME as REPOSITIONING_POLICY_NAME,
    POLICY_VERSION as REPOSITIONING_POLICY_VERSION,
    TrainTODRepositioningManager,
    load_train_demand_reference,
)

CONFIG_REL = Path("stage4/config/final_experiment_execution.json")
OUTPUT_FILES = (
    "scenario_config.json",
    "run_status.json",
    "summary.json",
    "request_outcomes.parquet",
    "assignment_log.parquet",
    "epoch_stats.parquet",
    "exposure_state.parquet",
    "runtime_diagnostics.json",
)
PHASE_A_IDS = (
    "MAIN_Q25_M_P70",
    "MAIN_Q50_M_P70",
    "ODD_Q50_M_P70_REFERENCE",
)
SCIENTIFIC_HASH_FIELDS = (
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
)
TRANSIENT_ERROR_NAMES = {
    "ConnectionError",
    "TimeoutError",
    "BrokenPipeError",
    "ConnectionResetError",
}


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def load_execution_config(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    config = _read_json(root / CONFIG_REL)
    if config["registry_commit"] != "b82cd3fda60a39c7474de49bdf4f205850a6725d":
        raise FleetPyCompatibilityError("S5B registry commit is not frozen")
    if config["fleetpy_commit"] != FLEETPY_COMMIT:
        raise FleetPyCompatibilityError("FleetPy commit is not frozen")
    if int(config["max_parallel_scenarios"]) < 1:
        raise FleetPyCompatibilityError("max_parallel_scenarios must be positive")
    if int(config["expected_request_count_per_profile"]) != 30000:
        raise FleetPyCompatibilityError("final Test31 demand must remain 30,000 orders")
    return config


def _parse_optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def _parse_bool(value: str) -> bool:
    if value not in {"True", "False"}:
        raise FleetPyCompatibilityError(f"invalid registry boolean: {value}")
    return value == "True"


def load_registry(
    root: str | Path, config: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    config = config or load_execution_config(root)
    path = root / config["registry_path"]
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for name in (
                "requested_q_A",
                "acceptance_probability",
                "eta_cost_av_to_hv",
                "pickup_cost_epsilon",
                "H_base_exact",
                "achieved_q_A",
                "target_HV_vehicle_hours",
                "achieved_HV_vehicle_hours",
                "HV_vehicle_hour_error_pct",
            ):
                row[name] = float(row[name])
            for name in ("gamma_static", "gamma_dynamic", "gamma_speed"):
                row[name] = _parse_optional_float(row[name])
            for name in (
                "acceptance_seed",
                "AV_vehicle_count",
                "selected_HV_session_count",
            ):
                row[name] = int(row[name])
            for name in ("cost_enabled", "benchmark_flag"):
                row[name] = _parse_bool(row[name])
            rows.append(row)
    if (
        len(rows) != 42
        or sum(bool(row["reuse_source_scenario_id"]) for row in rows) != 1
    ):
        raise FleetPyCompatibilityError(
            "frozen registry must contain 42 rows and one reuse"
        )
    if len({row["scenario_id"] for row in rows}) != 42:
        raise FleetPyCompatibilityError("scenario_id must be unique")
    return rows


def scientific_configuration(row: dict[str, Any]) -> dict[str, Any]:
    return {name: row[name] for name in SCIENTIFIC_HASH_FIELDS}


def scenario_config_sha256(row: dict[str, Any]) -> str:
    payload = json.dumps(
        scientific_configuration(row),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scenario_dir(
    root: str | Path, scenario_id: str, config: dict[str, Any] | None = None
) -> Path:
    root = Path(root).resolve()
    config = config or load_execution_config(root)
    return root / config["output_root"] / scenario_id


def required_outputs_exist(directory: Path) -> bool:
    return all((directory / name).is_file() for name in OUTPUT_FILES)


def _process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        import psutil

        return psutil.pid_exists(int(pid)) and psutil.Process(int(pid)).is_running()
    except Exception:
        return False


def completed_is_reusable(
    root: str | Path,
    row: dict[str, Any],
    execution_commit: str,
    config: dict[str, Any] | None = None,
) -> bool:
    root = Path(root).resolve()
    config = config or load_execution_config(root)
    directory = scenario_dir(root, row["scenario_id"], config)
    status_path = directory / "run_status.json"
    if not status_path.is_file() or not required_outputs_exist(directory):
        return False
    status = _read_json(status_path)
    return (
        status.get("status") == "COMPLETED"
        and status.get("execution_commit") == execution_commit
        and status.get("registry_commit") == config["registry_commit"]
        and status.get("scenario_config_sha256") == scenario_config_sha256(row)
    )


def resume_action(
    root: str | Path,
    row: dict[str, Any],
    execution_commit: str,
    *,
    retry_failed: bool = False,
    config: dict[str, Any] | None = None,
) -> str:
    root = Path(root).resolve()
    config = config or load_execution_config(root)
    directory = scenario_dir(root, row["scenario_id"], config)
    status_path = directory / "run_status.json"
    if not status_path.is_file():
        return "RUN"
    status = _read_json(status_path)
    state = status.get("status")
    if state == "COMPLETED":
        return (
            "SKIP"
            if completed_is_reusable(root, row, execution_commit, config)
            else "STALE"
        )
    if state == "RUNNING":
        return "ACTIVE" if _process_alive(status.get("process_id")) else "STALE"
    if state == "FAILED":
        return "RUN" if retry_failed else "FAILED"
    if state == "PENDING":
        return "RUN"
    raise FleetPyCompatibilityError(f"unknown run status {state}")


def unique_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not row["reuse_source_scenario_id"]]


def phase_rows(rows: Iterable[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    rows = unique_rows(rows)
    phase = phase.upper()
    if phase == "A":
        wanted = set(PHASE_A_IDS)
        return [row for row in rows if row["scenario_id"] in wanted]
    if phase == "B":
        return [
            row
            for row in rows
            if row["experiment_block"] in {"MAIN_STRUCTURAL", "BENCHMARK"}
        ]
    if phase == "C":
        return [
            row
            for row in rows
            if row["experiment_block"] in {"ODD_POLICY", "COST_ROBUSTNESS"}
        ]
    if phase == "ALL":
        return rows
    raise ValueError("phase must be A, B, C, or ALL")


def _memory_peak_mb() -> float | None:
    try:
        import psutil

        memory = psutil.Process().memory_info()
        value = getattr(memory, "peak_wset", memory.rss)
        return float(value) / (1024.0 * 1024.0)
    except Exception:
        return None


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _exposure_kpis(
    assignments: pd.DataFrame,
    exposure: pd.DataFrame,
    row: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    av = assignments.loc[assignments["vehicle_type"].astype(str).eq("AV")].copy()
    result: dict[str, Any] = {}
    for family in ("static", "dynamic", "speed"):
        assigned = pd.to_numeric(
            av.get(f"exposure_{family}", pd.Series(dtype=float)), errors="coerce"
        )
        cumulative = pd.to_numeric(
            exposure.get(f"cumulative_mean_{family}_excess", pd.Series(dtype=float)),
            errors="coerce",
        )
        gamma = row[f"gamma_{family}"]
        result[f"mean_assigned_exposure_{family}"] = (
            float(assigned.mean()) if len(assigned) else 0.0
        )
        result[f"positive_assigned_exposure_share_{family}"] = (
            float(assigned.gt(0.0).mean()) if len(assigned) else 0.0
        )
        result[f"final_cumulative_mean_exposure_{family}"] = (
            float(cumulative.iloc[-1]) if len(cumulative) else 0.0
        )
        result[f"maximum_cumulative_mean_exposure_{family}"] = (
            float(cumulative.max()) if len(cumulative) else 0.0
        )
        if gamma is None:
            result[f"binding_epoch_count_{family}"] = None
            result[f"near_binding_epoch_count_{family}"] = None
            result[f"minimum_slack_{family}"] = None
            result[f"mean_slack_{family}"] = None
        else:
            slack = float(gamma) - cumulative
            result[f"binding_epoch_count_{family}"] = int(slack.le(tolerance).sum())
            result[f"near_binding_epoch_count_{family}"] = int(
                slack.le(10.0 * tolerance).sum()
            )
            result[f"minimum_slack_{family}"] = (
                float(slack.min()) if len(slack) else float(gamma)
            )
            result[f"mean_slack_{family}"] = (
                float(slack.mean()) if len(slack) else float(gamma)
            )
            if (
                len(cumulative)
                and float(cumulative.iloc[-1]) > float(gamma) + tolerance
            ):
                raise FleetPyCompatibilityError(f"final {family} Gamma budget violated")
    return result


def _collect_summary(
    row: dict[str, Any],
    config: dict[str, Any],
    runtime_config: dict[str, Any],
    base: dict[str, Any],
    outcomes: pd.DataFrame,
    assignments: pd.DataFrame,
    epoch: pd.DataFrame,
    exposure: pd.DataFrame,
    execution_commit: str,
    config_hash: str,
    started_at: str,
    ended_at: str,
) -> dict[str, Any]:
    requests = int(base["matching"]["requests"])
    matched = int(base["matching"]["matched"])
    hv = int(base["assignment"]["hv"])
    av = int(base["assignment"]["av"])
    accepted = int(outcomes["passenger_accepts_av"].fillna(False).sum())
    fleet = base["fleet"]
    assignment_cost = pd.to_numeric(
        assignments.get("normalized_operating_cost", pd.Series(dtype=float)),
        errors="coerce",
    )
    pickup_eta = pd.to_numeric(
        assignments.get("pickup_eta_s", pd.Series(dtype=float)), errors="coerce"
    )
    summary: dict[str, Any] = {
        "scenario_id": row["scenario_id"],
        "experiment_block": row["experiment_block"],
        "execution_commit": execution_commit,
        "S5B_registry_commit": config["registry_commit"],
        "FleetPy_commit": config["fleetpy_commit"],
        "scenario_config_sha256": config_hash,
        "execution_timestamp": started_at,
        "execution_end_timestamp": ended_at,
        "horizon_start": config["horizon_start"],
        "horizon_end": config["horizon_end"],
        "H_base_exact": float(fleet["h_base_exact"]),
        "profile_id": row["profile_id"],
        "gamma_policy": row["gamma_policy"],
        "acceptance_seed": int(row["acceptance_seed"]),
        "request_count": requests,
        "matched": matched,
        "completed": int(base["matching"]["completed"]),
        "patience_expired": int(base["matching"]["patience_expired"]),
        "service_rate": _safe_div(matched, requests),
        "first_window_match_rate": float(base["queue"]["first_window_match_rate"]),
        "carry_over_entry_rate": float(base["queue"]["carry_over_entry_rate"]),
        "carry_over_recovery_rate": float(base["queue"]["carry_over_recovery_rate"]),
        "critical_recovery": float(base["queue"]["critical_order_recovery_rate"]),
        "critical_order_count": int(base["queue"]["critical_order_count"]),
        "request_to_pickup_mean": base["waiting"]["request_to_pickup_mean_s"],
        "request_to_pickup_p50": base["waiting"]["p50_s"],
        "request_to_pickup_p90": base["waiting"]["p90_s"],
        "request_to_pickup_p95": base["waiting"]["p95_s"],
        "HV_assignments": hv,
        "AV_assignments": av,
        "HV_assignment_share": _safe_div(hv, matched),
        "AV_assignment_share": _safe_div(av, matched),
        "requested_q_A": float(fleet["requested_q_a"]),
        "achieved_q_A": float(fleet["achieved_q_a"]),
        "AV_vehicle_count": int(fleet["av_count"]),
        "target_HV_vehicle_hours": float(fleet["target_hv_vehicle_hours"]),
        "achieved_HV_vehicle_hours": float(fleet["achieved_hv_vehicle_hours"]),
        "HV_vehicle_hour_error_pct": float(fleet["vehicle_hour_error_pct"]),
        "selected_HV_session_count": int(fleet["selected_hv_session_count"]),
        "target_p_A": float(row["acceptance_probability"]),
        "accepted_order_count": accepted,
        "realized_accepted_order_share": _safe_div(accepted, requests),
        "mean_attempts": float(base["queue"]["mean_matching_attempts"]),
        "expanded_radius_match_share": float(
            base["queue"]["expanded_radius_match_share"]
        ),
        "normalized_operating_cost_total": float(assignment_cost.sum()),
        "normalized_operating_cost_per_matched_order": _safe_div(
            float(assignment_cost.sum()), matched
        ),
        "HV_equivalent_operating_seconds": float(assignment_cost.sum()),
        "pickup_ETA_objective_value": float(pickup_eta.sum()),
        "relative_pickup_ETA_degradation_vs_epsilon_0_reference": None,
        "cost_enabled": bool(row["cost_enabled"]),
        "eta_cost_av_to_hv": float(row["eta_cost_av_to_hv"]),
        "pickup_cost_epsilon": float(row["pickup_cost_epsilon"]),
        "benchmark_flag": bool(row["benchmark_flag"]),
        "scientific_role": row["scientific_role"],
        "runtime": {
            **base["computation"],
            "wall_clock_runtime_s": float(base["computation"]["total_runtime_s"]),
            "peak_rss_mb": _memory_peak_mb(),
            "enabled_gamma_constraint_count": int(
                epoch.get("enabled_gamma_constraint_count", pd.Series(dtype=int)).max()
            )
            if len(epoch)
            else 0,
            "cost_level_solve_count": int(
                epoch.get("cost_level_solved", pd.Series(dtype=bool)).sum()
            ),
        },
        "integrity": {
            "runtime_guard_exceeded": bool(base["failures"]["runtime_guard_exceeded"]),
            "base_failures": base["failures"],
            "assignment_conservation": hv + av == matched,
            "request_count_matches_horizon": requests
            == int(config["expected_request_count_per_profile"]),
            "sparse_representation": runtime_config["assignment_matrix_representation"],
            "gpu_usage": runtime_config["gpu_usage"],
        },
    }
    summary.update(
        _exposure_kpis(
            assignments,
            exposure,
            row,
            float(runtime_config["solver_numerical_tolerance"]),
        )
    )
    return summary


def execute_scenario(
    root: str | Path,
    fleetpy_root: str | Path,
    row: dict[str, Any],
    config: dict[str, Any],
    execution_commit: str,
) -> dict[str, Any]:
    root = Path(root).resolve()
    directory = scenario_dir(root, row["scenario_id"], config)
    directory.mkdir(parents=True, exist_ok=True)
    base_config = _read_json(root / "stage4/config/rolling_or_baseline.json")
    start = pd.Timestamp(config["horizon_start"])
    demand_end = pd.Timestamp(config["horizon_end"])
    matching_end = demand_end + pd.Timedelta(
        seconds=int(base_config["max_pickup_wait_s"])
    )
    runtime_config = {
        **base_config,
        "benchmark_start_time": config["horizon_start"],
        "benchmark_end_time": config["horizon_end"],
        "profile_id": row["profile_id"],
        "av_vehicle_hour_share": float(row["requested_q_A"]),
        "passenger_acceptance_policy": "STABLE_ORDER_CRN",
        "passenger_acceptance_rate": float(row["acceptance_probability"]),
        "passenger_acceptance_seed": int(row["acceptance_seed"]),
        "gamma_static": row["gamma_static"],
        "gamma_dynamic": row["gamma_dynamic"],
        "gamma_speed": row["gamma_speed"],
        "eta_cost_av_to_hv": float(row["eta_cost_av_to_hv"]),
        "cost_level_enabled": bool(row["cost_enabled"]),
        "pickup_cost_epsilon": float(row["pickup_cost_epsilon"]),
        "solver_numerical_tolerance": 1e-7,
        "benchmark_runtime_guard_s": float(config["full_day_runtime_guard_s"]),
        "assignment_matrix_representation": config["assignment_matrix_representation"],
        "gpu_usage": config["gpu_usage"],
        "matching_end_s": int((matching_end - start).total_seconds()),
        "prospective_gate_logging": bool(
            config.get("prospective_gate_logging", False)
        ),
        "gate_diagnostic_bin_minutes": int(
            config.get("gate_diagnostic_bin_minutes", 15)
        ),
        "repositioning_enabled": bool(config.get("repositioning_enabled", False)),
    }
    repositioning_reference = None
    repositioning_manifest = None
    if runtime_config["repositioning_enabled"]:
        repositioning_reference, repositioning_manifest = load_train_demand_reference(root)
        expected_reference_sha = config.get("repositioning_reference_sha256")
        if (
            expected_reference_sha
            and repositioning_manifest["reference_sha256"] != expected_reference_sha
        ):
            raise FleetPyCompatibilityError("repositioning reference SHA mismatch")
    config_hash = scenario_config_sha256(row)
    scenario_config = {
        "scenario_id": row["scenario_id"],
        "scientific_configuration": scientific_configuration(row),
        "scenario_config_sha256": config_hash,
        "horizon_start": config["horizon_start"],
        "horizon_end": config["horizon_end"],
        "expected_request_count": config["expected_request_count_per_profile"],
        "execution_commit": execution_commit,
        "registry_commit": config["registry_commit"],
        "FleetPy_commit": config["fleetpy_commit"],
        "runtime_configuration": runtime_config,
    }
    if repositioning_manifest is not None:
        scenario_config["repositioning"] = {
            "policy_name": REPOSITIONING_POLICY_NAME,
            "policy_version": REPOSITIONING_POLICY_VERSION,
            "train_reference": repositioning_manifest,
            "empty_route_odd_qualification": "OPERATIONAL_ABSTRACTION_NOT_ODD_CERTIFIED",
        }
    _atomic_json(scenario_config, directory / "scenario_config.json")
    requests = load_all_test31_requests(
        root, start=start, end=demand_end, profile_id=row["profile_id"]
    )
    if len(requests) != int(config["expected_request_count_per_profile"]):
        raise FleetPyCompatibilityError(
            f"{row['scenario_id']} loaded {len(requests)} requests, expected 30000"
        )
    max_service = max(request.realized_service_time_s for request in requests)
    step = int(runtime_config["dispatch_interval_s"])
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
        requested_q_a=float(row["requested_q_A"]),
        seed=int(runtime_config["fleet_sampling_seed"]),
        max_hv_hour_error_pct=float(runtime_config["max_hv_vehicle_hour_error_pct"]),
    )
    bindings = load_fleetpy_bindings(fleetpy_root)
    registry = CoordinateRegistry()
    attach_fleetpy_requests(requests, bindings, registry)
    network = create_native_network(bindings, registry)
    demand = create_native_demand(bindings, requests, registry, network, directory)
    vehicles, native_output = create_native_vehicles(
        fleet.native_fixtures,
        bindings,
        registry,
        demand.rq_db,
        directory / "runtime",
        native_movement=True,
        routing_engine=network,
    )
    eta = SparseValhallaMatrixAdapter(root)
    repositioning_manager = None
    if runtime_config["repositioning_enabled"]:
        repositioning_manager = TrainTODRepositioningManager(
            bindings=bindings,
            runtimes=vehicles,
            network=network,
            eta_adapter=eta,
            reference=repositioning_reference,
            start=start,
            policy_end=demand_end,
        )
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
        repositioning_manager,
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
    started_at = _now()
    started = time.perf_counter()
    guard_exceeded = False
    try:
        simulation.run()
    except RollingRuntimeGuardExceeded:
        guard_exceeded = True
    runtime_s = time.perf_counter() - started
    control.reconcile()
    ended_at = _now()
    outcomes = _outcomes(requests, control, start)
    outcomes["passenger_accepts_av"] = outcomes["order_id"].map(
        {request.order_id: bool(request.passenger_accepts_av) for request in requests}
    )
    outcomes["acceptance_source"] = outcomes["order_id"].map(
        {request.order_id: request.acceptance_source for request in requests}
    )
    assignments = pd.DataFrame(control.assignment_rows)
    epoch = pd.DataFrame(control.epoch_rows)
    exposure = pd.DataFrame(control.exposure_rows)
    routing_audit = {
        "matrix_failed_arc_events": len(eta.failed_arc_records),
        "sampled_matrix_failures": 0,
    }
    base = _summary(
        runtime_config,
        fleet,
        outcomes,
        control,
        eta,
        runtime_s,
        matching_end,
        guard_exceeded,
        routing_audit,
    )
    if any(bool(value) for value in base["failures"].values()):
        raise FleetPyCompatibilityError(
            f"scenario invariant failure: {base['failures']}"
        )
    summary = _collect_summary(
        row,
        config,
        runtime_config,
        base,
        outcomes,
        assignments,
        epoch,
        exposure,
        execution_commit,
        config_hash,
        started_at,
        ended_at,
    )
    if repositioning_manager is not None:
        horizon_seconds = float((demand_end - start).total_seconds())
        repositioning_summary = repositioning_manager.summary(
            av_count=int(fleet.accounting["av_count"]), horizon_seconds=horizon_seconds
        )
        summary["repositioning"] = {
            "enabled": True,
            "train_reference_sha256": repositioning_manifest["reference_sha256"],
            **repositioning_summary,
        }
        if repositioning_summary["position_reconciliation_failure_count"]:
            raise FleetPyCompatibilityError("repositioning position reconciliation failed")
    integrity = summary["integrity"]
    if not (
        summary["matched"] <= summary["request_count"]
        and summary["completed"] <= summary["matched"]
        and integrity["assignment_conservation"]
        and integrity["request_count_matches_horizon"]
        and 0.0 <= summary["realized_accepted_order_share"] <= 1.0
        and 0.0 <= summary["service_rate"] <= 1.0
    ):
        raise FleetPyCompatibilityError("minimal final scenario integrity check failed")
    outcomes.to_parquet(directory / "request_outcomes.parquet", index=False)
    assignments.to_parquet(directory / "assignment_log.parquet", index=False)
    epoch.to_parquet(directory / "epoch_stats.parquet", index=False)
    exposure.to_parquet(directory / "exposure_state.parquet", index=False)
    if repositioning_manager is not None:
        pd.DataFrame(repositioning_manager.trip_rows).to_parquet(
            directory / "repositioning_log.parquet", index=False
        )
        pd.DataFrame(repositioning_manager.epoch_rows).to_parquet(
            directory / "repositioning_epoch.parquet", index=False
        )
        pd.DataFrame(repositioning_manager.distribution_rows).to_parquet(
            directory / "idle_av_distribution.parquet", index=False
        )
    _atomic_json(summary["runtime"], directory / "runtime_diagnostics.json")
    _atomic_json(summary, directory / "summary.json")
    return summary


def _status_template(
    row: dict[str, Any],
    config: dict[str, Any],
    execution_commit: str,
    state: str,
) -> dict[str, Any]:
    return {
        "scenario_id": row["scenario_id"],
        "status": state,
        "start_time": None,
        "end_time": None,
        "execution_commit": execution_commit,
        "registry_commit": config["registry_commit"],
        "scenario_config_sha256": scenario_config_sha256(row),
        "process_id": None,
        "retry_count": 0,
        "error_type": None,
        "error_message": None,
        "traceback_tail": None,
    }


def run_one(
    root: str | Path,
    fleetpy_root: str | Path,
    scenario_id: str,
    *,
    retry_failed: bool = False,
) -> int:
    root = Path(root).resolve()
    config = load_execution_config(root)
    rows = load_registry(root, config)
    row = next((item for item in rows if item["scenario_id"] == scenario_id), None)
    if row is None:
        raise FleetPyCompatibilityError(f"scenario not found: {scenario_id}")
    if row["reuse_source_scenario_id"]:
        raise FleetPyCompatibilityError("reuse-only row must not launch a simulation")
    execution_commit = _git_head(root)
    directory = scenario_dir(root, scenario_id, config)
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "run_status.json"
    action = resume_action(
        root,
        row,
        execution_commit,
        retry_failed=retry_failed,
        config=config,
    )
    if action == "SKIP":
        return 0
    if action in {"ACTIVE", "FAILED", "STALE"}:
        raise FleetPyCompatibilityError(
            f"scenario {scenario_id} requires explicit state resolution: {action}"
        )
    previous_retry = 0
    if status_path.is_file():
        previous_retry = int(_read_json(status_path).get("retry_count", 0))
    status = _status_template(row, config, execution_commit, "RUNNING")
    status.update(
        {
            "start_time": _now(),
            "process_id": os.getpid(),
            "retry_count": previous_retry,
        }
    )
    _atomic_json(status, status_path)
    try:
        execute_scenario(root, fleetpy_root, row, config, execution_commit)
    except Exception as exc:
        status.update(
            {
                "status": "FAILED",
                "end_time": _now(),
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:2000],
                "traceback_tail": "\n".join(traceback.format_exc().splitlines()[-20:]),
            }
        )
        _atomic_json(status, status_path)
        return 1
    status.update({"status": "COMPLETED", "end_time": _now()})
    _atomic_json(status, status_path)
    return 0


def _available_ram_gb() -> float:
    try:
        import psutil

        return float(psutil.virtual_memory().available) / (1024.0**3)
    except Exception:
        return float("inf")


def batch_partition(
    root: str | Path,
    rows: Iterable[dict[str, Any]],
    execution_commit: str,
    *,
    retry_failed: bool,
    config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    result = {name: [] for name in ("RUN", "SKIP", "FAILED", "ACTIVE", "STALE")}
    for row in rows:
        action = resume_action(
            root,
            row,
            execution_commit,
            retry_failed=retry_failed,
            config=config,
        )
        result[action].append(row)
    return result


def run_batch(
    root: str | Path,
    fleetpy_root: str | Path,
    *,
    phase: str,
    max_parallel: int | None = None,
    resume: bool = True,
    retry_failed: bool = False,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> dict[str, Any]:
    root = Path(root).resolve()
    fleetpy_root = Path(fleetpy_root).resolve()
    config = load_execution_config(root)
    execution_commit = _git_head(root)
    selected = phase_rows(load_registry(root, config), phase)
    if not resume:
        existing = [
            row["scenario_id"]
            for row in selected
            if (
                scenario_dir(root, row["scenario_id"], config) / "run_status.json"
            ).is_file()
        ]
        if existing:
            raise FleetPyCompatibilityError(
                f"existing scenario status requires --resume: {existing[:5]}"
            )
    partition = batch_partition(
        root,
        selected,
        execution_commit,
        retry_failed=retry_failed,
        config=config,
    )
    if partition["ACTIVE"]:
        raise FleetPyCompatibilityError("active scenario processes already exist")
    if partition["STALE"]:
        raise FleetPyCompatibilityError(
            "stale or provenance-mismatched scenarios require explicit resolution: "
            + ",".join(row["scenario_id"] for row in partition["STALE"])
        )
    pending = list(partition["RUN"])
    workers = int(max_parallel or config["max_parallel_scenarios"])
    if workers < 1:
        raise ValueError("max_parallel must be positive")
    running: dict[str, tuple[Any, dict[str, Any], float]] = {}
    results: dict[str, int] = {}
    started = time.perf_counter()
    while pending or running:
        while pending and len(running) < workers:
            if _available_ram_gb() < float(
                config["min_available_ram_gb_before_launch"]
            ):
                break
            row = pending.pop(0)
            directory = scenario_dir(root, row["scenario_id"], config)
            directory.mkdir(parents=True, exist_ok=True)
            pending_status = _status_template(row, config, execution_commit, "PENDING")
            _atomic_json(pending_status, directory / "run_status.json")
            command = [
                sys.executable,
                "-m",
                "stage4.dispatch.final_experiment_runner",
                "run-one",
                "--root",
                str(root),
                "--fleetpy-root",
                str(fleetpy_root),
                "--scenario-id",
                row["scenario_id"],
            ]
            if retry_failed:
                command.append("--retry-failed")
            process = popen_factory(
                command,
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            running[row["scenario_id"]] = (process, row, time.perf_counter())
        finished: list[str] = []
        for scenario_id, (process, row, launched) in running.items():
            return_code = process.poll()
            if return_code is None:
                continue
            results[scenario_id] = int(return_code)
            finished.append(scenario_id)
            status_path = scenario_dir(root, scenario_id, config) / "run_status.json"
            terminal_state = (
                _read_json(status_path).get("status") if status_path.is_file() else None
            )
            if terminal_state not in {"COMPLETED", "FAILED"}:
                status = _status_template(row, config, execution_commit, "FAILED")
                status.update(
                    {
                        "start_time": None,
                        "end_time": _now(),
                        "error_type": "WorkerExitError",
                        "error_message": f"worker exited {return_code}",
                        "runtime_until_failure_s": time.perf_counter() - launched,
                    }
                )
                _atomic_json(status, status_path)
        for scenario_id in finished:
            running.pop(scenario_id)
        if pending or running:
            time.sleep(float(config["poll_interval_s"]))
    statuses: dict[str, int] = {"COMPLETED": 0, "FAILED": 0}
    for row in selected:
        path = scenario_dir(root, row["scenario_id"], config) / "run_status.json"
        state = _read_json(path)["status"] if path.is_file() else "FAILED"
        if state in statuses:
            statuses[state] += 1
    return {
        "phase": phase.upper(),
        "selected_unique_scenarios": len(selected),
        "completed": statuses["COMPLETED"],
        "failed": statuses["FAILED"],
        "skipped_completed": len(partition["SKIP"]),
        "preexisting_failed_not_retried": len(partition["FAILED"]),
        "launched": len(results),
        "max_parallel_scenarios": workers,
        "batch_wall_clock_s": time.perf_counter() - started,
        "execution_commit": execution_commit,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run-one")
    one.add_argument("--root", type=Path, default=Path("."))
    one.add_argument("--fleetpy-root", type=Path, required=True)
    one.add_argument("--scenario-id", required=True)
    one.add_argument("--retry-failed", action="store_true")
    batch = sub.add_parser("run-batch")
    batch.add_argument("--root", type=Path, default=Path("."))
    batch.add_argument("--fleetpy-root", type=Path, required=True)
    batch.add_argument("--phase", choices=("A", "B", "C", "ALL"), required=True)
    batch.add_argument("--max-parallel", type=int, default=None)
    batch.add_argument("--resume", action="store_true")
    batch.add_argument("--retry-failed", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "run-one":
        raise SystemExit(
            run_one(
                args.root,
                args.fleetpy_root,
                args.scenario_id,
                retry_failed=args.retry_failed,
            )
        )
    result = run_batch(
        args.root,
        args.fleetpy_root,
        phase=args.phase,
        max_parallel=args.max_parallel,
        resume=args.resume,
        retry_failed=args.retry_failed,
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
