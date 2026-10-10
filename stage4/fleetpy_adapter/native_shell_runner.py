"""Run the Stage4-S2 FleetPy-owned native rolling simulation shell."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .mixed_fleet_adapter import create_native_vehicles, load_mixed_fleet_fixture
from .native_demand import create_native_demand
from .native_fleet_control import create_native_fleet_control
from .native_network import create_native_network
from .native_simulation import create_native_simulation
from .test31_demand_adapter import attach_fleetpy_requests, load_test31_requests
from .upstream import (
    CoordinateRegistry,
    FLEETPY_COMMIT,
    FLEETPY_REPOSITORY,
    FleetPyCompatibilityError,
    load_fleetpy_bindings,
)
from .valhalla_time_adapter import ValhallaPickupTimeAdapter

TIMEZONE = "Asia/Shanghai"
CONFIG_REL = Path("stage4/config/fleetpy_native_shell.json")
OUTPUT_REL = Path("stage4/output/fleetpy_native_shell")
REPORT_REL = Path("stage4/docs/fleetpy_native_shell/stage4_s2_native_shell_summary.md")


def _load_config(root: Path, config_path: str | Path | None) -> dict[str, Any]:
    path = Path(config_path) if config_path else root / CONFIG_REL
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "test_date",
        "simulation_start_time",
        "simulation_end_time",
        "simulation_time_step_s",
        "profile_id",
        "engineering_request_count",
        "engineering_hv_count",
        "engineering_av_count",
        "selection_seed",
        "av_availability_policy",
        "dispatch_policy",
        "fleetpy_commit",
    }
    missing = required - set(config)
    if missing:
        raise FleetPyCompatibilityError(
            f"native shell config missing {sorted(missing)}"
        )
    if config["fleetpy_commit"] != FLEETPY_COMMIT:
        raise FleetPyCompatibilityError("native shell FleetPy commit is not pinned")
    if int(config["simulation_time_step_s"]) <= 0:
        raise FleetPyCompatibilityError("simulation timestep must be positive")
    return config


def _timestamps(config: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(
        f"{config['test_date']} {config['simulation_start_time']}", tz=TIMEZONE
    )
    end = pd.Timestamp(
        f"{config['test_date']} {config['simulation_end_time']}", tz=TIMEZONE
    )
    if end <= start:
        raise FleetPyCompatibilityError("native shell end must follow start")
    return start, end


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _request_log(requests: list, fleet_control: Any) -> pd.DataFrame:
    activation = {
        int(row["native_request_id"]): row for row in fleet_control.activation_rows
    }
    assignment = {
        int(row["native_request_id"]): row for row in fleet_control.assignment_rows
    }
    rows = []
    for request in requests:
        native = request.native_request
        active = activation.get(request.native_id, {})
        assigned = assignment.get(request.native_id, {})
        rows.append(
            {
                "native_request_id": request.native_id,
                "order_id": request.order_id,
                "profile_id": request.profile_id,
                "historical_request_time": request.request_time,
                "native_activation_time": active.get("native_activation_time", pd.NaT),
                "activation_lag_s": active.get("activation_lag_s", np.nan),
                "selected_vehicle_id": assigned.get("vehicle_id"),
                "selected_vehicle_type": assigned.get("vehicle_type"),
                "assignment_time": assigned.get("assignment_time", pd.NaT),
                "pickup_eta_s": assigned.get("pickup_eta_s", np.nan),
                "pickup_time": (
                    pd.NaT
                    if native.pu_time is None
                    else request.request_time
                    - pd.Timedelta(seconds=request.sim_time_s)
                    + pd.Timedelta(seconds=float(native.pu_time))
                ),
                "service_end_time": (
                    pd.NaT
                    if native.do_time is None
                    else request.request_time
                    - pd.Timedelta(seconds=request.sim_time_s)
                    + pd.Timedelta(seconds=float(native.do_time))
                ),
                "native_service_vehicle_id": native.service_vid,
                "native_completed": native.do_time is not None,
                "hard_state": request.hard_state,
                "evidence_complete": request.evidence_complete,
                "rho_static": request.rho_static,
                "rho_dynamic": request.rho_dynamic,
                "rho_speed": request.rho_speed,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["historical_request_time", "order_id"], kind="mergesort"
    )


def _diagnostics(
    *,
    bindings: Any,
    config: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    requests: list,
    fixtures: list,
    fleet_control: Any,
    eta_adapter: ValhallaPickupTimeAdapter,
    native_output: list[dict],
    simulation: Any,
) -> dict[str, Any]:
    assignments = pd.DataFrame(fleet_control.assignment_rows)
    activation = pd.DataFrame(fleet_control.activation_rows)
    completed = assignments.get("completed", pd.Series(dtype=bool)).fillna(False)
    pickup = assignments.get("pickup_eta_s", pd.Series(dtype=float))
    overrun = np.asarray(fleet_control.hv_realized_overrun_seconds, dtype=float)
    failures = {
        "av_availability_violations": fleet_control.av_availability_violations,
        "position_reconciliation_failures": fleet_control.position_reconciliation_failures,
        "request_state_reconciliation_failures": fleet_control.request_state_reconciliation_failures,
        "vehicle_state_reconciliation_failures": fleet_control.vehicle_state_reconciliation_failures,
    }
    recommendation = (
        "GO_NATIVE_FLEETPY_SHELL"
        if all(int(value) == 0 for value in failures.values())
        and len(activation) == len(requests)
        else "REASSESS_FLEETPY_NATIVE_SHELL"
    )
    return {
        "phase_status": "STAGE4_S2_NATIVE_FLEETPY_SHELL_COMPLETE",
        "recommendation": recommendation,
        "fleetpy": {
            "repository": FLEETPY_REPOSITORY,
            "commit": bindings.commit,
            "core_modifications": "NONE",
        },
        "native_path": {
            "simulation_class": simulation.__class__.__name__,
            "run_method": simulation.run.__func__.__qualname__,
            "step_method": simulation.step.__func__.__qualname__,
            "fleet_control_class": fleet_control.__class__.__name__,
            "fleet_control_hook": "Stage4NativeFleetControl.time_trigger",
            "demand_hook": "Demand.get_new_travelers",
            "broker_class": simulation.broker.__class__.__name__,
            "vehicle_progression_hook": "SimulationVehicle.update_veh_state",
            "network_class": simulation.routing_engine.__class__.__name__,
            "network_movement_hook": "Stage4ValhallaNetworkBridge.move_along_route",
            "s1_custom_event_loop_used": False,
        },
        "interval": {"start": start.isoformat(), "end": end.isoformat()},
        "simulation_time_step_s": int(config["simulation_time_step_s"]),
        "engineering_fixture": {
            "profile_id": config["profile_id"],
            "request_count": len(requests),
            "hv_count": sum(item.vehicle_type == "HV" for item in fixtures),
            "av_count": sum(item.vehicle_type == "AV" for item in fixtures),
            "scientifically_normalized": False,
        },
        "diagnostics": {
            "requests_loaded": len(requests),
            "requests_natively_activated": len(activation),
            "requests_assigned": len(assignments),
            "requests_completed": int(completed.sum()),
            "hv_assignments": int(
                assignments.get("vehicle_type", pd.Series(dtype=str)).eq("HV").sum()
            ),
            "av_assignments": int(
                assignments.get("vehicle_type", pd.Series(dtype=str)).eq("AV").sum()
            ),
            "activation_lag_s_mean": float(activation["activation_lag_s"].mean()),
            "activation_lag_s_max": float(activation["activation_lag_s"].max()),
            "corrected_pickup_eta_s_mean": float(pickup.mean()),
            "corrected_pickup_eta_s_max": float(pickup.max()),
            "candidate_arc_evaluations": fleet_control.candidate_arc_evaluations,
            "valhalla_calls": len(eta_adapter.call_log),
            "valhalla_cache_hits": eta_adapter.cache_hit_count,
            "valhalla_route_failures": fleet_control.routing_failures,
            "hv_session_end_candidate_arc_exclusions": len(
                fleet_control.hv_session_end_exclusions
            ),
            "hv_realized_session_overrun_count": int(len(overrun)),
            "hv_realized_session_overrun_seconds_mean": (
                float(overrun.mean()) if len(overrun) else 0.0
            ),
            "hv_realized_session_overrun_seconds_max": (
                float(overrun.max()) if len(overrun) else 0.0
            ),
            **failures,
            "native_route_leg_rows": len(native_output),
        },
        "availability_semantics": {
            "hv": "RECONSTRUCTED_S0_SESSION_WINDOW",
            "av": "FULL_SIMULATION_HORIZON_UNLESS_BUSY",
            "hv_realized_overrun": "FINISH_ACCEPTED_TRIP_THEN_PERMANENTLY_OFFLINE",
        },
        "scientific_service_rate_interpretation": False,
        "rolling_or_dispatch_started": False,
    }


def _write_report(root: Path, summary: dict[str, Any]) -> None:
    native = summary["native_path"]
    run = summary["diagnostics"]
    fixture = summary["engineering_fixture"]
    lines = [
        "# Stage4 S2 FleetPy Native Shell Summary",
        "",
        "## Native FleetPy path",
        "",
        f"- Simulation class: `{native['simulation_class']}`",
        f"- Run method: `{native['run_method']}`",
        f"- Step method: `{native['step_method']}`",
        f"- Fleet-control hook: `{native['fleet_control_hook']}`",
        f"- Demand/request hook: `{native['demand_hook']}`",
        f"- Broker: `{native['broker_class']}`",
        f"- Vehicle progression: `{native['vehicle_progression_hook']}`",
        f"- Network/routing hook: `{native['network_class']}.{native['network_movement_hook'].split('.')[-1]}`",
        "",
        "FleetPy owns the simulation clock, request activation, vehicle route-leg progression, busy/available transitions, broker callbacks, and native leg logging.",
        "",
        "## Reused upstream modules",
        "",
        "- `FleetSimulationBase.run`",
        "- `ImmediateDecisionsSimulation.step`",
        "- `Demand.get_new_travelers`",
        "- `BrokerBasic`",
        "- `FleetControlBase` subclass contract",
        "- `SimulationVehicle.update_veh_state`",
        "- `VehicleRouteLeg` and `VRL_STATES`",
        "- `NetworkBase` subclass contract",
        "",
        "## Project adapters",
        "",
        "- deterministic Test31 demand population",
        "- HV/AV fleet-window eligibility",
        "- WGS84 Valhalla plus frozen 15-minute beta network bridge",
        "- realized occupied-service leg timing",
        "- minimum-corrected-pickup-ETA fleet-control stub",
        "- optional silent progress shim when the existing Valhalla environment lacks progress-only `tqdm`",
        "",
        "## Availability semantics",
        "",
        "- `HV = reconstructed S0 session window`",
        "- `AV = full simulation horizon, unavailable only while busy`",
        "- An admitted HV trip that realizes past session end is completed; the vehicle is then permanently ineligible for new assignments.",
        "- The 40/10 fixture is not scientifically normalized.",
        "",
        "## Native replay result",
        "",
        f"- Interval: `{summary['interval']['start']}` to `{summary['interval']['end']}`",
        f"- Native timestep: {summary['simulation_time_step_s']} s",
        f"- Requests loaded/activated: {run['requests_loaded']}/{run['requests_natively_activated']}",
        f"- HV/AV fixture: {fixture['hv_count']}/{fixture['av_count']}",
        f"- Assigned/completed: {run['requests_assigned']}/{run['requests_completed']}",
        f"- HV/AV assignments: {run['hv_assignments']}/{run['av_assignments']}",
        f"- Activation lag mean/max (s): {run['activation_lag_s_mean']:.3f}/{run['activation_lag_s_max']:.3f}",
        f"- Corrected pickup ETA mean/max (s): {run['corrected_pickup_eta_s_mean']:.3f}/{run['corrected_pickup_eta_s_max']:.3f}",
        f"- Candidate arc evaluations: {run['candidate_arc_evaluations']}",
        f"- Valhalla calls/cache hits/failures: {run['valhalla_calls']}/{run['valhalla_cache_hits']}/{run['valhalla_route_failures']}",
        f"- HV session-end candidate exclusions: {run['hv_session_end_candidate_arc_exclusions']}",
        f"- HV realized overruns count/mean/max (s): {run['hv_realized_session_overrun_count']}/{run['hv_realized_session_overrun_seconds_mean']:.3f}/{run['hv_realized_session_overrun_seconds_max']:.3f}",
        f"- AV availability violations: {run['av_availability_violations']}",
        "",
        "No scientific service-rate conclusion is drawn from assigned/completed counts.",
        "",
        "## Lifecycle reconciliation",
        "",
        f"- Position failures: {run['position_reconciliation_failures']}",
        f"- Request-state failures: {run['request_state_reconciliation_failures']}",
        f"- Vehicle-state failures: {run['vehicle_state_reconciliation_failures']}",
        f"- Native completed route-leg rows: {run['native_route_leg_rows']}",
        "",
        "FleetPy request states, vehicle states, native route legs, assignments, completion positions, and availability policies reconcile when all failure counts are zero.",
        "",
        "## S1 loop status",
        "",
        "The S1 project-side event loop remains only as compatibility-spike lineage and is not part of the S2 formal native simulation path.",
        "",
        "## FleetPy core changes",
        "",
        "`NONE`",
        "",
        "## Recommendation",
        "",
        f"`{summary['recommendation']}`",
        "",
        "The shell proves native FleetPy ownership of the one-hour engineering replay. Full-day replay, scientific fleet normalization, candidate pruning, and the OR dispatch model remain unauthorized.",
        "",
    ]
    path = root / REPORT_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_native_shell(
    root: str | Path,
    fleetpy_root: str | Path,
    config_path: str | Path | None = None,
    *,
    eta_actor: Any | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    config = _load_config(root, config_path)
    start, end = _timestamps(config)
    bindings = load_fleetpy_bindings(fleetpy_root)
    registry = CoordinateRegistry()
    requests = load_test31_requests(
        root,
        start=start,
        end=end,
        profile_id=config["profile_id"],
        request_count=int(config["engineering_request_count"]),
        seed=int(config["selection_seed"]),
    )
    attach_fleetpy_requests(requests, bindings, registry)
    network = create_native_network(bindings, registry)
    output = root / OUTPUT_REL
    output.mkdir(parents=True, exist_ok=True)
    demand = create_native_demand(bindings, requests, registry, network, output)
    fixtures = load_mixed_fleet_fixture(
        root,
        start=start,
        end=end,
        hv_count=int(config["engineering_hv_count"]),
        av_count=int(config["engineering_av_count"]),
        seed=int(config["selection_seed"]),
    )
    vehicles, native_output = create_native_vehicles(
        fixtures,
        bindings,
        registry,
        demand.rq_db,
        output / "runtime",
        native_movement=True,
        routing_engine=network,
    )
    eta_adapter = ValhallaPickupTimeAdapter(root, actor=eta_actor)
    fleet_control = create_native_fleet_control(
        bindings,
        vehicles,
        requests,
        demand,
        network,
        eta_adapter,
        start,
        end,
    )
    simulation_end_s = int((end - start).total_seconds())
    simulation = create_native_simulation(
        bindings,
        simulation_end_s=simulation_end_s,
        time_step_s=int(config["simulation_time_step_s"]),
        demand=demand,
        vehicles=[runtime.native_vehicle for runtime in vehicles],
        fleet_control=fleet_control,
        network=network,
        native_output=native_output,
    )
    fleet_simulation_module = importlib.import_module("src.FleetSimulationBase")
    fleet_simulation_module.PROGRESS_LOOP = "off"
    simulation.run()
    fleet_control.reconcile()
    summary = _diagnostics(
        bindings=bindings,
        config=config,
        start=start,
        end=end,
        requests=requests,
        fixtures=fixtures,
        fleet_control=fleet_control,
        eta_adapter=eta_adapter,
        native_output=native_output,
        simulation=simulation,
    )
    failures = summary["diagnostics"]
    if summary["recommendation"] != "GO_NATIVE_FLEETPY_SHELL":
        raise FleetPyCompatibilityError(
            f"native shell reconciliation failed: {failures}"
        )
    request_log = _request_log(requests, fleet_control)
    _write_parquet(request_log, output / "native_request_log.parquet")
    _write_parquet(
        pd.DataFrame(fleet_control.vehicle_rows),
        output / "native_vehicle_log.parquet",
    )
    _write_parquet(
        pd.DataFrame(fleet_control.assignment_rows),
        output / "native_assignment_log.parquet",
    )
    _write_parquet(
        pd.DataFrame(native_output),
        output / "native_route_leg_log.parquet",
    )
    _write_json(summary, output / "native_shell_summary.json")
    _write_report(root, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--fleetpy-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            run_native_shell(args.root, args.fleetpy_root, args.config),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
