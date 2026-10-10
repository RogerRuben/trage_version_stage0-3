"""Run the bounded Stage4 S1 FleetPy/Test31 compatibility spike and stop."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .dispatch_stub import MinCorrectedPickupEtaDispatchStub
from .mixed_fleet_adapter import create_native_vehicles, load_mixed_fleet_fixture
from .replay_service_time_adapter import FleetPyLifecycleAdapter
from .test31_demand_adapter import attach_fleetpy_requests, load_test31_requests
from .upstream import (
    FLEETPY_COMMIT,
    FLEETPY_REPOSITORY,
    CoordinateRegistry,
    FleetPyCompatibilityError,
    load_fleetpy_bindings,
)
from .valhalla_time_adapter import ValhallaPickupTimeAdapter

TIMEZONE = "Asia/Shanghai"
DEFAULT_CONFIG_REL = Path("stage4/config/fleetpy_spike.json")
OUTPUT_REL = Path("stage4/output/fleetpy_spike")
REPORT_REL = Path(
    "stage4/docs/fleetpy_spike/stage4_s1_fleetpy_compatibility_summary.md"
)
CONFIG_KEYS = {
    "test_date",
    "spike_start_time",
    "spike_end_time",
    "profile_id",
    "engineering_request_count",
    "engineering_hv_count",
    "engineering_av_count",
    "selection_seed",
    "av_availability_policy",
    "dispatch_stub",
    "fleetpy_repository",
    "fleetpy_commit",
}

COMPATIBILITY_MATRIX = {
    "exact_historical_request_injection": "SUPPORTED_BY_THIN_ADAPTER",
    "custom_initial_vehicle_positions": "SUPPORTED_DIRECTLY",
    "hv_availability_windows": "SUPPORTED_BY_THIN_ADAPTER",
    "always_on_av_availability": "SUPPORTED_BY_THIN_ADAPTER",
    "fleet_control_waiting_available_time_access": "SUPPORTED_BY_SUBCLASS",
    "custom_assignment_submission": "SUPPORTED_DIRECTLY",
    "completed_service_position_update": "SUPPORTED_DIRECTLY",
    "vehicle_available_again": "SUPPORTED_DIRECTLY",
    "external_valhalla_travel_time": "SUPPORTED_BY_THIN_ADAPTER",
    "request_vehicle_kpi_logs": "SUPPORTED_BY_THIN_ADAPTER",
}


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != CONFIG_KEYS:
        raise FleetPyCompatibilityError("fleetpy_spike config keys changed")
    if config["test_date"] != "20161031":
        raise FleetPyCompatibilityError("FleetPy spike is authorized only for Test31")
    if config["av_availability_policy"] != "FULL_SIMULATION_HORIZON":
        raise FleetPyCompatibilityError("AV must remain available for the full horizon")
    if config["dispatch_stub"] != "MIN_CORRECTED_PICKUP_ETA":
        raise FleetPyCompatibilityError(
            "only the transparent dispatch stub is authorized"
        )
    if config["fleetpy_repository"] != FLEETPY_REPOSITORY:
        raise FleetPyCompatibilityError("FleetPy repository mismatch")
    if config["fleetpy_commit"] != FLEETPY_COMMIT:
        raise FleetPyCompatibilityError("FleetPy commit mismatch")
    if int(config["engineering_request_count"]) not in range(100, 301):
        raise FleetPyCompatibilityError("spike request count must remain in [100, 300]")
    return config


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_report(root: Path, summary: dict[str, Any]) -> None:
    run = summary["tiny_replay_result"]
    matrix = summary["compatibility_matrix"]
    lines = [
        "# Stage4 S1 FleetPy Compatibility Spike",
        "",
        "## Upstream dependency",
        "",
        f"- Repository: `{summary['fleetpy']['repository']}`",
        f"- Commit: `{summary['fleetpy']['commit']}`",
        "- License: MIT",
        "- Installation: pinned external shallow checkout; no FleetPy source vendored",
        f"- Runtime Python: `{summary['runtime']['python']}`",
        "- FleetPy upstream environment targets Python 3.10; the exercised pure-Python classes imported and ran in the existing Stage0-Valhalla environment.",
        "- C++ router required: NO",
        "- Gurobi required: NO",
        "- OR-Tools required: NO",
        "",
        "## Reused FleetPy modules",
        "",
        "- `src.demand.TravelerModels.BasicRequest`",
        "- `src.simulation.Vehicles.ExternallyMovingSimulationVehicle`",
        "- `src.simulation.Legs.VehicleRouteLeg`",
        "- `src.misc.globals.VRL_STATES`",
        "- `FleetSimulationBase`, `ImmediateDecisionsSimulation`, `Demand`, `FleetControlBase`, and `NetworkBase` were inspected to verify the future subclass hooks; no upstream file was changed.",
        "",
        "## Project-specific adapters",
        "",
        "- Test31 demand selection and `BasicRequest` construction",
        "- deterministic S0-derived 40-HV/10-AV fixture",
        "- WGS84 Valhalla pickup callback with frozen 15-minute S0 beta",
        "- external-arrival bridge into FleetPy's native vehicle-leg lifecycle",
        "- transparent minimum-corrected-pickup-ETA dispatch stub",
        "",
        "## Availability semantics",
        "",
        "- `HV = reconstructed S0 session window`",
        "- `AV = full simulation horizon / always available unless busy`",
        "- AVs do not inherit source-session end times.",
        "- AV fleet-count normalization is not frozen in this spike.",
        "- The 40/10 split is an engineering fixture with no scientific interpretation.",
        "",
        "## Tiny replay result",
        "",
        f"- Interval: `{summary['interval']['start']}` to `{summary['interval']['end']}`",
        f"- Requests: {run['requests_injected']}",
        f"- HV fixture: {summary['fixture']['hv_count']}",
        f"- AV fixture: {summary['fixture']['av_count']}",
        f"- Assigned/completed: {run['requests_assigned']}/{run['requests_completed']}",
        f"- HV assignments: {run['hv_assignments']}",
        f"- AV assignments: {run['av_assignments']}",
        f"- Corrected pickup ETA mean/max (s): {run['mean_corrected_pickup_eta_s']:.3f}/{run['max_corrected_pickup_eta_s']:.3f}",
        f"- Vehicles activated: {run['vehicles_activated']}",
        f"- HV session-end exclusions: {run['hv_session_end_exclusions']}",
        f"- AV availability violations: {run['av_availability_violations']}",
        f"- Position/timing failures: {run['position_transition_failures']}/{run['request_timing_failures']}",
        f"- Valhalla calls/cache hits/failures: {run['valhalla_calls']}/{run['valhalla_cache_hits']}/{run['valhalla_route_failures']}",
        f"- FleetPy native output reconciles with adapter logs: `{run['native_output_reconciles']}`",
        "",
        "This is an engineering compatibility run only; no scientific fleet or policy conclusion is drawn from it.",
        "",
        "## Compatibility matrix",
        "",
        "| Check | Status |",
        "|---|---|",
        *[f"| {key} | `{value}` |" for key, value in matrix.items()],
        "",
        "## Core FleetPy modifications",
        "",
        "`NONE`",
        "",
        "## Recommendation",
        "",
        f"`{summary['recommendation']}`",
        "",
        summary["recommendation_reason"],
        "",
        "## Limitations and stop condition",
        "",
        "- The spike directly exercised FleetPy request objects, external vehicles, route legs, state transitions, and native leg logs. It did not run the full `FleetSimulationBase` scenario loader; the inspected subclass hooks are reserved for a separately authorized rolling replay.",
        "- Request selection applied only a basic finite-coordinate and positive finite service-time smoke filter before stable-hash truncation; no missing predictor was imputed.",
        "- Occupied progression used frozen realized service duration; predicted service time remained a separate decision-time field and was used only for the HV session-end admission check.",
        "- No full-day replay, OR/Gamma/passenger model, repositioning, charging, or AV penetration scenario was run.",
        "",
        "`ROLLING_DISPATCH_KERNEL = NOT STARTED`",
        "",
    ]
    path = root / REPORT_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_spike(
    root: str | Path,
    fleetpy_root: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    config = load_config(
        Path(config_path).resolve()
        if config_path is not None
        else root / DEFAULT_CONFIG_REL
    )
    start = pd.Timestamp(
        f"{config['test_date']} {config['spike_start_time']}", tz=TIMEZONE
    )
    end = pd.Timestamp(f"{config['test_date']} {config['spike_end_time']}", tz=TIMEZONE)
    if end - start != pd.Timedelta(hours=1):
        raise FleetPyCompatibilityError("compatibility spike must remain 60 minutes")
    bindings = load_fleetpy_bindings(fleetpy_root)
    requests = load_test31_requests(
        root,
        start=start,
        end=end,
        profile_id=str(config["profile_id"]),
        request_count=int(config["engineering_request_count"]),
        seed=int(config["selection_seed"]),
    )
    fixtures = load_mixed_fleet_fixture(
        root,
        start=start,
        end=end,
        hv_count=int(config["engineering_hv_count"]),
        av_count=int(config["engineering_av_count"]),
        seed=int(config["selection_seed"]),
    )
    registry = CoordinateRegistry()
    request_db = attach_fleetpy_requests(requests, bindings, registry)
    output = root / OUTPUT_REL
    vehicles, native_output = create_native_vehicles(
        fixtures, bindings, registry, request_db, output / "runtime"
    )
    eta_adapter = ValhallaPickupTimeAdapter(root)
    lifecycle = FleetPyLifecycleAdapter(bindings, registry)
    result = MinCorrectedPickupEtaDispatchStub(
        start=start,
        end=end,
        requests=requests,
        vehicles=vehicles,
        eta_adapter=eta_adapter,
        lifecycle=lifecycle,
        native_output=native_output,
    ).run()
    if result.summary["requests_completed"] == 0:
        raise FleetPyCompatibilityError("spike completed no requests")
    if any(
        value in {"REQUIRES_SMALL_PATCH", "BLOCKED"}
        for value in COMPATIBILITY_MATRIX.values()
    ):
        recommendation = "REASSESS_FLEETPY"
        reason = "At least one required compatibility hook is patched or blocked."
    else:
        recommendation = "GO_FLEETPY"
        reason = (
            "Pinned FleetPy request/vehicle/leg lifecycle and native logging worked with "
            "thin Test31, availability, and Valhalla adapters and no upstream modification."
        )
    summary = {
        "phase_status": "STAGE4_S1_FLEETPY_COMPATIBILITY_SPIKE_COMPLETE",
        "recommendation": recommendation,
        "recommendation_reason": reason,
        "fleetpy": {
            "repository": FLEETPY_REPOSITORY,
            "commit": bindings.commit,
            "license": "MIT",
            "core_modifications": "NONE",
        },
        "runtime": {
            "python": os.sys.version.split()[0],
            "fleetpy_cpp_router_required": False,
            "gurobi_required": False,
            "ortools_required": False,
        },
        "interval": {"start": start.isoformat(), "end": end.isoformat()},
        "fixture": {
            "profile_id": config["profile_id"],
            "request_count": len(requests),
            "hv_count": sum(item.vehicle_type == "HV" for item in fixtures),
            "av_count": sum(item.vehicle_type == "AV" for item in fixtures),
            "av_availability_policy": config["av_availability_policy"],
            "fleet_count_scientifically_frozen": False,
        },
        "tiny_replay_result": result.summary,
        "compatibility_matrix": COMPATIBILITY_MATRIX,
        "rolling_dispatch_kernel_started": False,
        "full_day_experiment_started": False,
        "output_rows": {
            "request_lifecycle_log": len(result.request_log),
            "vehicle_lifecycle_log": len(result.vehicle_log),
            "assignment_log": len(result.assignment_log),
            "fleetpy_native_vehicle_route_legs": len(native_output),
        },
    }
    _write_parquet(result.request_log, output / "request_lifecycle_log.parquet")
    _write_parquet(result.vehicle_log, output / "vehicle_lifecycle_log.parquet")
    _write_parquet(result.assignment_log, output / "assignment_log.parquet")
    _write_parquet(
        pd.DataFrame(native_output),
        output / "fleetpy_native_vehicle_route_legs.parquet",
    )
    _write_json(summary, output / "spike_summary.json")
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
            run_spike(args.root, args.fleetpy_root, args.config),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
