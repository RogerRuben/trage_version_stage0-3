"""Freeze a controlled one-hour input for Stage4/FleetPy cross-validation.

The generated package is simulator-neutral. Both engines must read the same
request and vehicle tables. This script prepares inputs only; it never claims
that a FleetPy run happened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from stage4.simulator_v3.routing_engine import haversine_m


DEFAULT_DEMAND = Path("stage4/data/decoupled_abm/demand_20161023_RT-Base.parquet")
DEFAULT_FLEET = Path("stage4/output/decoupled_environment/replication=1/simulation_fleet.parquet")
DEFAULT_OUTPUT = Path("stage4/output/fleetpy_cross_validation/input")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_key_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    values = frame[columns].astype(str).agg("|".join, axis=1).tolist()
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demand", type=Path, default=DEFAULT_DEMAND)
    parser.add_argument("--fleet", type=Path, default=DEFAULT_FLEET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window-start", default="2016-10-23T01:00:00Z")
    parser.add_argument("--window-duration-sec", type=int, default=3600)
    parser.add_argument("--max-orders", type=int, default=2000)
    parser.add_argument(
        "--vehicle-types",
        default="HV",
        help="Comma-separated types. HV-only isolates the kernel because FleetPy has no Stage4 ODD gate.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare(args: argparse.Namespace) -> dict:
    if not 500 <= int(args.max_orders) <= 2000:
        raise ValueError("FleetPy cross-validation requires 500-2,000 orders")
    start = pd.Timestamp(args.window_start)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = start + pd.Timedelta(seconds=int(args.window_duration_sec))

    demand = pd.read_parquet(args.demand)
    demand["simulated_request_time"] = pd.to_datetime(demand["simulated_request_time"], utc=True)
    requests = demand[
        (demand["simulated_request_time"] >= start)
        & (demand["simulated_request_time"] < end)
    ].copy()
    # FleetPy treats a same-node trip as a trivial request and omits it from
    # user statistics.  Exclude those requests so both engines evaluate the
    # same non-trivial execution lifecycle.
    requests = requests[
        requests["origin_zone"].astype(str).ne(requests["destination_zone"].astype(str))
    ].copy()
    requests = requests.sort_values(["simulated_request_time", "order_id"], kind="mergesort")
    # Cover the complete hour.  Taking head(N) creates an artificial burst in
    # the first few minutes and compares queueing policies rather than the
    # execution kernels requested by the validation protocol.
    if len(requests) > int(args.max_orders):
        positions = np.linspace(0, len(requests) - 1, int(args.max_orders), dtype=int)
        requests = requests.iloc[positions].copy()
    if len(requests) < 500:
        raise ValueError(f"Selected window contains only {len(requests)} orders")
    if requests["order_id"].astype(str).duplicated().any():
        raise ValueError("Request order_id is not unique")

    fleet = pd.read_parquet(args.fleet)
    fleet["online_start"] = pd.to_datetime(fleet["online_start"], utc=True)
    fleet["online_end"] = pd.to_datetime(fleet["online_end"], utc=True)
    vehicle_types = {x.strip() for x in str(args.vehicle_types).split(",") if x.strip()}
    # The comparison uses vehicles online for the complete controlled hour.
    # This gives both engines the same fixed supply and avoids comparing
    # FleetPy's static fleet with Stage4 session-boundary behaviour.
    vehicles = fleet[
        fleet["vehicle_type"].astype(str).isin(vehicle_types)
        & (fleet["online_start"] <= start)
        & (fleet["online_end"] >= end)
    ].copy()
    vehicles = vehicles.sort_values(["vehicle_type", "vehicle_id"], kind="mergesort")
    if vehicles.empty:
        raise ValueError("No vehicles overlap the selected window")
    if vehicles["vehicle_id"].astype(str).duplicated().any():
        raise ValueError("Vehicle vehicle_id is not unique")

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} exists; pass --overwrite")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    request_columns = [
        "order_id", "simulated_request_time", "origin_lon", "origin_lat", "origin_zone",
        "destination_lon", "destination_lat", "destination_zone", "predicted_service_time_sec",
        "realized_service_time_sec", "route_length_m", "eta_source",
    ]
    missing_request = sorted(set(request_columns) - set(requests.columns))
    if missing_request:
        raise ValueError(f"Demand missing canonical columns: {missing_request}")
    canonical_requests = requests[request_columns].copy()

    # Quantise both engines onto the same operational-zone graph.  FleetPy
    # then receives a compact 25-ish node road network while Stage4 receives
    # zone-time priors whose distance/time formula is identical to each edge.
    points = pd.concat([
        requests[["origin_zone", "origin_lon", "origin_lat"]].rename(
            columns={"origin_zone": "zone", "origin_lon": "lon", "origin_lat": "lat"}
        ),
        requests[["destination_zone", "destination_lon", "destination_lat"]].rename(
            columns={"destination_zone": "zone", "destination_lon": "lon", "destination_lat": "lat"}
        ),
        vehicles[["initial_zone", "initial_lon", "initial_lat"]].rename(
            columns={"initial_zone": "zone", "initial_lon": "lon", "initial_lat": "lat"}
        ),
    ], ignore_index=True)
    points["zone"] = points["zone"].astype(str)
    centers = points.groupby("zone", as_index=False)[["lon", "lat"]].median()
    center_map = centers.set_index("zone")[["lon", "lat"]].to_dict("index")

    def snap(frame: pd.DataFrame, zone_col: str, lon_col: str, lat_col: str) -> None:
        frame[zone_col] = frame[zone_col].astype(str)
        frame[lon_col] = frame[zone_col].map(lambda z: float(center_map[z]["lon"]))
        frame[lat_col] = frame[zone_col].map(lambda z: float(center_map[z]["lat"]))

    snap(canonical_requests, "origin_zone", "origin_lon", "origin_lat")
    snap(canonical_requests, "destination_zone", "destination_lon", "destination_lat")
    controlled_circuity = 1.20
    controlled_speed_mps = 8.0
    canonical_requests["route_length_m"] = canonical_requests.apply(
        lambda r: haversine_m(r.origin_lon, r.origin_lat, r.destination_lon, r.destination_lat)
        * controlled_circuity,
        axis=1,
    )
    canonical_requests["predicted_service_time_sec"] = (
        canonical_requests["route_length_m"] / controlled_speed_mps
    ).clip(lower=1.0)
    canonical_requests["realized_service_time_sec"] = canonical_requests["predicted_service_time_sec"]
    canonical_requests["eta_source"] = "fleetpy_controlled_zone_graph"
    canonical_requests["request_time_sec"] = (
        canonical_requests["simulated_request_time"] - start
    ).dt.total_seconds().round().astype(int)
    canonical_requests["simulated_request_time"] = (
        start + pd.to_timedelta(canonical_requests["request_time_sec"], unit="s")
    )
    canonical_requests["latest_pickup_time_sec"] = canonical_requests["request_time_sec"] + 480.0
    canonical_requests.to_parquet(args.output / "canonical_requests.parquet", index=False, compression="zstd")
    canonical_requests.to_csv(args.output / "fleetpy_requests.csv", index=False)

    # Private Stage4 data root: it contains the exact selected requests plus
    # the unchanged routing/ODD priors. This prevents --max-orders tie ordering
    # from silently changing the cross-validation request set.
    stage4_data = args.output / "stage4_data"
    stage4_data.mkdir(parents=True, exist_ok=True)
    controlled_demand = requests.copy().set_index("order_id")
    controlled = canonical_requests.set_index("order_id")
    for column in [
        "origin_lon", "origin_lat", "origin_zone", "destination_lon", "destination_lat",
        "destination_zone", "predicted_service_time_sec", "realized_service_time_sec",
        "route_length_m", "eta_source", "simulated_request_time",
    ]:
        controlled_demand[column] = controlled[column]
    controlled_demand.reset_index().to_parquet(
        stage4_data / "demand_20161023_RT-Base.parquet", index=False, compression="zstd"
    )
    speed_rows = [
        {"origin_zone": zone, "time_bin": time_bin, "empty_speed_mps": controlled_speed_mps}
        for zone in centers["zone"] for time_bin in range(48)
    ]
    pd.DataFrame(speed_rows).to_parquet(stage4_data / "pickup_empty_speed_by_zone_time.parquet", index=False)
    pd.DataFrame({
        "origin_zone": centers["zone"],
        "circuity_factor": controlled_circuity,
    }).to_parquet(stage4_data / "pickup_circuity_by_zone.parquet", index=False)
    # HV-only validation does not apply ODD, but the Stage4 loader requires a
    # complete proxy table.  Its provenance is explicit and never interpreted
    # as empirical AV capability.
    pair_rows = [
        {"origin_zone": a, "destination_zone": b, "pickup_odd_feasible": True,
         "pickup_odd_proxy_source": "fleetpy_controlled_validation_only"}
        for a in centers["zone"] for b in centers["zone"]
    ]
    pd.DataFrame(pair_rows).to_parquet(stage4_data / "pickup_odd_zone_pair_proxy.parquet", index=False)
    source_zone = args.demand.parent / "operational_zone_system.json"
    shutil.copy2(source_zone, stage4_data / "operational_zone_system.json")

    vehicle_columns = [
        "vehicle_id", "vehicle_type", "initial_lon", "initial_lat", "initial_zone",
        "online_start", "online_end",
    ]
    missing_vehicle = sorted(set(vehicle_columns) - set(vehicles.columns))
    if missing_vehicle:
        raise ValueError(f"Fleet missing canonical columns: {missing_vehicle}")
    canonical_vehicles = vehicles[vehicle_columns].copy()
    snap(canonical_vehicles, "initial_zone", "initial_lon", "initial_lat")
    # Controlled kernel validation supplies one initially co-located idle
    # vehicle per request (remaining vehicles keep their frozen origin).  This
    # isolates request/leg execution from FleetPy's insertion heuristic and
    # Stage4's window-level global assignment policy.
    n_colocated = min(len(canonical_requests), len(canonical_vehicles))
    canonical_vehicles.loc[canonical_vehicles.index[:n_colocated], "initial_zone"] = (
        canonical_requests["origin_zone"].iloc[:n_colocated].to_numpy()
    )
    snap(canonical_vehicles, "initial_zone", "initial_lon", "initial_lat")
    canonical_vehicles["online_start"] = start
    canonical_vehicles["online_end"] = end + pd.Timedelta(hours=2)
    canonical_vehicles["online_start_sec"] = (
        canonical_vehicles["online_start"].clip(lower=start) - start
    ).dt.total_seconds()
    canonical_vehicles["online_end_sec"] = (
        canonical_vehicles["online_end"].clip(upper=end) - start
    ).dt.total_seconds()
    canonical_vehicles.to_parquet(args.output / "canonical_vehicles.parquet", index=False, compression="zstd")
    canonical_vehicles.to_csv(args.output / "fleetpy_vehicles.csv", index=False)

    env_dir = args.output / "stage4_environment" / "replication=1"
    env_dir.mkdir(parents=True, exist_ok=True)
    controlled_fleet = vehicles.copy().set_index("vehicle_id")
    controlled_vehicle_index = canonical_vehicles.set_index("vehicle_id")
    for column in ["initial_lon", "initial_lat", "initial_zone", "online_start", "online_end"]:
        controlled_fleet[column] = controlled_vehicle_index[column]
    controlled_fleet.reset_index().to_parquet(
        env_dir / "simulation_fleet.parquet", index=False, compression="zstd"
    )
    controlled_preassignment = {
        "version": "fleetpy_controlled_cross_validation",
        "preassignment_enabled_default": False,
        "preassignment_horizon_sec": 300,
        "max_reserved_orders_per_vehicle": 1,
        "driver_response_delay_sec": 0,
    }
    (args.output / "controlled_preassignment_config.json").write_text(
        json.dumps(controlled_preassignment, indent=2), encoding="utf-8"
    )

    centers = centers.sort_values("zone", kind="mergesort").reset_index(drop=True)
    centers["node_index"] = np.arange(len(centers), dtype=int)
    centers.to_csv(args.output / "controlled_zone_nodes.csv", index=False)

    manifest = {
        "status": "INPUT_PREPARED",
        "comparison_scope": "one_hour_controlled_kernel_cross_validation",
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "window_duration_sec": int(args.window_duration_sec),
        "request_count": int(len(canonical_requests)),
        "vehicle_count": int(len(canonical_vehicles)),
        "vehicle_types": sorted(vehicle_types),
        "controlled_zone_count": int(len(centers)),
        "controlled_circuity_factor": controlled_circuity,
        "controlled_empty_speed_mps": controlled_speed_mps,
        "passenger_patience_sec": 480,
        "operation": "Stay",
        "preassignment": False,
        "strategy": "Safe GlobalMatch-MinPickup / closest-vehicle control",
        "stage4_environment_variables": {"SIMULATOR_V3_MIN_LEG_DURATION_SEC": "0"},
        "demand_source": str(args.demand.resolve()),
        "demand_source_sha256": file_sha256(args.demand),
        "fleet_source": str(args.fleet.resolve()),
        "fleet_source_sha256": file_sha256(args.fleet),
        "request_key_sha256": frame_key_hash(canonical_requests, ["order_id", "simulated_request_time"]),
        "vehicle_key_sha256": frame_key_hash(canonical_vehicles, ["vehicle_id", "online_start", "online_end"]),
        "stage4_command": [
            "python", "stage4/scripts/run_simulator_v3.py", "--replication", "1",
            "--strategy", "Safe GlobalMatch-MinPickup", "--operation", "O0",
            "--request-time-scenario", "RT-Base", "--data-root",
            str((args.output / "stage4_data").resolve()), "--environment-root",
            str((args.output / "stage4_environment").resolve()), "--min-request-time",
            start.isoformat(), "--max-orders", str(len(canonical_requests)),
            "--decision-epoch-sec", "1", "--preassignment-config",
            str((args.output / "controlled_preassignment_config.json").resolve()),
            "--output-root", str((args.output.parent / "stage4_results").resolve()),
            "--results-dir", str((args.output.parent / "stage4_metrics").resolve()), "--overwrite",
        ],
        "fleetpy_install_required": True,
        "fleetpy_repository": "https://github.com/TUM-VT/FleetPy.git",
        "fleetpy_expected_version": "1.0.2-or-pinned-commit",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    print(json.dumps(prepare(parse_args()), indent=2))


if __name__ == "__main__":
    main()
