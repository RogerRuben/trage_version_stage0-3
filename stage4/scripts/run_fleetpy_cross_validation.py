"""Run an actual FleetPy 1.0.2 controlled-hour comparison workload.

The script builds a compact complete operational-zone graph, freezes exact
vehicle initial nodes, and invokes FleetPy's own ``run_scenarios`` entrypoint.
It never substitutes a local mock for FleetPy.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from stage4.simulator_v3.routing_engine import haversine_m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("stage4/output/fleetpy_cross_validation/input"))
    parser.add_argument("--fleetpy-root", type=Path, default=Path("stage4/output/fleetpy_runtime"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows[1:], columns=rows[0]).to_csv(path, index=False)


def prepare_runtime(args: argparse.Namespace) -> tuple[Path, Path, dict]:
    manifest = json.loads((args.input / "manifest.json").read_text(encoding="utf-8"))
    requests = pd.read_parquet(args.input / "canonical_requests.parquet")
    vehicles = pd.read_parquet(args.input / "canonical_vehicles.parquet")
    nodes = pd.read_csv(args.input / "controlled_zone_nodes.csv")
    nodes["zone"] = nodes["zone"].astype(str)
    node_map = nodes.set_index("zone")["node_index"].astype(int).to_dict()

    network_name = "stage4_cv_network"
    demand_name = "stage4_cv_demand"
    study_name = "stage4_cv"
    scenario_name = "stage4_cv_fleetpy"
    base = args.fleetpy_root / "data" / "networks" / network_name / "base"
    demand_dir = args.fleetpy_root / "data" / "demand" / demand_name / "matched" / network_name
    vehicle_dir = args.fleetpy_root / "data" / "vehicles"
    scenario_dir = args.fleetpy_root / "studies" / study_name / "scenarios"
    initial_dir = args.fleetpy_root / "studies" / study_name / "results" / "stage4_cv_initial"
    result_dir = args.fleetpy_root / "studies" / study_name / "results" / scenario_name
    for path in [base, demand_dir, vehicle_dir, scenario_dir, initial_dir]:
        path.mkdir(parents=True, exist_ok=True)
    if result_dir.exists() and args.overwrite:
        shutil.rmtree(result_dir)
    elif result_dir.exists():
        raise FileExistsError(f"{result_dir} exists; pass --overwrite")

    fleet_nodes = pd.DataFrame({
        "node_index": nodes["node_index"].astype(int),
        "is_stop_only": False,
        "pos_x": nodes["lon"].astype(float),
        "pos_y": nodes["lat"].astype(float),
    })
    fleet_nodes.to_csv(base / "nodes.csv", index=False)
    n = len(nodes)
    distance = np.zeros((n, n), dtype=float)
    travel_time = np.zeros((n, n), dtype=float)
    edges: list[dict] = []
    circ = float(manifest["controlled_circuity_factor"])
    speed = float(manifest["controlled_empty_speed_mps"])
    for a in nodes.itertuples(index=False):
        for b in nodes.itertuples(index=False):
            if int(a.node_index) == int(b.node_index):
                continue
            d = haversine_m(a.lon, a.lat, b.lon, b.lat) * circ
            tt = max(d / speed, 0.001)
            distance[int(a.node_index), int(b.node_index)] = d
            travel_time[int(a.node_index), int(b.node_index)] = tt
            edges.append({
                "from_node": int(a.node_index), "to_node": int(b.node_index),
                "distance": d, "travel_time": tt,
                "source_edge_id": f"{int(a.node_index)}_{int(b.node_index)}",
            })
    pd.DataFrame(edges).to_csv(base / "edges.csv", index=False)
    np.save(base / "dis_matrix.npy", distance)
    np.save(base / "tt_matrix.npy", travel_time)
    (base / "crs.info").write_text("EPSG:4326", encoding="utf-8")

    fleetpy_requests = pd.DataFrame({
        "rq_time": requests["request_time_sec"].round().astype(int),
        "start": requests["origin_zone"].astype(str).map(node_map).astype(int),
        "end": requests["destination_zone"].astype(str).map(node_map).astype(int),
        "request_id": np.arange(len(requests), dtype=int),
    })
    fleetpy_requests.to_csv(demand_dir / "stage4_cv_requests.csv", index=False)
    pd.DataFrame({
        "vtype_name_full": ["stage4_cv_vehicle"],
        "maximum_passengers": [1],
        "daily_fix_cost [cent]": [0],
        "per_km_cost [cent]": [0],
        "battery_size [kWh]": [100],
        "range [km]": [100000],
        "source": ["Stage4 controlled FleetPy cross-validation"],
    }).T.to_csv(vehicle_dir / "stage4_cv_vehicle.csv", header=False)

    init = pd.DataFrame({
        "operator_id": 0,
        "vehicle_id": np.arange(len(vehicles), dtype=int),
        "final_soc": 1.0,
        "final_node_index": vehicles["initial_zone"].astype(str).map(node_map).astype(int),
        "final_time": 0.0,
    })
    init.to_csv(initial_dir / "final_state.csv")

    constant = [
        ["Input_Parameter_Name", "Parameter_Value"],
        ["initial_state_scenario", "stage4_cv_initial"],
        ["sim_env", "ImmediateDecisionsSimulation"],
        ["network_type", "NetworkBasicWithStore"],
        ["network_name", network_name],
        ["demand_name", demand_name],
        ["random_seed", 1],
        ["start_time", 0],
        ["end_time", int(manifest["window_duration_sec"])],
        ["time_step", 1],
        ["route_output_flag", True],
        ["replay_flag", True],
        ["nr_mod_operators", 1],
        ["rq_type", "BasicRequest"],
        ["user_max_decision_time", 0],
        ["op_min_wait_time", 0],
        ["op_max_wait_time", int(manifest["passenger_patience_sec"])],
        ["op_max_detour_time_factor", 1],
        ["op_const_boarding_time", 0],
        ["op_add_boarding_time", 0],
        ["op_base_fare", 0],
        ["op_distance_fare", 0],
        ["op_time_fare", 0],
        ["op_min_standard_fare", 0],
        ["op_vr_control_func_dict", "func_key:distance_and_user_times;vot:1.0"],
        ["op_reoptimisation_timestep", 1],
    ]
    write_csv(scenario_dir / "constant_config.csv", constant)
    write_csv(scenario_dir / "scenarios.csv", [
        ["op_module", "scenario_name", "rq_file", "op_fleet_composition"],
        ["PoolingIRSOnly", scenario_name, "stage4_cv_requests.csv", f"stage4_cv_vehicle:{len(vehicles)}"],
    ])
    return scenario_dir / "constant_config.csv", scenario_dir / "scenarios.csv", {
        "result_dir": str(result_dir.resolve()),
        "request_count": int(len(requests)),
        "vehicle_count": int(len(vehicles)),
        "network_node_count": int(n),
        "network_edge_count": int(len(edges)),
    }


def main() -> None:
    args = parse_args()
    constant, scenarios, run_manifest = prepare_runtime(args)
    root = args.fleetpy_root.resolve()
    sys.path.insert(0, str(root))
    os.environ["PYTHONPATH"] = str(root)
    from run_scenarios import run_scenarios  # type: ignore

    run_scenarios(
        str(constant.resolve()), str(scenarios.resolve()),
        n_parallel_sim=1, n_cpu_per_sim=1, evaluate=0,
        log_level="warning", keep_old=False,
    )
    result_dir = Path(run_manifest["result_dir"])
    user_files = list(result_dir.glob("*user-stats.csv")) + list(result_dir.glob("*user_stats.csv"))
    op_files = list(result_dir.glob("*op-stats.csv"))
    if not user_files or not op_files:
        raise RuntimeError(f"FleetPy did not produce required actual outputs in {result_dir}")
    run_manifest.update({
        "status": "FLEETPY_ACTUAL_RUN_COMPLETED",
        "fleetpy_source_commit": "053aa9d4",
        "user_stats": [str(p.resolve()) for p in user_files],
        "operator_stats": [str(p.resolve()) for p in op_files],
    })
    out = args.input.parent / "fleetpy_actual_run_manifest.json"
    out.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(json.dumps(run_manifest, indent=2))


if __name__ == "__main__":
    main()
