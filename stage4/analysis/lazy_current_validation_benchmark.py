"""Four fixed known-parked as-of subgraphs, real C checks, no native replay."""
from __future__ import annotations

from time import perf_counter
STARTED = perf_counter()

import argparse
from datetime import datetime
import gc
import hashlib
import json
from math import floor
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from .competitive_chain_benchmark import atomic_json, load_inputs, sha
from .scheme_a_layout_mechanism import ParkedInventory
from stage4.dispatch.candidate_graph import SpatialVehicle, SparseCandidateIndex, search_radius_m
from stage4.dispatch.city_pickup_support import CityPickupSupportValidator
from stage4.dispatch.controlled_fleet import build_controlled_fleet
from stage4.dispatch.controlled_routes import ControlledEmptyRouter
from stage4.dispatch.deterministic_routing import SINGLE_SOURCE_MATRIX
from stage4.dispatch.lazy_current_validation import CheckedAction, select_with_validation
from stage4.dispatch.routing_v4 import StaticRawRoutingAdapter
from stage4.dispatch.scheme_a_city_graph import CoarseHistoricalLibrary, RouteState
from stage4.dispatch.scheme_a_city_master import CurrentAction
from stage4.dispatch.scheme_a_two_scale_value import CoarseServiceValue
from stage4.dispatch.solver import AssignmentArc
from stage4.dispatch.waiting_contract import assignment_feasible
from stage4.fleetpy_adapter.upstream import CoordinateRegistry

CONFIG = Path("stage4/config/lazy_current_validation_v1.json")
OUT = Path("stage4/output/capability_chain_planning/lazy_current_validation_v1/run_001")
DOC = Path("stage4/docs/capability_chain_planning/lazy_current_validation_v1")
DAY_START = pd.Timestamp("2016-10-31T00:00:00+08:00")
BASE = Path("stage4/input/replay_foundation/stage4_order_replay_base.parquet")
ROUTES = Path("stage4/output/symmetric_flexibility_v1/input/test31_research_routes.parquet")


def stamp():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds")


class ResourceGuard:
    def __init__(self, cfg):
        self.cfg, self.process, self.peak = cfg, psutil.Process(), 0.
        self.check(initial=True)

    def check(self, initial=False):
        memory = self.process.memory_info()
        self.peak = max(self.peak, memory.rss/2**20, getattr(memory, "peak_wset", memory.rss)/2**20)
        available = psutil.virtual_memory().available/2**20
        if self.peak > self.cfg["maximum_rss_mib"]:
            raise MemoryError("fixed 768 MiB process cap exceeded")
        threshold = self.cfg["minimum_initial_available_ram_mib"] if initial else self.cfg["minimum_remaining_ram_mib"]
        if available < threshold:
            raise MemoryError("available RAM below the fixed resource reserve")
        if perf_counter()-STARTED >= self.cfg["batch_time_limit_s"]:
            raise TimeoutError("fixed 600-second prototype budget exhausted")


def load_source(root, cfg):
    source = root/cfg["source_native"]
    summary = json.loads((source/"summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "TWO_SCALE_CITY_GROUP_COMPLETE":
        raise ValueError("completed source run required")
    protocol = dict(summary["frozen_protocol"])
    if (protocol["requested_q_a"] != .1 or protocol["supply_seed"] != 20261013
            or protocol["profile_id"] != "C" or protocol["max_pickup_eta_s"] != 600):
        raise ValueError("fixed source fleet/profile differs")
    binding = {CONFIG.as_posix(): sha(root/CONFIG)}
    inputs = (source/"summary.json", source/"assignments.parquet",
              source/"empty_movements.parquet", source/"cohort_outcomes.parquet",
              root/BASE, root/ROUTES, root/protocol["source_layout"],
              root/"stage4/input/replay_foundation/replay_fleet_template.parquet")
    for path in inputs:
        binding[path.relative_to(root).as_posix()] = sha(path)
    jobs = pd.read_parquet(source/"assignments.parquet", columns=[
        "simulation_time_s", "native_vehicle_id", "native_request_id", "vehicle_id", "vehicle_type",
        "service_end_time_s", "service_end_time", "completion_lon_wgs84", "completion_lat_wgs84"])
    moves = pd.read_parquet(source/"empty_movements.parquet", columns=[
        "native_vehicle_id", "vehicle_type", "start_time_s", "end_time_s", "end_lon_wgs84",
        "end_lat_wgs84", "start_lon_wgs84", "start_lat_wgs84", "status", "shape_polyline6",
        "geometry_timestamp", "actual_distance_m"])
    outcomes = pd.read_parquet(source/"cohort_outcomes.parquet", columns=[
        "order_id", "native_request_id", "release_s", "C_compatible", "passenger_accepts_av",
        "first_attempt_s"])
    base = pd.read_parquet(root/BASE, columns=[
        "order_id", "profile_id", "pickup_lon_wgs84", "pickup_lat_wgs84", "dropoff_lon_wgs84", "dropoff_lat_wgs84"])
    # The existing replay base has one copy per C/M/A, not three requests.
    base = base.loc[base.profile_id.eq("C")].drop(columns="profile_id")
    predicted = pd.read_parquet(root/ROUTES, columns=["order_id", "predicted_route_time_p50_s"])
    requests = outcomes.merge(base, on="order_id", validate="one_to_one").merge(
        predicted, on="order_id", validate="one_to_one")
    if len(requests) != len(outcomes):
        raise ValueError("source request/coordinate identity mismatch")
    fleet = build_controlled_fleet(root, benchmark_start=DAY_START,
        simulation_end=DAY_START+pd.Timedelta(seconds=protocol["physical_drain_limit_s"]),
        requested_q_a=.1, seed=20261013)
    plans = fleet.scenario_fleet.copy()
    mapping = plans.set_index("native_id").vehicle_id.astype(str).to_dict()
    if any(mapping.get(int(r.native_vehicle_id)) != str(r.vehicle_id) for r in jobs.itertuples(index=False)):
        raise ValueError("source native vehicle identity differs")
    layout = json.loads((root/protocol["source_layout"]).read_text(encoding="utf-8"))["placements"]["compatible_hot"]
    av = plans.vehicle_type.eq("AV")
    for index, row in plans.loc[av].iterrows():
        point = layout[str(row.vehicle_id)]["position"]
        plans.loc[index, ["initial_lon_wgs84", "initial_lat_wgs84"]] = point
    plans["activation_s"] = (plans.availability_start_time-DAY_START).dt.total_seconds()
    plans["admission_end_s"] = (plans.availability_end_time-DAY_START).dt.total_seconds()
    del fleet, base, predicted
    return protocol, binding, jobs, moves, requests, plans


def reconstruct(now, jobs, moves, requests, plans, cfg):
    # Remove actions at the current epoch; keep only already-started histories.
    past_jobs = jobs.loc[jobs.simulation_time_s.lt(now)]
    past_moves = moves.loc[moves.start_time_s.lt(now)]
    inventory = ParkedInventory(plans, past_jobs, past_moves)
    inventory.advance(now)
    mask = inventory.admitted & (inventory.busy == 0) & inventory.known
    parked = plans.loc[mask].copy()
    parked[["current_lon", "current_lat"]] = inventory.points[mask]
    booked = set(past_jobs.native_request_id.astype(int))
    pending = requests.loc[requests.release_s.le(now)
        & (requests.release_s+300>now) & requests.first_attempt_s.le(now)
        & ~requests.native_request_id.isin(booked)].copy()
    pending["selection"] = pending.order_id.astype(str).map(
        lambda order: hashlib.sha256(f"{cfg['selection_seed']}|{now}|{order}".encode()).hexdigest())
    pending = pending.sort_values(["selection", "native_request_id"], kind="stable").head(
        cfg["maximum_waiting_requests"])
    return parked, pending, past_jobs, past_moves, dict(
        all_pending_before_cap=int(((requests.release_s.le(now)) & (requests.release_s+300>now)
            & requests.first_attempt_s.le(now) & ~requests.native_request_id.isin(booked)).sum()),
        selected_pending=len(pending), known_parked=len(parked),
        unknown_free_positions=inventory.unknown_free,
        busy_or_moving_excluded=int(np.count_nonzero(inventory.admitted & (inventory.busy != 0))),
        exact_native_epoch_reproduction=False)


def make_context(now, parked, pending, past_jobs, past_moves, requests, protocol, eta):
    registry = CoordinateRegistry()
    by_rid = {}
    for row in requests.itertuples(index=False):
        rid = int(row.native_request_id)
        by_rid[rid] = NS(native_id=rid, order_id=str(row.order_id), sim_time_s=float(row.release_s),
            pickup_lon_wgs84=float(row.pickup_lon_wgs84), pickup_lat_wgs84=float(row.pickup_lat_wgs84),
            dropoff_lon_wgs84=float(row.dropoff_lon_wgs84), dropoff_lat_wgs84=float(row.dropoff_lat_wgs84),
            dropoff_position=registry.position_for(float(row.dropoff_lon_wgs84), float(row.dropoff_lat_wgs84)),
            predicted_service_time_s=float(row.predicted_route_time_p50_s))
    runtime, states, spatial = {}, {}, []
    for row in parked.itertuples(index=False):
        vid = int(row.native_id)
        point = float(row.current_lon), float(row.current_lat)
        fixture = NS(native_id=vid, vehicle_id=str(row.vehicle_id), vehicle_type=str(row.vehicle_type))
        runtime[vid] = NS(fixture=fixture, native_vehicle=NS(pos=registry.position_for(*point),
                            status=0, assigned_route=[]))
        states[vid] = RouteState(point, now, float(row.admission_end_s),
                                "C" if row.vehicle_type == "AV" else "HV")
        spatial.append(SpatialVehicle(str(row.vehicle_id), vid, str(row.vehicle_type), *point))
    meta = {}
    for row in pending.itertuples(index=False):
        rid = int(row.native_request_id)
        meta[rid] = dict(order_id=str(row.order_id), pickup_deadline_s=float(row.release_s)+300,
            research_route_compatible=bool(row.C_compatible),
            passenger_accepts_av=bool(row.passenger_accepts_av),
            failed_round_count=max(0, int((now-float(row.first_attempt_s))//30)),
            carry_over_flag=float(row.first_attempt_s)<now,
            critical=0< float(row.release_s)+300-now <=30)
    assignment_rows, completed = [], set()
    for row in past_jobs.sort_values("simulation_time_s").itertuples(index=False):
        observed = float(row.service_end_time_s)<=now
        if observed:
            completed.add(int(row.native_request_id))
        assignment_rows.append(dict(native_vehicle_id=int(row.native_vehicle_id),
            native_request_id=int(row.native_request_id), completed=observed,
            service_end_time=row.service_end_time if observed else None))
    observed_moves = []
    for row in past_moves.itertuples(index=False):
        if float(row.end_time_s)>now:
            continue
        move = row._asdict()
        move["destination_position"] = registry.position_for(float(row.end_lon_wgs84), float(row.end_lat_wgs84))
        observed_moves.append(move)
    last_move_start = {}
    for row in past_moves.itertuples(index=False):
        vid = int(row.native_vehicle_id)
        if float(row.actual_distance_m or 0.) > 0:
            last_move_start[vid] = max(last_move_start.get(vid, -float("inf")), float(row.start_time_s))
    c = NS(config=protocol, routing_engine=registry, runtime_by_vid=runtime,
        request_by_rid=by_rid, request_meta=meta, assignment_rows=assignment_rows,
        completed_rids=completed, eta_adapter=eta, _timestamp=lambda s:DAY_START+pd.Timedelta(seconds=float(s)),
        repositioning_manager=NS(rows=observed_moves, active={},
            has_movement_since=lambda vid, finish:last_move_start.get(int(vid), -float("inf")) >= finish),
        city_departure_contexts={},
        request_decisions=None)
    return c, states, spatial


def qvalue(provider, state):
    return int(floor(3*provider.value(state)+.5))


def graph_inputs(now, c, states, spatial, pending, eta, provider, cfg):
    index = SparseCandidateIndex(spatial, cached_geometry=True)
    original, base, base_units, unknown, upper_units = {}, [], {}, [], {}
    for vid, state in sorted(states.items()):
        aid = f"V{vid}:WAIT"
        base.append(CurrentAction(aid, vid, "WAIT", None)); base_units[aid] = 0
    wait_values = {vid:qvalue(provider, state) for vid, state in states.items()}
    for row in pending.itertuples(index=False):
        rid, request = int(row.native_request_id), c.request_by_rid[int(row.native_request_id)]
        info = c.request_meta[rid]
        radius = search_radius_m(info["failed_round_count"], 2000., 1000., 8000.)
        found, _ = index.query(request.pickup_lon_wgs84, request.pickup_lat_wgs84, radius,
            cfg["current_top_k"], info["research_route_compatible"] and info["passenger_accepts_av"])
        vehicles = [vehicle for vehicle, _ in found]
        estimates = eta.estimate_many(vehicles, request.pickup_lon_wgs84,
                                     request.pickup_lat_wgs84, c._timestamp(now))
        for vehicle in vehicles:
            estimate = estimates.get(vehicle.native_vehicle_id)
            if estimate is None or not assignment_feasible(c.config, now, info["pickup_deadline_s"],
                                                            estimate.corrected_pickup_eta_s):
                continue
            vid, state = int(vehicle.native_vehicle_id), states[int(vehicle.native_vehicle_id)]
            aid = f"V{vid}:SERVE:{rid}"
            arc = AssignmentArc(vid, rid, float(estimate.corrected_pickup_eta_s),
                bool(info["critical"]), bool(info["carry_over_flag"]),
                (c.runtime_by_vid[vid], request, estimate), vehicle_type=vehicle.vehicle_type)
            original[aid] = arc
            after = RouteState((request.dropoff_lon_wgs84, request.dropoff_lat_wgs84),
                now+request.predicted_service_time_s+(0. if vehicle.vehicle_type=="AV" else arc.pickup_eta_s),
                state.admission_end_s, state.profile_id)
            delta = qvalue(provider, after)-wait_values[vid]
            if vehicle.vehicle_type == "AV":
                unknown.append(CurrentAction(aid, vid, "SERVE", rid, arc.critical, arc.carry_over, 0.))
                upper_units[aid] = delta
            else:
                base.append(CurrentAction(aid, vid, "SERVE", rid, arc.critical, arc.carry_over,
                                          float(estimate.route_distance_m)))
                base_units[aid] = delta
    return original, base, base_units, unknown, upper_units, wait_values


class RealOracle:
    def __init__(self, validator, c, now, states, provider, original, wait_values, guard):
        self.validator, self.c, self.now = validator, c, now
        self.states, self.provider, self.original = states, provider, original
        self.wait_values, self.guard = wait_values, guard
        self.outcomes = {}

    def __call__(self, ids):
        self.guard.check()
        arcs = [self.original[aid] for aid in ids]
        valid = self.validator(self.c, arcs, self.now)
        if self.validator.last_diagnostics["rejection_reason_counts"].get(
                "PICKUP_SUPPORT_ROUTING_BUDGET_EXHAUSTED", 0):
            raise RuntimeError("unresolved routing budget cannot be treated as infeasibility")
        accepted = {(int(arc.vehicle_id), int(arc.request_id)): arc for arc in valid}
        answers = {}
        for aid in ids:
            source = self.original[aid]
            arc = accepted.get((source.vehicle_id, source.request_id))
            if arc is None:
                answers[aid] = None
                self.outcomes[aid] = dict(admitted=False)
                continue
            request, state = arc.payload[1], self.states[arc.vehicle_id]
            after = RouteState((request.dropoff_lon_wgs84, request.dropoff_lat_wgs84),
                self.now+arc.pickup_eta_s+request.predicted_service_time_s,
                state.admission_end_s, state.profile_id)
            delta = qvalue(self.provider, after)-self.wait_values[arc.vehicle_id]
            action = CurrentAction(aid, arc.vehicle_id, "SERVE", arc.request_id, arc.critical, arc.carry_over,
                float(arc.payload[2].route_distance_m),
                dict(validated_arc=arc, pickup_path=self.validator.pickup_paths.get((arc.vehicle_id, arc.request_id))))
            answers[aid] = CheckedAction(action, delta)
            self.outcomes[aid] = dict(admitted=True, distance_m=action.empty_distance_m,
                                     units=delta, eta_s=arc.pickup_eta_s)
        self.guard.check()
        return answers


def reset_oracle(validator, empty, c):
    validator.cache.clear()
    empty.cache.clear()
    validator.seam_evidence._idle_routes.clear()
    c.city_departure_contexts = {}


def run(root, config=CONFIG, prepare_only=False):
    cfg = json.loads((root/config).read_text(encoding="utf-8"))
    fixed = dict(anchors_s=[27000,30600,61200,64800],selection_seed=20261014,
        maximum_waiting_requests=24,current_top_k=20,modes=["EAGER","LAZY_CERTIFIED"],
        mode_time_limit_s=10,batch_time_limit_s=600,maximum_rss_mib=768,
        minimum_initial_available_ram_mib=1536,minimum_remaining_ram_mib=512,
        route_workers=1,unknown_distance_lower_m=0.,native_replay=False,gpu_used=False)
    if any(cfg.get(key) != value for key,value in fixed.items()):
        raise ValueError("fixed graph selection changed")
    guard = ResourceGuard(cfg)
    protocol, binding, jobs, moves, requests, plans = load_source(root, cfg)
    if prepare_only:
        rows = [dict(tick=now, **reconstruct(now,jobs,moves,requests,plans,cfg)[4]) for now in cfg["anchors_s"]]
        guard.check()
        print(json.dumps(dict(scope="READ_ONLY_INPUT_RECONSTRUCTION_NO_ROUTING", states=rows,
            peak_rss_mib=guard.peak),ensure_ascii=False,indent=2),flush=True)
        return
    output, public = root/OUT, root/DOC
    if output.exists():
        raise FileExistsError("preserve existing fixed prototype; no silent rerun")
    output.mkdir(parents=True)
    code_sha = subprocess.check_output(["git","-c",f"safe.directory={root.as_posix()}","rev-parse","HEAD"],
                                      cwd=root,text=True).strip()
    record = dict(status="RUNNING", code_sha=code_sha, started_at=stamp(), pid=psutil.Process().pid,
        inputs=binding, expected_graphs=4, graph_scope="KNOWN_PARKED_ASOF_SUBGRAPH_NOT_EXACT_NATIVE_REPLAY",
        frozen_city_controller_modified=False, rows=[], gpu_used=False)
    eta = None
    try:
        with threadpool_limits(limits=1):
            templates, reference, _ = load_inputs(root, protocol, binding)
            library = CoarseHistoricalLibrary(templates, protocol)
            provider = CoarseServiceValue(library, reference, templates, protocol)
            eta = StaticRawRoutingAdapter(root, routing_mode=SINGLE_SOURCE_MATRIX, route_workers=cfg["route_workers"],
                persistent_cache_size=5000, raw_od_cache_size=5000, static_raw_od=True)
            empty = ControlledEmptyRouter(root, eta)
            validator = CityPickupSupportValidator(empty, geometry_timestamp=DAY_START, routing_budget_s=60.)
            print(json.dumps(dict(event="STARTED",time=stamp(),pid=record["pid"],code_sha=code_sha)),flush=True)
            for case_index, now in enumerate(cfg["anchors_s"]):
                guard.check()
                before = perf_counter()
                parked, pending, past_jobs, past_moves, inventory = reconstruct(now,jobs,moves,requests,plans,cfg)
                c, states, spatial = make_context(now,parked,pending,past_jobs,past_moves,requests,protocol,eta)
                future = provider.refresh(now, list(states.values()))
                matrix_before = eta.routing_queries
                original, base, units, unknown, upper, wait = graph_inputs(
                    now,c,states,spatial,pending,eta,provider,cfg)
                wanted = {str(c.request_by_rid[arc.request_id].order_id) for arc in original.values()
                          if arc.vehicle_type=="AV"}
                by_vid = {arc.vehicle_id for arc in original.values() if arc.vehicle_type=="AV"}
                for row in reversed(c.assignment_rows):
                    if row["native_vehicle_id"] in by_vid and row["completed"]:
                        request = c.request_by_rid.get(row["native_request_id"])
                        if request is not None:
                            wanted.add(request.order_id)
                        by_vid.remove(row["native_vehicle_id"])
                setup = validator.prepare_private_identity_store(wanted)
                common_s = perf_counter()-before
                case_id = f"H{now//3600:02d}{now%3600//60:02d}"
                case = dict(case_id=case_id, **inventory, known_actions=len(base), unknown_C_arcs=len(unknown),
                    common_setup_time_s=common_s, common_matrix_backend_queries=eta.routing_queries-matrix_before,
                    coarse_available=future["available"], coarse_status=future["status"],
                    modes={}, private_identity_setup=setup)
                order = cfg["modes"] if case_index%2==0 else list(reversed(cfg["modes"]))
                for mode in order:
                    reset_oracle(validator, empty, c)
                    oracle = RealOracle(validator,c,now,states,provider,original,wait,guard)
                    before_diag = empty.diagnostics()
                    idle_before = validator.seam_evidence.departure_context_route_count
                    result = select_with_validation(base,units,unknown,upper,oracle,mode=mode,
                        time_limit_s=cfg["mode_time_limit_s"],check=guard.check)
                    after_diag = empty.diagnostics()
                    row = dict(case_id=case_id,mode=mode,known_parked=inventory["known_parked"],
                        pending=inventory["selected_pending"],unknown_C_arcs=len(unknown),
                        objective=list(result["score"]),matched=result["score"][1],
                        full_objective_certified=result["full_current_objective_certified"],
                        oracle_calls=result["oracle_calls"],unqueried=result["unqueried_actions"],
                        query_batches=result["query_batches"],mode_time_s=result["total_time_s"],
                        flow_time_s=result["solve_time_s"],validation_time_s=result["oracle_time_s"],
                        scalar_route_calls=after_diag["scalar_route_calls"]-before_diag["scalar_route_calls"],
                        edge_walk_calls=after_diag["edge_walk_calls"]-before_diag["edge_walk_calls"],
                        idle_context_routes=validator.seam_evidence.departure_context_route_count-idle_before,
                        wall_overrun=result["wall_overrun"],coarse_available=future["available"])
                    case["modes"][mode] = dict(result=row,selected_ids=[a.action_id for a in result["selected"]],
                        outcomes=oracle.outcomes,trace=result["trace"])
                    record["rows"].append(row)
                    print(json.dumps(dict(event="MODE_COMPLETE",time=stamp(),**row)),flush=True)
                eager,lazy = case["modes"]["EAGER"],case["modes"]["LAZY_CERTIFIED"]
                case["all_current_objectives_equal"] = eager["result"]["objective"]==lazy["result"]["objective"]
                # Validate order-independence of every queried lazy oracle result.
                for aid, answer in lazy["outcomes"].items():
                    if answer != eager["outcomes"].get(aid):
                        raise ValueError("fixed oracle depends on query order/subset")
                if (eager["result"]["full_objective_certified"] and lazy["result"]["full_objective_certified"]
                        and not case["all_current_objectives_equal"]):
                    raise ValueError("two certified fixed-graph objectives differ")
                compact = lambda a, vals: dict(action_id=a.action_id,vehicle_id=a.vehicle_id,
                    request_id=a.request_id,kind=a.kind,critical=a.critical,carry_over=a.carry_over,
                    distance_m=a.empty_distance_m,units=vals[a.action_id])
                atomic_json(output/case_id/"graph.json",dict(known=[compact(a,units) for a in base],
                    unknown=[compact(a,upper) for a in unknown],case=case))
                atomic_json(output/"summary.json",dict(record,completed_graphs=case_index+1,
                    peak_rss_mib=guard.peak))
                gc.collect()
            record["status"] = "FOUR_FIXED_GRAPHS_COMPLETE"
    except Exception as error:
        record.update(status="FAILED_OR_LIMITED",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        if eta is not None:
            eta.close(); eta.actor=None
        record.update(finished_at=stamp(),runtime_s=perf_counter()-STARTED,peak_rss_mib=guard.peak)
        atomic_json(output/"summary.json",record)
        public.mkdir(parents=True,exist_ok=True)
        atomic_json(public/"summary.json",record)
        if record["rows"]:
            temp=public/"results.csv.tmp"
            pd.DataFrame(record["rows"]).to_csv(temp,index=False); temp.replace(public/"results.csv")
    return record


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("."))
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    run(args.root.resolve(),prepare_only=args.prepare_only)
