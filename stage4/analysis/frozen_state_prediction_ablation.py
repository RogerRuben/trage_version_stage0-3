"""Bounded fixed-state prediction-to-decision ablation for Stage4-PAPER-ENH-P2.

This is deliberately not a simulation.  It reconstructs ten pre-registered
states from one frozen canonical trajectory, routes each sparse candidate arc
once, and solves three one-epoch counterfactual assignment problems without
advancing vehicles or requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage4.dispatch.acceptance import passenger_acceptance
from stage4.dispatch.candidate_graph import SparseCandidateIndex, SpatialVehicle, search_radius_m
from stage4.dispatch.deterministic_routing import ArcDeterministicValhallaAdapter, SINGLE_SOURCE_MATRIX
from stage4.dispatch.exposure import CumulativeExposureState, exposure_excess
from stage4.dispatch.fleet_normalization import build_fleet_scenario
from stage4.dispatch.solver import AssignmentArc, solve_lexicographic
from stage4.fleetpy_adapter.test31_demand_adapter import load_all_test31_requests


SCENARIO_ID = "ODD_Q50_M_P70_REFERENCE"
OUTPUT_REL = Path("stage4/output/paper_enhancement/frozen_state_prediction_ablation")
SCENARIO_REL = Path("stage4/output/final_experiments") / SCENARIO_ID
PROFILE_REL = Path("stage3/config/stage3_av_capability_profiles.json")
TRAIN_DESCRIPTOR_REL = Path("stage3/output/odd_tod/s3/train_dynamic_route_descriptors.parquet")
TEST_DESCRIPTOR_REL = Path("stage3/output/odd_tod/s4/test31_original_route_descriptors.parquet")
TEST_PREDICTION_REL = Path("stage3/output/odd_tod/s4/test31_m3_predictions.parquet")
ORACLE_DIR_REL = Path("stage2/output_v4/route_conditioned_dataset/oracle_timing")

EPOCH_CLOCKS = ("07:30", "08:30", "12:00", "13:00", "17:00", "17:30", "18:00", "18:30", "21:00", "23:00")
DIMENSIONS = ("crawl", "stop", "speed_cv", "acceleration_rms")
LABEL_COLUMNS = {
    "crawl": "crawl_time_share",
    "stop": "stop_time_share",
    "speed_cv": "speed_cv_bounded",
    "acceleration_rms": "acceleration_rms_bounded",
}
PRED_COLUMNS = {
    "crawl": "pred_crawl",
    "stop": "pred_stop",
    "speed_cv": "pred_speed_cv",
    "acceleration_rms": "pred_acceleration_rms",
}
VARIANTS = ("P", "H", "D0")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def _profile_m(root: Path) -> dict[str, Any]:
    value = json.loads((root / PROFILE_REL).read_text(encoding="utf-8"))
    rows = value.get("profiles", value) if isinstance(value, dict) else value
    if isinstance(rows, dict):
        return rows["M"]
    return next(item for item in rows if item["profile_id"] == "M")


def _aggregate_oracle_day(path: Path) -> pd.DataFrame:
    columns = ["order_id", "decision_time", "route_part_length_m", *LABEL_COLUMNS.values()]
    frame = pd.read_parquet(path, columns=columns)
    weight = pd.to_numeric(frame["route_part_length_m"], errors="coerce").fillna(0.0).clip(lower=0.0)
    base = frame[["order_id", "decision_time"]].groupby("order_id", sort=False).agg(decision_time=("decision_time", "first"))
    base["route_token_count"] = frame.groupby("order_id", sort=False).size()
    for dimension, column in LABEL_COLUMNS.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.notna() & np.isfinite(values) & weight.gt(0)
        numerator = (values.loc[valid] * weight.loc[valid]).groupby(frame.loc[valid, "order_id"], sort=False).sum()
        denominator = weight.loc[valid].groupby(frame.loc[valid, "order_id"], sort=False).sum()
        base[dimension] = numerator / denominator
    return base.reset_index()


def build_train_historical_reference(root: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """Build a small Train-only TOD/route-scale median reference, one day at a time."""
    days: list[pd.DataFrame] = []
    for date in range(20161009, 20161025):
        path = root / ORACLE_DIR_REL / f"day={date}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        days.append(_aggregate_oracle_day(path))
    train = pd.concat(days, ignore_index=True)
    stamp = pd.to_datetime(train["decision_time"], unit="s", utc=True).dt.tz_convert("Asia/Shanghai")
    train["tod_bin"] = stamp.dt.hour * 4 + stamp.dt.minute // 15
    token_edges = np.unique(np.quantile(train["route_token_count"], [0.0, 0.25, 0.5, 0.75, 1.0]))
    if len(token_edges) < 3:
        token_edges = np.asarray([0.0, float(train["route_token_count"].median()), float(train["route_token_count"].max()) + 1.0])
    token_edges[0] = -np.inf
    token_edges[-1] = np.inf
    train["length_bin"] = np.digitize(train["route_token_count"], token_edges[1:-1], right=True)
    reference = train.groupby(["tod_bin", "length_bin"], as_index=False)[list(DIMENSIONS)].median()
    fallback = train.groupby("length_bin", as_index=False)[list(DIMENSIONS)].median()
    fallback["tod_bin"] = -1
    reference = pd.concat([reference, fallback], ignore_index=True)
    distributions = {
        dimension: np.sort(pd.to_numeric(train[dimension], errors="coerce").dropna().to_numpy(dtype=float))
        for dimension in DIMENSIONS
    }
    return reference, distributions, token_edges


def _mid_cdf(value: float, sorted_values: np.ndarray) -> float:
    left = int(np.searchsorted(sorted_values, value, side="left"))
    right = int(np.searchsorted(sorted_values, value, side="right"))
    return (left + right) / (2.0 * len(sorted_values)) if len(sorted_values) else float("nan")


def _historical_values(reference: pd.DataFrame, distributions: dict[str, np.ndarray], token_edges: np.ndarray, timestamp: pd.Timestamp, token_count: float) -> tuple[dict[str, float], float]:
    tod_bin = int(timestamp.hour * 4 + timestamp.minute // 15)
    length_bin = int(np.digitize([token_count], token_edges[1:-1], right=True)[0])
    row = reference.loc[(reference["tod_bin"].eq(tod_bin)) & (reference["length_bin"].eq(length_bin))]
    if row.empty:
        row = reference.loc[(reference["tod_bin"].eq(-1)) & (reference["length_bin"].eq(length_bin))]
    if len(row) != 1:
        raise RuntimeError(f"historical reference missing for TOD={tod_bin}, length={length_bin}")
    raw = {dimension: float(row.iloc[0][dimension]) for dimension in DIMENSIONS}
    profile = _profile_m(_historical_values.root)  # type: ignore[attr-defined]
    ratios = []
    for dimension in DIMENSIONS:
        z = _mid_cdf(raw[dimension], distributions[dimension])
        ratios.extend(z / float(profile["dynamic_caps"][dimension][metric]) for metric in ("E", "Q", "C"))
    return raw, float(max(ratios))


def _vehicle_state(fixtures: list[Any], assignments: pd.DataFrame, timestamp: pd.Timestamp) -> list[SpatialVehicle]:
    prior = assignments.loc[pd.to_datetime(assignments["assignment_time"]).le(timestamp)].copy()
    active_ids = set(prior.loc[pd.to_datetime(prior["service_end_time"]).gt(timestamp), "vehicle_id"].astype(str))
    completed = prior.loc[pd.to_datetime(prior["service_end_time"]).le(timestamp)].sort_values(["service_end_time", "assignment_time"])
    last = completed.groupby("vehicle_id", sort=False).tail(1).set_index("vehicle_id")
    vehicles: list[SpatialVehicle] = []
    for fixture in fixtures:
        if not (pd.Timestamp(fixture.availability_start_time) <= timestamp < pd.Timestamp(fixture.availability_end_time)):
            continue
        if str(fixture.vehicle_id) in active_ids:
            continue
        if str(fixture.vehicle_id) in last.index:
            row = last.loc[str(fixture.vehicle_id)]
            lon, lat = float(row.completion_lon_wgs84), float(row.completion_lat_wgs84)
        else:
            lon, lat = float(fixture.initial_lon_wgs84), float(fixture.initial_lat_wgs84)
        vehicles.append(SpatialVehicle(str(fixture.vehicle_id), int(fixture.native_id), str(fixture.vehicle_type), lon, lat))
    return vehicles


def _waiting_requests(requests: list[Any], assignments: pd.DataFrame, simulation_time_s: int) -> list[tuple[Any, int, bool, bool]]:
    assigned = set(assignments.loc[pd.to_numeric(assignments["simulation_time_s"]).lt(simulation_time_s), "order_id"].astype(str))
    rows = []
    for request in requests:
        if request.order_id in assigned or request.sim_time_s > simulation_time_s or simulation_time_s >= request.sim_time_s + 300:
            continue
        first_epoch = int(math.ceil(request.sim_time_s / 30.0) * 30)
        failed_rounds = max(0, (simulation_time_s - first_epoch) // 30)
        remaining = request.sim_time_s + 300 - simulation_time_s
        rows.append((request, failed_rounds, failed_rounds > 0, 0 < remaining <= 30))
    return sorted(rows, key=lambda item: (item[0].request_time, item[0].native_id))


def _exposure_before(exposure: pd.DataFrame, simulation_time_s: int) -> CumulativeExposureState:
    prior = exposure.loc[pd.to_numeric(exposure["simulation_time_s"]).lt(simulation_time_s)]
    if prior.empty:
        return CumulativeExposureState()
    row = prior.sort_values("simulation_time_s").iloc[-1]
    return CumulativeExposureState(int(row.cumulative_av_assignments), float(row.cumulative_static_excess), float(row.cumulative_dynamic_excess), float(row.cumulative_speed_excess))


def _weighted_order_values(frame: pd.DataFrame, columns: dict[str, str], weight_column: str) -> pd.DataFrame:
    weight = pd.to_numeric(frame[weight_column], errors="coerce").fillna(0.0).clip(lower=0.0)
    result = pd.DataFrame(index=pd.Index(frame["order_id"].astype(str).unique(), name="order_id"))
    for dimension, column in columns.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.notna() & np.isfinite(values) & weight.gt(0)
        numerator = (values.loc[valid] * weight.loc[valid]).groupby(frame.loc[valid, "order_id"].astype(str), sort=False).sum()
        denominator = weight.loc[valid].groupby(frame.loc[valid, "order_id"].astype(str), sort=False).sum()
        result[dimension] = numerator / denominator
    return result.reset_index()


def run(root: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(root).resolve()
    output = root / OUTPUT_REL
    scenario = root / SCENARIO_REL
    config = json.loads((scenario / "scenario_config.json").read_text(encoding="utf-8"))["runtime_configuration"]
    start = pd.Timestamp("2016-10-31T00:00:00+08:00")
    end = start + pd.Timedelta(days=1)
    requests = load_all_test31_requests(root, start=start, end=end, profile_id="M")
    request_by_order = {item.order_id: item for item in requests}
    assignments = pd.read_parquet(scenario / "assignment_log.parquet")
    for column in ("assignment_time", "service_end_time"):
        assignments[column] = pd.to_datetime(assignments[column], utc=True).dt.tz_convert("Asia/Shanghai")
    exposure_log = pd.read_parquet(scenario / "exposure_state.parquet")
    max_service = max(item.realized_service_time_s for item in requests)
    fleet = build_fleet_scenario(root, benchmark_start=start, simulation_end=end + pd.Timedelta(seconds=max_service + 60), requested_q_a=0.5, seed=20260824, max_hv_hour_error_pct=2.0)
    descriptors = pd.read_parquet(root / TEST_DESCRIPTOR_REL, columns=["order_id", "route_token_count"])
    token_count = descriptors.set_index(descriptors["order_id"].astype(str))["route_token_count"].to_dict()
    reference, distributions, token_edges = build_train_historical_reference(root)
    _historical_values.root = root  # type: ignore[attr-defined]
    gammas = {family: config[f"gamma_{family}"] for family in ("static", "dynamic", "speed")}
    adapter = ArcDeterministicValhallaAdapter(root, routing_mode=SINGLE_SOURCE_MATRIX)

    registry_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    selected_order_ids: set[str] = set()
    for clock in EPOCH_CLOCKS:
        timestamp = pd.Timestamp(f"2016-10-31T{clock}:00+08:00")
        sim_s = int((timestamp - start).total_seconds())
        vehicles = _vehicle_state(fleet.native_fixtures, assignments, timestamp)
        waiting = _waiting_requests(requests, assignments, sim_s)
        state = _exposure_before(exposure_log, sim_s)
        index = SparseCandidateIndex(vehicles)
        candidate_sets: dict[tuple[int, str], list[SpatialVehicle]] = {}
        estimates_by_request: dict[int, dict[int, Any]] = {}
        opportunity_counts = {variant: {"n0": 0, "n1": 0, "n2": 0, "n3": 0} for variant in VARIANTS}
        h_by_order: dict[str, tuple[dict[str, float], float]] = {}
        for request, failed_rounds, _carry, _critical in waiting:
            h_by_order[request.order_id] = _historical_values(reference, distributions, token_edges, timestamp, float(token_count.get(request.order_id, 0.0)))
            radius = search_radius_m(failed_rounds, config["search_radius_initial_m"], config["search_radius_step_m"], config["search_radius_cap_m"])
            nearby_av = index.count_vehicle_type_within(request.pickup_lon_wgs84, request.pickup_lat_wgs84, radius, "AV")
            acceptance = passenger_acceptance(request.order_id, 0.7, 20260827).passenger_accepts_av
            for variant in VARIANTS:
                opportunity_counts[variant]["n0"] += nearby_av
                if acceptance:
                    opportunity_counts[variant]["n1"] += nearby_av
                structural = request.hard_state == "FEASIBLE" and str(request.selected_route_type).upper() not in {"", "NONE", "NAN", "NULL"}
                if acceptance and structural:
                    opportunity_counts[variant]["n2"] += nearby_av
                rho_dynamic = request.rho_dynamic if variant in ("P", "D0") else h_by_order[request.order_id][1]
                evidence = structural and np.isfinite(request.rho_static) and np.isfinite(request.rho_speed) and (variant == "D0" or np.isfinite(rho_dynamic)) and np.isfinite(request.predicted_service_time_s) and request.predicted_service_time_s > 0
                if acceptance and evidence:
                    opportunity_counts[variant]["n3"] += nearby_av
                av_eligible = bool(acceptance and evidence)
                candidates, _ = index.query(request.pickup_lon_wgs84, request.pickup_lat_wgs84, radius, int(config["candidate_top_k"]), av_eligible)
                candidate_sets[(request.native_id, variant)] = [item[0] for item in candidates]
            union = {vehicle.native_vehicle_id: vehicle for variant in VARIANTS for vehicle in candidate_sets[(request.native_id, variant)]}
            estimates_by_request[request.native_id] = adapter.estimate_many(list(union.values()), request.pickup_lon_wgs84, request.pickup_lat_wgs84, timestamp)

        state_payload = {
            "timestamp": timestamp.isoformat(),
            "waiting": [item[0].order_id for item in waiting],
            "vehicles": [(v.vehicle_id, v.vehicle_type, round(v.lon_wgs84, 7), round(v.lat_wgs84, 7)) for v in vehicles],
            "exposure": state.__dict__,
            "candidate_rules": {key: config[key] for key in ("search_radius_initial_m", "search_radius_step_m", "search_radius_cap_m", "candidate_top_k", "max_pickup_wait_s")},
        }
        registry_rows.append({"epoch_id": clock.replace(":", ""), "timestamp": timestamp, "period": "EVENING_PEAK" if 17 <= timestamp.hour <= 18 else ("MORNING_PEAK" if 7 <= timestamp.hour <= 8 else ("MIDDAY" if 12 <= timestamp.hour <= 13 else "OFF_PEAK")), "waiting_order_count": len(waiting), "available_vehicle_count": len(vehicles), "available_av_count": sum(v.vehicle_type == "AV" for v in vehicles), "available_hv_count": sum(v.vehicle_type == "HV" for v in vehicles), "cumulative_av_assignments": state.av_assignments, "cumulative_static_excess": state.static, "cumulative_dynamic_excess": state.dynamic, "cumulative_speed_excess": state.speed, "state_sha256": _sha(state_payload), "source_scenario_id": SCENARIO_ID})

        for variant in VARIANTS:
            arcs: list[AssignmentArc] = []
            for request, _failed_rounds, carry, critical in waiting:
                remaining = float(request.sim_time_s + 300 - sim_s)
                h_raw, h_rho = h_by_order[request.order_id]
                rho_dynamic = request.rho_dynamic if variant == "P" else (h_rho if variant == "H" else 1.0)
                excess = exposure_excess(request.rho_static, rho_dynamic, request.rho_speed)
                for vehicle in candidate_sets[(request.native_id, variant)]:
                    estimate = estimates_by_request[request.native_id].get(vehicle.native_vehicle_id)
                    if estimate is None or not np.isfinite(estimate.corrected_pickup_eta_s) or estimate.corrected_pickup_eta_s > remaining:
                        continue
                    if vehicle.vehicle_type == "HV":
                        # Mirror the production HV session-window gate: an HV
                        # arc without decision-time service duration is not admissible.
                        if not np.isfinite(request.predicted_service_time_s):
                            continue
                        fixture = fleet.native_fixtures[vehicle.native_vehicle_id]
                        predicted_end = timestamp + pd.Timedelta(seconds=float(estimate.corrected_pickup_eta_s + request.predicted_service_time_s))
                        if predicted_end > pd.Timestamp(fixture.availability_end_time):
                            continue
                    arc_exposure = excess if vehicle.vehicle_type == "AV" else None
                    arcs.append(AssignmentArc(vehicle.native_vehicle_id, request.native_id, float(estimate.corrected_pickup_eta_s), critical, carry, payload=(request.order_id, vehicle.vehicle_id), vehicle_type=vehicle.vehicle_type, exposure_static=arc_exposure.static if arc_exposure else 0.0, exposure_dynamic=arc_exposure.dynamic if arc_exposure else 0.0, exposure_speed=arc_exposure.speed if arc_exposure else 0.0))
            variant_gammas = dict(gammas)
            if variant == "D0":
                variant_gammas["dynamic"] = None
            result = solve_lexicographic(arcs, exposure_state=state, gammas=variant_gammas, cost_level_enabled=False)
            selected = [arcs[index_value] for index_value in result.selected_indices]
            selected_av = [arc for arc in selected if arc.vehicle_type == "AV"]
            selected_hv = [arc for arc in selected if arc.vehicle_type == "HV"]
            selected_orders = sorted(request_by_order[next(item.order_id for item in requests if item.native_id == arc.request_id)].order_id for arc in []) if False else sorted(str(arc.payload[0]) for arc in selected)
            selected_order_ids.update(selected_orders)
            for arc in selected:
                assignment_rows.append({"epoch_id": clock.replace(":", ""), "variant": variant, "order_id": str(arc.payload[0]), "vehicle_id": str(arc.payload[1]), "vehicle_type": arc.vehicle_type, "pickup_eta_s": arc.pickup_eta_s, "exposure_static": arc.exposure_static, "exposure_dynamic": arc.exposure_dynamic, "exposure_speed": arc.exposure_speed})
            decision_rows.append({"epoch_id": clock.replace(":", ""), "timestamp": timestamp, "variant": variant, "passenger_compatible_opportunities": opportunity_counts[variant]["n1"], "structurally_ready_opportunities": opportunity_counts[variant]["n2"], "evidence_complete_opportunities": opportunity_counts[variant]["n3"], "solver_eligible_av_arcs": sum(arc.vehicle_type == "AV" for arc in arcs), "n5_n0_conversion": (sum(arc.vehicle_type == "AV" for arc in arcs) / opportunity_counts[variant]["n0"] if opportunity_counts[variant]["n0"] else np.nan), "selected_av_assignments": len(selected_av), "selected_hv_assignments": len(selected_hv), "matched_order_count": len(selected), "matched_order_id_set": json.dumps(selected_orders, separators=(",", ":")), "selected_static_exposure_mean": np.mean([arc.exposure_static for arc in selected_av]) if selected_av else 0.0, "selected_dynamic_exposure_mean": np.mean([arc.exposure_dynamic for arc in selected_av]) if selected_av else 0.0, "selected_speed_exposure_mean": np.mean([arc.exposure_speed for arc in selected_av]) if selected_av else 0.0, "pickup_objective_s": result.pickup_eta_optimum_s, "state_sha256": registry_rows[-1]["state_sha256"], "solver_backend": result.backend})

    registry = pd.DataFrame(registry_rows)
    decisions = pd.DataFrame(decision_rows)
    selected_assignments = pd.DataFrame(assignment_rows)
    overlap_rows = []
    for epoch_id in registry["epoch_id"]:
        group = selected_assignments.loc[selected_assignments["epoch_id"].eq(epoch_id)]
        for alternative in ("H", "D0"):
            p = group.loc[group["variant"].eq("P")]
            alt = group.loc[group["variant"].eq(alternative)]
            po, ao = set(p["order_id"]), set(alt["order_id"])
            p_all = set(zip(p["vehicle_id"], p["order_id"])); a_all = set(zip(alt["vehicle_id"], alt["order_id"]))
            p_av = p.loc[p["vehicle_type"].eq("AV")]; a_av = alt.loc[alt["vehicle_type"].eq("AV")]
            pa = set(zip(p_av["vehicle_id"], p_av["order_id"])); aa = set(zip(a_av["vehicle_id"], a_av["order_id"]))
            overlap_rows.append({"epoch_id": epoch_id, "reference_variant": "P", "alternative_variant": alternative, "selected_order_jaccard": len(po & ao) / len(po | ao) if po | ao else 1.0, "selected_av_arc_jaccard": len(pa & aa) / len(pa | aa) if pa | aa else 1.0, "assignment_set_changed": p_all != a_all, "vehicle_type_changed_order_count": sum(1 for order in po & ao if set(p.loc[p.order_id.eq(order), "vehicle_type"]) != set(alt.loc[alt.order_id.eq(order), "vehicle_type"]))})
    overlap = pd.DataFrame(overlap_rows)

    # Prediction-level metrics are evaluated only for orders present in the ten states.
    evaluation_ids = sorted({item[0].order_id for clock in EPOCH_CLOCKS for item in _waiting_requests(requests, assignments, int((pd.Timestamp(f"2016-10-31T{clock}:00+08:00") - start).total_seconds()))})
    filters = [("order_id", "in", evaluation_ids)]
    pred = pd.read_parquet(root / TEST_PREDICTION_REL, columns=["order_id", "allocated_distance_m", *PRED_COLUMNS.values()], filters=filters)
    observed = pd.read_parquet(root / ORACLE_DIR_REL / "day=20161031.parquet", columns=["order_id", "route_part_length_m", *LABEL_COLUMNS.values()], filters=filters)
    p_values = _weighted_order_values(pred, PRED_COLUMNS, "allocated_distance_m").set_index("order_id")
    y_values = _weighted_order_values(observed, LABEL_COLUMNS, "route_part_length_m").set_index("order_id")
    metric_rows = []
    for epoch in registry.itertuples(index=False):
        ids = [item[0].order_id for item in _waiting_requests(requests, assignments, int((pd.Timestamp(epoch.timestamp) - start).total_seconds()))]
        for variant in ("P", "H"):
            for dimension in DIMENSIONS:
                pairs = []
                for order_id in ids:
                    if order_id not in y_values.index or pd.isna(y_values.at[order_id, dimension]):
                        continue
                    if variant == "P":
                        prediction = p_values.at[order_id, dimension] if order_id in p_values.index else np.nan
                    else:
                        prediction = _historical_values(reference, distributions, token_edges, pd.Timestamp(epoch.timestamp), float(token_count.get(order_id, 0.0)))[0][dimension]
                    if np.isfinite(prediction):
                        pairs.append(abs(float(prediction) - float(y_values.at[order_id, dimension])))
                metric_rows.append({"epoch_id": epoch.epoch_id, "variant": variant, "metric": dimension, "valid_order_count": len(pairs), "mae": float(np.mean(pairs)) if pairs else np.nan, "median_absolute_error": float(np.median(pairs)) if pairs else np.nan, "evaluation_only_realized_test31": True})
    prediction_metrics = pd.DataFrame(metric_rows)

    comparisons = overlap.merge(decisions.loc[decisions.variant.eq("P"), ["epoch_id", "solver_eligible_av_arcs", "selected_dynamic_exposure_mean", "pickup_objective_s"]].rename(columns=lambda c: f"p_{c}" if c != "epoch_id" else c), on="epoch_id").merge(decisions.loc[decisions.variant.ne("P"), ["epoch_id", "variant", "solver_eligible_av_arcs", "selected_dynamic_exposure_mean", "pickup_objective_s"]], left_on=["epoch_id", "alternative_variant"], right_on=["epoch_id", "variant"])
    h = comparisons.loc[comparisons.alternative_variant.eq("H")]
    changed_share = float(h["assignment_set_changed"].mean()) if len(h) else 0.0
    mean_jaccard = float(h["selected_av_arc_jaccard"].mean()) if len(h) else 1.0
    classification = "DECISION-RELEVANT" if changed_share >= 0.25 and mean_jaccard < 0.9 else ("DECISION-MODEST" if changed_share > 0 or mean_jaccard < 0.98 else "DECISION-NEGLIGIBLE")
    summary = pd.DataFrame([{"source_scenario_id": SCENARIO_ID, "epoch_count": len(registry), "variants": "P|H|D0", "runtime_s": time.perf_counter() - started, "routing_mode": SINGLE_SOURCE_MATRIX, "routed_arc_evaluations": adapter.routing_arc_evaluations, "routing_failure_count": adapter.routing_failures, "p_vs_h_epoch_assignment_change_share": changed_share, "p_vs_h_mean_selected_av_arc_jaccard": mean_jaccard, "p_vs_h_mean_solver_eligible_av_arc_difference": float((h["solver_eligible_av_arcs"] - h["p_solver_eligible_av_arcs"]).mean()) if len(h) else 0.0, "p_vs_h_mean_dynamic_exposure_difference": float((h["selected_dynamic_exposure_mean"] - h["p_selected_dynamic_exposure_mean"]).mean()) if len(h) else 0.0, "p_vs_h_mean_pickup_objective_difference_s": float((h["pickup_objective_s"] - h["p_pickup_objective_s"]).mean()) if len(h) else 0.0, "classification": classification, "full_day_service_rate_reported": False, "gpu_used": False}])
    _atomic_csv(registry, output / "frozen_epoch_registry.csv")
    _atomic_csv(prediction_metrics, output / "prediction_variant_metrics.csv")
    _atomic_csv(decisions, output / "decision_variant_metrics.csv")
    _atomic_csv(overlap, output / "assignment_overlap.csv")
    _atomic_csv(summary, output / "ablation_summary.csv")
    _atomic_csv(selected_assignments, output / "selected_assignments_diagnostic.csv")
    return summary.iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    print(json.dumps(run(parser.parse_args().root), indent=2, default=str))


if __name__ == "__main__":
    main()
