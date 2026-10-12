"""One completed v5 C comparison against the retained v4 group; no new replay.

Reads projected event tables sequentially and exports aggregates only. Physical
admission accounting reuses the existing clipped ledger, not busy-time proxies.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import pandas as pd
import psutil

from .capability_chain_instances import atomic_json
from .c_operations_diagnostic import DAY_S, KINDS, START, TOL, _hours, clipped_ledger, overlap, sha
from stage4.dispatch.controlled_fleet import build_controlled_fleet

GROUP = "CHAIN_LAYOUT_BASELINE"
OUT = Path("stage4/output/capability_chain_planning")
DOC = Path("stage4/docs/capability_chain_planning/operations_v5")
ORDER_COLUMNS = ["order_id", "release_s", "C_compatible", "passenger_accepts_av",
    "completed", "matched", "expired", "vehicle_type", "wait_s", "matching_wait_s", "pickup_wait_s"]
TYPED_COLUMNS = ["typed_last_final_AV_count", "typed_last_final_HV_count",
    "typed_HV_assigned_with_final_AV_same_epoch", "typed_AV_alternatives_idle_after_choice",
    "typed_AV_alternatives_serving_other_request", "typed_AV_alternatives_moving_after_choice",
    "typed_rejection_counts_json"]


def _peak():
    memory = psutil.Process().memory_info()
    peak = max(memory.rss, getattr(memory, "peak_wset", memory.rss)) / 2**20
    if peak > 512:
        raise MemoryError("projected result analysis exceeds its 512-MiB cap")
    return peak


def typed_partition(frame):
    """Final-epoch exclusive states, with separate nonexclusive lifetime witnesses."""
    eligible = frame.C_compatible & frame.passenger_accepts_av
    hv = frame.loc[eligible & frame.completed & frame.vehicle_type.eq("HV")]
    final = hv.typed_last_final_AV_count.gt(0)
    idle = hv.typed_AV_alternatives_idle_after_choice.fillna(0).astype(int)
    other = hv.typed_AV_alternatives_serving_other_request.fillna(0).astype(int)
    moving = hv.typed_AV_alternatives_moving_after_choice.fillna(0).astype(int)
    if (not (idle + other + moving).equals(hv.typed_last_final_AV_count.astype(int))
            or not hv.typed_HV_assigned_with_final_AV_same_epoch.astype(bool).equals(final)):
        raise ValueError("HV final-C alternative states do not reconcile")
    categories = dict(NO_FINAL_C_CANDIDATE=int((~final).sum()),
        FINAL_C_LEFT_UNOCCUPIED=int((final & idle.gt(0)).sum()),
        FINAL_C_ALL_SERVING_OTHER_REQUESTS=int((final & idle.eq(0) & moving.eq(0)).sum()),
        FINAL_C_MOVING_OR_MIXED_COMMITTED=int((final & idle.eq(0) & moving.gt(0)).sum()))
    if sum(categories.values()) != len(hv):
        raise ValueError("exclusive HV final-C categories do not partition eligible requests")
    witnesses = {}
    for name, subset in (("ELIGIBLE_HV_NO_FINAL_C", hv.loc[~final]),
                         ("ELIGIBLE_EXPIRED", frame.loc[eligible & frame.expired])):
        attempts, requests, phases = Counter(), Counter(), Counter()
        for value in subset.typed_rejection_counts_json:
            codes = json.loads(value)
            seen = set()
            for code, count in codes.items():
                if not isinstance(count, int) or count < 0:
                    raise ValueError("invalid typed diagnostic counter")
                if not code.startswith("AV:") or not count:
                    continue
                public = code if re.fullmatch(r"[A-Z][A-Z0-9_:-]{0,100}", code) else "AV:OTHER_TEXT_REASON"
                attempts[public] += count
                seen.add(public)
            requests.update(seen)
            phases.update({code.split(":")[1] for code in seen})
        witnesses[name] = dict(request_count=len(subset),
            reason_request_counts=dict(sorted(requests.items())),
            reason_attempt_counts=dict(sorted(attempts.items())),
            phase_request_counts=dict(sorted(phases.items())))
    return dict(eligible_orders=int(eligible.sum()), eligible_served_C=int((eligible & frame.vehicle_type.eq("AV")).sum()),
        eligible_served_HV=len(hv), eligible_expired=int((eligible & frame.expired).sum()),
        HV_with_final_C_count=int(final.sum()), HV_with_final_C_share=float(final.mean()) if len(hv) else None,
        exclusive_final_HV_categories=categories, lifetime_rejection_witnesses=witnesses,
        final_scope="ADMITTED_TOPK_GRAPH_AT_ASSIGNMENT_NOT_ALL_POSSIBLE_VEHICLES",
        rejection_witnesses_are_nonexclusive_not_counterfactual_causes=True)


def _admission(plans, assignments, moves, expected):
    activities = defaultdict(list)
    full_customer, calendar_customer, movement_time, movement_distance = (Counter() for _ in range(4))
    serving = set()
    for row in assignments.itertuples(index=False):
        vid = int(row.native_vehicle_id)
        kind, a, b = plans[vid]
        if (kind != row.vehicle_type or not a - TOL <= row.simulation_time_s < b
                or not row.simulation_time_s <= row.pickup_time_s <= row.service_end_time_s):
            raise ValueError("customer commitment/time chain differs from source admission")
        activities[vid].extend(((row.simulation_time_s, row.pickup_time_s, "PICKUP"),
                               (row.pickup_time_s, row.service_end_time_s, "LOADED")))
        full_customer[kind] += row.service_end_time_s - row.simulation_time_s
        calendar_customer[kind] += overlap(row.simulation_time_s, row.service_end_time_s, 0, DAY_S)
        serving.add(vid)
    for row in moves.itertuples(index=False):
        vid = int(row.native_vehicle_id)
        kind = plans[vid][0]
        if (kind != row.vehicle_type or row.status not in ("COMPLETED", "INTERRUPTED_BY_CUSTOMER")
                or abs(row.end_time_s - row.start_time_s - row.actual_duration_s) > TOL):
            raise ValueError("incomplete movement physical event")
        activities[vid].append((row.start_time_s, row.end_time_s, "MOVE"))
        movement_time[kind] += row.actual_duration_s
        movement_distance[kind] += row.actual_distance_m
    totals = {kind: Counter() for kind in ("AV", "HV")}
    hourly = {kind: [Counter() for _ in range(24)] for kind in totals}
    slots = {kind: Counter() for kind in totals}
    for vid, (kind, a, b) in plans.items():
        ledger = clipped_ledger(activities[vid], a, b)
        keys = ("admission_s", *KINDS, "idle_s")
        totals[kind].update({key: ledger[key] for key in keys})
        if ledger["admission_s"]:
            slots[kind]["day_slots"] += 1
            slots[kind]["never_served_slots"] += int(vid not in serving)
            slots[kind]["fully_idle_slots"] += int(ledger["idle_s"] >= ledger["admission_s"] - TOL)
            slots[kind]["slots_with_idle_gap_over_300s"] += int(ledger["longest_idle_s"] > 300)
        for hour in range(max(0, int(max(a, 0) // 3600)), min(24, int(min(b, DAY_S) // 3600) + 1)):
            part = clipped_ledger(activities[vid], a, b, hour * 3600., (hour + 1) * 3600.)
            hourly[kind][hour].update({key: part[key] for key in keys})
    result = {}
    for kind, total in totals.items():
        busy = total["PICKUP"] + total["LOADED"]
        result[kind] = dict(**_hours(total), **slots[kind], full_committed_customer_hours=full_customer[kind] / 3600.,
            loaded_share=total["LOADED"] / total["admission_s"] if total["admission_s"] else None,
            calendar_customer_after_admission_hours=max(0., calendar_customer[kind] - busy) / 3600.,
            customer_outside_calendar_hours=(full_customer[kind] - calendar_customer[kind]) / 3600.,
            full_movement_hours=movement_time[kind] / 3600., full_movement_km=movement_distance[kind] / 1000.,
            partition_error_s=abs(total["admission_s"] - busy - total["MOVE"] - total["idle_s"]))
        planned = "achieved_av_vehicle_hours" if kind == "AV" else "achieved_hv_vehicle_hours"
        if abs(result[kind]["admission_hours"] - expected[planned]) > TOL:
            raise ValueError("calendar admission denominator changed")
    return result, hourly


def _collect(root, version, summary, plans, *, directory=None):
    directory = root / OUT / f"two_scale_city_{version}" / GROUP if directory is None else Path(directory)
    cfg = summary["frozen_protocol"]
    typed = bool(cfg.get("typed_request_diagnostics", False))
    outcome = pd.read_parquet(directory / "cohort_outcomes.parquet", columns=ORDER_COLUMNS + (TYPED_COLUMNS if typed else []))
    assignments = pd.read_parquet(directory / "assignments.parquet", columns=["order_id", "native_vehicle_id",
        "vehicle_type", "simulation_time_s", "pickup_time_s", "service_end_time_s", "completed", "pickup_route_distance_m"])
    moves = pd.read_parquet(directory / "empty_movements.parquet", columns=["native_vehicle_id", "vehicle_type",
        "start_time_s", "end_time_s", "actual_duration_s", "actual_distance_m", "status"])
    served = outcome.loc[outcome.completed]
    if (len(outcome) != 28367 or not outcome.order_id.is_unique or not assignments.order_id.is_unique
            or not (outcome.matched ^ outcome.expired).all() or not outcome.completed.equals(outcome.matched)
            or set(assignments.order_id.astype(str)) != set(served.order_id.astype(str))
            or not assignments.completed.all() or len(served) != summary["completed_orders"]):
        raise ValueError("final cohort/service/physical-drain reconciliation failed")
    if (not served.matching_wait_s.ge(0).all() or not served.matching_wait_s.lt(300).all()
            or not served.pickup_wait_s.between(0, 600 + TOL).all()
            or not np.allclose(served.wait_s, served.matching_wait_s + served.pickup_wait_s, rtol=0, atol=TOL)):
        raise ValueError("separate assignment and pickup waiting clocks failed")
    physics = summary["physical_accounting"]
    if physics["duplicate_orders"] or physics["vehicle_task_overlap"] or physics["maximum_time_error_s"] > TOL:
        raise ValueError("reported physical conservation failed")
    admission, hourly_admission = _admission(plans, assignments, moves, summary["all_day_supply"]["accounting"])
    eligible = outcome.C_compatible & outcome.passenger_accepts_av
    pickup = assignments.groupby("vehicle_type").pickup_route_distance_m.sum().to_dict()
    trace = pd.read_parquet(directory / "solver_trace.parquet")
    epochs = pd.read_parquet(directory / "epochs.parquet", columns=["candidate_generation_time_s", "routing_time_s"])
    row = dict(completed=len(served), expired=int(outcome.expired.sum()), service_rate=len(served) / len(outcome),
        served_C=int(outcome.vehicle_type.eq("AV").sum()), served_HV=int(outcome.vehicle_type.eq("HV").sum()),
        C_body_compatible_orders=int(outcome.C_compatible.sum()), C_after_passenger_acceptance_orders=int(eligible.sum()),
        C_eligible_served_by_HV=int((eligible & outcome.vehicle_type.eq("HV")).sum()),
        C_eligible_expired=int((eligible & outcome.expired).sum()),
        pickup_empty_km=float(assignments.pickup_route_distance_m.sum() / 1000.),
        idle_empty_km=float(moves.actual_distance_m.sum() / 1000.), actual_idle_moves=len(moves),
        C_pickup_empty_km=float(pickup.get("AV", 0) / 1000.), HV_pickup_empty_km=float(pickup.get("HV", 0) / 1000.),
        admission_state=admission, runtime_s=summary["total_runner_wall_s"], setup_time_s=summary["setup_time_s"],
        simulation_loop_wall_s=summary["simulation_loop_wall_s"], peak_rss_mib=summary["peak_rss_mib"],
        candidate_generation_time_s=float(epochs.candidate_generation_time_s.sum()),
        candidate_routing_time_s=float(epochs.routing_time_s.sum()), adapter_timings_s=summary["adapter_performance"]["timings_s"],
        timing_components_are_nested_do_not_sum=True,
        idle_movement_timings=dict(routing_wall_time_s=summary["idle_movement"]["routing_wall_time_s"],
                                  total_decision_time_s=summary["idle_movement"]["total_decision_time_s"]),
        actual_routing=summary["actual_routing"], idle_movement=summary["idle_movement"],
        request_decision_diagnostics=summary["request_decision_diagnostics"],
        maximum_current_actions=int(trace.variables.max()), maximum_current_residual_edges=int(trace.nonzeros.max()),
        decision_count=len(trace), current_flow_status=trace.opt_status.value_counts().to_dict(),
        adapter_wall_overrun_count=int(trace.adapter_wall_deadline_exceeded.sum()),
        explicit_fallback_decisions=int(trace.resource_fallback.sum()), physical_accounting=physics,
        source_code_sha=summary["code_sha"], waiting_contract=summary["waiting_contract"],
        future_connector_queries=summary["adapter_performance"]["future_connector_queries"])
    if row["future_connector_queries"]:
        raise ValueError("future-routing path was entered")
    row["total_empty_km"] = row["pickup_empty_km"] + row["idle_empty_km"]
    for name in ("matching_wait_s", "pickup_wait_s", "wait_s"):
        row[name] = dict(mean=float(served[name].mean()), p90=float(served[name].quantile(.9)), max=float(served[name].max()))
    row["hourly"] = []
    for hour, frame in outcome.groupby((outcome.release_s // 3600).astype(int), sort=True):
        row["hourly"].append(dict(hour=int(hour), requests=len(frame), completed=int(frame.completed.sum()),
            expired=int(frame.expired.sum()), served_C=int(frame.vehicle_type.eq("AV").sum()),
            served_HV=int(frame.vehicle_type.eq("HV").sum()),
            C_state=_hours(hourly_admission["AV"][int(hour)]) if hour < 24 else None,
            HV_state=_hours(hourly_admission["HV"][int(hour)]) if hour < 24 else None))
    if typed:
        row["typed_final_candidate_diagnostics"] = typed_partition(outcome)
        solved = trace.secondary_status.ne("NO_CURRENT_FLOW_SOLVE")
        row["secondary_distance"] = dict(solve_count=int(solved.sum()),
            optimal_count=int(trace.secondary_optimality_proved.fillna(False).sum()),
            statuses=trace.secondary_status.value_counts().to_dict(), time_s=float(trace.secondary_time_s.fillna(0).sum()),
            deadline_exceeded_count=int(trace.secondary_wall_deadline_exceeded.fillna(False).sum()),
            scope="ADMITTED_PRIMARY_OPTIMAL_CURRENT_GRAPH_MM_NOT_WHOLE_DAY")
    row["source_products_sha256"] = {name: sha(directory / name) for name in
        ("summary.json", "assignments.parquet", "cohort_outcomes.parquet", "empty_movements.parquet", "solver_trace.parquet")}
    private = outcome[ORDER_COLUMNS].set_index("order_id")
    private["pickup_distance_m"] = assignments.set_index("order_id").pickup_route_distance_m
    _peak()
    del assignments, moves, outcome, trace, epochs, served
    gc.collect()
    return row, private


def analyze(root):
    started = perf_counter()
    root = Path(root).resolve()
    summaries = {version: json.loads((root / OUT / f"two_scale_city_{version}" / GROUP / "summary.json").read_text(encoding="utf-8"))
        for version in ("v4", "v5")}
    for version, summary in summaries.items():
        if (summary["status"] != "TWO_SCALE_CITY_GROUP_COMPLETE" or not summary["full_day"]
                or summary["version"] != f"scheme_a_two_scale_city_{version}"
                or summary["group"] != GROUP or summary["policy"] != "SERVICE_PRESERVING" or summary["layout_mode"] != "joint"):
            raise ValueError("only the completed one-group fixed C comparison is in scope")
    before, after = (summaries[version] for version in ("v4", "v5"))
    if any(before["frozen_protocol"].get(key) != value for key, value in
            dict(test_date="20161031", profile_id="C", requested_q_a=.1, passenger_acceptance_rate=.7).items()):
        raise ValueError("the retained baseline is not the preregistered C/q10/P70 condition")
    controls = ("test_date", "profile_id", "requested_q_a", "passenger_acceptance_rate", "passenger_acceptance_seed",
        "supply_seed", "forecast_seed", "source_layout_sha256", "max_match_wait_s", "max_pickup_eta_s", "post_assignment_cancellation",
        "solver_time_limit_s", "scenario_timeout_s", "maximum_rss_mib", "rolling_step_s", "current_top_k",
        "measurement_end_s", "admission_end_s", "physical_drain_limit_s", "reposition_max_moves", "common_movement_wall_limit_s")
    if any(before["frozen_protocol"][key] != after["frozen_protocol"][key] for key in controls):
        raise ValueError("a fixed scientific/resource control changed")
    common_inputs = {path: digest for path, digest in before["inputs_sha256"].items() if path in after["inputs_sha256"]}
    if any(after["inputs_sha256"][path] != digest or sha(root / path) != digest for path, digest in common_inputs.items()):
        raise ValueError("a shared frozen input changed")
    state = json.loads((root / OUT / "two_scale_city_v5/status.json").read_text(encoding="utf-8"))
    if state["status"] != "TWO_SCALE_CITY_EXPERIMENT_COMPLETE" or state["completed_groups"] != [GROUP] or state["group_order"] != [GROUP]:
        raise ValueError("v5 did not complete exactly the preregistered single group")
    cfg = after["frozen_protocol"]
    fleet = build_controlled_fleet(root, benchmark_start=START,
        simulation_end=START + pd.Timedelta(seconds=cfg["physical_drain_limit_s"]), requested_q_a=cfg["requested_q_a"], seed=cfg["supply_seed"])
    label = fleet.accounting["type_label_sha256"]
    if any(summary["all_day_supply"]["accounting"]["type_label_sha256"] != label for summary in summaries.values()):
        raise ValueError("effective supply labels differ")
    plans = {int(row.native_id): (str(row.vehicle_type), float((row.availability_start_time - START).total_seconds()),
        float((row.availability_end_time - START).total_seconds())) for row in fleet.scenario_fleet.itertuples(index=False)}
    del fleet
    public, private = {}, {}
    for version, summary in summaries.items():
        public[version], private[version] = _collect(root, version, summary, plans)
    b, a = private["v4"].sort_index(), private["v5"].sort_index()
    if not b.index.equals(a.index) or not b[["release_s", "C_compatible", "passenger_accepts_av"]].equals(a[["release_s", "C_compatible", "passenger_accepts_av"]]):
        raise ValueError("common request identities/releases/body/passenger assumptions differ")
    both = b.completed & a.completed
    paired = dict(shared_served=int(both.sum()), v4_only_served=int((b.completed & ~a.completed).sum()),
        v5_only_served=int((a.completed & ~b.completed).sum()),
        HV_to_C=int((both & b.vehicle_type.eq("HV") & a.vehicle_type.eq("AV")).sum()),
        C_to_HV=int((both & b.vehicle_type.eq("AV") & a.vehicle_type.eq("HV")).sum()),
        shared_served_total_wait_difference_mean_s=float((a.loc[both, "wait_s"] - b.loc[both, "wait_s"]).mean()),
        shared_served_pickup_distance_difference_mean_m=float((a.loc[both, "pickup_distance_m"] - b.loc[both, "pickup_distance_m"]).mean()),
        shared_served_pickup_distance_difference_km=float((a.loc[both, "pickup_distance_m"] - b.loc[both, "pickup_distance_m"]).sum() / 1000.),
        v4_only_served_pickup_distance_km=float(b.loc[b.completed & ~a.completed, "pickup_distance_m"].sum() / 1000.),
        v5_only_served_pickup_distance_km=float(a.loc[a.completed & ~b.completed, "pickup_distance_m"].sum() / 1000.))
    deltas = {key: public["v5"][key] - public["v4"][key] for key in
        ("completed", "expired", "service_rate", "served_C", "served_HV", "pickup_empty_km", "idle_empty_km", "total_empty_km", "runtime_s")}
    result = dict(status="OPERATIONS_V5_SINGLE_COMPARISON_ANALYZED", group=GROUP,
        common_cohort_count=len(b), test_date="20161031", profile="C", planned_AV_hour_ratio=.1, passenger_acceptance=.7,
        groups=public, paired=paired, difference_v5_minus_v4=deltas, common_input_sha256=common_inputs,
        type_label_sha256=label, one_day_one_seed=True, bundled_policy_change_not_component_ablation=True,
        analysis_invoked_simulation=False, analysis_invoked_routing=False, analysis_invoked_inference=False,
        native_v5_execution_completed=True, baseline_native_rerun=False, raw_identifiers_or_coordinates_exported=False,
        verification_status="ANALYZED_NOT_REPRODUCTION_RERUN", statistical_significance_claimed=False,
        scientific_generalization_claimed=False, runtime_s=perf_counter() - started, peak_rss_mib=_peak())
    atomic_json(root / DOC / "full_day_comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = analyze(args.root)
    print(json.dumps(dict(status=result["status"], difference=result["difference_v5_minus_v4"], paired=result["paired"],
        runtime_s=result["runtime_s"], peak_rss_mib=result["peak_rss_mib"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
