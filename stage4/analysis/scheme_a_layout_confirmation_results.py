"""Read the six completed cases plus the existing two-case reference.

Only projected customer/movement events and aggregate comparisons are read.
No simulations, layouts, inference, routing, or extra experiment conditions.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import pandas as pd
import psutil

from .capability_chain_instances import atomic_json
from .c_operations_diagnostic import START, TOL, sha
from .scheme_a_layout_confirmation import CONDITIONS, DOC, GROUPS, OUT, REFERENCE
from .scheme_a_layout_mechanism import DELAYS_S, activation_decomposition, slot_accounting
from stage4.dispatch.controlled_fleet import build_controlled_fleet

JOB_COLS = ["native_vehicle_id", "vehicle_type", "simulation_time_s", "pickup_time_s", "service_end_time_s", "completed"]
MOVE_COLS = ["native_vehicle_id", "vehicle_type", "start_time_s", "end_time_s", "status"]


def first_customer_metrics(frame):
    working = frame.loc[frame.served]
    distribution = lambda values: None if not len(values) else dict(mean=float(values.mean()), p50=float(values.median()), p90=float(values.quantile(.9)))
    return dict(slots=len(frame), served_slots=int(frame.served.sum()), never_served_slots=int((~frame.served).sum()),
        first_assignment_delay_served_only_s=distribution(working.first_assignment_delay_s),
        first_boarding_delay_served_only_s=distribution(working.first_boarding_delay_s),
        pre_first_idle_h=float(frame.pre_first_idle_s.sum() / 3600),
        post_first_idle_h=float(frame.post_first_idle_s.sum() / 3600),
        curves_all_template_C_slots=[dict(delay_s=delay,
            assignment_share=float((frame.first_assignment_in_calendar_window & frame.first_assignment_delay_s.le(delay)).mean()),
            boarding_share=float((frame.first_boarding_in_calendar_window & frame.first_boarding_delay_s.le(delay)).mean())) for delay in DELAYS_S],
        unsuccessful_slots_retained=True, vehicle_activation_clock_not_passenger_wait=True,
        conditional_sample_composition_changes=True)


def project_slots(root, directory, av, expected, sources):
    jobs = pd.read_parquet(root / directory / "assignments.parquet", columns=JOB_COLS)
    moves = pd.read_parquet(root / directory / "empty_movements.parquet", columns=MOVE_COLS)
    if not jobs.completed.all() or not moves.status.isin(["COMPLETED", "INTERRUPTED_BY_CUSTOMER"]).all():
        raise ValueError("incomplete physical event ledger")
    for name in ("summary.json", "assignments.parquet", "empty_movements.parquet"):
        path = directory / name
        digest = expected["source_products_sha256"][name]
        if sha(root / path) != digest:
            raise ValueError("completed source product changed")
        sources[path.as_posix()] = digest
    jobs = jobs.loc[jobs.vehicle_type.eq("AV")]
    moves = moves.loc[moves.vehicle_type.eq("AV")]
    j = {int(vid): [(float(r.simulation_time_s), float(r.pickup_time_s), float(r.service_end_time_s))
        for r in values.itertuples(index=False)] for vid, values in jobs.groupby("native_vehicle_id")}
    m = {int(vid): [(float(r.start_time_s), float(r.end_time_s)) for r in values.itertuples(index=False)]
        for vid, values in moves.groupby("native_vehicle_id")}
    records = []
    for r in av.itertuples(index=False):
        start, end = (r.availability_start_time - START).total_seconds(), (r.availability_end_time - START).total_seconds()
        metrics = slot_accounting(start, end, j.get(int(r.native_id), []), m.get(int(r.native_id), []))
        records.append(dict(metrics, native_id=int(r.native_id), activation_s=start, admission_end_s=end))
    frame = pd.DataFrame(records).set_index("native_id").sort_index()
    accounting = expected["admission_state"]["AV"]
    if (int(frame.orders.sum()) != expected["served_C"]
            or int((~frame.served).sum()) != accounting["never_served_slots"]
            or abs(frame.loaded_s.sum() / 3600 - accounting["LOADED_hours"]) > TOL
            or abs(frame.idle_s.sum() / 3600 - accounting["idle_hours"]) > TOL):
        raise ValueError("first-customer events do not reconcile with the completed comparison")
    return frame


def collect(root):
    started = perf_counter()
    root = Path(root).resolve()
    target = root / DOC / "six_case_results.json"
    if target.exists():
        raise ValueError("existing aggregate retained; no silent overwrite")
    entries = [("Q10_REFERENCE_EXISTING_NOT_RERUN", REFERENCE,
        Path("stage4/output/capability_chain_planning/layout_comparison_300_600_v1")),
        *((name, DOC / name / "native/comparison.json", OUT / name / "native") for name, _, _ in CONDITIONS)]
    conditions, sources, native_shas = {}, {}, set()
    for name, comparison_path, directory in entries:
        comparison = json.loads((root / comparison_path).read_text(encoding="utf-8"))
        sources[comparison_path.as_posix()] = sha(root / comparison_path)
        if comparison["status"] != "PREPARED_LAYOUT_COMPARISON_ANALYZED":
            raise ValueError("requires completed pair analysis")
        summary = json.loads((root / directory / GROUPS[0] / "summary.json").read_text(encoding="utf-8"))
        cfg = summary["frozen_protocol"]
        if name != entries[0][0]:
            state = json.loads((root / directory / "status.json").read_text(encoding="utf-8"))
            if state["status"] != "TWO_SCALE_CITY_EXPERIMENT_COMPLETE" or state["completed_groups"] != list(GROUPS):
                raise ValueError("the prescribed two native groups are not both complete")
            native_shas.add(comparison["execution_binding"]["code_sha"])
        fleet = build_controlled_fleet(root, benchmark_start=START,
            simulation_end=START + pd.Timedelta(seconds=cfg["physical_drain_limit_s"]),
            requested_q_a=cfg["requested_q_a"], seed=cfg["supply_seed"])
        if fleet.accounting["type_label_sha256"] != comparison["type_label_sha256"]:
            raise ValueError("supply label reconstruction differs")
        accounting = {key: fleet.accounting[key] for key in ("requested_q_a", "q_a_realized", "achieved_av_vehicle_hours",
            "achieved_hv_vehicle_hours", "h_base_day_exact", "av_count", "seed", "type_label_sha256")}
        av = fleet.scenario_fleet.loc[fleet.scenario_fleet.vehicle_type.eq("AV")].copy()
        del fleet
        frames, groups = {}, {}
        for group in GROUPS:
            expected = comparison["groups"][group]
            frames[group] = project_slots(root, directory / group, av, expected, sources)
            frame = frames[group]
            admission_h, loaded_h = float(frame.admission_s.sum() / 3600), float(frame.loaded_s.sum() / 3600)
            if abs(admission_h - accounting["achieved_av_vehicle_hours"]) > TOL:
                raise ValueError("normalization denominator differs from actual C admission hours")
            groups[group] = dict(C_loaded_h=loaded_h, C_admission_h=admission_h, C_loaded_utilization=loaded_h / admission_h,
                completed_orders=expected["completed"], expired_orders=expected["expired"], served_C=expected["served_C"],
                served_HV=expected["served_HV"], total_empty_km=expected["total_empty_km"],
                runtime_s=expected["runtime_s"], setup_time_s=expected["setup_time_s"], peak_rss_mib=expected["peak_rss_mib"],
                current_flow_status=expected["current_flow_status"], physical_accounting=expected["physical_accounting"],
                first_customer=first_customer_metrics(frame))
        before, after = (frames[group] for group in GROUPS)
        if not before.index.equals(after.index) or not before[["activation_s", "admission_end_s"]].equals(after[["activation_s", "admission_end_s"]]):
            raise ValueError("pair did not retain identical physical C windows")
        b, a = (groups[group] for group in GROUPS)
        delta = dict(C_loaded_utilization_pp=100 * (a["C_loaded_utilization"] - b["C_loaded_utilization"]),
            C_loaded_h=a["C_loaded_h"] - b["C_loaded_h"], served_C=a["served_C"] - b["served_C"],
            completed_orders=a["completed_orders"] - b["completed_orders"], total_empty_km=a["total_empty_km"] - b["total_empty_km"],
            newly_served_slots=int((~before.served & after.served).sum()), lost_service_slots=int((before.served & ~after.served).sum()),
            both_served_slots=int((before.served & after.served).sum()), neither_served_slots=int((~before.served & ~after.served).sum()),
            pre_first_idle_h=float((after.pre_first_idle_s - before.pre_first_idle_s).sum() / 3600),
            post_first_idle_h=float((after.post_first_idle_s - before.post_first_idle_s).sum() / 3600),
            activation_product_decomposition=activation_decomposition(before, after))
        conditions[name] = dict(supply=accounting, groups=groups, joint_minus_compatible=delta,
            native_producer_sha=comparison["execution_binding"]["code_sha"], existing_reference=name == entries[0][0])
        del av, before, after, frames
        gc.collect()
    if len(native_shas) != 1:
        raise ValueError("the six new cases used different producer commits")
    if any(sha(root / path) != digest for path, digest in sources.items()):
        raise ValueError("a completed product changed while aggregating")
    memory = psutil.Process().memory_info()
    peak = max(memory.rss, getattr(memory, "peak_wset", memory.rss)) / 2**20
    if peak > 512:
        raise MemoryError("projected result aggregation exceeded 512 MiB")
    result = dict(status="SIX_NATIVE_CASES_COMPLETE_AND_ANALYZED", origin_skill="academic-research-suite",
        verification_status="ANALYZED_COMPLETED_EXECUTIONS_NOT_REPRODUCTION_RERUN",
        new_completed_native_cases=6, existing_reference_cases_reused=2, conditions=conditions,
        source_products_sha256=sources, analysis_source_sha256=sha(Path(__file__)),
        simulation_invoked=False, routing_invoked=False, inference_invoked=False, dense_matrix=False, gpu_used=False,
        raw_identifiers_or_coordinates_exported=False, multiday_generalization_or_significance_claimed=False,
        runtime_s=perf_counter() - started, peak_rss_mib=peak)
    atomic_json(target, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = collect(args.root)
    print(json.dumps(dict(status=result["status"], runtime_s=result["runtime_s"], peak_rss_mib=result["peak_rss_mib"])), flush=True)


if __name__ == "__main__":
    main()
