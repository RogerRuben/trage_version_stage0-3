"""Read completed paired products; no model fitting or native replay."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa

from stage3.scripts.traffic_state_batch1 import sha, write_json

BASE = Path("stage4/output/symmetric_flexibility_v1/full_day/MYOPIC")
NEW = Path("stage4/output/symmetric_flexibility_v1/accelerated_full_day/SERVICE_PRESERVING_LOOKAHEAD")
DOC = Path("stage4/docs/flexibility_dispatch/acceleration_v1")
RUN_RECORD = Path("stage4/output/symmetric_flexibility_v1/runner_logs/20261005T083905Z")


def analyze(root):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    summaries = [json.loads((root/p/"summary.json").read_text()) for p in (BASE, NEW)]
    if any(s["status"] != "COMPLETE" for s in summaries):
        raise ValueError("analysis requires two complete executions")
    launch = json.loads((root/RUN_RECORD/"launch.json").read_text())
    exit_record = json.loads((root/RUN_RECORD/"exit.json").read_text())
    if (exit_record["exit_code"] != 0 or launch["policy"] != summaries[1]["policy"]
            or launch["acceleration_config_sha256"] != summaries[1]["acceleration_config_sha256"]):
        raise ValueError("completed runner provenance differs")
    for summary in summaries:
        for path, expected in summary["inputs_sha256"].items():
            if sha(root/path) != expected:
                raise ValueError(f"frozen input changed: {path}")
    fleets = [json.loads((root/p/"fleet_accounting.json").read_text()) for p in (BASE, NEW)]
    if fleets[0] != fleets[1]:
        raise ValueError("paired fleet accounting differs")
    frames = [pd.read_parquet(root/p/"cohort_outcomes.parquet") for p in (BASE, NEW)]
    assignments = [pd.read_parquet(root/p/"assignments.parquet") for p in (BASE, NEW)]
    routes = pd.read_parquet(root/"stage4/output/symmetric_flexibility_v1/input/test31_research_routes.parquet",
        columns=["order_id", "reverse_token_count", "compatible_M"])
    for s, frame, assigned in zip(summaries, frames, assignments):
        if (len(frame) != 30000 or not frame.order_id.is_unique or not assigned.native_request_id.is_unique
                or len(assigned) != s["matched"] or int(frame.matched.sum()) != len(assigned)
                or not assigned.completed.all()
                or int(frame.matched.sum()+frame.expired.sum()+frame.excluded_input.sum()) != 30000
                or frame.wait_s.max() > 300+1e-6):
            raise ValueError("completed order/task accounting failed")
    trace = pd.read_parquet(root/NEW/"solver_trace.parquet")
    epochs = pd.read_parquet(root/NEW/"epochs.parquet")
    if not trace.current_face_preserved.all():
        raise ValueError("current service face violated")
    active = trace.loc[trace.variable_count.notna()]
    s = summaries[1]
    result = dict(status="ANALYZED_NOT_INDEPENDENT_RERUN", original_orders=30000,
        common_eligible=s["common_eligible"], paired_fleet_accounting_identical=True, fleet=fleets[0],
        inputs_unchanged=True, execution_code_sha=launch["code_sha"], runner_exit_code=exit_record["exit_code"],
        myopic_reused=True, native_scientific_conditions_added=0,
        fallback_reasons=trace.resource_fallback.value_counts().to_dict(),
        current_face_violation_epochs=int((~trace.current_face_preserved).sum()),
        active_two_stage_epochs=len(active), empty_current_graph_epochs=int(trace.current_arc_count.eq(0).sum()),
        variable_quantiles=active.variable_count.quantile([.5,.9,.95,1]).to_dict(),
        integer_variable_quantiles=active.integer_variable_count.quantile([.5,.9,.95,1]).to_dict(),
        timing_quantiles_s={c:trace[c].quantile([.5,.9,.95,.99,1]).to_dict() for c in
            ("solver_time_s","forecast_sampling_time_s","future_graph_time_s","problem_setup_time_s","optimization_time_s")},
        timing_percent={k:v/s["runtime_s"]*100 for k,v in s["timing_totals_s"].items()},
        other_time_s=s["runtime_s"]-sum(s["timing_totals_s"].values()),
        component_percent={k:v/s["runtime_s"]*100 for k,v in s["solver_pipeline_components_s"].items()},
        cache_hit_share=s["arc_cache_hits"]/s["arc_lookups"],
        exact_cross_epoch_hit_share=s["cross_epoch_exact_cache_hits"]/s["arc_lookups"],
        relative_service_increase=(s["matched"]/summaries[0]["matched"]-1)*100,
        mean_served_wait_increase_s=s["mean_wait_s"]-summaries[0]["mean_wait_s"],
        sampled_process_group_peak_mib=s["peak_process_group_rss_mib"],
        parent_peak_mib=s["peak_rss_mib"], waiting_is_conditional_on_service=True,
        one_day_one_profile_one_seed=True, statistical_significance_claim=False)
    result["by_policy"] = []
    for s, frame, assigned in zip(summaries, frames, assignments):
        annotated = assigned.merge(routes,on="order_id",validate="one_to_one")
        av = annotated.vehicle_type.eq("AV")
        result["by_policy"].append(dict(policy=s["policy"],
            reverse_route_assigned_by_type=annotated.loc[annotated.reverse_token_count.gt(0)].groupby("vehicle_type").size().to_dict(),
            av_capability_violations=int((av & ~annotated.compatible_M).sum()),
            wait_by_vehicle=assigned.merge(frame[["order_id","wait_s"]],on="order_id",validate="one_to_one").groupby(
                "vehicle_type").wait_s.agg(["count","mean","median"]).to_dict()))
    hourly = epochs.assign(hour=(epochs.simulation_time_s//3600).astype(int)).groupby("hour").agg(
        epochs=("simulation_time_s","size"), routing_s=("routing_time_s","sum"),
        candidate_s=("candidate_generation_time_s","sum"), pipeline_s=("solver_time_s","sum"))
    result["hourly_timing"] = hourly.to_dict("index")
    result["busy_forecast_occurrences"] = {c:int(trace[c].sum()) for c in
        ("busy_states","overdue_busy_states","conditional_busy_states","unsupported_busy_states_omitted")}
    write_json(root/DOC/"result_diagnostics.json",result)
    print(json.dumps(dict(status=result["status"],active_two_stage_epochs=len(active),
        fallback_reasons=result["fallback_reasons"],by_policy=result["by_policy"]),ensure_ascii=False))


if __name__ == "__main__":
    analyze(Path.cwd())
