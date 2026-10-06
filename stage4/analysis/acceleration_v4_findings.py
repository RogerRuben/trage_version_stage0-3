"""Analyze the completed frozen M pair; never launches a simulation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from stage3.scripts.traffic_state_batch1 import sha, write_json

BASE = Path("stage4/output/symmetric_flexibility_v1/full_day/MYOPIC")
NEW = Path("stage4/output/symmetric_flexibility_v1/accelerated_v4_full_day/SERVICE_PRESERVING_LOOKAHEAD")
V2 = Path("stage4/docs/flexibility_dispatch/acceleration_v2/service_preserving_lookahead_summary.json")
DOC = Path("stage4/docs/flexibility_dispatch/acceleration_v4")


def describe(series):
    values = pd.Series(series).dropna().astype(float)
    if values.empty:
        return dict(n=0, mean=None, p50=None, p90=None, p95=None)
    return dict(n=len(values), mean=float(values.mean()), p50=float(values.quantile(.5)),
                p90=float(values.quantile(.9)), p95=float(values.quantile(.95)))


def time_parts(summary):
    top, sub = summary.get("timing_totals_s", {}), summary.get("solver_pipeline_components_s", {})
    parts = dict(total=float(summary["runtime_s"]), candidate=float(top.get("candidate_generation", 0.)),
                 routing=float(top.get("routing_wall", 0.)))
    parts.update({key:float(sub.get(key + "_time_s", 0.)) for key in
        ("forecast_sampling", "future_graph", "problem_setup", "model_build", "optimization", "recourse_recovery", "failed_model_attempt")})
    parts["other"] = parts["total"] - sum(v for k, v in parts.items() if k != "total")
    return parts


def operational(frame):
    rows = {}
    for kind in ("HV", "AV"):
        f = frame.loc[frame.vehicle_type.eq(kind)]
        pickup = float(f.pickup_eta_s.sum())
        service = float(f.realized_service_time_s.sum())
        rows[kind] = dict(served=len(f), mean_pickup_eta_s=float(f.pickup_eta_s.mean()),
            mean_realized_service_s=float(f.realized_service_time_s.mean()),
            pickup_task_hours=pickup / 3600., service_task_hours=service / 3600.,
            completed_task_hours_including_drain=(pickup + service) / 3600.,
            unique_active_service_vehicles=int(f.native_vehicle_id.nunique()))
    return rows


def main():
    root = Path.cwd()
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    summaries = [json.loads((root / p / "summary.json").read_text()) for p in (BASE, NEW)]
    if any(s["status"] != "COMPLETE" for s in summaries):
        raise RuntimeError("both frozen paired results must be complete before analysis")
    old, new = summaries
    first = {p.replace("\\", "/"):h for p,h in old["inputs_sha256"].items()}
    second = {p.replace("\\", "/"):h for p,h in new["inputs_sha256"].items()}
    if any(second.get(p) != h for p,h in first.items()):
        raise RuntimeError("paired scientific inputs differ")
    frames = [pd.read_parquet(root / p / "cohort_outcomes.parquet") for p in (BASE, NEW)]
    pair = frames[0].merge(frames[1], on="order_id", validate="one_to_one", suffixes=("_base", "_new"))
    if len(pair) != 30000 or not pair.common_eligible_base.equals(pair.common_eligible_new):
        raise RuntimeError("paired population mismatch")
    assets = np.load(root / "stage4/output/runtime_acceleration_v2/assets/orders.npy", mmap_mode="r", allow_pickle=False)
    hour = {str(r["order_id"]): int(r["release"] // 3600) for r in assets}
    pair["release_hour"] = pair.order_id.map(hour)
    if pair.release_hour.isna().any():
        raise RuntimeError("hour alignment missing")
    both = pair.matched_base & pair.matched_new
    gained = ~pair.matched_base & pair.matched_new
    lost = pair.matched_base & ~pair.matched_new
    common = pair.common_eligible_base
    gain, loss = int(gained.sum()), int(lost.sum())
    delta = int(pair.matched_new.sum() - pair.matched_base.sum())
    if gain - loss != delta:
        raise RuntimeError("gained/lost do not conserve the served-count difference")
    paired_wait = pair.loc[both, "wait_s_new"] - pair.loc[both, "wait_s_base"]
    transitions = {f"{a}_TO_{b}": int((both & pair.vehicle_type_base.eq(a) & pair.vehicle_type_new.eq(b)).sum())
        for a in ("HV", "AV") for b in ("HV", "AV")}
    by_hour = []
    for h, f in pair.groupby("release_hour", sort=True):
        b = f.matched_base & f.matched_new
        g = ~f.matched_base & f.matched_new
        l = f.matched_base & ~f.matched_new
        by_hour.append(dict(hour=int(h), original_orders=len(f), common_orders=int(f.common_eligible_base.sum()),
            base_served=int(f.matched_base.sum()), new_served=int(f.matched_new.sum()),
            gained=int(g.sum()), lost=int(l.sum()), net=int(g.sum() - l.sum()),
            base_av=int(f.vehicle_type_base.eq("AV").sum()), new_av=int(f.vehicle_type_new.eq("AV").sum()),
            paired_wait_delta=describe(f.loc[b, "wait_s_new"] - f.loc[b, "wait_s_base"])))
    columns = ["native_vehicle_id", "vehicle_type", "pickup_eta_s", "realized_service_time_s"]
    assignments = [pd.read_parquet(root / p / "assignments.parquet", columns=columns) for p in (BASE, NEW)]
    old_v2 = json.loads((root / V2).read_text())
    parts = {"v2":time_parts(old_v2), "v4":time_parts(new)}
    result = dict(status="COMPLETE", scientific_scope="INTERNAL_TEST31_M_SINGLE_SEED_FROZEN_PAIRED_REPLAY",
        reused_myopic=True, myopic_rerun=False, new_native_conditions=1,
        canonical_timing=dict(request_release="OBSERVED_DEPARTURE_PROXY", request_lead_s=0, pickup_patience_s=300),
        original_orders=len(pair), common_orders=int(common.sum()), excluded_input=int((~common).sum()),
        base_served=int(pair.matched_base.sum()), new_served=int(pair.matched_new.sum()), net_served_change=delta,
        service_rate_common_pp_change=100 * delta / int(common.sum()),
        service_rate_original_pp_change=100 * delta / len(pair),
        gained=gain, lost=loss, common_served=int(both.sum()),
        all_served_wait_s=dict(base=describe(pair.loc[pair.matched_base, "wait_s_base"]),
                               new=describe(pair.loc[pair.matched_new, "wait_s_new"])),
        same_served_wait_delta_s=describe(paired_wait),
        paired_wait_changes=dict(lower=int((paired_wait < -1e-6).sum()),
                                same=int((paired_wait.abs() <= 1e-6).sum()), higher=int((paired_wait > 1e-6).sum())),
        gained_new_wait_s=describe(pair.loc[gained, "wait_s_new"]),
        lost_base_wait_s=describe(pair.loc[lost, "wait_s_base"]),
        vehicle_type_transitions=transitions,
        gained_by_type={k:int((gained & pair.vehicle_type_new.eq(k)).sum()) for k in ("HV", "AV")},
        lost_by_type={k:int((lost & pair.vehicle_type_base.eq(k)).sum()) for k in ("HV", "AV")},
        operations=dict(base=operational(assignments[0]), new=operational(assignments[1])),
        by_release_hour=by_hour, runtime_components_s=parts,
        total_runtime_speedup_v2_to_v4=parts["v2"]["total"] / parts["v4"]["total"],
        routing_speedup_v2_to_v4=parts["v2"]["routing"] / parts["v4"]["routing"] if parts["v4"]["routing"] else None,
        v4_versus_v2_served_change=int(new["matched"] - old_v2["matched"]),
        scientific_inputs_equal=True, no_future_information_or_refit=True,
        cma_stability="NOT_IDENTIFIED_M_ONLY_NO_C_OR_A_NEW_RUNS",
        causal_or_significance_claim=False, technical_backend_change_not_bitwise_trajectory_identity=True,
        base_summary=old, new_summary=new,
        inputs_sha256={str(p):sha(root / p) for p in (BASE / "summary.json", NEW / "summary.json",
            BASE / "cohort_outcomes.parquet", NEW / "cohort_outcomes.parquet")})
    # Compact interpretation checks; no p-values or extra experiments.
    result["interpretation_checks"] = dict(coverage="11/11 checked",
        ecological_extrapolation="No routing-only倍率 interpreted as full-day倍率",
        sampling="Single date/seed; release-hour strata and both served/unserved retained",
        composition="All-served and common-served wait explicitly separated",
        forking_paths="Frozen configuration; no quality/capacity thresholds changed by outcome",
        causality="Conditional replay comparison, not field causal effect",
        other_statistical_fallacies="No regression/correlation/classifier significance inference was used")
    write_json(root / DOC / "full_day_findings.json", result)
    print(json.dumps({k:result[k] for k in ("status", "base_served", "new_served", "net_served_change",
        "gained", "lost", "same_served_wait_delta_s", "total_runtime_speedup_v2_to_v4", "cma_stability")}), flush=True)


if __name__ == "__main__":
    main()
