"""Exactly two serial M continuations; reuse all existing reference results."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import pandas as pd
import psutil

from stage3.scripts.traffic_state_batch1 import sha, write_json
from stage4.analysis.flexibility_diagnostics import DOC, MODEL, NEW_CONFIG, OUTPUT
from stage4.analysis.flexibility_prepare import CONFIG, OUT as INPUT
from stage4.analysis.flexibility_windows import OUTPUT as OLD_WINDOWS, condition
from stage4.dispatch.remaining_time import TrainRemainingTime
from stage4.fleetpy_adapter.upstream import load_fleetpy_bindings

BASELINES = ("MYOPIC", "AV_FIRST", "LOOKAHEAD")


def analyze(root):
    cfg = json.loads((root / NEW_CONFIG).read_text())
    complete = json.loads((root / OUTPUT / "summary.json").read_text())
    if complete["status"] != "COMPLETE" or len(complete["rows"]) != 2:
        raise ValueError("both declared native continuations must complete")
    groups = []
    for cut in cfg["cuts_s"]:
        name = f'M_{cut}_{cfg["policy"]}'
        new = pd.read_parquet(root / OUTPUT / name / "cohort_outcomes.parquet").set_index("order_id")
        summaries = {}
        paired = {}
        for policy in BASELINES:
            directory = root / OLD_WINDOWS / f"M_{cut}_{policy}"
            summaries[policy] = json.loads((directory / "summary.json").read_text())
            old = pd.read_parquet(directory / "cohort_outcomes.parquet").set_index("order_id")
            if len(old) != len(new) or set(old.index) != set(new.index):
                raise ValueError("reference cohort identity mismatch")
            old = old.reindex(new.index)
            common = new.matched & old.matched
            paired[policy] = dict(gained=int((new.matched & ~old.matched).sum()),
                lost=int((~new.matched & old.matched).sum()), net=int(new.matched.sum()-old.matched.sum()),
                common_served=int(common.sum()),
                common_served_mean_wait_delta_s=float((new.loc[common, "wait_s"]-old.loc[common, "wait_s"]).mean()))
        summaries[cfg["policy"]] = json.loads((root / OUTPUT / name / "summary.json").read_text())
        trace = pd.read_parquet(root / OUTPUT / name / "solver_trace.parquet")
        violation = int((~trace.current_face_preserved).sum())
        if violation:
            raise ValueError("current service face was not preserved")
        groups.append(dict(cut=cut, cohort=len(new),
            matched={p: r["matched"] for p, r in summaries.items()},
            av_served={p: r["av"] for p, r in summaries.items()},
            mean_wait_among_served_s={p: r["mean_wait_s"] for p, r in summaries.items()},
            paired_new_vs_reference=paired, decision_epochs=len(trace), current_face_violation_epochs=violation,
            busy_state_evaluations=int(trace.busy_states.fillna(0).sum()),
            overdue_busy_state_evaluations=int(trace.overdue_busy_states.fillna(0).sum()),
            conditional_busy_state_evaluations=int(trace.conditional_busy_states.fillna(0).sum()),
            unsupported_busy_states_omitted=int(trace.unsupported_busy_states_omitted.fillna(0).sum()),
            mean_conditional_ready_shift_s=float(trace.busy_ready_shift_sum_s.fillna(0).sum()/max(1,trace.conditional_busy_states.fillna(0).sum())),
            maximum_model_variables=int(trace.variable_count.max()), maximum_model_nonzeros=int(trace.nonzeros.max())))
    totals = {p: sum(g["matched"][p] for g in groups) for p in (*BASELINES, cfg["policy"])}
    net = totals[cfg["policy"]] - totals["MYOPIC"]
    deltas = [g["paired_new_vs_reference"]["MYOPIC"]["net"] for g in groups]
    classification = ("BOUNDED_WINDOW_IMPROVEMENT" if net > 0 and min(deltas) >= 0 else
                      "MIXED_WINDOW_RESULT" if net > 0 else "NO_CLEAR_SERVICE_GAIN")
    result = dict(status="COMPLETE", classification=classification, groups=groups, cohort=sum(g["cohort"] for g in groups),
        total_matched=totals, net_vs_myopic=net,
        resources=dict(runtime_s=complete["runtime_s"], peak_rss_mib=complete["peak_rss_mib"],
            routing_failures=sum(r["routing_failures"] for r in complete["rows"]),
            resource_fallback_epochs=sum(r["resource_fallback_epochs"] for r in complete["rows"]),
            dense_request_vehicle_matrix=False, gpu_used=False),
        interpretation=dict(new_native_conditions=2, reused_reference_conditions=6,
            combined_objective_and_remaining_time_change_not_a_factorial_ablation=True,
            epoch_current_service_protection_not_window_dominance=True, no_full_day_superiority_claim=True,
            test31_not_used_to_fit_remaining_time_or_demand=True, no_more_experiments_automatically_started=True,
            gamma_cost_repositioning_still_disabled=True,
            statistical_fallacy_coverage="11/11; paired exploratory windows, no significance test",
            conditional_model_uses_train_gps_duration_proxy_not_true_boarding_events=True))
    write_json(root / DOC / "analysis.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def run(root, fleetpy, resume):
    cfg = json.loads((root / NEW_CONFIG).read_text())
    base = json.loads((root / CONFIG).read_text())
    if (cfg["profile"] != "M" or cfg["cuts_s"] != [37800, 63000]
            or cfg["policy"] != "SERVICE_PRESERVING_LOOKAHEAD" or cfg["additional_conditions"] != 2
            or cfg["refit_m3"] or cfg["refit_demand"] or cfg["parameter_search"]):
        raise ValueError("this runner is restricted to the authorized two-condition protocol")
    prepared = json.loads((root / INPUT / "preparation_summary.json").read_text())
    if prepared["config_sha256"] != sha(root / CONFIG):
        raise ValueError("existing base input/config mismatch")
    load_fleetpy_bindings(fleetpy)
    model = TrainRemainingTime(json.loads((root / MODEL).read_text()))
    checkpoints = {r["cut"]: r for r in json.loads((root / "stage4/output/traffic_mechanism_v1/states/checkpoints.json").read_text())["rows"] if r["profile"] == "M"}
    protected_paths = [CONFIG, NEW_CONFIG, MODEL, INPUT/"test31_research_routes.parquet", INPUT/"train_request_templates.parquet",
        Path("stage3/config/stage3_av_capability_profiles.json"), Path("stage2/output_v5_2/development/M3/epoch_004.pt")]
    for cut in cfg["cuts_s"]:
        protected_paths.append(Path(checkpoints[cut]["path"]))
        for policy in BASELINES:
            protected_paths += [OLD_WINDOWS/f"M_{cut}_{policy}"/p for p in ("summary.json", "cohort_outcomes.parquet")]
    protected = {str(p):sha(root/p) for p in protected_paths}
    routes = pd.read_parquet(root / INPUT / "test31_research_routes.parquet")
    templates = pd.read_parquet(root / INPUT / "train_request_templates.parquet")
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists() and not resume:
        raise ValueError("output already started; explicit --resume is required")
    summary = dict(status="RUNNING", active=None, rows=[], inputs_sha256=protected,
                   new_policy=cfg["policy"], frozen_original_baselines_preserved=True)
    started = time.monotonic()
    try:
        for cut in cfg["cuts_s"]:
            name = f'M_{cut}_{cfg["policy"]}'
            directory = output / name
            done = directory / "summary.json"
            if resume and done.is_file() and json.loads(done.read_text()).get("status") == "COMPLETE":
                summary["rows"].append(json.loads(done.read_text()))
                continue
            directory.mkdir(exist_ok=resume)
            summary["active"] = name
            write_json(output / "summary.json", summary)
            row = condition(root, checkpoints[cut], base, cut, "M", cfg["policy"], directory, routes, templates, model)
            summary["rows"].append(row)
            write_json(output / "summary.json", summary)
            print(json.dumps(dict(completed=name, matched=row["matched"], av=row["av"], runtime_s=row["runtime_s"])), flush=True)
            gc.collect()
        if protected != {p:sha(root/p) for p in protected}:
            raise RuntimeError("declared input or existing baseline changed during execution")
        summary.update(status="COMPLETE", active=None, runtime_s=time.monotonic()-started,
                       peak_rss_mib=psutil.Process().memory_info().peak_wset / 2**20)
    except Exception as error:
        summary.update(status="STOPPED", error=repr(error), runtime_s=time.monotonic()-started)
        write_json(output / "summary.json", summary)
        raise
    write_json(output / "summary.json", summary)
    write_json(root / DOC / "summary.json", summary)
    analyze(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleetpy-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(Path.cwd(), args.fleetpy_root, args.resume)
