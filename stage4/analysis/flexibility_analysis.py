"""Compact paired-cohort reporting; no new runs or parameter selection."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stage3.scripts.traffic_state_batch1 import write_json
from stage4.analysis.flexibility_prepare import CONFIG, DOC, OUT as INPUT
from stage4.analysis.flexibility_windows import OUTPUT


def run(root):
    cfg = json.loads((root / CONFIG).read_text())
    complete = json.loads((root / OUTPUT / "summary.json").read_text())
    comparison = json.loads((root / DOC / "native_window_comparison.json").read_text())
    expected = {(cut, profile, policy) for cut in cfg["cuts_s"]
                for profile in cfg["profiles"] for policy in cfg["policies"]}
    rows = complete["rows"]
    actual = {(r["cut"], r["profile"], r["policy"]) for r in rows}
    if complete["status"] != "COMPLETE" or actual != expected or len(rows) != len(expected):
        raise ValueError("native window set is incomplete")
    if any(not r["physical_reconciliation"] or r["cohort"] != r["matched"] + r["expired"] for r in rows):
        raise ValueError("physical/cohort reconciliation failed")
    routes = pd.read_parquet(root / INPUT / "test31_research_routes.parquet",
        columns=["order_id", "research_data_ready", "compatible_C", "compatible_M", "compatible_A"])
    nested = bool((~routes.compatible_C | routes.compatible_M).all()
                  and (~routes.compatible_M | routes.compatible_A).all())
    if routes.order_id.duplicated().any() or not nested:
        raise ValueError("route identity/compatibility nesting failed")
    indexed = {(r["cut"], r["profile"], r["policy"]): r for r in rows}
    paired = {(r["cut"], r["profile"], r["policy"]): r for r in comparison["pairs"]}
    groups = []
    for cut in cfg["cuts_s"]:
        for profile in cfg["profiles"]:
            conditions = {policy: indexed[(cut, profile, policy)] for policy in cfg["policies"]}
            n = conditions["MYOPIC"]["cohort"]
            if any(r["cohort"] != n for r in conditions.values()):
                raise ValueError("cohort size differs across policies")
            groups.append(dict(cut=cut, profile=profile, cohort=n,
                matched={p: r["matched"] for p, r in conditions.items()},
                service_rate={p: r["matched"] / n for p, r in conditions.items()},
                av_served={p: r["av"] for p, r in conditions.items()},
                mean_wait_among_served_s={p: r["mean_wait_s"] for p, r in conditions.items()},
                paired_vs_myopic={p: paired[(cut, profile, p)] for p in ("AV_FIRST", "LOOKAHEAD")}))
    aggregates = []
    for profile in cfg["profiles"]:
        group = [r for r in groups if r["profile"] == profile]
        n = sum(r["cohort"] for r in group)
        matched = {p: sum(r["matched"][p] for r in group) for p in cfg["policies"]}
        aggregates.append(dict(profile=profile, cohort=n, matched=matched,
            service_rate={p: count / n for p, count in matched.items()},
            net_vs_myopic={p: matched[p] - matched["MYOPIC"] for p in ("AV_FIRST", "LOOKAHEAD")},
            percentage_points_vs_myopic={p: 100 * (matched[p] - matched["MYOPIC"]) / n
                                        for p in ("AV_FIRST", "LOOKAHEAD")}))
    diagnostics = []
    timing = {k: 0.0 for k in ("candidate_generation_time_s", "routing_time_s", "solver_time_s")}
    max_topk = max_arcs = 0
    for row in rows:
        trace = pd.read_parquet(root / OUTPUT / f'{row["profile"]}_{row["cut"]}_{row["policy"]}/solver_trace.parquet')
        epochs = pd.read_parquet(root / OUTPUT / f'{row["profile"]}_{row["cut"]}_{row["policy"]}/epochs.parquet',
            columns=[*timing, "candidate_topk_pairs", "valid_or_arcs"])
        for column in timing:
            timing[column] += float(epochs[column].sum())
        max_topk = max(max_topk, int(epochs.candidate_topk_pairs.max()))
        max_arcs = max(max_arcs, int(epochs.valid_or_arcs.max()))
        diagnostics.append(dict(cut=row["cut"], profile=row["profile"], policy=row["policy"],
            decision_epochs=len(trace),
            zero_selected_with_current_arcs=int((trace.current_arc_count.gt(0) & trace.current_selected.eq(0)).sum()),
            interpretation="Observation only; not a same-state maximum-matching counterfactual."))
    resource = dict(runtime_s=complete["runtime_s"], peak_rss_mib=complete["peak_rss_mib"],
        routing_failures=sum(r["routing_failures"] for r in rows),
        routing_arc_evaluations=sum(r["routing_arc_evaluations"] for r in rows),
        resource_fallback_epochs=sum(r["resource_fallback_epochs"] for r in rows),
        maximum_successful_model_variables=max(r["maximum_model_variables"] for r in rows),
        maximum_successful_model_nonzeros=max(r["maximum_model_nonzeros"] for r in rows),
        adapter_time_s=sum(r["solver_time_s"] for r in rows),
        current_epoch_timing_s=timing, maximum_current_topk_pairs=max_topk,
        maximum_current_valid_arcs=max_arcs,
        gpu_used=False, dense_request_vehicle_matrix=False)
    result = dict(status="COMPLETE", classification="BOUNDED_EXPLORATORY_NATIVE_COMPARISON",
        groups=groups, aggregates_by_profile=aggregates, resources=resource,
        descriptive_epoch_diagnostics=diagnostics,
        necessary_qa=dict(completed_conditions=len(rows), expected_conditions=len(expected),
            physical_reconciliation=True, route_compatibility_nested=nested,
            forecast_dates=cfg["forecast_train_dates"], forecast_fit_on_test31=False,
            frozen_profile_checkpoint_config_unchanged=True),
        interpretation=dict(independent_window_count=len(cfg["cuts_s"]),
            profiles_are_not_independent_replications=True, paired_gains_not_significance_tests=True,
            future_eta_is_train_prediction_heuristic=True, recourse_counts_not_realized_service=True,
            full_day_superiority_not_established=True, no_additional_experiment=True,
            gamma_operating_cost_repositioning_disabled=True))
    write_json(root / DOC / "native_window_analysis.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run(Path.cwd())
