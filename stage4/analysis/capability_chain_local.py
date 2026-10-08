"""Dense common candidates and complete LOCAL windows, not a city replay.

Region/candidates depend only on earlier all-demand history. Both controls use
the same full local requests, scenarios, initial layout and physical rules.
No C-stratification, favorable-case search, native run or new model fitting.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa

from stage4.analysis.capability_chain_instances import TEMPLATES, atomic_json, atomic_parquet, grid_keys, sha
from stage4.analysis.capability_chain_rolling import (
    ConnectionProvider, HistoricalPredictionTimeProxy, InstanceRouter, MixedDateEvidence,
    compact_policy, replay_policy, shared_initial_layout, task_record, visible_tasks, _gap_m,
)

CONFIG = Path("stage4/config/capability_chain_local_v2.json")
OUT = Path("stage4/output/capability_chain_planning/local_v2")
DOC = Path("stage4/docs/capability_chain_planning/local_v2")


def dense_sites(history, cfg):
    """All-demand region and occupied subcells; no capability-based site ranking."""
    work = history.copy()
    work["cell"] = grid_keys(work, "start_lon_wgs84", "start_lat_wgs84", cfg)
    counts = work.groupby("cell").size().reset_index(name="count").sort_values(
        ["count", "cell"], ascending=[False, True], kind="stable")
    region = str(counts.iloc[0].cell)
    local = work.loc[work.cell.eq(region)].copy()
    subcfg = dict(cfg, grid_size_degrees=cfg["site_subgrid_degrees"])
    local["subcell"] = grid_keys(local, "start_lon_wgs84", "start_lat_wgs84", subcfg)
    sites = []
    for i, (key, group) in enumerate(local.groupby("subcell", sort=True)):
        group = group.copy()
        x = group.start_lon_wgs84.to_numpy(float) * math.cos(math.radians(34.25))
        y = group.start_lat_wgs84.to_numpy(float)
        group["center_gap"] = (x-x.mean())**2 + (y-y.mean())**2
        anchor = group.sort_values(["center_gap", "date", "order_id"], kind="stable").iloc[0]
        sites.append(dict(site_id=f"S{i+1:02d}", cell=region, subcell=str(key), historical_pickups=len(group),
            lon_wgs84=float(anchor.start_lon_wgs84), lat_wgs84=float(anchor.start_lat_wgs84)))
    return region, sites, local


def complete_window(templates, region, date, cut, cfg):
    work = templates.loc[templates.date.eq(date) & templates.common_eligible
        & templates.release_second.ge(cut) & templates.release_second.lt(cut+cfg["horizon_s"])].copy()
    work["cell"] = grid_keys(work, "start_lon_wgs84", "start_lat_wgs84", cfg)
    work = work.loc[work.cell.eq(region)].sort_values(["release_second", "order_id"], kind="stable").reset_index(drop=True)
    work["task_id"] = [("A" if date == cfg["replay_date"] else f"F{date[-2:]}_")+f"{i+1:03d}" for i in range(len(work))]
    return work


def prepared_cases(templates, region, cfg):
    cases = []
    for cut in cfg["window_starts_s"]:
        parts, cohorts, counts = [], [], {}
        for date in [cfg["replay_date"], *cfg["history_dates"]]:
            chosen = complete_window(templates, region, date, cut, cfg)
            records = [task_record(row, cut, cfg, date == cfg["replay_date"]) for row in chosen.itertuples(index=False)]
            counts[date] = dict(all_local_template_orders=len(chosen), C_customer_orders=int(chosen.compatible_C.sum()),
                no_order_subsampling=True, no_C_stratification=True)
            if date == cfg["replay_date"]:
                actual = records
            else:
                cohorts.append(dict(scenario_id=date, weight=1/len(cfg["history_dates"]), tasks=records))
            parts.append(chosen)
        # Static computational sizing, never sent to the policy as future data.
        worst = max((len(visible_tasks(actual, now, {})) + len([t for t in scene["tasks"] if t["release_s"] > now])
            for now in range(0, cfg["horizon_s"], cfg["step_s"]) for scene in cohorts), default=0)
        if worst > cfg["task_limit_per_scenario"]:
            raise ValueError("complete local window exceeds the declared task bound; no truncation permitted")
        cases.append(dict(case_id=f"LOCAL_{cfg['replay_date']}_{cut//3600:02d}{cut%3600//60:02d}",
            cut=cut, universe=pd.concat(parts, ignore_index=True), actual_tasks=actual, cohorts=cohorts,
            counts=counts, worst_case_scenario_tasks_without_prior_service=worst))
    return cases


def preparation(root, cfg):
    templates = pd.read_parquet(root / TEMPLATES)
    templates["date"] = templates.date.astype(str)
    templates["order_id"] = templates.order_id.astype(str)
    history = templates.loc[templates.date.isin(cfg["history_dates"])].copy()
    region, sites, local = dense_sites(history, cfg)
    cases = prepared_cases(templates, region, cfg)
    pairs = [dict(source=a["site_id"], target=b["site_id"],
        distance_m=_gap_m((a["lon_wgs84"], a["lat_wgs84"]), (b["lon_wgs84"], b["lat_wgs84"])))
        for i,a in enumerate(sites) for b in sites[i+1:]]
    neighbor = {s["site_id"]:[] for s in sites}
    for pair in pairs:
        if pair["distance_m"] <= cfg["reposition_radius_m"]:
            neighbor[pair["source"]].append(pair["target"])
            neighbor[pair["target"]].append(pair["source"])
    meta = dict(status="LOCAL_INSTANCE_PREPARED_BEFORE_POLICY_EXECUTION", region_cell=region,
        selection="EARLIER_ALL_DEMAND_TOP_CELL_NOT_C_OR_POLICY_WINNER", history_dates=cfg["history_dates"],
        historical_local_all_day_orders=len(local), site_count=len(sites), observed_history_anchor_points=True,
        minimum_site_separation_m=min((p["distance_m"] for p in pairs), default=None),
        undirected_pairs_within_2km=sum(p["distance_m"] <= cfg["reposition_radius_m"] for p in pairs),
        minimum_geometric_neighbor_count=min(map(len, neighbor.values()), default=0),
        case_counts=[{k:case[k] for k in ("case_id", "counts", "worst_case_scenario_tasks_without_prior_service")} for case in cases],
        templates_sha256=sha(root / TEMPLATES), config=cfg, actual_future_not_sent_to_control=True)
    atomic_json(root / DOC / "preparation.json", meta)
    atomic_parquet(root / OUT / "private_candidate_sites.parquet", pd.DataFrame(sites))
    for case in cases:
        atomic_parquet(root / OUT / case["case_id"] / "private_source_mapping.parquet", case["universe"])
    return sites, local, cases, meta


def mobility_graph(provider, sites, cfg, budget_check):
    rows = []
    for source in sites:
        sid = source["site_id"]
        candidates = [(target["site_id"], _gap_m(provider.locations[sid], provider.locations[target["site_id"]]))
            for target in sites if target["site_id"] != sid]
        candidates = sorted((p for p in candidates if 1 < p[1] <= cfg["reposition_radius_m"]), key=lambda p:(p[1],p[0]))
        for target, chord in candidates[:cfg["reposition_top_k"]]:
            budget_check()
            link = provider.relocation(sid, target)
            for profile in ("HV", "C"):
                rows.append(dict(source=sid, target=target, profile_id=profile, chord_m=chord,
                    supported=link["supported"], proxy_eta_s=link["travel_time_s"],
                    feasible_under_existing_move_rule=bool(link["supported"] and profile in link["compatible_profiles"]
                        and link["travel_time_s"] <= cfg["reposition_max_eta_s"]),
                    reason_codes=link["reason_codes"], C_reason_codes=link.get("C_reason_codes", [])))
    return pd.DataFrame(rows)


def summarize(root, result):
    for case in result["cases"]:
        destination = root / case["private_output"]
        before = json.loads((destination / "SERVICE_PRESERVING_full_result.json").read_text(encoding="utf-8"))
        after = json.loads((destination / "CHAIN_DEFER_full_result.json").read_text(encoding="utf-8"))
        case["policies"] = [compact_policy(before), compact_policy(after)]
        b, a = before["committed_jobs"], after["committed_jobs"]
        shared = sorted(set(b)&set(a))
        case["comparison"] = dict(realized_service_gain=len(a)-len(b), gained_jobs=sorted(set(a)-set(b)),
            lost_jobs=sorted(set(b)-set(a)), common_served_count=len(shared),
            paired_mean_pickup_wait_change_s=float(np.mean([a[j]["pickup_s"]-b[j]["pickup_s"] for j in shared])) if shared else None,
            executed_sequences_identical=[{k:v for k,v in e.items() if k != "policy"} for e in before["executed_events"]]
                == [{k:v for k,v in e.items() if k != "policy"} for e in after["executed_events"]])
        for policy, full in [("SERVICE_PRESERVING",before),("CHAIN_DEFER",after)]:
            diagnostics = dict(total_expanded_prefixes=0, total_dominated_prefixes=0,
                total_emitted_columns=0, total_retained_columns=0, maximum_single_decision_s=0)
            for epoch in full["epochs"]:
                stats = epoch.get("model", {}).get("compression_statistics", {})
                for output, field in [("total_expanded_prefixes","expanded_prefixes"),
                    ("total_dominated_prefixes","dominated_prefixes"), ("total_emitted_columns","emitted_columns"),
                    ("total_retained_columns","retained_columns")]:
                    diagnostics[output] += int(stats.get(field,0))
                diagnostics["maximum_single_decision_s"] = max(diagnostics["maximum_single_decision_s"],epoch["timing_s"]["total"])
            case.setdefault("exact_representation_diagnostics", {})[policy] = diagnostics
    atomic_json(root / OUT / "run_summary.json", result)
    atomic_json(root / DOC / "summary.json", result)
    return result


def run(root, cfg_path=CONFIG, prepare_only=False):
    started = perf_counter()
    cfg = json.loads((root / cfg_path).read_text(encoding="utf-8"))
    if (any(d >= cfg["replay_date"] for d in cfg["history_dates"]) or cfg["replay_date"] > "20161024"
        or cfg["full_day_or_native"] or cfg["parameter_search"] or cfg["model_retraining"]):
        raise ValueError("only the prespecified earlier-history local prototype is authorized")
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    sites, local, cases, meta = preparation(root,cfg)
    print(json.dumps(dict(status=meta["status"], sites=meta["site_count"],
        geometric_pairs_within_2km=meta["undirected_pairs_within_2km"], case_counts=meta["case_counts"])), flush=True)
    if prepare_only:
        return meta
    proxy_path = root / cfg["time_proxy_source"]
    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    previous = json.loads((root / "stage4/docs/capability_chain_planning/rolling_v1/summary.json").read_text(encoding="utf-8"))
    if (proxy["history_dates"] != cfg["history_dates"] or proxy["replay_date_used"] or proxy["Test31_used"]
        or previous["templates_sha256"] != meta["templates_sha256"]):
        raise ValueError("earlier-history time proxy does not match the fixed source data")
    adapter = HistoricalPredictionTimeProxy(root)
    adapter.factors = {int(row["window_start_s"]):float(row["factor"]) for row in proxy["factors"]}
    router = InstanceRouter(root,adapter)
    process = psutil.Process()

    def check_budget():
        if perf_counter()-started > cfg["maximum_runtime_s"]:
            raise RuntimeError("local prototype exceeded declared total runtime limit")
        if process.memory_info().rss/2**20 > cfg["maximum_rss_mib"]:
            raise RuntimeError("local prototype exceeded declared RSS bound")
        if router.route_request_count > cfg["maximum_route_queries"]:
            raise RuntimeError("local prototype exceeded route query bound")

    result = dict(status="RUNNING", kind="FULL_LOCAL_WINDOW_INDEPENDENT_PROTOTYPE_NOT_CITY_OR_NATIVE",
        config_sha256=sha(root/cfg_path), preparation=meta,
        reused_time_proxy_path=cfg["time_proxy_source"], reused_time_proxy_sha256=sha(proxy_path),
        time_proxy_refit=False, actual_empty_time_calibration=False, frozen_native_baseline_modified=False, cases=[])
    atomic_json(root/OUT/"run_summary.json",result)
    for case in cases:
        cut, destination = case["cut"], root/OUT/case["case_id"]
        evidence = MixedDateEvidence(root,router,case["universe"],cfg)
        provider = ConnectionProvider(router,evidence,case["universe"],sites,cfg,cut)
        graph = mobility_graph(provider,sites,cfg,check_budget)
        atomic_parquet(destination/"prepared_directed_mobility_graph.parquet",graph)
        graph_counts = {profile:int(graph.loc[graph.profile_id.eq(profile),"feasible_under_existing_move_rule"].sum()) for profile in ("HV","C")}
        print(json.dumps(dict(case=case["case_id"], actual_directed_mobility=graph_counts,
            rss_mib=process.memory_info().rss/2**20)),flush=True)
        ranking = sorted([s["site_id"] for s in sites], key=lambda sid:(-len(local.loc[
            local.subcell.eq(next(s["subcell"] for s in sites if s["site_id"] == sid))
            & local.release_second.ge(cut) & local.release_second.lt(cut+cfg["horizon_s"])]),sid))
        initial, layout = shared_initial_layout(provider,case["cohorts"],sites,ranking,cfg)
        atomic_json(destination/"input.json",dict(actual_tasks=case["actual_tasks"], forecast_cohorts=case["cohorts"],
            sites=sites, shared_initial_resources=initial, config=cfg))
        atomic_json(destination/"historical_layout_solution.json",layout)
        for policy in cfg["policies"]:
            replay = replay_policy(case["actual_tasks"],case["cohorts"],initial,provider,policy,cfg,check_budget)
            atomic_json(destination/f"{policy}_full_result.json",replay)
            atomic_parquet(destination/f"{policy}_events.parquet",pd.DataFrame(replay["executed_events"]))
            del replay
        atomic_parquet(destination/"sparse_connections.parquet",pd.DataFrame(provider.cache.values()))
        atomic_parquet(destination/"idle_move_connections.parquet",pd.DataFrame(provider.moves.values()))
        result["cases"].append(dict(case_id=case["case_id"], private_output=str(destination.relative_to(root)),
            counts=case["counts"], prepared_feasible_directed_mobility_arcs=graph_counts,
            shared_initial_layout={r["resource_id"]:r["location_id"] for r in initial},
            layout_model={k:layout["model"][k] for k in ("columns","nonzeros","compression_statistics")},
            layout_runtime_s=layout["runtime_s"], cached_sparse_connections=len(provider.cache),
            cached_independent_idle_connections=len(provider.moves), shared_cache_hits=provider.hits,
            common_input_failure_reasons=dict(Counter(reason for link in provider.cache.values()
                if not link["supported"] for reason in link["reason_codes"]))))
        atomic_json(root/OUT/"run_summary.json",result)
        del provider,evidence,graph,layout
        gc.collect()
        check_budget()
    result.update(status="FULL_LOCAL_WINDOW_COMPARISON_COMPLETE", runtime_s=perf_counter()-started,
        peak_rss_mib=process.memory_info().peak_wset/2**20, routing=router.diagnostics(),
        new_calibration_route_queries=0, gpu_used=False, native_runs=0, full_day_runs=0,
        runtime_comparison_caveat="SHARED_LAZY_CACHE_NOT_A_PAIRED_ROUTING_SPEED_COMPARISON")
    summarize(root,result)
    print(json.dumps(dict(status=result["status"], runtime_s=result["runtime_s"], peak_rss_mib=result["peak_rss_mib"])),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--config",type=Path,default=CONFIG)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--summarize-existing",action="store_true")
    args=parser.parse_args()
    root=args.root.resolve()
    if args.summarize_existing:
        summarize(root,json.loads((root/OUT/"run_summary.json").read_text(encoding="utf-8")))
        print(json.dumps(dict(status="EXISTING_LOCAL_RESULTS_AGGREGATED_NO_RERUN")),flush=True)
    else:
        try:
            run(root,args.config,args.prepare_only)
        except Exception as exc:
            # A stopped process must not remain advertised as RUNNING. Keep the
            # partial local artifacts; never retry or turn a failure into PASS.
            path = root/OUT/"run_summary.json"
            if path.is_file() and not args.prepare_only:
                failed=json.loads(path.read_text(encoding="utf-8"))
                if failed.get("status") == "RUNNING":
                    failed.update(status="FAILED",failure_type=type(exc).__name__,failure_detail=str(exc))
                    atomic_json(path,failed)
            raise


if __name__ == "__main__":
    main()
