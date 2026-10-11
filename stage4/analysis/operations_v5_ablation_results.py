"""Read the completed one-group movement ablation and retained v4/v5 products.

No native replay, inference, routing or policy selection is launched here.
Projected products are read sequentially; only aggregates leave memory.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from .capability_chain_instances import atomic_json
from .c_operations_diagnostic import START, sha
from .operations_v5_results import GROUP, OUT, _collect, _peak
from stage4.dispatch.controlled_fleet import build_controlled_fleet

DOC = Path("stage4/docs/capability_chain_planning/operations_v5_ablation")
VERSIONS = ("v4", "v5_common", "v5")
ADMIN_FIELDS = {"version", "base_config", "base_config_sha256", "disk_route_cache"}


def paired_counts(before, after):
    b, a = before.sort_index(), after.sort_index()
    fixed = ["release_s", "C_compatible", "passenger_accepts_av"]
    if not b.index.equals(a.index) or not b[fixed].equals(a[fixed]):
        raise ValueError("request identities, releases or eligibility changed")
    both = b.completed & a.completed
    return dict(shared_served=int(both.sum()),
        before_only_served=int((b.completed & ~a.completed).sum()),
        after_only_served=int((a.completed & ~b.completed).sum()),
        HV_to_C=int((both & b.vehicle_type.eq("HV") & a.vehicle_type.eq("AV")).sum()),
        C_to_HV=int((both & b.vehicle_type.eq("AV") & a.vehicle_type.eq("HV")).sum()),
        shared_served_total_wait_difference_mean_s=float((a.loc[both, "wait_s"] - b.loc[both, "wait_s"]).mean()),
        shared_served_pickup_distance_difference_km=float((a.loc[both, "pickup_distance_m"] - b.loc[both, "pickup_distance_m"]).sum() / 1000.))


def metric_differences(before, after):
    delta = {key: after[key] - before[key] for key in ("completed", "expired", "service_rate", "served_C", "served_HV",
        "pickup_empty_km", "idle_empty_km", "total_empty_km", "actual_idle_moves", "runtime_s")}
    delta["total_empty_relative_change"] = after["total_empty_km"] / before["total_empty_km"] - 1
    for key in ("wait_s", "matching_wait_s", "pickup_wait_s"):
        delta[key + "_mean"] = after[key]["mean"] - before[key]["mean"]
    for key in ("LOADED_hours", "PICKUP_hours", "MOVE_hours", "idle_hours", "loaded_share", "idle_share"):
        delta["C_" + key] = after["admission_state"]["AV"][key] - before["admission_state"]["AV"][key]
    return delta


def analyze(root):
    started = perf_counter()
    root = Path(root).resolve()
    summaries = {version: json.loads((root / OUT / f"two_scale_city_{version}" / GROUP / "summary.json").read_text(encoding="utf-8"))
        for version in VERSIONS}
    for version, summary in summaries.items():
        if (summary["status"] != "TWO_SCALE_CITY_GROUP_COMPLETE" or not summary["full_day"]
                or summary["version"] != f"scheme_a_two_scale_city_{version}"
                or summary["group"] != GROUP or summary["policy"] != "SERVICE_PRESERVING" or summary["layout_mode"] != "joint"):
            raise ValueError("requires three complete retained versions of the same group")
    cfgs = {version: s["frozen_protocol"] for version, s in summaries.items()}
    b, c = cfgs["v5_common"], cfgs["v5"]
    changed = {key for key in set(b) | set(c) if b.get(key) != c.get(key)}
    if changed != ADMIN_FIELDS | {"idle_movement_mode"}:
        raise ValueError("common vs typed changed more than the target rule and administrative identities")
    a = cfgs["v4"]
    old_changed = {key for key in set(a) | set(b) if a.get(key) != b.get(key)}
    allowed = ADMIN_FIELDS | {"typed_request_diagnostics", "current_empty_distance_mode",
        "typed_reposition_reference", "typed_reposition_manifest"}
    if old_changed - allowed:
        raise ValueError("the retained v4 differs in another scientific or resource condition")
    if (b["test_date"], b["profile_id"], b["requested_q_a"], b["passenger_acceptance_rate"],
            b["max_match_wait_s"], b["max_pickup_eta_s"]) != ("20161031", "C", .1, .7, 300, 600):
        raise ValueError("outside the fixed C/q10/P70/300/600 experiment")
    if (a.get("current_empty_distance_mode", "LOCAL_STABLE") != "LOCAL_STABLE"
            or any(cfg["current_empty_distance_mode"] != "PRIMARY_FACE_MM" for cfg in (b, c))
            or any(cfg["idle_movement_mode"] != "COMMON_HISTORICAL_DEFICIT" for cfg in (a, b))
            or c["idle_movement_mode"] != "TYPE_CONDITIONED_HISTORICAL_DEFICIT"):
        raise ValueError("the intended component settings do not match")
    directory = root / OUT / "two_scale_city_v5_common"
    state = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    receipt = json.loads((directory / "execution_receipts" / f"{GROUP}.json").read_text(encoding="utf-8"))
    if (state["status"] != "TWO_SCALE_CITY_EXPERIMENT_COMPLETE" or state["group_order"] != [GROUP]
            or state["completed_groups"] != [GROUP] or receipt["exit_code"] != 0 or receipt["parent_hard_timeout"]):
        raise ValueError("the single new native run did not complete cleanly")
    shared_paths = set.intersection(*(set(s["inputs_sha256"]) for s in summaries.values()))
    shared = {path: summaries["v4"]["inputs_sha256"][path] for path in sorted(shared_paths)}
    if any(sha(root / path) != digest or any(s["inputs_sha256"][path] != digest for s in summaries.values())
           for path, digest in shared.items()):
        raise ValueError("a shared frozen source changed")
    if len({s["layout_sha256"] for s in summaries.values()}) != 1:
        raise ValueError("initial layout differs across the component comparison")
    fleet = build_controlled_fleet(root, benchmark_start=START,
        simulation_end=START + pd.Timedelta(seconds=b["physical_drain_limit_s"]), requested_q_a=.1, seed=b["supply_seed"])
    label = fleet.accounting["type_label_sha256"]
    if any(s["all_day_supply"]["accounting"]["type_label_sha256"] != label for s in summaries.values()):
        raise ValueError("effective supply labels differ")
    plans = {int(row.native_id): (str(row.vehicle_type), float((row.availability_start_time - START).total_seconds()),
        float((row.availability_end_time - START).total_seconds())) for row in fleet.scenario_fleet.itertuples(index=False)}
    del fleet
    public, private = {}, {}
    for version in VERSIONS:
        public[version], private[version] = _collect(root, version, summaries[version], plans)
        _peak()
        gc.collect()
    pairs = {}
    for name, before, after in (("common_movement_empty_handling", "v4", "v5_common"),
                               ("typed_movement_given_primary_face", "v5_common", "v5"),
                               ("combined_v4_to_v5", "v4", "v5")):
        pairs[name] = dict(before=before, after=after, difference_after_minus_before=metric_differences(public[before], public[after]),
            paired=paired_counts(private[before], private[after]))
    result = dict(status="OPERATIONS_V5_COMPONENT_COMPARISON_ANALYZED", verification_status="ANALYZED_NOT_REPRODUCTION_RERUN",
        group=GROUP, test_date="20161031", common_requests=28367, groups=public, comparisons=pairs,
        shared_input_sha256=shared, layout_sha256=summaries["v4"]["layout_sha256"], type_label_sha256=label,
        frozen_native_code_sha={v: s["code_sha"] for v, s in summaries.items()},
        new_native_run_count=1, retained_baseline_rerun=False, analysis_invoked_simulation=False,
        analysis_invoked_routing=False, analysis_invoked_inference=False,
        one_day_one_seed=True, complete_two_factor_interaction_identified=False,
        statistical_significance_claimed=False, global_city_optimality_claimed=False,
        raw_identifiers_or_coordinates_exported=False, runtime_s=perf_counter() - started, peak_rss_mib=_peak())
    atomic_json(root / DOC / "component_comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = analyze(args.root)
    print(json.dumps(dict(status=result["status"], comparisons=result["comparisons"], runtime_s=result["runtime_s"],
        peak_rss_mib=result["peak_rss_mib"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
