"""Read the three fixed prepared-layout replays; never run extra experiments."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from .capability_chain_instances import atomic_json
from .c_operations_diagnostic import START, sha
from .operations_v5_results import _collect, _peak
from .operations_v5_ablation_results import metric_differences, paired_counts
from .scheme_a_two_scale_experiment import (LAYOUT_RULES, _group_binding, completed_group,
    document_root, load_protocol, output_root)
from stage4.dispatch.controlled_fleet import build_controlled_fleet

CONFIG = Path("stage4/config/scheme_a_layout_comparison_300_600_v1.json")


def analyze(root, config=CONFIG):
    started = perf_counter()
    root = Path(root).resolve()
    cfg = load_protocol(root, config)
    directory = root / output_root(cfg)
    state = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    binding = json.loads((directory / "input_binding.json").read_text(encoding="utf-8"))
    groups = tuple(LAYOUT_RULES)
    if (state["status"] != "TWO_SCALE_CITY_EXPERIMENT_COMPLETE"
            or state["group_order"] != list(groups) or state["completed_groups"] != list(groups)):
        raise ValueError("the fixed three prepared-layout groups are not all complete")
    if any(sha(root / path) != digest for path, digest in binding["inputs_sha256"].items()):
        raise ValueError("a shared replay input changed")
    fleet = build_controlled_fleet(root, benchmark_start=START,
        simulation_end=START + pd.Timedelta(seconds=cfg["physical_drain_limit_s"]),
        requested_q_a=cfg["requested_q_a"], seed=cfg["supply_seed"])
    label = fleet.accounting["type_label_sha256"]
    plans = {int(row.native_id): (str(row.vehicle_type), float((row.availability_start_time - START).total_seconds()),
        float((row.availability_end_time - START).total_seconds())) for row in fleet.scenario_fleet.itertuples(index=False)}
    del fleet
    public, private, child_receipts = {}, {}, {}
    for group in groups:
        summary = completed_group(directory / group, _group_binding(binding, cfg, group))
        receipt = json.loads((directory / "execution_receipts" / f"{group}.json").read_text(encoding="utf-8"))
        if receipt["exit_code"] != 0 or receipt["parent_hard_timeout"]:
            raise ValueError("a native group did not exit cleanly")
        if summary["all_day_supply"]["accounting"]["type_label_sha256"] != label:
            raise ValueError("effective supply labels changed")
        public[group], private[group] = _collect(root, group, summary, plans, directory=directory / group)
        child_receipts[group] = dict(child_wall_time_s=receipt["child_wall_time_s"], exit_code=receipt["exit_code"],
            started_utc=receipt["started_utc"], ended_utc=receipt["ended_utc"])
        gc.collect()
    comparisons = {}
    for before, after in ((groups[0], groups[1]), (groups[1], groups[2]), (groups[0], groups[2])):
        comparisons[f"{before}_to_{after}"] = dict(before=before, after=after,
            difference_after_minus_before=metric_differences(public[before], public[after]),
            paired=paired_counts(private[before], private[after]))
    result = dict(status="PREPARED_LAYOUT_COMPARISON_ANALYZED", verification_status="ANALYZED_NOT_REPRODUCTION_RERUN",
        test_date="20161031", profile_id="C", requested_AV_baseline_hour_ratio=.1,
        passenger_acceptance_rate=.7, request_to_assignment_limit_s=300, independent_pickup_eta_cap_s=600,
        common_requests=28367, groups=public, comparisons=comparisons, child_receipts=child_receipts,
        execution_binding=binding, type_label_sha256=label, native_run_count=3,
        old_layout_or_results_overwritten=False, layout_preparation_repeated=False,
        online_control="COMMON_HISTORICAL_DEFICIT_AND_PRIMARY_FACE_MM_SERVICE_PRESERVING",
        analysis_invoked_simulation=False, analysis_invoked_routing=False, analysis_invoked_inference=False,
        one_day_one_seed=True, statistical_significance_claimed=False, full_city_optimality_claimed=False,
        raw_identifiers_or_coordinates_exported=False, runtime_s=perf_counter() - started, peak_rss_mib=_peak())
    atomic_json(root / document_root(cfg) / "comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = analyze(args.root)
    print(json.dumps(dict(status=result["status"], groups={group: {key: row[key] for key in
        ("completed", "served_C", "total_empty_km", "runtime_s", "peak_rss_mib")} for group, row in result["groups"].items()},
        runtime_s=result["runtime_s"], peak_rss_mib=result["peak_rss_mib"])), flush=True)


if __name__ == "__main__":
    main()
