"""Prepare three integer AV birth layouts; never launch a native experiment.

Only predetermined supply and earlier-history templates/reference are read.
The existing 300/600 sparse time oracle and Train-M3 chord pace are reused.
Future service columns are a planning approximation, not road certificates.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import json
from math import ceil, floor
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import pandas as pd
import psutil
from scipy.spatial import cKDTree

from .capability_chain_instances import atomic_json
from .c_operations_diagnostic import START, sha
from .scheme_a_city_experiment import load_history_templates
from .scheme_a_two_scale_experiment import load_protocol as load_runtime_protocol
from stage4.dispatch.coarse_layout import compatible_head_weights, integer_quotas, solve_placement
from stage4.dispatch.controlled_fleet import FLEET_REL, build_controlled_fleet
from stage4.dispatch.flexibility_native import TrainDemandForecast
from stage4.dispatch.repositioning_policy import REFERENCE_REL, REFERENCE_MANIFEST_REL, load_train_demand_reference
from stage4.dispatch.scheme_a_city_adapter import _rank, sites_for
from stage4.dispatch.scheme_a_city_graph import CoarseHistoricalLibrary, _xy
from stage4.dispatch.scheme_a_match_deadline_value import MatchDeadlineSceneDAG

CONFIG = Path("stage4/config/scheme_a_layout_300_600_v1.json")
OUT = Path("stage4/output/capability_chain_planning/layout_300_600_v1")
DOC = Path("stage4/docs/capability_chain_planning/layout_300_600_v1")
MODES = ("all_hot", "compatible_hot", "opportunity_joint")


def load_config(root, path=CONFIG):
    overrides = json.loads((root / path).read_text(encoding="utf-8"))
    required = dict(version="scheme_a_layout_300_600_v1",
        base_config="stage4/config/scheme_a_two_scale_city_v5_common.json",
        layout_method="SHARED_SCENARIO_SPARSE_LP_INTEGER_PLACEMENT_RECOVERY", layout_bin_wall_limit_s=30.,
        layout_preparation_timeout_s=900, layout_preparation_rss_mib=512, layout_generators=list(MODES),
        native_execution=False, new_routing=False, new_inference=False)
    if (set(overrides) != set(required) | {"base_config_sha256"}
            or any(overrides.get(key) != value for key, value in required.items())
            or sha(root / overrides["base_config"]) != overrides["base_config_sha256"]):
        raise ValueError("outside the fixed layout-only 300/600 preparation")
    return dict(load_runtime_protocol(root, Path(overrides["base_config"])), **overrides)


def _time_group(row, now):
    ready = int(ceil((max(now, row["activation_s"]) - now) / 300))
    end = min(6, int(floor((row["admission_end_s"] - now) / 300 + 1e-10)))
    return ready, end


def _allocate(rows, sites, counts):
    choices = [site for site, count in zip(sites, counts) for _ in range(int(count))]
    if len(rows) != len(choices):
        raise RuntimeError("placement count does not match the original slots")
    return {str(row["vehicle_id"]): dict(site) for row, site in zip(rows, choices)}


def prepare(root, config=CONFIG, *, resume=False):
    started = perf_counter()
    root = Path(root).resolve()
    cfg = load_config(root, config)
    source_paths = (Path(config), Path(cfg["base_config"]), FLEET_REL, REFERENCE_REL, REFERENCE_MANIFEST_REL,
        Path(cfg["source_history_templates"]), Path(cfg["source_history_manifest"]), Path(cfg["frozen_profile_config"]))
    inputs = {path.as_posix(): sha(root / path) for path in source_paths}
    code = subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"], text=True).strip()
    binding = dict(code_sha=code, inputs_sha256=inputs, waiting_contract=dict(max_match_wait_s=300, max_pickup_eta_s=600))
    directory = root / OUT
    checkpoint_path = directory / "checkpoint.json"
    if (directory / "placements.json").exists():
        raise ValueError("completed layouts retained; no overwrite or new preparation")
    if checkpoint_path.exists() and not resume:
        raise ValueError("existing partial preparation retained; use explicit same-input --resume")
    memory = psutil.virtual_memory().available / 2**20
    if memory < cfg["minimum_initial_available_ram_mib"]:
        raise MemoryError("insufficient startup RAM; other applications are not closed")

    def check():
        proc = psutil.Process().memory_info()
        if max(proc.rss, proc.peak_wset) / 2**20 > cfg["layout_preparation_rss_mib"]:
            raise MemoryError("layout preparation exceeds the 512-MiB cap")
        if perf_counter() - started > cfg["layout_preparation_timeout_s"]:
            raise TimeoutError("layout-only administrative time limit exceeded")
        if psutil.disk_usage(root.anchor).free / 2**20 < cfg["minimum_disk_reserve_mib"]:
            raise OSError("layout preparation disk reserve is insufficient")

    reference, _ = load_train_demand_reference(root)
    templates, inventory = load_history_templates(root, cfg)
    library = CoarseHistoricalLibrary(templates, cfg)
    pace = TrainDemandForecast(templates, dict(cfg, forecast_horizon_s=1800, fast_forecast=True), 86434)
    del templates
    fleet = build_controlled_fleet(root, benchmark_start=START,
        simulation_end=START + pd.Timedelta(seconds=cfg["physical_drain_limit_s"]), requested_q_a=.1, seed=cfg["supply_seed"])
    labels = fleet.accounting["type_label_sha256"]
    slots = [dict(vehicle_id=str(row.vehicle_id), native_id=int(row.native_id),
        profile_id="C" if row.vehicle_type == "AV" else "HV", vehicle_type=row.vehicle_type,
        activation_s=float((row.availability_start_time - START).total_seconds()),
        admission_end_s=float((row.availability_end_time - START).total_seconds()),
        position=(float(row.initial_lon_wgs84), float(row.initial_lat_wgs84)))
        for row in fleet.scenario_fleet.itertuples(index=False)]
    del fleet
    av = [row for row in slots if row["vehicle_type"] == "AV"]
    bins = sorted({int(row["activation_s"] // 1800) * 1800 for row in av})
    maps, records = {mode: {} for mode in MODES}, []
    if checkpoint_path.exists():
        prior = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if prior["binding"] != binding or prior["type_label_sha256"] != labels:
            raise ValueError("partial layouts have different code/inputs/supply")
        maps, records = prior["placements"], prior["records"]
        if [record["bin_s"] for record in records] != bins[:len(records)]:
            raise ValueError("checkpoint is not a complete stable bin prefix")
    status = dict(status="PREPARING_LAYOUTS_ONLY", binding=binding, type_label_sha256=labels,
        bins_total=len(bins), bins_complete=len(records), native_execution=False)
    atomic_json(directory / "status.json", status)
    for now in bins[len(records):]:
        check()
        sites = sites_for(reference, now, cfg)
        ids = [site["site_id"] for site in sites]
        xy = np.asarray([_xy(site["position"]) for site in sites])
        tree = cKDTree(xy)
        fresh = [row for row in av if int(row["activation_s"] // 1800) * 1800 == now]
        fresh.sort(key=lambda row: _rank(row["vehicle_id"]))
        models = {name: MatchDeadlineSceneDAG(tasks, sites, now, now + 1800, pace, cfg, check)
            for name, tasks in library.view(now).items()}
        head_counts, groups, short = {}, defaultdict(list), []
        for row in fresh:
            group = _time_group(row, now)
            if not 0 <= group[0] < group[1] <= 6:
                short.append(row)
            else:
                groups[group].append(row)
        all_weights = np.asarray([site["demand_share"] for site in sites])
        for group, rows in groups.items():
            score = compatible_head_weights(models, library.weights, group, len(sites))
            head_counts[group] = integer_quotas(score if score.sum() > 0 else all_weights, len(rows), ids)
            maps["all_hot"].update(_allocate(rows, sites, integer_quotas(all_weights, len(rows), ids)))
            maps["compatible_hot"].update(_allocate(rows, sites, head_counts[group]))
        generic_short = _allocate(short, sites, integer_quotas(all_weights, len(short), ids))
        for mode in MODES:
            maps[mode].update(generic_short)
        fresh_ids = {row["native_id"] for row in fresh}
        fixed_supply = Counter()
        far = 0
        for row in slots:
            if row["native_id"] in fresh_ids or not row["admission_end_s"] > now or not row["activation_s"] < now + 1800:
                continue
            ready, end = _time_group(row, now)
            if ready >= end or ready >= 6:
                continue
            point = maps["opportunity_joint"].get(row["vehicle_id"], {}).get("position", row["position"])
            distance, zone = tree.query(np.asarray(_xy(point)))
            far += int(distance > 2000)
            fixed_supply[row["profile_id"], int(zone), ready, end] += 1
        counts, info = solve_placement(models, library.weights, fixed_supply,
            {group: len(rows) for group, rows in groups.items()}, ids, cfg, check, tie_targets=head_counts)
        for group, rows in groups.items():
            maps["opportunity_joint"].update(_allocate(rows, sites, counts[group]))
        records.append(dict(bin_s=now, newly_placed_AV_slots=len(fresh), candidate_sites=len(sites),
            short_window_slots=len(short), zero_compatible_head_groups=int(sum(
                compatible_head_weights(models, library.weights, group, len(sites)).sum() == 0 for group in groups)),
            fixed_supply_states=len(fixed_supply), preparation_position_projections_over_2km=far,
            forecast_nodes={name: len(model.tasks) for name, model in models.items()}, model=info))
        atomic_json(checkpoint_path, dict(binding=binding, type_label_sha256=labels, placements=maps, records=records))
        print(json.dumps(dict(bins_complete=len(records), bins_total=len(bins), placed_AV_slots=len(maps["opportunity_joint"]),
            runtime_s=perf_counter() - started, rss_mib=psutil.Process().memory_info().rss / 2**20)), flush=True)
        del models
        gc.collect()
    expected = {row["vehicle_id"] for row in av}
    if any(set(value) != expected for value in maps.values()) or len(av) != 851:
        raise RuntimeError("new layouts do not cover every original AV slot exactly once")
    if any(sha(root / path) != digest for path, digest in inputs.items()):
        raise RuntimeError("a frozen preparation source changed")
    check()
    payload = dict(status="LAYOUTS_PREPARED_300_600", binding=binding, type_label_sha256=labels,
        placements=maps, records=records, native_execution=False, target_day_requests_loaded=False,
        current_runtime_control="COMMON_HISTORICAL_DEFICIT_AND_PRIMARY_FACE_MM",
        preparation_supply_positions_not_observed_midday_locations=True, no_future_road_certificates=True,
        time_grid_s=300, max_predicted_services_per_path=3, exact_physical_slot_times_preserved=True,
        global_or_integer_service_optimality_claimed=False, runtime_s=perf_counter() - started,
        peak_rss_mib=psutil.Process().memory_info().peak_wset / 2**20, historical_inventory=inventory)
    atomic_json(directory / "placements.json", payload)
    summary = dict(payload)
    summary.pop("placements")
    summary.update(AV_slots_per_layout={mode: len(values) for mode, values in maps.items()},
        placements_file_sha256=sha(directory / "placements.json"),
        pairwise_equal_slot_counts={f"{a}_vs_{b}": sum(maps[a][key]["site_id"] == maps[b][key]["site_id"] for key in expected)
            for a, b in (("all_hot", "compatible_hot"), ("compatible_hot", "opportunity_joint"), ("all_hot", "opportunity_joint"))})
    atomic_json(root / DOC / "preparation_summary.json", summary)
    atomic_json(directory / "status.json", dict(status, status=payload["status"], bins_complete=len(records),
        runtime_s=payload["runtime_s"], peak_rss_mib=payload["peak_rss_mib"]))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = prepare(args.root, args.config, resume=args.resume)
    print(json.dumps({key: summary[key] for key in ("status", "runtime_s", "peak_rss_mib", "AV_slots_per_layout", "pairwise_equal_slot_counts")}), flush=True)


if __name__ == "__main__":
    main()
