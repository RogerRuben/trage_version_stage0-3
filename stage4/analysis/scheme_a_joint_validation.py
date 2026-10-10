"""Four finite same-domain layout/chain solves, not a new replay experiment.

Reuses earlier-history tasks and certified directed connectors already saved
by local_v2. No Valhalla, actual future requests, M3 inference, native simulator,
or all-day data is queried. Missing cached links stay outside the stated domain.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import subprocess
from time import perf_counter

import pandas as pd
import psutil
import pyarrow as pa

from stage4.analysis.capability_chain_instances import atomic_json, grid_keys, sha
from stage4.dispatch.scheme_a_joint_planner import SchemeAJointPlanner


CONFIG = Path("stage4/config/scheme_a_joint_v1.json")
OUT = Path("stage4/output/capability_chain_planning/scheme_a_joint_v1")
DOC = Path("stage4/docs/capability_chain_planning/scheme_a_joint_v1")


def _records(path):
    frame = pd.read_parquet(path)
    records = frame.to_dict("records")
    for row in records:
        for key in ("compatible_profiles", "reason_codes", "C_reason_codes"):
            value = row.get(key)
            if value is not None and hasattr(value, "tolist"):
                row[key] = value.tolist()
    return records


class CachedConnectorSnapshot:
    """Explicit supplied finite domain; never approximate missing physical links."""
    def __init__(self, source, sites, universe):
        self.cache = {(r["origin_id"], r["target_job_id"]): r
            for r in _records(source / "sparse_connections.parquet")}
        self.moves = {(r["origin_id"], r["site_id"]): r
            for r in _records(source / "idle_move_connections.parquet")}
        self.sites = {s["site_id"]: s for s in sites}
        self.locations = {r.task_id: (float(r.end_lon_wgs84), float(r.end_lat_wgs84))
            for r in universe.itertuples(index=False)}
        self.locations.update({s["site_id"]: (s["lon_wgs84"], s["lat_wgs84"]) for s in sites})
        for move in self.moves.values():
            self.locations[move["arrival_location_id"]] = self.locations[move["site_id"]]
        self.missing_queries = Counter()

    def connection(self, origin, target, allowed_targets):
        if target not in allowed_targets:
            raise ValueError("connector request outside revealed/historical task set")
        if (origin, target) in self.cache:
            return self.cache[origin, target]
        self.missing_queries["customer"] += 1
        return dict(origin_id=origin, target_job_id=target, supported=False,
            compatible_profiles=[], travel_time_s=None, empty_distance_m=None,
            reason_codes=["OUTSIDE_FROZEN_FINITE_CONNECTION_SNAPSHOT"])

    def relocation(self, origin, site):
        if site not in self.sites:
            raise ValueError("relocation site outside identical common candidate pool")
        if (origin, site) in self.moves:
            return self.moves[origin, site]
        self.missing_queries["relocation"] += 1
        return dict(origin_id=origin, site_id=site, arrival_location_id=site,
            supported=False, compatible_profiles=[], travel_time_s=None,
            empty_distance_m=None, reason_codes=["OUTSIDE_FROZEN_FINITE_CONNECTION_SNAPSHOT"])


def historical_hotspot_ranking(universe, sites, fixed_cfg):
    history = universe.loc[universe.date.astype(str).isin(fixed_cfg["history_dates"])].copy()
    if history.empty:
        raise ValueError("no earlier-history customers for all-demand hotspot baseline")
    subcfg = dict(fixed_cfg, grid_size_degrees=fixed_cfg["site_subgrid_degrees"])
    history["subcell"] = grid_keys(history, "start_lon_wgs84", "start_lat_wgs84", subcfg)
    counts = history.groupby("subcell").size().to_dict()
    return sorted((s["site_id"] for s in sites),
        key=lambda sid: (-int(counts.get(next(s["subcell"] for s in sites if s["site_id"] == sid), 0)), sid))


def compact_solution(solution):
    paths = solution["recourse_paths"]
    return dict(layout_mode=solution["layout_mode"], policy=solution["policy"],
        opt_status=solution["opt_status"], expected_planned_services=solution["expected_served"],
        expected_planned_empty_distance_m=solution["expected_empty_distance_m"],
        selected_layout={a["resource_id"]: a["location_id"] for a in solution["selected_actions"]},
        maximum_customer_chain_length=max((p["total_served"] for p in paths), default=0),
        maximum_C_customer_chain_length=max((p["total_served"] for p in paths if p["resource_id"] == "C1"), default=0),
        C_scenario_service_counts={p["scenario_id"]: p["total_served"] for p in paths if p["resource_id"] == "C1"},
        model=solution["model"], quality_certificate=solution["quality_certificate"],
        timing_s=solution["timing_s"], validation=solution["validation"],
        scheme_a_scope=solution["scheme_a_scope"])


def run(root, config_path=CONFIG):
    root = Path(root).resolve()
    cfg = json.loads((root / config_path).read_text(encoding="utf-8"))
    if (cfg["planning_horizon_s"] != 1800 or cfg["step_s"] != 30
        or cfg["cases"] != ["LOCAL_20161024_0730", "LOCAL_20161024_1700"]
        or cfg["layout_modes"] != ["HOTSPOT_FIXED", "CHAIN_JOINT"]
        or any(cfg[k] for k in ("full_day", "native_execution", "city_scale", "formal_q10",
            "parameter_search", "model_retraining", "new_routing", "future_target_requests_used_for_layout"))):
        raise ValueError("outside the fixed finite Scheme-A quality batch")
    if (root / OUT / "summary.json").exists():
        raise ValueError("existing quality run found; no implicit overwrite or retry")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    process = psutil.Process()
    if psutil.virtual_memory().available / 2**20 < cfg["maximum_rss_mib"]:
        raise MemoryError("insufficient available RAM for fixed finite batch")
    started = perf_counter()
    result = dict(status="RUNNING", kind=cfg["first_batch_scope"],
        execution_code_sha=subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}",
            "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        config_sha256=sha(root / config_path), config=cfg, cases=[],
        new_routing_queries=0, new_M3_inference=False, raw_target_future_used=False,
        native_runs=0, full_day_runs=0, dense_matrix=False, gpu_used=False)
    atomic_json(root / OUT / "summary.json", result)

    def guard():
        if perf_counter() - started > cfg["maximum_runtime_s"]:
            raise TimeoutError("finite batch exceeded fixed 120s total budget")
        if process.memory_info().rss / 2**20 > cfg["maximum_rss_mib"]:
            raise MemoryError("finite batch exceeded fixed RSS cap")

    try:
        for case_id in cfg["cases"]:
            guard()
            source = root / cfg["source_local_output"] / case_id
            inputs = json.loads((source / "input.json").read_text(encoding="utf-8"))
            fixed = inputs["config"]
            if fixed["history_dates"] != cfg["history_dates"] or fixed["replay_date"] != cfg["replay_date"]:
                raise ValueError("source dates changed from earlier-history protocol")
            parameters = dict(fixed, planning_horizon_s=cfg["planning_horizon_s"])
            sources = {name: sha(source / name) for name in ("input.json", "private_source_mapping.parquet",
                "sparse_connections.parquet", "idle_move_connections.parquet")}
            universe = pd.read_parquet(source / "private_source_mapping.parquet")
            ranking = historical_hotspot_ranking(universe, inputs["sites"], fixed)
            provider = CachedConnectorSnapshot(source, inputs["sites"], universe)
            planner = SchemeAJointPlanner(provider, inputs["forecast_cohorts"], parameters,
                policy=cfg["finite_quality_policy"])
            case = dict(case_id=case_id, source=str(source.relative_to(root)), sources_sha256=sources,
                candidate_site_count=len(inputs["sites"]), historical_scenario_task_counts={
                    s["scenario_id"]: len(s["tasks"]) for s in inputs["forecast_cohorts"]},
                resources=3, resource_profiles=["HV", "HV", "C"], models=[])
            for mode in cfg["layout_modes"]:
                guard()
                _, solution, problem = planner.plan_initial_layout(inputs["shared_initial_resources"],
                    inputs["sites"], ranking, mode=mode)
                if any(t["source_kind"] != "FORECAST" for s in problem["scenarios"] for t in s["tasks"]):
                    raise RuntimeError("actual target-day customer leaked into layout problem")
                for action in solution["selected_actions"]:
                    template = next(r for r in inputs["shared_initial_resources"] if r["resource_id"] == action["resource_id"])
                    if template["profile_id"] == "HV" and action["location_id"] != template["location_id"]:
                        raise RuntimeError("HV initial position was rearranged by AV layout mode")
                destination = root / OUT / case_id
                atomic_json(destination / f"{mode}_problem.json", problem)
                atomic_json(destination / f"{mode}_solution.json", solution)
                case["models"].append(compact_solution(solution))
                print(json.dumps(dict(case=case_id, mode=mode,
                    expected_services=solution["expected_served"],
                    variables=solution["model"]["columns"], runtime_s=solution["runtime_s"])), flush=True)
                guard()
            if sources != {name: sha(source / name) for name in sources}:
                raise RuntimeError("source snapshot changed during finite quality solve")
            before, after = case["models"]
            case["joint_minus_fixed_expected_services"] = after["expected_planned_services"] - before["expected_planned_services"]
            case["joint_minus_fixed_expected_empty_m"] = after["expected_planned_empty_distance_m"] - before["expected_planned_empty_distance_m"]
            case["missing_snapshot_queries"] = dict(provider.missing_queries)
            result["cases"].append(case)
            atomic_json(root / OUT / "summary.json", result)
            del universe, provider, planner, solution, problem
            gc.collect()
        result.update(status="SCHEME_A_FINITE_JOINT_QUALITY_COMPLETE", runtime_s=perf_counter()-started,
            peak_rss_mib=process.memory_info().peak_wset / 2**20,
            scientific_scope="HISTORICAL_SCENARIO_MODEL_QUALITY_NOT_REALIZED_SERVICE_GAIN",
            city_joint_validation_superseded=False)
        atomic_json(root / OUT / "summary.json", result)
        atomic_json(root / DOC / "summary.json", result)
        print(json.dumps(dict(status=result["status"], runtime_s=result["runtime_s"],
            peak_rss_mib=result["peak_rss_mib"])), flush=True)
        return result
    except Exception as error:
        result.update(status="STOPPED", error=repr(error), runtime_s=perf_counter()-started)
        atomic_json(root / OUT / "summary.json", result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    run(args.root, args.config)


if __name__ == "__main__":
    main()
