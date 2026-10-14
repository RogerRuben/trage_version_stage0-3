"""One fixed twelve-instance comparison; no native simulation/routing/inference."""
from __future__ import annotations

from time import perf_counter
PROCESS_STARTED = perf_counter()

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from fractions import Fraction
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import threading
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from stage4.dispatch.coarse_layout import compatible_head_weights, integer_quotas, solve_placement
from stage4.dispatch.competitive_chain import (
    FiniteCatalogue, certified_upper, independent_placement, reset_prices, solve_bound_recovery,
)
from stage4.dispatch.flexibility_native import TrainDemandForecast
from stage4.dispatch.scheme_a_city_adapter import sites_for
from stage4.dispatch.scheme_a_city_graph import CoarseHistoricalLibrary
from stage4.dispatch.scheme_a_match_deadline_value import MatchDeadlineSceneDAG


CONFIG = Path("stage4/config/competitive_chain_benchmark_v1.json")
OUTPUT = Path("stage4/output/capability_chain_planning/competitive_chain_algorithm_v1/run_001")
PUBLIC = Path("stage4/docs/capability_chain_planning/competitive_chain_algorithm_v1/results")
REFERENCE = Path("stage4/output/paper_enhancement/repositioning_robustness/train_tod_demand_reference.parquet")
EXPECTED_SCALES = [
    dict(name="SMALL", tasks_per_scene=12, hv=3, c=1, sites=4),
    dict(name="MEDIUM", tasks_per_scene=24, hv=4, c=2, sites=6),
    dict(name="LARGE", tasks_per_scene=36, hv=6, c=2, sites=8),
]
METHODS = ("INDEPENDENT_VALUE", "EXISTING_RESTRICTED", "BOUND_RECOVERY", "EXACT_DECLARED_DOMAIN")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def clock_text():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds")


class Guard:
    def __init__(self, cfg):
        self.cfg, self.process = cfg, psutil.Process()
        self.started, self.peak = PROCESS_STARTED, 0.
        self.stop = threading.Event()
        self.failure = None
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.sample()
        if psutil.virtual_memory().available / 2**20 < cfg["minimum_initial_available_ram_mib"]:
            raise MemoryError("available RAM below the fixed 1.5 GiB start threshold")
        self.thread.start()

    def sample(self):
        memory = self.process.memory_info()
        high = max(memory.rss, getattr(memory, "peak_wset", memory.rss)) / 2**20
        self.peak = max(self.peak, high)
        if high > self.cfg["maximum_rss_mib"]:
            self.failure = MemoryError("fixed 512 MiB process cap exceeded")

    def _sample(self):
        while not self.stop.wait(.05):
            self.sample()

    def check(self):
        self.sample()
        if self.failure:
            raise self.failure
        if perf_counter() - self.started >= self.cfg["batch_wall_limit_s"]:
            raise TimeoutError("fixed 1800-second batch budget exhausted")

    def close(self):
        self.stop.set()
        self.thread.join(1.)
        self.sample()


def load_configuration(root, path):
    checked = {}

    def read(relative, seen=()):
        if relative in seen:
            raise ValueError("configuration cycle")
        file = root / relative
        raw = json.loads(file.read_text(encoding="utf-8"))
        checked[relative.as_posix()] = sha(file)
        base = raw.get("base_config")
        if base:
            base = Path(base)
            expected = raw.get("base_config_sha256")
            if expected and sha(root / base) != str(expected).lower():
                raise ValueError(f"existing base configuration changed: {base}")
            return dict(read(base, (*seen, relative)), **raw)
        return raw

    cfg = read(path)
    fixed = dict(version="competitive_chain_benchmark_v1",
        anchors_s=[27000, 30600, 61200, 64800], selection_seed=20261014, scales=EXPECTED_SCALES,
        method_wall_limit_s=30, integer_evaluation_limit_s=10, exact_reference_limit_s=30,
        batch_wall_limit_s=1800, maximum_rss_mib=512, minimum_initial_available_ram_mib=1536,
        maximum_variables=20000, maximum_nonzeros=150000, maximum_pricing_rounds=6,
        maximum_exchanges=8, new_method_lp_reserved_limit_s=18,
        gpu_used=False, new_routing=False, new_inference=False, native_simulation=False)
    if any(cfg.get(key) != value for key, value in fixed.items()):
        raise ValueError("outside the already specified twelve-instance plan")
    if (cfg["forecast_train_dates"] != ["20161010", "20161017", "20161024"]
            or cfg.get("max_pickup_eta_s") != 600 or cfg["patience_s"] != 300):
        raise ValueError("historical dates or 300/600 time contract changed")
    cfg.update(measurement_end_s=86400, native_execution=False, full_day=False,
               layout_bin_wall_limit_s=30., coarse_value_max_pricing_rounds=6)
    return cfg, checked


def load_inputs(root, cfg, binding):
    path, manifest = root / cfg["source_history_templates"], root / cfg["source_history_manifest"]
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if metadata.get("status") != "COMPLETE" or metadata.get("full_day_Train_library_not_test31_future_demand") is not True:
        raise ValueError("existing complete historical template source is required")
    columns = list(CoarseHistoricalLibrary._COLUMNS) + ["research_data_ready"]
    templates = pd.read_parquet(path, columns=columns)
    templates = templates.loc[templates.date.astype(str).isin(cfg["forecast_train_dates"])].copy()
    counts = {str(day): int(size) for day, size in templates.groupby(templates.date.astype(str)).size().items()}
    declared = {str(item["date"]): int(item["templates"]) for item in metadata["train_templates"]}
    if counts != {date: declared[date] for date in cfg["forecast_train_dates"]}:
        raise ValueError("historical source row counts differ")
    if templates.duplicated(["date", "order_id"]).any():
        raise ValueError("duplicate original history identity")
    reference_path = root / REFERENCE
    reference_manifest = reference_path.with_suffix(".manifest.json")
    ref_meta = json.loads(reference_manifest.read_text(encoding="utf-8"))
    if (ref_meta.get("test31_included") is not False
            or ref_meta.get("train_dates") != [f"201610{d:02d}" for d in range(9, 25)]
            or ref_meta.get("reference_sha256") != sha(reference_path)):
        raise ValueError("existing historical demand reference changed")
    reference = pd.read_parquet(reference_path, columns=[
        "time_bin_index", "node_id", "demand_share", "lon_wgs84", "lat_wgs84"])
    for file in (path, manifest, reference_path, reference_manifest):
        binding[file.relative_to(root).as_posix()] = sha(file)
    # Only already-frozen model identity is bound; weights/checkpoint are not loaded.
    profile_path = root / cfg["frozen_profile_config"]
    binding[profile_path.relative_to(root).as_posix()] = sha(profile_path)
    return templates, reference, counts


def stable_select(tasks, scene, count, seed):
    selected = sorted(tasks, key=lambda task: hashlib.sha256(
        f"{seed}|{scene}|{task.request_id}".encode()).digest())[:count]
    return tuple(sorted(selected, key=lambda task: (task.release_s, task.request_id)))


def witness(catalogue, vector):
    vector, _, exact, scenes = catalogue.validate(vector)
    return dict(deployment={f"{g[0]}:{g[1]}": values.tolist()
                            for g, values in catalogue.deployment(vector).items()},
        service_exact=[exact.numerator, exact.denominator], scenario_services=scenes,
        paths=[dict(scene=root.scene, kind=root.kind, root=list(root.key), tasks=list(path))
               for index, (root, path) in enumerate(catalogue.columns, len(catalogue.placement)) if vector[index]])


def run_case(now, scale, source_scenes, reference, pace, cfg, guard, output):
    case_id = f"H{int(now//3600):02d}{int(now%3600//60):02d}_{scale['name']}"
    started = perf_counter()
    scenes = {name: stable_select(tasks, name, scale["tasks_per_scene"], cfg["selection_seed"])
              for name, tasks in source_scenes.items()}
    weights = {name: Fraction(1, len(scenes)) for name in scenes}
    local_cfg = dict(cfg, layout_candidate_site_count=scale["sites"])
    sites = sites_for(reference, now, local_cfg)
    site_ids = [site["site_id"] for site in sites]
    hot = np.asarray([site["demand_share"] for site in sites])
    hv = integer_quotas(hot, scale["hv"], site_ids)
    fixed = {("HV", zone, 0, 6): int(count) for zone, count in enumerate(hv) if count}
    fresh = {(0, 6): scale["c"]}
    build_started = perf_counter()
    models = {name: MatchDeadlineSceneDAG(tasks, sites, now, now+1800, pace, local_cfg, guard.check)
              for name, tasks in scenes.items()}
    catalogue = FiniteCatalogue(models, weights, fixed, fresh, site_ids, guard.check,
                               cfg["maximum_variables"], cfg["maximum_nonzeros"])
    build_s = perf_counter()-build_started
    private_tasks = {name: [dict(asdict(task), compatible_profiles=sorted(task.compatible_profiles))
                            for task in tasks] for name, tasks in scenes.items()}
    private_input = dict(now=now, scale=scale, sites=sites, tasks=private_tasks,
                         fixed_supply=[dict(key=list(key), count=count) for key, count in fixed.items()],
                         fresh_supply=[dict(group=list(group), count=count) for group, count in fresh.items()])
    input_hash = hashlib.sha256(json.dumps(private_input, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    atomic_json(output / case_id / "input.json", private_input)
    tie_targets = {}
    for group, count in fresh.items():
        values = compatible_head_weights(models, {n: float(w) for n, w in weights.items()},
                                         group, len(sites))
        tie_targets[group] = integer_quotas(values if values.sum() else hot, count, site_ids).astype(float)
    case = dict(case_id=case_id, input_sha256=input_hash, now_s=now, scale=scale["name"],
                task_counts={name: len(tasks) for name, tasks in scenes.items()},
                c_eligible_counts={name: int(np.count_nonzero(model.allowed["C"]))
                                   for name, model in models.items()},
                sites=len(sites), hv=scale["hv"], c=scale["c"], common_build_s=build_s,
                catalogue=catalogue.info, methods={})
    rows = []
    for method in METHODS[:3]:
        guard.check()
        reset_prices(models)
        method_started = perf_counter()
        receipt, internal_witness = {}, None
        if method == "INDEPENDENT_VALUE":
            placement = independent_placement(models, weights, fresh, site_ids, guard.check)
            receipt["pricing_closed"] = None
        elif method == "EXISTING_RESTRICTED":
            placement, receipt = solve_placement(models, {n: float(w) for n, w in weights.items()},
                fixed, fresh, site_ids,
                dict(local_cfg, layout_bin_wall_limit_s=max(.001, 30.-build_s)), guard.check,
                tie_targets=tie_targets)
        else:
            placement, internal_witness, receipt = solve_bound_recovery(models, weights, fixed, fresh,
                site_ids, local_cfg, catalogue, guard.check, tie_targets=tie_targets, common_build_s=build_s)
        method_s = perf_counter()-method_started
        guard.check()
        # All final placements receive exactly the same complete-domain evaluator.
        vector, evaluation = catalogue.solve_integer(placement, cfg["integer_evaluation_limit_s"],
                                                      internal_witness)
        certificate = (dict(upper_bound=receipt["upper_bound"], upper_bound_exact=receipt["upper_bound_exact"])
                       if method == "BOUND_RECOVERY" else
                       certified_upper(models, weights, fixed, fresh, site_ids, guard.check))
        if catalogue.validate(vector)[2] > Fraction(*certificate["upper_bound_exact"]):
            raise RuntimeError("final feasible witness exceeds the method dual certificate")
        stored = dict(receipt=receipt, evaluation=evaluation, witness=witness(catalogue, vector),
                      bound={key: value for key, value in certificate.items() if key != "value"})
        case["methods"][method] = stored
        rows.append(dict(case_id=case_id, method=method, scale=scale["name"],
            tasks_total=sum(case["task_counts"].values()),
            c_eligible_total=sum(case["c_eligible_counts"].values()),
            service_value=evaluation["service_value"], method_upper=certificate["upper_bound"],
            method_time_s=method_s, charged_common_build_s=build_s, method_end_to_end_s=build_s+method_s,
            final_evaluation_time_s=evaluation["runtime_s"],
            bound_time_s=receipt.get("bound_time_s"), recovery_time_s=receipt.get("recovery_time_s"),
            recovery_gain=receipt.get("recovery_gain"),
            pricing_rounds=receipt.get("pricing_rounds", receipt.get("lp_info", {}).get("pricing_rounds", 0)),
            pricing_closed=receipt.get("pricing_closed", receipt.get("lp_info", {}).get("pricing_closed")),
            exchange_attempts=receipt.get("exchange_attempts", 0),
            accepted_exchanges=receipt.get("accepted_exchanges", 0),
            final_evaluation_optimal=evaluation["optimal_solver_termination"],
            method_wall_overrun=build_s+method_s>cfg["method_wall_limit_s"],
            variables=catalogue.info["variables"], nonzeros=catalogue.info["nonzeros"],
            current_process_peak_mib=guard.peak))
    guard.check()
    reset_prices(models)
    reference_started = perf_counter()
    vector, evaluation = catalogue.solve_integer(time_limit_s=max(.001, 30.-build_s))
    remaining = max(.001, cfg["exact_reference_limit_s"]-build_s-(perf_counter()-reference_started))
    common_certificate, lp_info = catalogue.complete_lp(min(10., remaining))
    if common_certificate is None:
        reset_prices(models)
        common_certificate = certified_upper(models, weights, fixed, fresh, site_ids, guard.check)
    if catalogue.validate(vector)[2] > common_certificate["value"]:
        raise RuntimeError("reference integer witness exceeds complete-domain upper")
    reference_s = perf_counter()-reference_started
    case["methods"]["EXACT_DECLARED_DOMAIN"] = dict(evaluation=evaluation, lp_info=lp_info,
        witness=witness(catalogue, vector),
        bound={k: v for k, v in common_certificate.items() if k != "value"})
    rows.append(dict(case_id=case_id, method="EXACT_DECLARED_DOMAIN", scale=scale["name"],
        tasks_total=sum(case["task_counts"].values()),
        c_eligible_total=sum(case["c_eligible_counts"].values()),
        service_value=evaluation["service_value"], method_upper=common_certificate["upper_bound"],
        method_time_s=reference_s, charged_common_build_s=build_s, method_end_to_end_s=build_s+reference_s,
        final_evaluation_time_s=0., bound_time_s=lp_info["runtime_s"], recovery_time_s=None,
        recovery_gain=None, pricing_rounds=None, pricing_closed=None,
        exchange_attempts=0, accepted_exchanges=0,
        final_evaluation_optimal=evaluation["optimal_solver_termination"],
        method_wall_overrun=build_s+reference_s>cfg["exact_reference_limit_s"],
        variables=catalogue.info["variables"], nonzeros=catalogue.info["nonzeros"],
        current_process_peak_mib=guard.peak))
    for row in rows:
        row["common_certificate_upper"] = common_certificate["upper_bound"]
        row["certificate_gap"] = common_certificate["upper_bound"]-row["service_value"]
        row["certificate_relative_gap"] = row["certificate_gap"]/max(1., abs(common_certificate["upper_bound"]))
        row["reference_integer_value"] = evaluation["service_value"]
        row["reference_solver_optimal"] = evaluation["optimal_solver_termination"]
        if evaluation["optimal_solver_termination"] and row["service_value"] > evaluation["service_value"]+1e-9:
            raise RuntimeError("a method beats the exact same-domain optimal reference")
    case["runtime_s"] = perf_counter()-started
    atomic_json(output / case_id / "case_results.json", case)
    print(json.dumps(dict(event="CASE_COMPLETE", time=clock_text(), case_id=case_id,
        values={row["method"]: row["service_value"] for row in rows}, runtime_s=case["runtime_s"],
        peak_rss_mib=guard.peak), ensure_ascii=False), flush=True)
    return rows, {key: value for key, value in case.items() if key != "methods"}


def execute(root, config=CONFIG):
    cfg, binding = load_configuration(root, config)
    output, public = root/OUTPUT, root/PUBLIC
    if output.exists() or (public/"summary.json").exists():
        raise FileExistsError("this fixed run already has products; no silent rerun/overwrite")
    output.mkdir(parents=True)
    guard = Guard(cfg)
    started = clock_text()
    code = subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}",
                                   "rev-parse", "HEAD"], cwd=root, text=True).strip()
    rows, cases = [], []
    result = dict(status="RUNNING", started_at=started, pid=psutil.Process().pid, code_sha=code,
        config_sha256=sha(root/config), inputs=binding, expected_instances=12,
        current_city_controller_modified=False, native_simulation=False, routing_queries=0,
        new_M3_inference=False, gpu_used=False, threads=1, results_analysis_is_descriptive=True)
    print(json.dumps(dict(event="STARTED", time=started, pid=result["pid"], code_sha=code,
                          available_ram_mib=psutil.virtual_memory().available/2**20,
                          rss_mib=guard.peak), ensure_ascii=False), flush=True)
    atomic_json(output/"summary.json", result)
    try:
        with threadpool_limits(limits=1):
            templates, reference, counts = load_inputs(root, cfg, binding)
            library = CoarseHistoricalLibrary(templates, cfg)
            # Same prediction-only pace constructor, no new calibration.
            pace = TrainDemandForecast(templates, dict(cfg, fast_forecast=False), 86434)
            result.update(history_template_counts=counts,
                          sampling="EXISTING_FROZEN_POISSON_LIBRARY_THEN_HASH_PREFIX_20261014",
                          common_build_charged_to_each_method_but_actual_batch_counted_once=True,
                          integer_recovery_uses_bounded_complete_catalogue=True)
            for now in cfg["anchors_s"]:
                source_scenes = library.view(now)
                for scale in cfg["scales"]:
                    guard.check()
                    case_rows, case = run_case(now, scale, source_scenes, reference, pace, cfg, guard, output)
                    rows.extend(case_rows); cases.append(case)
                    gc.collect()
                    result.update(completed_instances=len(cases), completed_method_rows=len(rows),
                                  cases=cases, peak_rss_mib=guard.peak)
                    atomic_json(output/"summary.json", result)
            result["status"] = "TWELVE_INSTANCES_COMPLETED"
    except Exception as error:
        result.update(status="FAILED_OR_LIMITED", error_type=type(error).__name__, error=str(error),
                      completed_instances=len(cases), remaining_instances=12-len(cases))
        raise
    finally:
        guard.close()
        result.update(finished_at=clock_text(), processing_runtime_s=perf_counter()-PROCESS_STARTED,
                      peak_rss_mib=guard.peak, completed_instances=len(cases))
        atomic_json(output/"summary.json", result)
        if rows:
            public.mkdir(parents=True, exist_ok=True)
            temp = public/"results.csv.tmp"
            pd.DataFrame(rows).to_csv(temp, index=False)
            temp.replace(public/"results.csv")
            atomic_json(public/"summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    result = execute(args.root.resolve(), args.config)
    print(json.dumps(dict(event="FINISHED", status=result["status"],
        completed_instances=result["completed_instances"], runtime_s=result["processing_runtime_s"],
        peak_rss_mib=result["peak_rss_mib"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
