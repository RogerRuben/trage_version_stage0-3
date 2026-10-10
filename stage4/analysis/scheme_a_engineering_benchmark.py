"""Bounded engineering comparison, NOT another native/whole-day experiment.

Three fine-time epochs use existing earlier-history templates and synthetic
current SERVE options at those historical dropoffs. Connections are the same
deterministic geometric scalar mock on both sides. The integer master is
captured, not solved, so this measures chain-construction engineering only.
The complete supplied column payloads must match exactly at every epoch.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.util
import json
from math import ceil
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from time import perf_counter

import numpy as np
import pandas as pd
import psutil

from stage4.analysis.capability_chain_instances import atomic_json, sha
from stage4.dispatch import scheme_a_city_adapter as modern_adapter
from stage4.dispatch import scheme_a_city_graph as modern_graph
from stage4.dispatch.scheme_a_city_master import CurrentAction


def load_reference(name, path, *, root=None, commit=None):
    if commit is not None:
        root = Path(root).resolve()
        relative = Path(path).resolve().relative_to(root).as_posix()
        source = subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}",
            "-C", str(root), "show", f"{commit}:{relative}"])
        module = ModuleType(f"stage4.dispatch.{name}")
        module.__package__ = "stage4.dispatch"
        module.__file__ = str(path)
        sys.modules[module.__name__] = module
        exec(compile(source, f"{path} [git {commit[:8]}]", "exec"), module.__dict__)
        module.__engineering_source_sha256__ = hashlib.sha256(source).hexdigest()
        return module
    spec = importlib.util.spec_from_file_location(f"stage4.dispatch.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.__engineering_source_sha256__ = sha(path)
    return module


class ScalarMock:
    def __init__(self):
        self.timings, self.counts = Counter(), Counter()

    def __call__(self, state, target, departure, profile):
        self.counts["calls"] += 1
        position = target["position"] if isinstance(target, dict) else target.pickup
        distance = float(np.linalg.norm(np.asarray(modern_graph._xy(state.position))
            - np.asarray(modern_graph._xy(position))))
        return dict(supported=True, compatible_profiles=("HV", "C"),
            travel_time_s=distance / 7.5, empty_distance_m=distance,
            arrival_context=dict(kind="EMPTY_ARRIVAL", point=position,
                edge_uids=("BENCHMARK_ONLY_SYNTHETIC_DIRECTED_EDGE",), geometry_timestamp="MOCK"))


def column_digest(chains):
    encoded = json.dumps([asdict(chain) for chain in chains], sort_keys=True,
        allow_nan=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_side(graph, adapter, templates, source_rows, settings):
    library = graph.CoarseHistoricalLibrary(templates, settings)
    connector = ScalarMock()
    digests, rows = [], []
    def capture(actions, chains, *args, **kwargs):
        digests.append(column_digest(chains))
        return dict(selected_chain_ids=(), runtime_s=0., model={})
    original_master = adapter.solve_city_master
    adapter.solve_city_master = capture
    started = perf_counter()
    try:
        for now in (61200., 61230., 61260.):
            scenes = library.view(now)
            actions, states = [], {}
            sites = [dict(site_id=f"S{i}", position=(float(r.end_lon_wgs84), float(r.end_lat_wgs84)))
                     for i, r in enumerate(source_rows)]
            for family, record in enumerate(source_rows):
                allowed = frozenset({"HV"} | {p for p in ("C", "M", "A")
                    if bool(getattr(record, f"compatible_{p}"))})
                source = graph.ChainTask(1000000 + family, float(record.release_second),
                    float(record.release_second) + 300, float(record.predicted_route_time_p50_s),
                    (float(record.start_lon_wgs84), float(record.start_lat_wgs84)),
                    (float(record.end_lon_wgs84), float(record.end_lat_wgs84)), allowed,
                    source_date=str(record.date), source_order_id=str(record.order_id))
                profile = "C" if "C" in allowed and family % 3 == 0 else "HV"
                ready_grid = float(ceil((now + 30. + source.service_time_s) / 30) * 30)
                for alternative in range(8):
                    vid = family * 8 + alternative
                    action = CurrentAction(f"V{vid}:SERVE:{source.request_id}", vid, "SERVE", source.request_id)
                    actions.append(action)
                    states[action.action_id] = graph.RouteState(source.dropoff,
                        ready_grid - 20. + alternative, now + 2400. + vid, profile,
                        dict(kind="CUSTOMER", task=source))
            result = adapter.build_and_solve(actions, states, scenes, library.weights,
                connector, sites, settings, now, "CHAIN_DEFER")
            rows.append(dict(epoch_s=now, current_actions=len(actions),
                columns=result["restricted_graph"]["generated_columns"],
                chain_builds_executed=result["restricted_graph"].get("chain_builds_executed"),
                chain_builds_reused=result["restricted_graph"].get("chain_builds_reused", 0),
                graph_wall_s=result["graph_wall_time_s"],
                continuation_cache_bytes=result["restricted_graph"].get("continuation_cache_bytes", 0)))
    finally:
        adapter.solve_city_master = original_master
    return dict(wall_s=perf_counter()-started, connector_calls=connector.counts["calls"],
        column_digests=digests, epochs=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True,
        help="Unmodified baseline source/data root; read-only.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-commit", help="Read baseline code from Git even after the data checkout is upgraded.")
    args = parser.parse_args(argv)
    start = perf_counter()
    root = args.reference_root.resolve()
    available = psutil.virtual_memory().available / 2**20
    if available < 1536:
        raise RuntimeError("engineering check requires 1536 MiB available; no application is closed")
    graph_file = root / "stage4/dispatch/scheme_a_city_graph.py"
    adapter_file = root / "stage4/dispatch/scheme_a_city_adapter.py"
    baseline_graph = load_reference("_engineering_reference_graph", graph_file, root=root, commit=args.reference_commit)
    baseline_adapter = load_reference("_engineering_reference_adapter", adapter_file, root=root, commit=args.reference_commit)
    baseline_adapter.build_restricted_chains = baseline_graph.build_restricted_chains
    config_path = root / "stage4/config/scheme_a_city_v1.json"
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    settings.update(admission_end_s=86760, measurement_end_s=86434)
    source_path = root / settings["source_history_templates"]
    templates = pd.read_parquet(source_path, columns=list(modern_graph.CoarseHistoricalLibrary._COLUMNS))
    ordered = templates.loc[templates.date.astype(str).eq(settings["forecast_train_dates"][0])
        & templates.release_second.lt(61200)
        & templates.predicted_route_time_p50_s.le(1200)].sort_values("order_id", kind="stable")
    # Fixed all-demand / C-capable workload strata, never selected by outcomes.
    source_rows = list(ordered.loc[ordered.compatible_C].head(4).itertuples(index=False))
    source_rows += list(ordered.loc[~ordered.compatible_C].head(8).itertuples(index=False))
    if len(source_rows) != 12:
        raise ValueError("the fixed historical engineering workload is incomplete")
    old = run_side(baseline_graph, baseline_adapter, templates, source_rows, settings)
    new = run_side(modern_graph, modern_adapter, templates, source_rows, settings)
    if old["column_digests"] != new["column_digests"]:
        raise AssertionError("engineering optimization changed the supplied chain columns")
    info = psutil.Process().memory_info()
    result = dict(status="ENGINEERING_COLUMN_EQUIVALENCE_PASS", native_simulation=False,
        whole_day_experiment=False, master_solved=False, real_valhalla_queries=0,
        connector_model="IDENTICAL_GEOMETRIC_SCALAR_MOCK_NOT_PHYSICAL_ROUTE_EVIDENCE",
        historical_template_rows=len(templates), epochs=3, actions_per_epoch=96,
        reference_code_sha=args.reference_commit or subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}",
            "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        input_sha256={"config":sha(config_path), "templates":sha(source_path),
            "reference_graph":baseline_graph.__engineering_source_sha256__,
            "reference_adapter":baseline_adapter.__engineering_source_sha256__},
        old=old, optimized=new, construction_speed_ratio=old["wall_s"]/new["wall_s"],
        connector_calls_avoided=old["connector_calls"]-new["connector_calls"],
        peak_rss_mib=getattr(info, "peak_wset", info.rss)/2**20,
        python_main_wall_s=perf_counter()-start, full_day_speedup_not_identified=True)
    atomic_json(args.output, result)
    print(json.dumps(result, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
