"""Deterministic scaling benchmarks for all v5 production hotspots."""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import time
import tracemalloc
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import psutil

from stage2.v5.aggregation import RouteDimension, aggregate_route_dimensions
from stage2.v5.cdf import EmpiricalCDFIndex, map_empirical_cdf
from stage2.v5.history import causal_window_mean


def _measure(name: str, rows: int, function: Callable[[], object]) -> tuple[dict[str, float | int | str], object]:
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    tracemalloc.start()
    started = time.perf_counter()
    value = function()
    runtime = time.perf_counter() - started
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = process.memory_info().rss
    peak_delta = max(traced_peak, rss_after - rss_before, 0)
    return (
        {
            "hotspot": name,
            "rows": int(rows),
            "wall_time_s": runtime,
            "rows_per_s": rows / max(runtime, 1e-12),
            "peak_rss_delta_mb": peak_delta / (1024 * 1024),
            "process_rss_mb": rss_after / (1024 * 1024),
        },
        value,
    )


def _cdf_case(rows: int, unique_edges: int) -> Callable[[], object]:
    codes = np.arange(rows, dtype=np.int64) % unique_edges
    keys = np.char.add("e", codes.astype(str))
    sample = np.linspace(0.01, 1.0, 32)
    samples = {f"e{index}": sample for index in range(unique_edges)}
    support = {key: 32 for key in samples}
    index = EmpiricalCDFIndex(("edge",), {"edge": samples}, {"edge": support})
    values = (codes % 100) / 100.0
    return lambda: map_empirical_cdf(values, {"edge": keys}, index, minimum_support=5)


def _aggregation_case(rows: int) -> Callable[[], object]:
    route_length = 20
    order = np.arange(rows) // route_length
    sequence = np.arange(rows) % route_length
    frame = pd.DataFrame(
        {
            "split": "benchmark",
            "date": "20161025",
            "order_id": order.astype(str),
            "route_sequence": sequence,
            "pct": (sequence % 11) / 10.0,
            "prob": (sequence % 7) / 6.0,
            "value": (sequence % 13) / 12.0,
            "weight": 1.0 + sequence,
        }
    )
    spec = (RouteDimension("rts", "pct", "prob", "value", "weight"),)
    return lambda: aggregate_route_dimensions(frame, spec)


def _merge_case(rows: int) -> Callable[[], object]:
    shard_count = 10
    per_shard = int(np.ceil(rows / shard_count))
    shards = [pd.DataFrame({"row_id": np.arange(index * per_shard, min((index + 1) * per_shard, rows)), "prediction": float(index)}) for index in range(shard_count)]
    return lambda: pd.concat(shards, ignore_index=True, copy=False).sort_values("row_id", kind="stable", ignore_index=True)


def _history_case(rows: int) -> Callable[[], object]:
    event_rows = rows
    query_rows = rows
    edge_count = max(100, rows // 50)
    event_key = np.char.add("e", (np.arange(event_rows) % edge_count).astype(str))
    event_time = (np.arange(event_rows) // edge_count).astype(float) * 30.0
    event_value = 0.1 + (np.arange(event_rows) % 17) / 100.0
    query_key = np.char.add("e", (np.arange(query_rows) % edge_count).astype(str))
    decision = (np.arange(query_rows) // edge_count).astype(float) * 30.0 + 60.0
    return lambda: causal_window_mean(event_key, event_time, event_value, query_key, decision, window_s=3600.0)


def _scenario_aggregation_case(rows: int) -> Callable[[], object]:
    scenario_count = 16
    route_inverse = np.arange(rows, dtype=np.int64) // 20
    traversal = np.full((rows, scenario_count), 2.0, dtype=np.float64)
    def aggregate() -> np.ndarray:
        output = np.zeros((int(route_inverse.max() + 1), scenario_count), dtype=np.float64)
        np.add.at(output, route_inverse, traversal)
        return output
    return aggregate


def run_benchmarks(sizes: tuple[int, ...]) -> tuple[pd.DataFrame, dict[str, object]]:
    factories = {
        "cdf_mapping": lambda rows: _cdf_case(rows, min(10000, max(100, rows // 50))),
        "route_aggregation": _aggregation_case,
        "prediction_shard_merge": _merge_case,
        "history_lookup": _history_case,
        "scenario_aggregation": _scenario_aggregation_case,
    }
    records: list[dict[str, float | int | str]] = []
    for name, factory in factories.items():
        for rows in sizes:
            record, _ = _measure(name, rows, factory(rows))
            records.append(record)
    # Explicit N x K guard requested by the taskbook.
    for edges in (100, 1000, 10000):
        record, _ = _measure(f"cdf_mapping_k={edges}", 100000, _cdf_case(100000, edges))
        records.append(record)
    frame = pd.DataFrame(records)
    checks: list[dict[str, object]] = []
    for hotspot, group in frame.loc[~frame["hotspot"].str.contains("=")].groupby("hotspot", sort=False):
        lookup = group.set_index("rows")["wall_time_s"]
        for low, high in ((10000, 50000), (100000, 500000)):
            if low in lookup.index and high in lookup.index:
                ratio = float(lookup.loc[high] / max(lookup.loc[low], 1e-12))
                checks.append({"hotspot": hotspot, "low_rows": low, "high_rows": high, "runtime_ratio": ratio, "status": "PASS" if ratio <= 8.0 else "FAIL"})
    report = {
        "schema_version": "stage2_v5_performance_benchmarks.1",
        "status": "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL",
        "scaling_checks": checks,
        "maximum_process_rss_mb": float(frame["process_rss_mb"].max()),
        "maximum_peak_rss_delta_mb": float(frame["peak_rss_delta_mb"].max()),
    }
    return frame, report


def _write_profile(output: Path) -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    _aggregation_case(100000)()
    _cdf_case(100000, 1000)()
    _history_case(100000)()
    _scenario_aggregation_case(100000)()
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(30)
    output.write_text(stream.getvalue(), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[10000, 50000, 100000, 500000])
    parser.add_argument("--csv", default="stage2/docs/v5/performance_benchmarks.csv")
    parser.add_argument("--json", default="stage2/docs/v5/stage2_v5_performance_benchmarks.json")
    parser.add_argument("--profile", default="stage2/docs/v5/performance_profile_hotspots.txt")
    args = parser.parse_args()
    frame, report = run_benchmarks(tuple(args.sizes))
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_profile(Path(args.profile))
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
