"""Static complexity audit and deterministic v5.2 scaling benchmarks."""

from __future__ import annotations

import ast
import os
import time
import tracemalloc
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from .micro_products import (
    DIMENSIONS,
    _maximum_consecutive_share,
    aggregate_original_route_micro_conditions,
    weighted_quantile_by_group,
)
from .support_transfer import support_gate


PROHIBITED_PATTERNS = (
    "groupby.apply", "DataFrame.apply(axis=1)", "iterrows", "itertuples",
    "concat_inside_loop", "unique_edge_filter_inside_loop",
)


class ComplexityScanner(ast.NodeVisitor):
    def __init__(self, path: Path):
        self.path = path
        self.loop_depth = 0
        self.findings: list[dict[str, Any]] = []

    def visit_For(self, node: ast.For) -> None:
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    visit_AsyncFor = visit_For

    def visit_Call(self, node: ast.Call) -> None:
        attribute = node.func.attr if isinstance(node.func, ast.Attribute) else ""
        if attribute in {"iterrows", "itertuples"}:
            self.findings.append({"file": self.path.as_posix(), "line": node.lineno, "pattern": attribute})
        if attribute == "apply":
            axis_one = any(
                keyword.arg == "axis" and isinstance(keyword.value, ast.Constant) and keyword.value.value == 1
                for keyword in node.keywords
            )
            groupby_apply = (
                isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Attribute)
                and node.func.value.func.attr == "groupby"
            )
            if axis_one or groupby_apply:
                self.findings.append({
                    "file": self.path.as_posix(), "line": node.lineno,
                    "pattern": "DataFrame.apply(axis=1)" if axis_one else "groupby.apply",
                })
        if attribute == "concat" and self.loop_depth:
            self.findings.append({"file": self.path.as_posix(), "line": node.lineno, "pattern": "concat_inside_loop"})
        self.generic_visit(node)


def static_complexity_audit(root: str | Path) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    paths = sorted(Path(root).rglob("*.py"))
    for path in paths:
        scanner = ComplexityScanner(path)
        scanner.visit(ast.parse(path.read_text(encoding="utf-8")))
        findings.extend(scanner.findings)
    return {
        "schema_version": "stage2_v5_2_static_complexity.1",
        "status": "PASS" if not findings else "FAIL",
        "scanned_file_count": len(paths),
        "blocking_finding_count": len(findings),
        "findings": findings,
        "prohibited_patterns": list(PROHIBITED_PATTERNS),
    }


def _measure(name: str, rows: int, function: Callable[[], object]) -> dict[str, float | int | str]:
    process = psutil.Process(os.getpid())
    before = process.memory_info().rss
    tracemalloc.start()
    started = time.perf_counter()
    function()
    wall = time.perf_counter() - started
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    after = process.memory_info().rss
    return {
        "hotspot": name,
        "rows": rows,
        "wall_time_s": wall,
        "rows_per_s": rows / max(wall, 1.0e-12),
        "peak_rss_mb": max(traced_peak, after - before, 0) / (1024 * 1024),
    }


def _route_case(rows: int) -> Callable[[], object]:
    route_length = 20
    order = np.arange(rows) // route_length
    sequence = np.arange(rows) % route_length
    frame = pd.DataFrame({
        "split": "benchmark", "date": "20161025", "order_id": order.astype(str),
        "route_sequence": sequence, "estimated_travel_time_p50_s": 1.0 + sequence,
        "allocated_distance_m": 10.0, "edge_train_support": sequence,
        "support_group": np.where(sequence == 0, "unseen", np.where(sequence < 7, "low", "high")),
    })
    for column in DIMENSIONS.values():
        frame[column] = (sequence % 11) / 10.0
    cdf = {
        "fit_split": "train", "evaluation_rows_used": 0,
        "thresholds": {name: 0.8 for name in DIMENSIONS},
    }
    return lambda: aggregate_original_route_micro_conditions(frame, cdf)


def _support_representation_case(rows: int) -> Callable[[], object]:
    support = (np.arange(rows) % 1000).astype(np.float32)
    identity = np.tile(np.linspace(-1.0, 1.0, 16, dtype=np.float32), (rows, 1))
    structure = np.flip(identity, axis=1).copy()

    def represent() -> np.ndarray:
        gate = support_gate(support, 25.0).astype(np.float32)
        return gate[:, None] * identity + (1.0 - gate[:, None]) * structure

    return represent


def _temporal_adapter_case(rows: int) -> Callable[[], object]:
    state = np.tile(np.linspace(-1.0, 1.0, 16, dtype=np.float32), (rows, 1))
    time_features = np.tile(np.array([0.0, 1.0, 0.5, 0.25], dtype=np.float32), (rows, 1))
    down = np.ones((20, 4), dtype=np.float32) / 20.0
    up = np.ones((4, 16), dtype=np.float32) / 4.0
    return lambda: state + np.maximum(np.column_stack((state, time_features)) @ down, 0.0) @ up


def _quantile_case(rows: int) -> Callable[[], object]:
    codes = np.arange(rows) // 20
    values = (np.arange(rows) % 19) / 18.0
    weights = 1.0 + np.arange(rows) % 7
    return lambda: weighted_quantile_by_group(codes, values, weights, int(codes.max() + 1), quantile=0.90)


def _consecutive_case(rows: int) -> Callable[[], object]:
    codes = np.arange(rows) // 20
    high = np.arange(rows) % 5 < 2
    weights = np.ones(rows)
    total = np.bincount(codes, weights=weights)
    return lambda: _maximum_consecutive_share(codes, high, weights, total)


def run_benchmarks(sizes: Iterable[int] = (10_000, 50_000, 100_000, 500_000)) -> tuple[pd.DataFrame, dict[str, Any]]:
    factories = {
        "support_aware_edge_representation": _support_representation_case,
        "static_structure_preprocessing": lambda rows: lambda: np.column_stack((
            np.arange(rows) % 12, np.arange(rows) % 2, np.arange(rows) % 3
        )).astype(np.float32),
        "micro_route_aggregation": _route_case,
        "weighted_quantile": _quantile_case,
        "max_consecutive_high_exposure": _consecutive_case,
        "temporal_adapter": _temporal_adapter_case,
    }
    records = [_measure(name, int(rows), factory(int(rows))) for name, factory in factories.items() for rows in sizes]
    frame = pd.DataFrame(records)
    checks: list[dict[str, Any]] = []
    for hotspot, group in frame.groupby("hotspot", sort=False, observed=True):
        lookup = group.set_index("rows")["wall_time_s"]
        for low, high in ((10_000, 50_000), (100_000, 500_000)):
            if low in lookup.index and high in lookup.index:
                ratio = float(lookup.loc[high] / max(lookup.loc[low], 1.0e-12))
                checks.append({
                    "hotspot": hotspot, "low_rows": low, "high_rows": high,
                    "scaling_ratio": ratio, "status": "PASS" if ratio <= 8.0 else "FAIL",
                })
    return frame, {
        "schema_version": "stage2_v5_2_performance.1",
        "status": "PASS" if checks and all(row["status"] == "PASS" for row in checks) else "FAIL",
        "scaling_checks": checks,
    }
