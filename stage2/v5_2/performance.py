"""Static complexity audit and deterministic v5.2 scaling benchmarks."""

from __future__ import annotations

import ast
import os
import threading
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
from .feature_binding import V51FeatureSchemaBinding
from .models.rc_mstnet_transfer import RCMSTNetTransfer
from .support_transfer import SupportAwareEdgeRepresentation
from .structure_features import build_static_structure_features, fit_static_structure_artifact
from .temporal_adapter import TemporalAdapter


PROHIBITED_PATTERNS = (
    "groupby.apply", "DataFrame.apply(axis=1)", "iterrows", "itertuples",
    "concat_inside_loop", "unique_edge_filter_inside_loop",
)


def benchmark_kernel_devices(*, torch_kernel: bool, cuda_available: bool) -> tuple[str, ...]:
    if not torch_kernel:
        return ("cpu",)
    return ("cpu", "cuda") if cuda_available else ("cpu",)


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


def _measure(
    name: str, rows: int, function: Callable[[], object], *, device: str,
    warmup_runs: int, repeat_runs: int, fixture_setup_rss: int,
) -> dict[str, float | int | str]:
    process = psutil.Process(os.getpid())
    baseline_rss = process.memory_info().rss
    peak_rss = baseline_rss
    stop = threading.Event()

    def sample_rss() -> None:
        nonlocal peak_rss
        while not stop.wait(0.005):
            peak_rss = max(peak_rss, process.memory_info().rss)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    import torch
    with torch.inference_mode():
        for _ in range(warmup_runs):
            function()
        if device == "cuda":
            torch.cuda.synchronize()
    timings: list[float] = []
    tracemalloc.start(); sampler.start()
    try:
        with torch.inference_mode():
            for _ in range(repeat_runs):
                started = time.perf_counter()
                function()
                if device == "cuda":
                    torch.cuda.synchronize()
                timings.append(time.perf_counter() - started)
    finally:
        stop.set()
        sampler.join()
        _, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_rss = max(peak_rss, process.memory_info().rss)
    return {
        "hotspot": name,
        "rows": rows,
        "device": device,
        "inference_mode": True,
        "warmup_runs": warmup_runs,
        "repeat_runs": repeat_runs,
        "wall_time_s": float(np.median(timings)),
        "rows_per_s": rows / max(float(np.median(timings)), 1.0e-12),
        "fixture_setup_rss_mb": fixture_setup_rss / (1024 * 1024),
        "execution_baseline_rss_mb": baseline_rss / (1024 * 1024),
        "rss_baseline_mb": baseline_rss / (1024 * 1024),
        "rss_peak_mb": max(traced_peak, peak_rss, baseline_rss) / (1024 * 1024),
        "rss_delta_mb": max(0, max(traced_peak, peak_rss) - baseline_rss) / (1024 * 1024),
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
        "protocol_id": "development",
        "model_id": "M4",
        "prediction_source": "benchmark_fixture",
        "route_track": "historical_original_service_route",
        "route_source": "frozen_stage1_route_parts",
        "route_product_version": "stage1_v3_route_sequence_context.1",
    })
    for column in DIMENSIONS.values():
        frame[column] = (sequence % 11) / 10.0
    cdf = {
        "fit_split": "train", "evaluation_rows_used": 0, "protocol_id": "development",
        "model_id": "M4", "prediction_source": "benchmark_fixture",
        "thresholds": {name: 0.8 for name in DIMENSIONS},
    }
    return lambda: aggregate_original_route_micro_conditions(frame, cdf)


def _support_representation_case(rows: int, device: str = "cpu") -> Callable[[], object]:
    import torch

    module = SupportAwareEdgeRepresentation(
        edge_vocabulary_size=1001,
        static_feature_count=8,
        embedding_dim=16,
        tau=25.0,
        mode="support_aware",
    ).eval().to(device)
    edge = torch.as_tensor(np.arange(rows) % 1001, dtype=torch.long, device=device)
    static = torch.ones((rows, 8), dtype=torch.float32, device=device)
    support = torch.as_tensor(np.arange(rows) % 1000, dtype=torch.float32, device=device)
    return lambda: module(edge, static, support)


def _temporal_adapter_case(rows: int, device: str = "cpu") -> Callable[[], object]:
    import torch

    module = TemporalAdapter(hidden_dim=16, time_feature_dim=4, bottleneck_dim=4).eval().to(device)
    state = torch.ones((rows, 16), dtype=torch.float32, device=device)
    time_features = torch.ones((rows, 4), dtype=torch.float32, device=device)
    return lambda: module(state, time_features)


def _full_transfer_forward_case(rows: int, device: str = "cpu") -> Callable[[], object]:
    import torch
    sequence_length = 32
    batch = max(1, int(np.ceil(rows / sequence_length)))
    categorical_sizes = (1001, 32, 32, 32, 32)
    binding = V51FeatureSchemaBinding(
        categorical_names=("edge", "highway", "time_bin", "position_bucket", "route_length_bucket"),
        edge_category_name="edge", edge_source_field="observed_directed_edge_uid",
        edge_column_index=0, pad_index=0, unseen_index=1,
        categorical_sizes=categorical_sizes, feature_artifact_sha256="f" * 64,
        edge_vocabulary_sha256="e" * 64, categorical_vocabulary_sha256="v" * 64,
    )
    model = RCMSTNetTransfer(
        numeric_feature_count=8, binding=binding, static_feature_count=8,
        support_tau=25.0, spatial_mode="support_aware", temporal_mode="zero_shot",
        backbone_kwargs={
            "hidden_dim": 16, "categorical_embedding_dim": 4, "transformer_layers": 1,
            "attention_heads": 4, "dropout": 0.0, "distribution_family": "monotonic_quantiles",
            "history_mode": "ordinary_concatenation",
        },
    ).eval().to(device)
    numeric = torch.ones((batch, sequence_length, 8), dtype=torch.float32, device=device)
    missing = torch.zeros_like(numeric, dtype=torch.bool)
    categorical = torch.stack([
        torch.as_tensor(np.arange(batch * sequence_length).reshape(batch, sequence_length) % size, dtype=torch.long, device=device)
        for size in categorical_sizes
    ], dim=-1)
    sequence = torch.arange(sequence_length, device=device).expand(batch, -1)
    pad = torch.zeros((batch, sequence_length), dtype=torch.bool, device=device)
    static = torch.ones((batch, sequence_length, 8), dtype=torch.float32, device=device)
    support = torch.ones((batch, sequence_length), dtype=torch.float32, device=device)
    temporal = {
        name: torch.ones((batch, sequence_length), dtype=torch.float32, device=device)
        for name in ("decision_hour_sin", "decision_hour_cos", "decision_weekday_index", "forecast_horizon_log1p")
    }
    return lambda: model(
        numeric, missing, categorical, sequence, pad, static_edge_features=static,
        edge_train_support=support, temporal_features=temporal,
    )


def _static_structure_case(rows: int) -> Callable[[], object]:
    frame = pd.DataFrame({
        "split": "train", "date": "20161009", "order_id": (np.arange(rows) // 20).astype(str),
        "route_sequence": np.arange(rows) % 20, "row_id": np.arange(rows),
        "canonical_highway": np.where(np.arange(rows) % 2, "primary", "secondary"),
        "road_class": np.where(np.arange(rows) % 3, "major", "minor"),
        "observed_direction": "forward", "bridge": False, "tunnel": False,
        "synthetic_reverse_edge": False, "osm_direction_disagreement": False,
    })
    artifact = fit_static_structure_artifact(
        [frame], protocol_id="benchmark", protocol_train_dates=("20161009",)
    )
    return lambda: build_static_structure_features(frame, artifact)


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


def run_benchmarks(
    sizes: Iterable[int] = (10_000, 50_000, 100_000, 500_000), *,
    warmup_runs: int = 2, repeat_runs: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    factories = {
        "support_aware_edge_representation": (_support_representation_case, True),
        "static_structure_preprocessing": (_static_structure_case, False),
        "micro_route_aggregation": (_route_case, False),
        "weighted_quantile": (_quantile_case, False),
        "max_consecutive_high_exposure": (_consecutive_case, False),
        "temporal_adapter": (_temporal_adapter_case, True),
        "full_transfer_model_forward": (_full_transfer_forward_case, True),
    }
    import torch
    devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
    records = []
    for name, (factory, torch_kernel) in factories.items():
        kernel_devices = benchmark_kernel_devices(
            torch_kernel=torch_kernel, cuda_available=torch.cuda.is_available()
        )
        for device in kernel_devices:
            for rows in sizes:
                fixture_setup_rss = psutil.Process(os.getpid()).memory_info().rss
                function = factory(int(rows), device) if torch_kernel else factory(int(rows))
                records.append(_measure(
                    name, int(rows), function, device=device,
                    warmup_runs=warmup_runs, repeat_runs=repeat_runs,
                    fixture_setup_rss=fixture_setup_rss,
                ))
    frame = pd.DataFrame(records)
    checks: list[dict[str, Any]] = []
    for (hotspot, device), group in frame.groupby(["hotspot", "device"], sort=False, observed=True):
        lookup = group.set_index("rows")["wall_time_s"]
        for low, high in ((10_000, 50_000), (100_000, 500_000)):
            if low in lookup.index and high in lookup.index:
                ratio = float(lookup.loc[high] / max(lookup.loc[low], 1.0e-12))
                checks.append({
                    "hotspot": hotspot, "device": device, "low_rows": low, "high_rows": high,
                    "scaling_ratio": ratio, "status": "PASS" if ratio <= 8.0 else "FAIL",
                })
    return frame, {
        "schema_version": "stage2_v5_2_performance.2",
        "status": "PASS" if checks and all(row["status"] == "PASS" for row in checks) else "FAIL",
        "timing_policy": {"inference_mode": True, "warmup_runs": warmup_runs, "repeat_runs": repeat_runs, "summary": "median"},
        "devices": sorted(frame["device"].unique().tolist()),
        "cpu_only_kernels": sorted(name for name, (_, torch_kernel) in factories.items() if not torch_kernel),
        "scaling_checks": checks,
    }
