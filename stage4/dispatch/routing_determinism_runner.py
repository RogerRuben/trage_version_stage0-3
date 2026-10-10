"""Compact routing determinism audit and deterministic-mode performance spike."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from stage4.dispatch.fleet_normalization import build_fleet_scenario
from stage4.fleetpy_adapter.test31_demand_adapter import load_all_test31_requests

OUTPUT_REL = Path("stage4/output/paper_enhancement/routing_determinism")
Q50_ASSIGNMENTS_REL = Path(
    "stage4/output/paper_enhancement/gate_decomposition/reruns/"
    "MAIN_Q50_M_P70/assignment_log.parquet"
)
STAGE3_CONFIG_REL = Path("stage3/config/stage3_finalization.json")
KNOWN_ORDER_ID = "ea90853f16ba4c0fcb2c0d8481cb6fd8"
SAMPLE_SEED = 20260901
REPEATS = 5
BATCH_SIZES = (2, 5, 10, 20)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _priority(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}|{SAMPLE_SEED}|{value}".encode()).hexdigest()


def _actor(root: Path) -> Any:
    from valhalla import Actor

    config = json.loads((root / STAGE3_CONFIG_REL).read_text(encoding="utf-8"))
    return Actor(str(Path(str(config["valhalla_config"])).resolve()))


def _all_q50_arcs(root: Path) -> pd.DataFrame:
    assignments = pd.read_parquet(root / Q50_ASSIGNMENTS_REL).sort_values(
        ["assignment_time", "order_id"], kind="mergesort"
    )
    start = pd.Timestamp("2016-10-31T00:00:00+08:00")
    end = pd.Timestamp("2016-11-01T00:00:00+08:00")
    requests = load_all_test31_requests(root, start=start, end=end, profile_id="M")
    request_by_order = {item.order_id: item for item in requests}
    fleet = build_fleet_scenario(
        root,
        benchmark_start=start,
        simulation_end=end + pd.Timedelta(days=1),
        requested_q_a=0.5,
        seed=20260824,
        max_hv_hour_error_pct=2.0,
    )
    position = {
        item.vehicle_id: (item.initial_lon_wgs84, item.initial_lat_wgs84)
        for item in fleet.native_fixtures
    }
    rows: list[dict[str, Any]] = []
    for row in assignments.itertuples(index=False):
        if row.vehicle_id not in position:
            raise RuntimeError(f"missing initial position for {row.vehicle_id}")
        request = request_by_order[str(row.order_id)]
        origin_lon, origin_lat = position[row.vehicle_id]
        stamp = pd.Timestamp(row.assignment_time).tz_convert("Asia/Shanghai")
        arc_id = hashlib.sha256(
            f"{row.order_id}|{row.vehicle_id}|{stamp.isoformat()}".encode()
        ).hexdigest()[:20]
        rows.append(
            {
                "arc_id": arc_id,
                "order_id": str(row.order_id),
                "vehicle_id": str(row.vehicle_id),
                "timestamp": stamp,
                "origin_lon_wgs84": float(origin_lon),
                "origin_lat_wgs84": float(origin_lat),
                "pickup_lon_wgs84": float(request.pickup_lon_wgs84),
                "pickup_lat_wgs84": float(request.pickup_lat_wgs84),
                "canonical_raw_time_s": float(row.valhalla_time_s),
                "canonical_corrected_eta_s": float(row.pickup_eta_s),
                "canonical_distance_m": float(row.pickup_route_distance_m),
            }
        )
        position[row.vehicle_id] = (
            float(row.completion_lon_wgs84),
            float(row.completion_lat_wgs84),
        )
    frame = pd.DataFrame(rows)
    if frame["arc_id"].duplicated().any() or len(frame) != len(assignments):
        raise RuntimeError("Q50 arc reconstruction identity failure")
    return frame


def build_arc_sample(root: str | Path) -> pd.DataFrame:
    root = Path(root).resolve()
    arcs = _all_q50_arcs(root)
    chosen: list[pd.DataFrame] = []
    known = arcs.loc[arcs["order_id"].eq(KNOWN_ORDER_ID)].copy()
    if len(known) != 1:
        raise RuntimeError("known divergent Q50 arc must resolve exactly once")
    known["sample_group"] = "KNOWN_DIVERGENT"
    chosen.append(known)
    used = set(known["arc_id"])

    remaining = arcs.loc[~arcs["arc_id"].isin(used)].copy()
    remaining["boundary_distance_s"] = (
        remaining["canonical_corrected_eta_s"] - 300.0
    ).abs()
    near = remaining.sort_values(
        ["boundary_distance_s", "arc_id"], kind="mergesort"
    ).head(20).copy()
    near["sample_group"] = "PATIENCE_BOUNDARY"
    chosen.append(near)
    used.update(near["arc_id"])

    remaining = arcs.loc[~arcs["arc_id"].isin(used)].copy()
    hour = remaining["timestamp"].dt.hour
    peak = remaining.loc[hour.isin([7, 8, 17, 18])].copy()
    peak["_priority"] = peak["arc_id"].map(lambda value: _priority("PEAK", value))
    peak = peak.sort_values(["_priority", "arc_id"], kind="mergesort").head(19)
    peak["sample_group"] = "PEAK_PERIOD"
    chosen.append(peak)
    used.update(peak["arc_id"])

    remaining = arcs.loc[~arcs["arc_id"].isin(used)].copy()
    remaining["_priority"] = remaining["arc_id"].map(
        lambda value: _priority("ORDINARY", value)
    )
    ordinary = remaining.sort_values(["_priority", "arc_id"], kind="mergesort").head(20)
    ordinary["sample_group"] = "ORDINARY_SUCCESS"
    chosen.append(ordinary)

    sample = pd.concat(chosen, ignore_index=True)
    sample["source_position_diagnostic"] = False
    diagnostic_ids = set(
        sample.sort_values(["sample_group", "arc_id"], kind="mergesort")
        .head(10)["arc_id"]
    )
    sample.loc[sample["arc_id"].isin(diagnostic_ids), "source_position_diagnostic"] = True
    keep = [
        "arc_id", "sample_group", "source_position_diagnostic", "order_id",
        "vehicle_id", "timestamp", "origin_lon_wgs84", "origin_lat_wgs84",
        "pickup_lon_wgs84", "pickup_lat_wgs84", "canonical_raw_time_s",
        "canonical_corrected_eta_s", "canonical_distance_m",
    ]
    sample = sample[keep].sort_values(["sample_group", "arc_id"], kind="mergesort")
    if len(sample) != 60 or sample["arc_id"].duplicated().any():
        raise RuntimeError("routing arc sample must contain 60 unique arcs")
    _atomic_csv(sample, root / OUTPUT_REL / "routing_arc_sample.csv")
    return sample


def _route_call(actor: Any, row: Any) -> dict[str, Any]:
    request = {
        "locations": [
            {"lon": float(row.origin_lon_wgs84), "lat": float(row.origin_lat_wgs84), "type": "break"},
            {"lon": float(row.pickup_lon_wgs84), "lat": float(row.pickup_lat_wgs84), "type": "break"},
        ],
        "costing": "auto", "units": "kilometers", "directions_type": "none",
        "date_time": {"type": 1, "value": pd.Timestamp(row.timestamp).strftime("%Y-%m-%dT%H:%M")},
    }
    try:
        trip = actor.route(request)["trip"]
        summary = trip["summary"]
        return {"success": True, "raw_time_s": float(summary["time"]), "distance_m": float(summary["length"]) * 1000.0, "returned_source_count": 1, "returned_target_count": 1, "failure_reason": ""}
    except Exception as exc:
        return {"success": False, "raw_time_s": np.nan, "distance_m": np.nan, "returned_source_count": 0, "returned_target_count": 0, "failure_reason": f"{type(exc).__name__}:{str(exc)[:160]}"}


def _matrix_call(actor: Any, row: Any, sources: list[tuple[float, float]], focal_index: int) -> dict[str, Any]:
    request = {
        "sources": [{"lon": float(lon), "lat": float(lat)} for lon, lat in sources],
        "targets": [{"lon": float(row.pickup_lon_wgs84), "lat": float(row.pickup_lat_wgs84)}],
        "costing": "auto", "units": "kilometers",
        "date_time": {"type": 1, "value": pd.Timestamp(row.timestamp).strftime("%Y-%m-%dT%H:%M")},
    }
    try:
        matrix = actor.matrix(request).get("sources_to_targets", [])
        returned_sources = len(matrix)
        returned_targets = len(matrix[0]) if matrix else 0
        cell = matrix[int(focal_index)][0]
        raw = float(cell["time"])
        if not np.isfinite(raw) or raw < 0.0:
            raise ValueError("invalid matrix time")
        return {"success": True, "raw_time_s": raw, "distance_m": float(cell["distance"]) * 1000.0, "returned_source_count": returned_sources, "returned_target_count": returned_targets, "failure_reason": ""}
    except Exception as exc:
        return {"success": False, "raw_time_s": np.nan, "distance_m": np.nan, "returned_source_count": locals().get("returned_sources", 0), "returned_target_count": locals().get("returned_targets", 0), "failure_reason": f"{type(exc).__name__}:{str(exc)[:160]}"}


def _context_sources(sample: pd.DataFrame, focal: Any, batch_size: int, position: str) -> tuple[list[tuple[float, float]], int]:
    pool = sample.loc[~sample["arc_id"].eq(focal.arc_id)].sort_values("arc_id")
    contexts = list(
        zip(pool["origin_lon_wgs84"].astype(float), pool["origin_lat_wgs84"].astype(float))
    )[: batch_size - 1]
    if position == "first":
        index = 0
    elif position == "last":
        index = batch_size - 1
    else:
        index = batch_size // 2
    contexts.insert(index, (float(focal.origin_lon_wgs84), float(focal.origin_lat_wgs84)))
    return contexts, index


def run_micro_audit(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    sample_path = root / OUTPUT_REL / "routing_arc_sample.csv"
    sample = pd.read_csv(sample_path, parse_dates=["timestamp"]) if sample_path.is_file() else build_arc_sample(root)
    actor = _actor(root)
    repeat_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    for focal in sample.itertuples(index=False):
        for repeat_id in range(REPEATS):
            repeat_rows.append({"arc_id": focal.arc_id, "routing_mode": "SCALAR_ROUTE", "batch_size": 1, "focal_source_position": "ONLY", "repeat_id": repeat_id, **_route_call(actor, focal)})
            repeat_rows.append({"arc_id": focal.arc_id, "routing_mode": "SINGLE_SOURCE_MATRIX", "batch_size": 1, "focal_source_position": "ONLY", "repeat_id": repeat_id, **_matrix_call(actor, focal, [(focal.origin_lon_wgs84, focal.origin_lat_wgs84)], 0)})
        for batch_size in BATCH_SIZES:
            positions = ("first", "middle", "last") if focal.source_position_diagnostic else ("middle",)
            for position in positions:
                sources, focal_index = _context_sources(sample, focal, batch_size, position)
                for repeat_id in range(REPEATS):
                    batch_rows.append({"arc_id": focal.arc_id, "routing_mode": "MULTI_SOURCE_MATRIX", "batch_size": batch_size, "focal_source_position": position.upper(), "focal_source_index": focal_index, "repeat_id": repeat_id, **_matrix_call(actor, focal, sources, focal_index)})
    repeat = pd.DataFrame(repeat_rows)
    batch = pd.DataFrame(batch_rows)
    _atomic_csv(repeat, root / OUTPUT_REL / "routing_repeatability.csv")
    _atomic_csv(batch, root / OUTPUT_REL / "routing_batch_context.csv")

    successful = repeat.loc[repeat["success"]].copy()
    ranges = successful.groupby(["routing_mode", "arc_id"])["raw_time_s"].agg(lambda values: float(values.max() - values.min()))
    m1 = repeat.loc[repeat["routing_mode"].eq("SINGLE_SOURCE_MATRIX")]
    scalar = repeat.loc[repeat["routing_mode"].eq("SCALAR_ROUTE")]
    m1_ref = m1.loc[m1["success"]].groupby("arc_id")["raw_time_s"].median()
    mb = batch.loc[batch["success"]].copy()
    mb["m1_reference_s"] = mb["arc_id"].map(m1_ref)
    mb["abs_difference_vs_m1_s"] = (mb["raw_time_s"] - mb["m1_reference_s"]).abs()
    position_spread = mb.groupby(["arc_id", "batch_size", "repeat_id"])["raw_time_s"].agg(lambda values: float(values.max() - values.min()))
    known_m1 = m1.loc[m1["arc_id"].eq(sample.loc[sample["order_id"].eq(KNOWN_ORDER_ID), "arc_id"].iloc[0])]
    decision = {
        "sample_size": int(len(sample)),
        "repeats_per_scalar_or_m1_arc": REPEATS,
        "scalar_failure_rate": float(1.0 - scalar["success"].mean()),
        "m1_failure_rate": float(1.0 - m1["success"].mean()),
        "mb_failure_rate": float(1.0 - batch["success"].mean()),
        "scalar_max_within_arc_range_s": float(ranges.get("SCALAR_ROUTE", pd.Series(dtype=float)).max()),
        "m1_max_within_arc_range_s": float(ranges.get("SINGLE_SOURCE_MATRIX", pd.Series(dtype=float)).max()),
        "mb_max_abs_difference_vs_m1_s": float(mb["abs_difference_vs_m1_s"].max()),
        "max_source_position_spread_s": float(position_spread.max()),
        "known_divergent_arc_m1_values_s": sorted(known_m1.loc[known_m1["success"], "raw_time_s"].unique().tolist()),
    }
    m1_pass = decision["m1_failure_rate"] == 0.0 and decision["m1_max_within_arc_range_s"] <= 1e-9 and len(decision["known_divergent_arc_m1_values_s"]) == 1
    scalar_pass = decision["scalar_failure_rate"] == 0.0 and decision["scalar_max_within_arc_range_s"] <= 1e-9
    decision["selected_routing_mode"] = "SINGLE_SOURCE_MATRIX" if m1_pass else ("SCALAR_ROUTE" if scalar_pass else "STOP_FOR_ROUTING_NONDETERMINISM")
    decision["performance_status"] = "PENDING"
    _atomic_json(decision, root / OUTPUT_REL / "routing_mode_decision.json")
    return decision


def run_performance_spike(root: str | Path, sample_size: int = 5000) -> pd.DataFrame:
    root = Path(root).resolve()
    arcs = _all_q50_arcs(root)
    arcs["_priority"] = arcs["arc_id"].map(lambda value: _priority("PERFORMANCE", value))
    arcs = arcs.sort_values(["_priority", "arc_id"], kind="mergesort").head(int(sample_size))
    process = psutil.Process()
    rows: list[dict[str, Any]] = []
    for mode in ("SINGLE_SOURCE_MATRIX", "SCALAR_ROUTE"):
        actor = _actor(root)
        failures = 0
        peak_rss = process.memory_info().rss
        started = time.perf_counter()
        for index, focal in enumerate(arcs.itertuples(index=False)):
            result = _matrix_call(actor, focal, [(focal.origin_lon_wgs84, focal.origin_lat_wgs84)], 0) if mode == "SINGLE_SOURCE_MATRIX" else _route_call(actor, focal)
            failures += int(not result["success"])
            if index % 100 == 0:
                peak_rss = max(peak_rss, process.memory_info().rss)
        elapsed = time.perf_counter() - started
        peak_rss = max(peak_rss, process.memory_info().rss)
        rows.append({"routing_mode": mode, "arc_count": int(len(arcs)), "wall_time_s": elapsed, "arcs_per_second": len(arcs) / elapsed, "peak_rss_mb": peak_rss / (1024.0 ** 2), "failure_count": failures, "failure_rate": failures / len(arcs), "cache_behavior": "DISABLED_UNIQUE_ARC_SPIKE"})
    frame = pd.DataFrame(rows)
    _atomic_csv(frame, root / OUTPUT_REL / "routing_mode_performance.csv")
    decision_path = root / OUTPUT_REL / "routing_mode_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    selected = decision["selected_routing_mode"]
    selected_row = frame.loc[frame["routing_mode"].eq(selected)]
    decision["performance_status"] = "PASS" if len(selected_row) == 1 and int(selected_row.iloc[0]["failure_count"]) == 0 else "FAIL"
    decision["performance"] = frame.to_dict(orient="records")
    _atomic_json(decision, decision_path)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-sample", "audit", "performance", "all"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--performance-sample-size", type=int, default=5000)
    args = parser.parse_args()
    if args.command in ("prepare-sample", "all"):
        print(build_arc_sample(args.root).groupby("sample_group").size().to_json())
    if args.command in ("audit", "all"):
        print(json.dumps(run_micro_audit(args.root), indent=2))
    if args.command in ("performance", "all"):
        print(run_performance_spike(args.root, args.performance_sample_size).to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
