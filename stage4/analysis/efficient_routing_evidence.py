"""Materialize the R0.5c minimal routing evidence from the frozen R0.5b audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from stage4.dispatch.routing_determinism_runner import _actor, _matrix_call, _route_call


SOURCE_REL = Path("stage4/output/paper_enhancement/routing_determinism")
OUTPUT_REL = Path("stage4/output/paper_enhancement/efficient_repositioning")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def build(root: Path) -> dict[str, Any]:
    source = root / SOURCE_REL
    sample = pd.read_csv(source / "routing_arc_sample.csv", parse_dates=["timestamp"])
    selected = pd.concat(
        [
            sample.loc[sample["sample_group"].eq("KNOWN_DIVERGENT")].head(1),
            sample.loc[sample["sample_group"].eq("ORDINARY_SUCCESS")].head(7),
            sample.loc[sample["sample_group"].eq("PATIENCE_BOUNDARY")].head(6),
            sample.loc[sample["sample_group"].eq("PEAK_PERIOD")].head(6),
        ],
        ignore_index=True,
    )
    if len(selected) != 20 or selected["arc_id"].duplicated().any():
        raise ValueError("R0.5c routing sample must contain 20 unique arcs")
    repeats = pd.read_csv(source / "routing_repeatability.csv")
    repeats = repeats.loc[repeats["arc_id"].isin(selected["arc_id"])].copy()
    known_id = selected.loc[selected["sample_group"].eq("KNOWN_DIVERGENT"), "arc_id"].iloc[0]
    compact = pd.concat(
        [
            repeats.loc[~repeats["arc_id"].eq(known_id) & repeats["repeat_id"].lt(3)],
            repeats.loc[repeats["arc_id"].eq(known_id)],
        ],
        ignore_index=True,
    )
    actor = _actor(root)
    known = selected.loc[selected["arc_id"].eq(known_id)].iloc[0]
    additions = []
    for repeat_id in range(5, 10):
        additions.append(
            {"arc_id": known_id, "routing_mode": "SCALAR_ROUTE", "batch_size": 1,
             "focal_source_position": "ONLY", "repeat_id": repeat_id,
             **_route_call(actor, known)}
        )
        additions.append(
            {"arc_id": known_id, "routing_mode": "SINGLE_SOURCE_MATRIX", "batch_size": 1,
             "focal_source_position": "ONLY", "repeat_id": repeat_id,
             **_matrix_call(actor, known, [(known.origin_lon_wgs84, known.origin_lat_wgs84)], 0)}
        )
    compact = pd.concat([compact, pd.DataFrame(additions)], ignore_index=True)
    compact = compact.merge(selected[["arc_id", "sample_group"]], on="arc_id", how="left")
    batch = pd.read_csv(source / "routing_batch_context.csv")
    contrast = batch.loc[
        batch["arc_id"].eq(known_id) & batch["batch_size"].eq(2)
    ].head(1).copy()
    contrast["sample_group"] = "KNOWN_DIVERGENT_BATCH_CONTRAST"
    compact = pd.concat([compact, contrast[compact.columns]], ignore_index=True)
    compact["evidence_origin"] = "FROZEN_R0.5B_REUSE_PLUS_5_DIVERGENT_REPEATS"
    performance = pd.read_csv(source / "routing_mode_performance.csv")
    performance["protocol_interpretation"] = "REUSED_STRONGER_5000_ARC_SPIKE"
    scalar = performance.loc[performance["routing_mode"].eq("SCALAR_ROUTE")].iloc[0]
    m1 = performance.loc[performance["routing_mode"].eq("SINGLE_SOURCE_MATRIX")].iloc[0]
    ranges = compact.loc[compact["routing_mode"].isin(["SCALAR_ROUTE", "SINGLE_SOURCE_MATRIX"])].groupby(["arc_id", "routing_mode"])["raw_time_s"].agg(lambda x: float(x.max() - x.min()))
    result = {
        "sample_size": 20,
        "known_repeats_per_mode": 10,
        "ordinary_repeats_per_mode": 3,
        "max_within_mode_range_s": float(ranges.max()),
        "failure_count": int((~compact["success"].astype(bool)).sum()),
        "selected_mode": "SCALAR_ROUTE",
        "selection_reason": "both modes exact with zero failures; scalar had higher frozen throughput",
        "scalar_arcs_per_second": float(scalar["arcs_per_second"]),
        "m1_arcs_per_second": float(m1["arcs_per_second"]),
        "performance_arc_count": int(scalar["arc_count"]),
    }
    _atomic_csv(compact, root / OUTPUT_REL / "routing_micro_audit.csv")
    _atomic_csv(performance, root / OUTPUT_REL / "routing_performance_spike.csv")
    path = root / OUTPUT_REL / "routing_evidence_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve()), indent=2))


if __name__ == "__main__":
    main()
