"""Projected, read-only postprocessing for the fixed D1--D3 figure design.

No planner, simulator, predictor, routing engine, external map, or new condition
is run. Coordinate-level figure tables remain in ignored stage4/output.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

BASE = Path("stage4/docs/capability_chain_planning/layout_confirmation_v1")
DOC = BASE / "visualization_v1"
PRIVATE = Path("stage4/output/capability_chain_planning/layout_confirmation_v1/visualization_v1")
OLD = Path("stage4/output/capability_chain_planning/layout_comparison_300_600_v1")
NEW = Path("stage4/output/capability_chain_planning/layout_confirmation_v1")
GROUPS = ("COMPATIBLE_HOT_COMMON_CONTROL", "OPPORTUNITY_JOINT_COMMON_CONTROL")
CASES = (("5%", "Q05_REFERENCE_LABELS"), ("原10%参考", "Q10_REFERENCE_EXISTING_NOT_RERUN"),
         ("新10%种子", "Q10_LABEL_SEED2"), ("20%", "Q20_REFERENCE_LABELS"))
STATES = ("C", "HV", "UNSERVED")
IDENTITY = Path("stage3/output/odd_tod/s4/test31_route_identity_resolution.parquet")
EDGES = Path("stage3/output/odd_tod/s2a/stage3_full_network_edges.parquet")
POSITIONS = Path("stage4/input/replay_foundation/stage4_order_replay_base.parquet")
REFERENCES = Path("stage4/output/symmetric_flexibility_v1/input/test31_research_routes.parquet")
BATCH = 32768


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def pair_dir(key):
    return OLD if key == CASES[1][1] else NEW / key / "native"


def states(path):
    values = pq.read_table(path, columns=["order_id", "vehicle_type", "completed"]).to_pydict()
    result = {}
    for order, kind, complete in zip(values["order_id"], values["vehicle_type"], values["completed"]):
        if order in result or complete not in (True, False):
            raise ValueError("duplicate order or missing completion state")
        if complete and kind not in ("AV", "HV"):
            raise ValueError("completed request has no known vehicle type")
        result[order] = ("C" if kind == "AV" else "HV") if complete else "UNSERVED"
    if len(result) != 28367:
        raise ValueError("request population differs from the fixed figure design")
    return result


def coordinates(root, orders):
    values = pq.read_table(root / POSITIONS, columns=["order_id", "pickup_lon_wgs84", "pickup_lat_wgs84",
        "dropoff_lon_wgs84", "dropoff_lat_wgs84"]).to_pydict()
    result = {}
    invalid = set()
    for order, *position in zip(*(values[name] for name in values)):
        if order not in orders:
            continue
        if not all(value is not None and np.isfinite(value) for value in position):
            invalid.add(order)
            continue
        position = tuple(float(value) for value in position)
        if order in result and result[order] != position:
            raise ValueError("profile copies disagree on request coordinates")
        result[order] = position
    if invalid.intersection(result):
        raise ValueError("profile copies disagree on coordinate availability")
    return result


def deployment(root, source, condition):
    # Reconstruct the existing supply labels only, not an offline placement.
    import pandas as pd
    from stage4.dispatch.controlled_fleet import build_controlled_fleet

    start = pd.Timestamp("2016-10-31", tz="Asia/Shanghai")
    protocol = json.loads((root / OLD / GROUPS[0] / "summary.json").read_text(encoding="utf-8"))["frozen_protocol"]
    supply = condition["supply"]
    fleet = build_controlled_fleet(root, benchmark_start=start,
        simulation_end=start + pd.Timedelta(seconds=protocol["physical_drain_limit_s"]),
        requested_q_a=supply["requested_q_a"], seed=supply["seed"])
    if fleet.accounting["type_label_sha256"] != supply["type_label_sha256"]:
        raise ValueError("reconstructed C labels do not match the completed run")
    hours = {r.vehicle_id: float(r.day_vehicle_hours) for r in fleet.scenario_fleet.itertuples(index=False)
             if r.vehicle_type == "AV"}
    layouts = json.loads((root / source).read_text(encoding="utf-8"))["placements"]
    totals = [defaultdict(float), defaultdict(float)]
    for index, mode in enumerate(("compatible_hot", "opportunity_joint")):
        if layouts[mode].keys() != hours.keys():
            raise ValueError("saved birth placements and C windows differ")
        for vehicle, value in layouts[mode].items():
            lon, lat = map(float, value["position"])
            key = (str(value["site_id"]), int(value["node_id"]), lon, lat)
            totals[index][key] += hours[vehicle]
    expected = float(supply["achieved_av_vehicle_hours"])
    if any(abs(sum(value.values()) - expected) > 1e-8 for value in totals):
        raise ValueError("birth hours do not conserve the C normalization denominator")
    rows = [dict(site_id=k[0], node_id=k[1], lon_wgs84=k[2], lat_wgs84=k[3],
        C_hours_H=totals[0].get(k, 0.), C_hours_J=totals[1].get(k, 0.),
        delta_hours=totals[1].get(k, 0.) - totals[0].get(k, 0.)) for k in sorted(set(totals[0]) | set(totals[1]))]
    del fleet, layouts
    gc.collect()
    return rows


def route_coverage(root, before, after, references):
    orders = set(before)
    c_orders = {o for o in orders if before[o] == "C" or after[o] == "C"}
    footprints = {o: set() for o in c_orders}
    geometry_uids = set()
    order_seen = set()
    types, skipped = Counter(), Counter()
    key_geometry = {}
    selected = pa.array(sorted(orders), type=pa.string())
    columns = ["order_id", "route_token_type", "resolved_stage3_edge_uid", "physical_forward_stage3_edge_uid"]
    for batch in pq.ParquetFile(root / IDENTITY).iter_batches(batch_size=BATCH, columns=columns):
        table = pa.Table.from_batches([batch])
        table = table.filter(pc.is_in(table["order_id"], value_set=selected))
        values = table.to_pydict()
        for order, kind, uid, physical in zip(*(values[name] for name in columns)):
            order_seen.add(order)
            types[kind] += 1
            if references.get(order) != f"ORIGINAL:{order}":
                skipped["unbound_route_reference_tokens"] += 1
                continue
            geometry = uid if kind == "FULL_NETWORK_EDGE" else physical if kind == "HISTORICAL_REVERSE_OVERLAY" else None
            if not geometry:
                skipped["missing_identity_tokens"] += 1
                if order in c_orders:
                    skipped["C_missing_identity_tokens"] += 1
                continue
            geometry_uids.add(geometry)
            if order in c_orders:
                key = (kind, geometry)
                footprints[order].add(key)
                key_geometry[key] = geometry
    counts = [Counter(), Counter()]
    for order, edges in footprints.items():
        for index, source in enumerate((before, after)):
            if source[order] == "C":
                counts[index].update(edges)
    rows = [dict(route_token_type=k[0], geometry_edge_uid=key_geometry[k],
        reverse_geometry=k[0] == "HISTORICAL_REVERSE_OVERLAY",
        C_orders_H=counts[0].get(k, 0), C_orders_J=counts[1].get(k, 0),
        delta_orders=counts[1].get(k, 0) - counts[0].get(k, 0))
        for k in sorted(set(counts[0]) | set(counts[1]))]
    scope = dict(cohort_token_type_counts=dict(types), skipped_tokens=dict(skipped),
        cohort_orders_without_identity=len(orders - order_seen),
        C_orders_union=len(c_orders), C_orders_without_bound_identity=sum(not v for v in footprints.values()),
        cohort_orders_without_original_reference=sum(references.get(o) != f"ORIGINAL:{o}" for o in orders),
        C_orders_without_original_reference=sum(references.get(o) != f"ORIGINAL:{o}" for o in c_orders),
        order_edge_pairs=sum(len(value) for value in footprints.values()),
        per_order_per_directed_identity_deduplicated=True)
    return rows, geometry_uids, scope


def spatial(root, condition, before, after):
    coords = coordinates(root, set(before))
    references = pq.read_table(root / REFERENCES, columns=["order_id", "selected_route_reference"]).to_pydict()
    references = dict(zip(references["order_id"], references["selected_route_reference"]))
    locations = deployment(root, Path("stage4/output/capability_chain_planning/layout_300_600_v1/placements.json"), condition)
    coverage, geometry_uids, scope = route_coverage(root, before, after, references)
    change_positions = Counter()
    changes, missing_change = Counter(), Counter()
    for order in before:
        kind = "GAIN" if before[order] == "UNSERVED" and after[order] != "UNSERVED" else (
               "LOSS" if before[order] != "UNSERVED" and after[order] == "UNSERVED" else None)
        if kind:
            changes[kind] += 1
            if order in coords:
                change_positions[(kind, *coords[order][:2])] += 1
            else:
                missing_change[kind] += 1
    position_rows = [dict(change_type=k[0], lon_wgs84=k[1], lat_wgs84=k[2], order_count=count)
                     for k, count in sorted(change_positions.items())]
    points = np.asarray([p for value in coords.values() for p in (value[:2], value[2:])], dtype=float)
    extent = [float(points[:, 0].min()), float(points[:, 1].min()),
              float(points[:, 0].max()), float(points[:, 1].max())]
    for row in locations:
        extent = [min(extent[0], row["lon_wgs84"]), min(extent[1], row["lat_wgs84"]),
                  max(extent[2], row["lon_wgs84"]), max(extent[3], row["lat_wgs84"])]
    found = set()
    for batch in pq.ParquetFile(root / EDGES).iter_batches(batch_size=BATCH, columns=["stage3_edge_uid", "geometry"]):
        for uid, encoded in zip(*batch.to_pydict().values()):
            if uid not in geometry_uids:
                continue
            line = np.asarray(json.loads(encoded), dtype=float)
            if len(line) < 2 or not np.isfinite(line).all():
                continue
            found.add(uid)
            extent = [min(extent[0], float(line[:, 0].min())), min(extent[1], float(line[:, 1].min())),
                      max(extent[2], float(line[:, 0].max())), max(extent[3], float(line[:, 1].max()))]
    for row in coverage:
        row["geometry_available"] = row["geometry_edge_uid"] in found
    scope.update(condition=CASES[1][1], cohort_orders=len(before), coordinates_missing=len(set(before) - set(coords)),
        service_changes=dict(changes), changes_missing_coordinates=dict(missing_change),
        C_hours_H=sum(r["C_hours_H"] for r in locations), C_hours_J=sum(r["C_hours_J"] for r in locations),
        location_delta_sum_h=sum(r["delta_hours"] for r in locations),
        saved_location_keys=len(locations), directed_coverage_keys=len(coverage),
        missing_coverage_geometry_keys=sum(not r["geometry_available"] for r in coverage),
        changed_keys=sum(r["delta_orders"] != 0 for r in coverage),
        cohort_geometry_uids=len(geometry_uids), cohort_geometry_uids_missing=len(geometry_uids - found),
        wgs84_extent=extent, extent_uses_common_routes_and_both_layouts=True)
    write_csv(root / PRIVATE / "deployment_delta.csv", locations, list(locations[0]))
    write_csv(root / PRIVATE / "route_coverage_delta.csv", coverage, list(coverage[0]))
    write_csv(root / PRIVATE / "service_change_origins.csv", position_rows,
              ["change_type", "lon_wgs84", "lat_wgs84", "order_count"])
    write_json(root / PRIVATE / "map_extent.json", dict(wgs84_extent=extent))
    return scope


def run(root):
    started = perf_counter()
    root = Path(root).resolve()
    target = root / DOC / "data_summary.json"
    if target.exists():
        raise ValueError("existing figure data retained; do not silently overwrite")
    aggregate = json.loads((root / BASE / "six_case_results.json").read_text(encoding="utf-8"))
    effects, decomposition, transitions, sources = [], [], [], {}
    common, reference = None, None
    for label, key in CASES:
        condition = aggregate["conditions"][key]
        pair = []
        for group in GROUPS:
            path = pair_dir(key) / group / "cohort_outcomes.parquet"
            pair.append(states(root / path))
            sources[path.as_posix()] = sha(root / path)
        before, after = pair
        if before.keys() != after.keys() or (common is not None and set(before) != common):
            raise ValueError("paired/common request populations differ")
        common = set(before)
        counts = Counter((before[o], after[o]) for o in before)
        values = condition["joint_minus_compatible"]
        gained = sum(counts[("UNSERVED", kind)] for kind in STATES[:2])
        lost = sum(counts[(kind, "UNSERVED")] for kind in STATES[:2])
        if gained - lost != values["completed_orders"]:
            raise ValueError("paired service transitions do not reconcile")
        for index, source in enumerate(pair):
            expected = condition["groups"][GROUPS[index]]
            if sum(s == "C" for s in source.values()) != expected["served_C"]:
                raise ValueError("C state marginal differs from the completed aggregate")
        transitions.extend(dict(condition=key, label=label, before=a, after=b, orders=counts[(a, b)])
                           for a in STATES for b in STATES)
        effects.append(dict(condition=key, label=label, existing_reference=condition["existing_reference"],
            **{name: values[name] for name in ("C_loaded_utilization_pp", "completed_orders", "total_empty_km")},
            service_gained=gained, service_lost=lost, C_orders_delta=values["served_C"]))
        parts = values["activation_product_decomposition"]
        if abs(parts["working_count_component_h"] + parts["conditional_intensity_component_h"] - values["C_loaded_h"]) > 1e-8:
            raise ValueError("loaded-hour accounting decomposition differs")
        decomposition.append(dict(condition=key, label=label, working_count_h=parts["working_count_component_h"],
            conditional_intensity_h=parts["conditional_intensity_component_h"], net_loaded_h=values["C_loaded_h"]))
        if key == CASES[1][1]:
            reference = pair
        print(json.dumps(dict(step="paired", condition=key, gained=gained, lost=lost)), flush=True)
        del before, after, pair
        gc.collect()
    scope = spatial(root, aggregate["conditions"][CASES[1][1]], *reference)
    write_csv(root / DOC / "effects.csv", effects, list(effects[0]))
    write_csv(root / DOC / "loaded_decomposition.csv", decomposition, list(decomposition[0]))
    write_csv(root / DOC / "service_transitions.csv", transitions, list(transitions[0]))
    for path in (BASE / "six_case_results.json", BASE / "visualization_design_v1.md", IDENTITY, EDGES, POSITIONS, REFERENCES,
                 Path("stage4/input/replay_foundation/replay_fleet_template.parquet"),
                 Path("stage4/output/capability_chain_planning/layout_300_600_v1/placements.json")):
        sources[path.as_posix()] = sha(root / path)
    memory = psutil.Process().memory_info()
    result = dict(status="FIGURE_DATA_COMPLETE_NOT_NEW_EXPERIMENT", source_files_sha256=sources,
        data_script_sha256=sha(__file__), spatial=scope, source_population=28367,
        producer_commits_are_preserved=True, simulation_invoked=False, inference_invoked=False, routing_invoked=False,
        gpu_used=False, dense_fleet_time_edge_array=False, raw_coordinates_exported_to_git=False,
        runtime_s=perf_counter() - started,
        peak_rss_mib=max(memory.rss, getattr(memory, "peak_wset", memory.rss)) / 2**20)
    write_json(target, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = run(args.root)
    print(json.dumps(dict(status=result["status"], runtime_s=result["runtime_s"],
                          peak_rss_mib=result["peak_rss_mib"], spatial=result["spatial"])), flush=True)


if __name__ == "__main__":
    main()
