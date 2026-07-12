"""Convert route-conditioned links to native movement-level IIS rows.

The production input is the same 15k/day route-conditioned estimated-time
dataset used by RC-MSTNet.  Older planned-route files are still supported via
column aliases, but the canonical output key is now explicit and stable:

date | order_id | movement_seq | from_link_id | node_id | to_link_id
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-conditioned-root", type=Path, default=Path("stage2/output/route_conditioned_dataset_15k/estimated_time_daily"))
    parser.add_argument("--planned-causal-root", type=Path, default=None, help="Deprecated alias for older planned-route files.")
    parser.add_argument("--actual-movement-root", type=Path, default=Path("stage0/output/fast_turn_movements"))
    parser.add_argument("--strict-target-root", type=Path, default=Path("stage2/output/strict_targets"))
    parser.add_argument("--roads", type=Path, default=Path("map_data/xian_2017/xian_2017_core_roads.parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/iis_movement_causal_dataset"))
    parser.add_argument("--dates", default="20161009,20161010,20161011,20161012,20161013,20161014,20161015,20161016,20161017,20161018,20161019")
    return parser.parse_args()


ROUTE_COLUMNS = [
    "order_id", "driver_id", "date", "route_link_id", "route_link_seq", "route_link_count",
    "route_link_length_m", "position_ratio", "distance_to_destination_ratio",
    "estimated_link_entry_time", "prediction_time_bin", "prediction_hour", "prediction_weekday",
    "prediction_is_weekend", "road_class", "area_grid", "endpoint_degree", "link_fragmentation",
    "minor_road", "activity_intensity_index", "strict_availability_check",
    "route_conditioned_time_check", "feature_availability_timestamp",
    "target_iis_raw", "target_iis_pct", "target_iis_tail90_raw", "target_iis_tail90_pct",
    "target_iis_uncertainty", "target_iis_valid",
]


def existing_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def canonical_route_path(args: argparse.Namespace, date: str) -> Path:
    root = args.planned_causal_root or args.route_conditioned_root
    direct = root / f"day={date}.parquet"
    if direct.exists():
        return direct
    nested = root / "estimated_time_daily" / f"day={date}.parquet"
    if nested.exists():
        return nested
    raise FileNotFoundError(f"No route-conditioned day file found for {date} under {root}")


def read_route_day(path: Path) -> pd.DataFrame:
    available = existing_columns(path)
    columns = [column for column in ROUTE_COLUMNS if column in available]
    columns += [column for column in available if column.startswith("poi_density_100m_")]
    columns += [
        column for column in available
        if any(column.startswith(prefix) for prefix in ["rolling_", "link_recent_", "area_recent_", "network_recent_", "upstream_recent_", "downstream_recent_"])
        and "timestamp" not in column
    ]
    frame = pd.read_parquet(path, columns=list(dict.fromkeys(columns)))
    rename = {
        "route_link_id": "planned_link_id",
        "route_link_seq": "planned_link_seq",
        "route_link_count": "planned_route_link_count",
        "route_link_length_m": "planned_link_length_m",
        "prediction_time_bin": "estimated_time_bin",
    }
    frame = frame.rename(columns={old: new for old, new in rename.items() if old in frame.columns})
    if "estimated_time_bin" not in frame.columns and "prediction_hour" in frame.columns:
        frame["estimated_time_bin"] = frame["prediction_hour"].astype(str)
    return frame


def load_actual(root: Path, target_root: Path, date: str) -> pd.DataFrame:
    movement_paths = sorted((root / f"day={date}").glob("*.parquet"))
    target_paths = sorted((target_root / f"day={date}").glob("*.parquet"))
    if not movement_paths or not target_paths:
        return pd.DataFrame()
    movements = pd.concat([pd.read_parquet(path) for path in movement_paths], ignore_index=True)
    targets = pd.concat([pd.read_parquet(path, columns=[
        "order_id", "link_id", "link_seq", "target_iis_raw", "target_iis_pct", "target_iis_tail90_raw",
        "target_iis_tail90_pct", "target_iis_uncertainty", "target_iis_valid",
    ]) for path in target_paths], ignore_index=True)
    actual = movements.merge(
        targets.rename(columns={"link_seq": "movement_seq", "link_id": "target_to_link_id"}),
        on=["order_id", "movement_seq"], how="left", validate="one_to_one",
    )
    return actual


def attach_topology(route: pd.DataFrame, roads: pd.DataFrame, degree: pd.Series) -> pd.DataFrame:
    route = route.sort_values(["order_id", "planned_link_seq"]).copy()
    movement = route.copy()
    movement["from_link_id"] = movement.groupby("order_id")["planned_link_id"].shift()
    movement = movement[movement["from_link_id"].notna()].copy()
    movement["to_link_id"] = movement["planned_link_id"]
    movement["movement_seq"] = movement["planned_link_seq"]

    left = roads.rename(columns={"link_id": "from_link_id", "from_node": "from_a", "to_node": "from_b"})
    right = roads.rename(columns={"link_id": "to_link_id", "from_node": "to_a", "to_node": "to_b"})
    movement = movement.merge(left, on="from_link_id", how="left")
    movement = movement.merge(right, on="to_link_id", how="left")
    conditions = [
        movement["from_a"].eq(movement["to_a"]),
        movement["from_a"].eq(movement["to_b"]),
        movement["from_b"].eq(movement["to_a"]),
        movement["from_b"].eq(movement["to_b"]),
    ]
    choices = [movement["from_a"], movement["from_a"], movement["from_b"], movement["from_b"]]
    movement["node_id"] = np.select(conditions, choices, default=np.nan)
    movement["planned_node_degree"] = movement["node_id"].map(degree).astype("float64")
    movement["movement_topology_valid"] = movement["node_id"].notna()
    return movement.drop(columns=["from_a", "from_b", "to_a", "to_b"], errors="ignore")


def add_movement_key(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    seq = pd.to_numeric(frame["movement_seq"], errors="coerce").astype("Int64").astype(str)
    node = pd.to_numeric(frame["node_id"], errors="coerce").round().astype("Int64").astype(str)
    frame["movement_key"] = (
        frame["date"].astype(str) + "|"
        + frame["order_id"].astype(str) + "|"
        + seq + "|"
        + frame["from_link_id"].astype(str) + "|"
        + node + "|"
        + frame["to_link_id"].astype(str)
    )
    return frame


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    roads = gpd.read_parquet(args.roads)[["link_id", "from_node", "to_node"]].copy()
    roads["link_id"] = roads["link_id"].astype(str)
    roads["from_node"] = pd.to_numeric(roads["from_node"], errors="coerce")
    roads["to_node"] = pd.to_numeric(roads["to_node"], errors="coerce")
    degree = pd.concat([roads["from_node"], roads["to_node"]]).dropna().astype("int64").value_counts()
    manifest = {"native_unit": "from_link-node-to_link movement", "days": {}}
    for date in [part.strip() for part in args.dates.split(",") if part.strip()]:
        route = read_route_day(canonical_route_path(args, date))
        movement = attach_topology(route, roads, degree)
        actual = load_actual(args.actual_movement_root, args.strict_target_root, date)
        movement = movement.drop(columns=[column for column in movement.columns if column.startswith("target_iis_")], errors="ignore")
        movement["movement_occurrence"] = movement.groupby(["order_id", "from_link_id", "to_link_id"]).cumcount()
        if not actual.empty:
            actual = actual.sort_values(["order_id", "movement_seq"]).copy()
            actual["movement_occurrence"] = actual.groupby(["order_id", "from_link_id", "to_link_id"]).cumcount()
            label_columns = [
                "order_id", "from_link_id", "to_link_id", "turn_angle", "turn_type", "node_degree", "junction_complexity",
                "target_iis_raw", "target_iis_pct", "target_iis_tail90_raw", "target_iis_tail90_pct",
                "target_iis_uncertainty", "target_iis_valid", "movement_occurrence",
            ]
            movement = movement.merge(
                actual[[column for column in label_columns if column in actual.columns]],
                on=["order_id", "from_link_id", "to_link_id", "movement_occurrence"],
                how="left", validate="many_to_one",
            )
        else:
            for column in ["turn_angle", "turn_type", "node_degree", "junction_complexity", "target_iis_raw", "target_iis_pct", "target_iis_tail90_raw", "target_iis_tail90_pct", "target_iis_uncertainty", "target_iis_valid"]:
                movement[column] = np.nan
        movement["iis_applicable"] = movement["planned_node_degree"].ge(3) & movement["movement_topology_valid"]
        movement["iis_observed"] = movement["target_iis_raw"].notna()
        movement = add_movement_key(movement)
        movement.to_parquet(args.output_root / f"day={date}.parquet", index=False, compression="zstd")
        manifest["days"][date] = {
            "route_orders": int(route["order_id"].nunique()),
            "movement_orders": int(movement["order_id"].nunique()),
            "route_links": int(len(route)),
            "rows": len(movement),
            "valid_topology_ratio": float(movement["movement_topology_valid"].mean()),
            "applicable_ratio": float(movement["iis_applicable"].mean()),
            "observed_ratio": float(movement["iis_observed"].mean()),
        }
        print(f"IIS movements day={date} {manifest['days'][date]}", flush=True)
    manifest.update({
        "applicability": "planned shared-node degree >= 3",
        "severity_target": "realized IIS only when planned from/to movement matches actual movement",
        "missing_iis_filled_zero": False,
        "canonical_key": "date|order_id|movement_seq|from_link_id|node_id|to_link_id",
    })
    (args.output_root / "iis_movement_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
