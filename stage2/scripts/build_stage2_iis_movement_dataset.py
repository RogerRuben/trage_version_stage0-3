"""Convert planned routes to native movement-level IIS prediction rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planned-causal-root", type=Path, default=Path("stage2/output/planned_route_causal_dataset"))
    parser.add_argument("--actual-movement-root", type=Path, default=Path("stage0/output/fast_turn_movements"))
    parser.add_argument("--strict-target-root", type=Path, default=Path("stage2/output/strict_targets"))
    parser.add_argument("--roads", type=Path, default=Path("map_data/xian_2017/xian_2017_core_roads.parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/iis_movement_causal_dataset"))
    parser.add_argument("--dates", default="20161009,20161010,20161011,20161012,20161013,20161014,20161015,20161016,20161017,20161018,20161019")
    return parser.parse_args()


def load_actual(root: Path, target_root: Path, date: str) -> pd.DataFrame:
    movements = pd.concat([pd.read_parquet(path) for path in sorted((root / f"day={date}").glob("*.parquet"))], ignore_index=True)
    targets = pd.concat([pd.read_parquet(path, columns=[
        "order_id", "link_id", "link_seq", "target_iis_raw", "target_iis_pct", "target_iis_tail90_raw",
        "target_iis_tail90_pct", "target_iis_uncertainty", "target_iis_valid",
    ]) for path in sorted((target_root / f"day={date}").glob("*.parquet"))], ignore_index=True)
    actual = movements.merge(
        targets.rename(columns={"link_seq": "movement_seq", "link_id": "target_to_link_id"}),
        on=["order_id", "movement_seq"], how="left", validate="one_to_one",
    )
    return actual


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    roads = gpd.read_parquet(args.roads)[["link_id", "from_node", "to_node"]].copy()
    roads["link_id"] = roads["link_id"].astype(str)
    lookup = roads.set_index("link_id")
    degree = pd.concat([roads["from_node"], roads["to_node"]]).value_counts()
    manifest = {"native_unit": "from_link-node-to_link movement", "days": {}}
    for date in [part.strip() for part in args.dates.split(",") if part.strip()]:
        route = pd.read_parquet(args.planned_causal_root / f"day={date}.parquet").sort_values(["order_id", "planned_link_seq"])
        movement = route.copy()
        movement["from_link_id"] = movement.groupby("order_id")["planned_link_id"].shift()
        movement = movement[movement["from_link_id"].notna()].copy()
        movement["to_link_id"] = movement["planned_link_id"]
        movement["movement_seq"] = movement["planned_link_seq"]
        shared_nodes, degrees = [], []
        for from_link, to_link in zip(movement["from_link_id"], movement["to_link_id"]):
            if from_link not in lookup.index or to_link not in lookup.index:
                shared_nodes.append(np.nan); degrees.append(np.nan); continue
            left, right = lookup.loc[from_link], lookup.loc[to_link]
            common = set([int(left.from_node), int(left.to_node)]) & set([int(right.from_node), int(right.to_node)])
            node = next(iter(common)) if common else np.nan
            shared_nodes.append(node)
            degrees.append(degree.get(node, np.nan) if np.isfinite(node) else np.nan)
        movement["node_id"] = shared_nodes
        movement["planned_node_degree"] = degrees
        actual = load_actual(args.actual_movement_root, args.strict_target_root, date)
        movement = movement.drop(columns=[column for column in movement.columns if column.startswith("target_iis_")], errors="ignore")
        movement["movement_occurrence"] = movement.groupby(["order_id", "from_link_id", "to_link_id"]).cumcount()
        actual = actual.sort_values(["order_id", "movement_seq"]).copy()
        actual["movement_occurrence"] = actual.groupby(["order_id", "from_link_id", "to_link_id"]).cumcount()
        label_columns = [
            "order_id", "from_link_id", "to_link_id", "turn_angle", "turn_type", "node_degree", "junction_complexity",
            "target_iis_raw", "target_iis_pct", "target_iis_tail90_raw", "target_iis_tail90_pct",
            "target_iis_uncertainty", "target_iis_valid",
        ]
        label_columns.append("movement_occurrence")
        movement = movement.merge(
            actual[label_columns], on=["order_id", "from_link_id", "to_link_id", "movement_occurrence"],
            how="left", validate="many_to_one",
        )
        movement["iis_applicable"] = movement["planned_node_degree"].ge(3)
        movement["iis_observed"] = movement["target_iis_raw"].notna()
        movement["planned_link_id"] = movement["to_link_id"]
        movement.to_parquet(args.output_root / f"day={date}.parquet", index=False, compression="zstd")
        manifest["days"][date] = {
            "rows": len(movement), "applicable_ratio": float(movement["iis_applicable"].mean()),
            "observed_ratio": float(movement["iis_observed"].mean()),
        }
        print(f"IIS movements day={date} {manifest['days'][date]}", flush=True)
    manifest.update({
        "applicability": "planned shared-node degree >= 3",
        "severity_target": "realized IIS only when planned from/to movement matches actual movement",
        "missing_iis_filled_zero": False,
    })
    (args.output_root / "iis_movement_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
