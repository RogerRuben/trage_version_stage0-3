"""Package audited Stage 0 smoke products into explicit canonical day files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0-root", type=Path, required=True)
    parser.add_argument("--dates", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def concatenate(files: list[Path], output: Path) -> int:
    if not files:
        raise FileNotFoundError(f"no source partitions for {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    rows = 0
    try:
        for path in files:
            table = pq.read_table(path)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
            rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    return rows


def main() -> None:
    args = arguments()
    dates = [item.strip() for item in args.dates.split(",") if item.strip()]
    result = {"status": "PASS", "dates": {}, "files": []}
    for date in dates:
        day = {}
        collections = {
            "link_traversals": args.stage0_root / "hmm_link_traversals" / f"day={date}",
            "turn_movements": args.stage0_root / "hmm_turn_movements" / f"day={date}",
        }
        for role, directory in collections.items():
            files = sorted(directory.glob("*.parquet"))
            target = args.output_root / role / f"day={date}.parquet"
            day[f"{role}_rows"] = concatenate(files, target)
            result["files"].append({"role": f"{role}_{date}", "path": target.as_posix()})

        route_files = sorted((args.stage0_root / "hmm_route_parts" / f"day={date}").glob("*.parquet"))
        routes = pd.concat([pd.read_parquet(path) for path in route_files], ignore_index=True)
        invalid = routes.groupby("order_id").transition_path_status.apply(lambda values: values.eq("gap").any())
        routes["route_direction_valid"] = ~routes.order_id.map(invalid).fillna(True)
        route_target = args.output_root / "routes" / f"day={date}.parquet"
        route_target.parent.mkdir(parents=True, exist_ok=True)
        routes.to_parquet(route_target, index=False, compression="zstd")
        result["files"].append({"role": f"routes_{date}", "path": route_target.as_posix()})
        day["route_rows"] = int(len(routes))
        day["route_valid_order_share"] = float((~invalid).mean())

        orders = pd.read_parquet(args.stage0_root / "order_base" / f"day={date}.parquet")
        orders["route_direction_valid"] = orders.order_id.map(~invalid).fillna(False)
        order_target = args.output_root / "orders" / f"day={date}.parquet"
        order_target.parent.mkdir(parents=True, exist_ok=True)
        orders.to_parquet(order_target, index=False, compression="zstd")
        result["files"].append({"role": f"orders_{date}", "path": order_target.as_posix()})
        day["orders"] = int(len(orders))
        result["dates"][date] = day
    manifest_path = args.output_root / "package_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
