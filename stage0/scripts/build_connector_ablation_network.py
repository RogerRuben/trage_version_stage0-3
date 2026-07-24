"""Create a diagnostic v4 view with graph-only connectors disabled."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    roads = gpd.read_parquet(args.roads)
    connector = roads.topology_connector.fillna(False).astype(bool)
    kept = roads.loc[~connector].copy()
    nodes = gpd.read_parquet(args.nodes)
    args.output_root.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(args.output_root / "roads.parquet", index=False)
    nodes.to_parquet(args.output_root / "nodes.parquet", index=False)
    audit = {
        "status": "DIAGNOSTIC_ONLY",
        "source_network": args.roads.as_posix(),
        "connectors_removed": int(connector.sum()),
        "links_retained": int(len(kept)),
        "purpose": "connector contribution ablation; not a new network version",
    }
    (args.output_root / "ablation_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
