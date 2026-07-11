"""Export a sampled actual matched-route oracle for planned-route auditing.

This product is audit-only. Actual links and actual entry times are post-trip
observations and are explicitly forbidden as deployable Stage2 inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order-od-root", type=Path, default=Path("stage0/output/order_od_audited"))
    parser.add_argument("--traversal-root", type=Path, default=Path("stage0/output/fast_link_traversals"))
    parser.add_argument("--strict-target-root", type=Path, default=Path("stage2/output/strict_targets"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/routes/actual_route_oracle"))
    parser.add_argument("--dates", default="20161009,20161010,20161011,20161012,20161013,20161014,20161015,20161016,20161017,20161018,20161019")
    parser.add_argument("--max-orders-per-day", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def parse_dates(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def read_parts(root: Path, date: str, orders: set[str]) -> pd.DataFrame:
    parts = []
    for path in sorted((root / f"day={date}").glob("*.parquet")):
        frame = pd.read_parquet(path)
        frame["order_id"] = frame["order_id"].astype(str)
        frame = frame[frame["order_id"].isin(orders)]
        if not frame.empty:
            parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "actual_route_oracle_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "route_source": "actual_matched_route_oracle",
            "deployable": False,
            "forbidden_model_inputs": ["actual_link_id", "actual_link_seq", "actual_link_entry_time", "actual_link_exit_time"],
            "days": {},
        }
    for number, date in enumerate(parse_dates(args.dates)):
        od = pd.read_parquet(args.order_od_root / f"day={date}.parquet")
        if "od_route_eligible" in od:
            od = od[od["od_route_eligible"]]
        if args.max_orders_per_day > 0 and len(od) > args.max_orders_per_day:
            od = od.sample(n=args.max_orders_per_day, random_state=args.seed + number)
        orders = set(od["order_id"].astype(str))
        actual = read_parts(args.traversal_root, date, orders)
        targets = read_parts(args.strict_target_root, date, orders) if (args.strict_target_root / f"day={date}").exists() else pd.DataFrame()
        if actual.empty:
            output = actual
        else:
            output = actual.sort_values(["order_id", "link_seq"]).rename(columns={
                "link_id": "actual_link_id", "link_seq": "actual_link_seq",
                "enter_time": "actual_link_entry_time", "exit_time": "actual_link_exit_time",
            })
            if not targets.empty:
                targets = targets.sort_values(["order_id", "link_seq"]).copy()
                targets["link_occurrence"] = targets.groupby(["order_id", "link_id"]).cumcount()
                output["link_occurrence"] = output.groupby(["order_id", "actual_link_id"]).cumcount()
                keep = [column for column in targets.columns if column.startswith("target_") or column in {"order_id", "link_id", "link_occurrence"}]
                output = output.merge(
                    targets[keep], left_on=["order_id", "actual_link_id", "link_occurrence"],
                    right_on=["order_id", "link_id", "link_occurrence"], how="left", validate="one_to_one",
                ).drop(columns="link_id", errors="ignore")
            output["route_source"] = "actual_matched_route_oracle"
            output["post_trip_oracle_only"] = True
        output.to_parquet(args.output_root / f"day={date}.parquet", index=False, compression="zstd")
        routed = int(output["order_id"].nunique()) if not output.empty else 0
        manifest["days"][date] = {
            "orders_input": int(len(od)), "orders_with_actual_route": routed,
            "actual_route_coverage": routed / max(len(od), 1), "rows": int(len(output)),
            "observed_link_ratio": float(output["traversal_quality"].ne("inferred_path").mean()) if "traversal_quality" in output else None,
        }
        print(f"actual route oracle day={date} {manifest['days'][date]}", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
