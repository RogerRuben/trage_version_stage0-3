"""Build validation-only Q0.9 service residuals for preassignment.

The default assets correspond to the rolling fold whose validation day is
2016-10-22 and test day is 2016-10-23.  No test-day duration or prediction is
read by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage4.simulator_v3.preassignment.safe_release_buffer import (
    RESIDUAL_DEFINITION,
    SafeReleaseBufferResolver,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-date", default="20161022")
    parser.add_argument(
        "--route-conditioned",
        type=Path,
        default=Path("stage2/output/route_conditioned_dataset_15k/estimated_time_daily/day=20161022.parquet"),
    )
    parser.add_argument(
        "--validation-predictions",
        type=Path,
        default=Path("stage2/output/deep_v3_stage3_rolling_100k/predictions/rc_mstnet/fold=7/validation_predictions.parquet"),
    )
    parser.add_argument(
        "--validation-orders",
        type=Path,
        default=Path("stage0/output/order_od_audited/day=20161022.parquet"),
    )
    parser.add_argument(
        "--zone-system",
        type=Path,
        default=Path("stage4/data/decoupled_abm/operational_zone_system.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("stage4/data/decoupled_abm/service_time_validation_residual_q90.parquet"),
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("stage4/output/decoupled_environment/validation_service_time_residual_rows.parquet"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("stage4/docs/results/simulator_v3/preassignment_release_buffer_audit.json"),
    )
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument("--test-date", default="20161023")
    return parser.parse_args()


def assign_zone(lon: pd.Series, lat: pd.Series, metadata: dict) -> pd.Series:
    grid_size = float(metadata["grid_size"])
    min_lon = float(metadata["min_lon"])
    min_lat = float(metadata["min_lat"])
    gx = np.floor((pd.to_numeric(lon, errors="coerce") - min_lon) / grid_size).astype("Int64")
    gy = np.floor((pd.to_numeric(lat, errors="coerce") - min_lat) / grid_size).astype("Int64")
    return "z" + gx.astype(str) + "_" + gy.astype(str)


def main() -> None:
    args = parse_args()
    if args.validation_date == args.test_date:
        raise ValueError("Validation residual calibration may not use the test date")
    for path in [args.route_conditioned, args.validation_predictions, args.validation_orders, args.zone_system]:
        if not path.exists():
            raise FileNotFoundError(path)

    route = pd.read_parquet(
        args.route_conditioned,
        columns=["order_id", "estimated_link_travel_time_sec", "prediction_time_bin", "route_link_seq"],
    )
    route["estimated_link_travel_time_sec"] = pd.to_numeric(
        route["estimated_link_travel_time_sec"], errors="coerce"
    )
    route_order = route.groupby("order_id", as_index=False).agg(
        predicted_service_time_sec=("estimated_link_travel_time_sec", "sum"),
        time_bin=("prediction_time_bin", "first"),
        predicted_link_count=("route_link_seq", "count"),
    )

    orders = pd.read_parquet(
        args.validation_orders,
        columns=["order_id", "date", "duration_sec", "origin_lon", "origin_lat"],
    )
    zone_meta = json.loads(args.zone_system.read_text(encoding="utf-8"))
    orders["zone"] = assign_zone(orders["origin_lon"], orders["origin_lat"], zone_meta)
    orders = orders.rename(columns={"duration_sec": "realized_service_time_sec"})

    prediction_columns = [
        "order_id",
        "pred_lcs_raw",
        "pred_pmis_raw",
        "pred_rts_raw",
    ]
    predictions = pd.read_parquet(args.validation_predictions, columns=prediction_columns)
    stress = predictions.groupby("order_id", as_index=False)[prediction_columns[1:]].mean()
    stress["stress_value"] = stress[prediction_columns[1:]].max(axis=1)
    rank = stress["stress_value"].rank(pct=True, method="average")
    stress["stress_bucket"] = pd.cut(
        rank,
        bins=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        labels=["low", "medium", "high"],
        include_lowest=True,
    ).astype(str)

    rows = route_order.merge(
        orders[["order_id", "date", "realized_service_time_sec", "zone"]],
        on="order_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        stress[["order_id", "stress_value", "stress_bucket"]],
        on="order_id",
        how="left",
        validate="one_to_one",
    )
    rows["stress_bucket"] = rows["stress_bucket"].fillna("unknown")
    rows["residual_sec"] = (
        pd.to_numeric(rows["predicted_service_time_sec"], errors="coerce")
        - pd.to_numeric(rows["realized_service_time_sec"], errors="coerce")
    )
    rows = rows.loc[
        rows["date"].astype(str).eq(args.validation_date)
        & rows["predicted_service_time_sec"].gt(0)
        & rows["realized_service_time_sec"].gt(0)
        & rows["time_bin"].notna()
        & rows["zone"].notna()
    ].copy()
    if rows.empty:
        raise ValueError("No validation service-time residual rows were assembled")

    source = (
        f"route={args.route_conditioned.as_posix()};"
        f"orders={args.validation_orders.as_posix()};"
        f"stress={args.validation_predictions.as_posix()}"
    )
    table = SafeReleaseBufferResolver.build_table(
        rows,
        quantile=args.quantile,
        validation_date=args.validation_date,
        source_dataset=source,
    )
    resolver = SafeReleaseBufferResolver(table)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output, index=False, compression="zstd")
    rows.to_parquet(args.raw_output, index=False, compression="zstd")
    audit = resolver.audit() | {
        "validation_date": args.validation_date,
        "test_date": args.test_date,
        "test_date_used": False,
        "validation_order_rows": int(len(rows)),
        "output": str(args.output),
        "raw_output": str(args.raw_output),
        "residual_q10_sec": float(rows["residual_sec"].quantile(0.1)),
        "residual_q50_sec": float(rows["residual_sec"].quantile(0.5)),
        "residual_q90_sec": float(rows["residual_sec"].quantile(0.9)),
        "residual_definition": RESIDUAL_DEFINITION,
    }
    args.audit_output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
