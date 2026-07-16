"""Build a demand-supply-decoupled Stage 4 engineering smoke environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical_pipeline.manifest import load_manifest, require_canonical_input


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--stage3-manifest", type=Path, required=True)
    parser.add_argument("--train-date", required=True)
    parser.add_argument("--test-date", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hv-count", type=int, default=475)
    parser.add_argument("--av-count", type=int, default=25)
    parser.add_argument("--depots", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--schema", type=Path, default=Path("config/artifact_manifest.schema.json"))
    return parser.parse_args()


def roles(manifest, root: Path) -> dict[str, Path]:
    return {item["role"]: root / item["path"] for item in manifest.data["files"]}


def order_endpoints(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path, columns=["driver_id", "order_id", "timestamp", "lon", "lat"])
    raw = raw.sort_values(["order_id", "timestamp"], kind="mergesort")
    return raw.groupby("order_id", sort=False).agg(
        historical_driver_id=("driver_id", "first"),
        origin_timestamp=("timestamp", "first"), destination_timestamp=("timestamp", "last"),
        origin_lon=("lon", "first"), origin_lat=("lat", "first"),
        destination_lon=("lon", "last"), destination_lat=("lat", "last"),
    ).reset_index()


def main() -> None:
    a = args(); root = Path.cwd().resolve()
    raw_manifest = load_manifest(a.raw_manifest, a.schema, root)
    stage3_manifest = load_manifest(a.stage3_manifest, a.schema, root)
    require_canonical_input(raw_manifest); require_canonical_input(stage3_manifest)
    raw = roles(raw_manifest, root); stage3 = roles(stage3_manifest, root)
    train = order_endpoints(raw[f"sample_{a.train_date}"])
    test = order_endpoints(raw[f"sample_{a.test_date}"])
    vector = pd.read_parquet(stage3[f"condition_vector_{a.test_date}"])
    demand = vector.merge(test, on="order_id", how="inner", validate="one_to_one")
    if len(demand) != 1000:
        raise ValueError(f"expected 1000 canonical demand orders, found {len(demand)}")
    demand["request_time"] = demand.decision_time
    demand["condition_available"] = True
    demand["actual_service_duration_decision_input"] = np.nan
    demand_path = a.output_root / "demand.parquet"; a.output_root.mkdir(parents=True, exist_ok=True)
    demand.to_parquet(demand_path, index=False, compression="zstd")

    initial = train.sort_values(["historical_driver_id", "origin_timestamp"], kind="mergesort").drop_duplicates(
        "historical_driver_id", keep="first"
    )
    if len(initial) < a.hv_count:
        raise ValueError("training-day supply has fewer unique drivers than requested HV count")
    initial = initial.assign(rank=initial.historical_driver_id.astype(str).map(
        lambda value: int.from_bytes(__import__("hashlib").sha256(f"{a.seed}|{value}".encode()).digest()[:8], "big")
    )).nsmallest(a.hv_count, "rank")
    start = pd.to_datetime(demand.request_time, utc=True).min().floor("D")
    end = start + pd.Timedelta(days=1, hours=2)
    hv = pd.DataFrame({
        "vehicle_id": [f"HV_{index:04d}" for index in range(a.hv_count)],
        "vehicle_type": "HV", "initial_lon": initial.origin_lon.to_numpy(),
        "initial_lat": initial.origin_lat.to_numpy(), "online_start": start, "online_end": end,
        "agent_source": "prior_train_day_initial_positions_exogenous_full_day_smoke_schedule",
        "source_driver_id": initial.historical_driver_id.astype(str).to_numpy(), "depot_id": pd.NA,
    })
    kmeans = KMeans(n_clusters=a.depots, random_state=a.seed, n_init=10).fit(train[["origin_lon", "origin_lat"]])
    depots = pd.DataFrame({
        "depot_id": [f"DEPOT_{index}" for index in range(a.depots)],
        "lon": kmeans.cluster_centers_[:, 0], "lat": kmeans.cluster_centers_[:, 1],
        "definition_source": "training_day_origin_kmeans_engineering_smoke",
        "source_date": a.train_date,
    })
    assignments = np.arange(a.av_count) % a.depots
    av = pd.DataFrame({
        "vehicle_id": [f"AV_{index:04d}" for index in range(a.av_count)],
        "vehicle_type": "AV", "initial_lon": depots.lon.to_numpy()[assignments],
        "initial_lat": depots.lat.to_numpy()[assignments], "online_start": start, "online_end": end,
        "agent_source": "training_day_depot_exogenous_smoke_fleet", "source_driver_id": pd.NA,
        "depot_id": depots.depot_id.to_numpy()[assignments],
    })
    fleet = pd.concat([hv, av], ignore_index=True)
    fleet_path = a.output_root / "fleet.parquet"; fleet.to_parquet(fleet_path, index=False, compression="zstd")
    depot_path = a.output_root / "depots.parquet"; depots.to_parquet(depot_path, index=False, compression="zstd")
    rng = np.random.default_rng(a.seed + 3001)
    scale = np.minimum(demand.service_time_uncertainty_sec.to_numpy(dtype=float) * 0.25,
                       demand.predicted_service_time_sec.to_numpy(dtype=float) * 0.5)
    residual = np.clip(rng.normal(0.0, scale), -0.5 * demand.predicted_service_time_sec, 2.0 * scale)
    residuals = pd.DataFrame({
        "order_id": demand.order_id, "service_time_residual_sec": residual,
        "random_stream": "rng_service_residual", "seed": a.seed + 3001,
        "source": "pre_generated_predicted_distribution_residual_not_historical_duration",
    })
    residual_path = a.output_root / "service_residuals.parquet"
    residuals.to_parquet(residual_path, index=False, compression="zstd")
    audit = {
        "status": "PASS", "demand_orders": int(len(demand)), "condition_available_orders": int(demand.condition_available.sum()),
        "realized_duration_decision_values": int(demand.actual_service_duration_decision_input.notna().sum()),
        "hv_count": int(len(hv)), "av_count": int(len(av)), "av_fleet_share": float(len(av) / len(fleet)),
        "supply_source_date": a.train_date, "demand_date": a.test_date,
        "test_day_supply_inputs": 0, "test_day_future_demand_depot_inputs": 0,
        "residual_source": "predicted_distribution", "files": [
            {"role": "stage4_demand", "path": demand_path.as_posix()},
            {"role": "stage4_fleet", "path": fleet_path.as_posix()},
            {"role": "stage4_depots", "path": depot_path.as_posix()},
            {"role": "stage4_service_residuals", "path": residual_path.as_posix()},
        ],
    }
    audit_path = a.output_root / "environment_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__": main()
