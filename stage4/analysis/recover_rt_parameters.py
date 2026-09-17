"""Recover legacy RT bounds from all-order endpoint manifests, not fitted outputs.

Calls the unchanged original chain/RT functions. No simulation or map matching.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from stage0.scripts.run_full_day_2017 import gcj02_to_wgs84
from stage4.scripts import build_decoupled_abm_environment as original


TRAIN_DATES = tuple(f"201610{d}" for d in range(19, 23))
PARAMETERS = dict(date="20161023", matching_response_sec=30.0,
                  minimum_pickup_sec=90.0, max_request_lead_sec=1800.0,
                  warmup_minutes=60, grid_size=0.02)
DOCS = Path("stage4/docs/paper_redesign")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def endpoint_frame(root: Path, date: str) -> tuple[pd.DataFrame, dict]:
    path = root / f"stage0/work_v6_final/candidate_manifests/date={date}.parquet"
    frame = pd.read_parquet(path, columns=[
        "order_id", "driver_id", "point_count", "valid_point_count", "start_time",
        "end_time", "start_lon", "start_lat", "end_lon", "end_lat"])
    # Prescan covers ALL orders, not only accepted Stage1 orders. No GPS validity
    # filter may have altered the endpoint population used by the old extractor.
    if not frame.point_count.eq(frame.valid_point_count).all():
        raise ValueError(f"{date}: invalid raw points require original extraction")
    if frame.order_id.duplicated().any() or frame.isna().any().any():
        raise ValueError(f"{date}: incomplete/duplicate endpoint manifest")
    frame = frame.sort_values("order_id", kind="stable").reset_index(drop=True)
    out = frame[["order_id", "driver_id"]].copy()
    for source, dest in (("start", "origin"), ("end", "destination")):
        out[f"{dest}_lon"], out[f"{dest}_lat"] = gcj02_to_wgs84(
            frame[f"{source}_lon"].to_numpy(), frame[f"{source}_lat"].to_numpy())
    out["observed_boarding_time"] = pd.to_datetime(frame.start_time, unit="s", utc=True)
    out["observed_dropoff_time"] = pd.to_datetime(frame.end_time, unit="s", utc=True)
    return out, dict(date=date, path=path.relative_to(root).as_posix(), sha256=sha256(path),
                     orders=len(frame), invalid_raw_point_count=0, population="all raw orders")


def fingerprint(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame([dict(scenario=name, orders=len(f),
        mean_lead_sec=float(f.latent_request_lead_sec.mean()),
        p50_lead_sec=float(f.latent_request_lead_sec.quantile(.5)),
        p90_lead_sec=float(f.latent_request_lead_sec.quantile(.9)),
        clipped_share=float(f.request_lead_clipped.mean())) for name, f in tables.items()])


def run(root: Path) -> dict:
    started = time.perf_counter()
    train, sources = {}, []
    for date in TRAIN_DATES:
        train[date], source = endpoint_frame(root, date)
        sources.append(source)
    spec = original.make_zone_spec(list(train.values()), PARAMETERS["grid_size"])
    chain, stats = original.build_training_chain_stats(train, spec)
    del chain, train
    demand, source = endpoint_frame(root, PARAMETERS["date"])
    sources.append(source)
    observed = fingerprint(original.attach_request_times(demand, stats, argparse.Namespace(**PARAMETERS)))
    reference_path = root / "stage4/docs/results/request_time_reconstruction_summary.csv"
    reference = pd.read_csv(reference_path)
    checks = []
    # Fixed before comparison; never optimize parameters against this fingerprint.
    for row in observed.to_dict("records"):
        expected = reference.set_index("scenario").loc[row["scenario"]]
        for field in ("orders", "mean_lead_sec", "p50_lead_sec", "p90_lead_sec", "clipped_share"):
            atol = 0.0 if field == "orders" else (1e-12 if field == "clipped_share" else 1e-6)
            actual, target = float(row[field]), float(expected[field])
            checks.append(dict(scenario=row["scenario"], metric=field, recovered=actual,
                reference=target, absolute_error=abs(actual-target), atol=atol,
                passed=bool(np.isclose(actual, target, rtol=0, atol=atol))))
    passed = all(row["passed"] for row in checks)
    source_files = ["stage4/scripts/build_decoupled_abm_environment.py",
                    "stage0/scripts/run_full_day_2017.py", "stage0/scripts/extract_order_od.py",
                    "stage0/v5/archive.py", "stage4/analysis/recover_rt_parameters.py"]
    result = dict(status="RECOVERED_FINGERPRINT_VERIFIED" if passed else "RECONSTRUCTION_MISMATCH",
        identification_status="latent RT scenario; not observed passenger request times",
        provenance_statement=("reconstructed from the original generator and verified against surviving frozen RT summary outputs"
                              if passed else "reconstruction did not reproduce surviving outputs; not frozen"),
        original_manifest_recovered=False, original_numeric_stats_independently_verified=False,
        train_dates=list(TRAIN_DATES), parameters=PARAMETERS, request_time_chain_stats=stats,
        endpoint_source="all-order prescan manifests; original GCJ02-to-WGS84 conversion",
        endpoint_caveat="Original OD files are missing. Raw-point validity and population are checked; endpoint tie selection is not independently recoverable. Aggregate fingerprint agreement is not per-order hash equivalence.",
        business_boundary_timezone="UTC (unchanged original convention)", sources=sources,
        code_sources={p: sha256(root/p) for p in source_files},
        fingerprint_source=dict(path=reference_path.relative_to(root).as_posix(), sha256=sha256(reference_path)),
        fingerprint_checks=checks, runtime_s=time.perf_counter()-started,
        full_day_simulation=False, parameter_optimization=False)
    directory = root / DOCS
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(directory / "recovered_rt_fingerprint_comparison.csv", index=False)
    target = directory / "recovered_rt_environment_parameters.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    result = run(parser.parse_args().root)
    if result["status"] != "RECOVERED_FINGERPRINT_VERIFIED":
        raise SystemExit(2)
