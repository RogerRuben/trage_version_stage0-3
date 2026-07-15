"""Compare actual Stage4 and FleetPy one-hour outputs.

Missing FleetPy outputs are an error. The script never writes NOT_RUN or a
synthetic PASS, so a published PASS necessarily comes from two result sets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS = {
    "completed_orders": 0.05,
    "cancelled_orders": 0.05,
    "mean_pickup_time_sec": 0.10,
    "mean_waiting_time_sec": 0.10,
    "vehicle_busy_time_sec": 0.10,
    "pickup_distance_m": 0.10,
    "service_distance_m": 0.10,
}


def first_present(frame: pd.DataFrame, aliases: list[str], required: bool = True) -> str | None:
    lookup = {str(c).lower(): str(c) for c in frame.columns}
    for name in aliases:
        if name.lower() in lookup:
            return lookup[name.lower()]
    if required:
        raise ValueError(f"None of the columns {aliases} found in {frame.columns.tolist()}")
    return None


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def timestamp_seconds(frame: pd.DataFrame, column: str) -> pd.Series:
    numeric = to_numeric(frame[column])
    if numeric.notna().mean() >= 0.95:
        return numeric
    stamps = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return stamps.astype("int64").where(stamps.notna()) / 1e9


def stage4_metrics(result_dir: Path) -> dict[str, float]:
    request_path = result_dir / "request_log.parquet"
    leg_path = result_dir / "vehicle_leg_log.parquet"
    if not request_path.exists() or not leg_path.exists():
        raise FileNotFoundError(f"Stage4 result is incomplete: {result_dir}")
    requests = pd.read_parquet(request_path)
    legs = pd.read_parquet(leg_path)
    completed = requests["final_status"].astype(str).eq("COMPLETED")
    request_sec = timestamp_seconds(requests, "request_time")
    boarding_sec = timestamp_seconds(requests, "boarding_time")
    pickup = legs[legs["leg_type"].astype(str).eq("PICKUP")]
    service = legs[legs["leg_type"].astype(str).eq("SERVICE")]
    active = legs[legs["leg_type"].astype(str).isin(["PICKUP", "SERVICE"])]
    return {
        "completed_orders": float(completed.sum()),
        "cancelled_orders": float(requests["final_status"].astype(str).eq("CANCELLED").sum()),
        "mean_pickup_time_sec": float(to_numeric(pickup["realized_time_sec"]).mean()),
        "mean_waiting_time_sec": float((boarding_sec[completed] - request_sec[completed]).mean()),
        "vehicle_busy_time_sec": float(to_numeric(active["realized_time_sec"]).sum()),
        "pickup_distance_m": float(to_numeric(pickup["distance_m"]).sum()),
        "service_distance_m": float(to_numeric(service["distance_m"]).sum()),
    }


def fleetpy_metrics(result_dir: Path) -> dict[str, float]:
    user_candidates = sorted(result_dir.glob("*user-stats.csv")) + sorted(result_dir.glob("*user_stats.csv"))
    op_candidates = sorted(result_dir.glob("*op-stats.csv"))
    if not user_candidates or not op_candidates:
        raise FileNotFoundError(
            f"Actual FleetPy outputs (*user_stats.csv and *op-stats.csv) not found in {result_dir}"
        )
    users = pd.read_csv(user_candidates[0])
    ops = pd.concat([pd.read_csv(path) for path in op_candidates], ignore_index=True)

    rq_col = first_present(users, ["rq_time", "request_time", "earliest_pickup_time"])
    pickup_col = first_present(users, ["pickup_time", "pu_time", "actual_pickup_time"])
    dropoff_col = first_present(users, ["dropoff_time", "do_time", "actual_dropoff_time"], required=False)
    operator_col = first_present(users, ["operator_id", "op_id", "service_operator"], required=False)
    rq_sec = timestamp_seconds(users, rq_col)
    pickup_sec = timestamp_seconds(users, pickup_col)
    if dropoff_col:
        completed = timestamp_seconds(users, dropoff_col).notna()
    elif operator_col:
        completed = to_numeric(users[operator_col]).fillna(-1).ge(0)
    else:
        completed = pickup_sec.notna()

    status_col = first_present(ops, ["status", "vehicle_status", "state"])
    start_col = first_present(ops, ["start_time", "start", "time_start"])
    end_col = first_present(ops, ["end_time", "end", "time_end"])
    distance_col = first_present(ops, ["driven_distance", "distance", "distance_m"])
    status = ops[status_col].astype(str).str.lower()
    duration = timestamp_seconds(ops, end_col) - timestamp_seconds(ops, start_col)
    distance = to_numeric(ops[distance_col]).fillna(0.0)
    pickup_mask = status.str.contains("route|pickup|approach", regex=True) & ~status.str.contains("service|reposition", regex=True)
    service_mask = status.str.contains("service|boarded|route.*customer", regex=True)
    onboard_col = first_present(ops, ["rq_on_board", "requests_on_board", "onboard_requests"], required=False)
    boarding_col = first_present(ops, ["rq_boarding", "boarding_requests"], required=False)
    if onboard_col:
        has_onboard = ops[onboard_col].fillna("").astype(str).str.strip().ne("")
        service_mask = service_mask | has_onboard
        pickup_mask = pickup_mask & ~has_onboard
    if boarding_col:
        pickup_mask = pickup_mask | ops[boarding_col].fillna("").astype(str).str.strip().ne("")

    return {
        "completed_orders": float(completed.sum()),
        "cancelled_orders": float((~completed).sum()),
        "mean_pickup_time_sec": float(duration[pickup_mask].mean()),
        "mean_waiting_time_sec": float((pickup_sec[completed] - rq_sec[completed]).mean()),
        "vehicle_busy_time_sec": float(duration[pickup_mask | service_mask].sum()),
        "pickup_distance_m": float(distance[pickup_mask].sum()),
        "service_distance_m": float(distance[service_mask].sum()),
    }


def relative_error(reference: float, candidate: float) -> float:
    if not np.isfinite(reference) or not np.isfinite(candidate):
        return float("inf")
    if abs(reference) < 1e-12:
        return 0.0 if abs(candidate) < 1e-12 else float("inf")
    return abs(candidate - reference) / abs(reference)


def compare(stage4: dict[str, float], fleetpy: dict[str, float]) -> pd.DataFrame:
    rows = []
    for metric, threshold in DEFAULT_THRESHOLDS.items():
        err = relative_error(fleetpy[metric], stage4[metric])
        rows.append({
            "metric": metric,
            "stage4_value": stage4[metric],
            "fleetpy_value": fleetpy[metric],
            "relative_error": err,
            "threshold": threshold,
            "status": "PASS" if err <= threshold else "FAIL",
        })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage4-result", type=Path, required=True)
    parser.add_argument("--fleetpy-result", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("stage4/docs/results/simulator_v3"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    if manifest.get("request_count", 0) < 500 or manifest.get("request_count", 0) > 2000:
        raise ValueError("Input manifest is outside the required 500-2,000 request range")
    s4 = stage4_metrics(args.stage4_result)
    fp = fleetpy_metrics(args.fleetpy_result)
    result = compare(s4, fp)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "fleetpy_cross_validation_summary.csv"
    result.to_csv(summary_path, index=False)
    overall = "PASS" if result["status"].eq("PASS").all() else "FAIL"
    report = [
        "# FleetPy cross-validation report", "", f"Overall status: **{overall}**", "",
        f"Controlled requests: {manifest['request_count']}",
        f"Controlled vehicles: {manifest['vehicle_count']}",
        f"Window: {manifest['window_start']} to {manifest['window_end']}", "",
        "Both engines used the frozen request/vehicle key hashes in the input manifest. "
        "This is a kernel consistency check (Stay, preassignment off, Safe/closest vehicle), "
        "not a validation of Stage4 ODD or pricing mechanisms.", "",
        result.to_markdown(index=False), "",
        "FleetPy source: https://github.com/TUM-VT/FleetPy", "",
    ]
    (args.output_dir / "fleetpy_cross_validation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": overall, "summary": str(summary_path)}, indent=2))
    if overall != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
