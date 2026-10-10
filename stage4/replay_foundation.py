"""Build the Stage4 S0 replay foundation without running dispatch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from stage0.v6.coordinates import gcj02_to_wgs84

TIMEZONE = "Asia/Shanghai"
FULL_ORDERS_REL = Path("stage0/work_v6_final/candidate_manifests/date=20161031.parquet")
REPLAY_ORDER_GLOB = (
    "stage1/input_v1/split=test/date=20161031/bucket=*/order_base.parquet"
)
STAGE3_INTERFACE_REL = Path(
    "stage3/output/odd_tod/final/test31_stage3_to_stage4_interface.parquet"
)
ORIGINAL_DESCRIPTOR_REL = Path(
    "stage3/output/odd_tod/s4/test31_original_route_descriptors.parquet"
)
STAGE3_FINALIZATION_CONFIG_REL = Path("stage3/config/stage3_finalization.json")
AUTO_ROUTE_CACHE_NAME = "historical_valhalla_auto_eta.parquet"
ROUTING_COORDINATE_SYSTEM = "WGS84_FROM_GCJ02"
OUTPUT_REL = Path("stage4/input/replay_foundation")
REPORT_REL = Path(
    "stage4/docs/replay_foundation/stage4_s0_replay_foundation_summary.md"
)
EXPECTED_REPLAY_ORDERS = 30_000
PRE_CORRECTION_BASELINE = {
    "session_count": 29_604,
    "fleet_size": 8_442,
    "supply_fit_mae": 24.59375,
    "global_beta": 1.9055717401529642,
    "global_beta_note": "invalidated because Valhalla received GCJ-02 as WGS84",
}
CONFIG_KEYS = frozenset(
    {
        "test_date",
        "session_gap_split_min",
        "time_bin_min",
        "fleet_sampling_seed",
        "pickup_eta_min_bin_sample",
    }
)


class ReplayFoundationError(RuntimeError):
    """Raised when a frozen input or S0 invariant is violated."""


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ReplayFoundationError(f"{name} missing columns: {missing}")


def _distribution(
    values: pd.Series, quantiles: Iterable[float]
) -> dict[str, float | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    result: dict[str, float | None] = {}
    for quantile in quantiles:
        key = f"p{int(round(100 * quantile)):02d}"
        result[key] = float(numeric.quantile(quantile)) if len(numeric) else None
    return result


def _timestamp_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        timestamps = pd.to_datetime(values, utc=True, errors="coerce")
    else:
        timestamps = pd.to_datetime(
            pd.to_numeric(values, errors="coerce"), unit="s", utc=True, errors="coerce"
        )
    return timestamps.dt.tz_convert(TIMEZONE)


def day_bounds(test_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(test_date, tz=TIMEZONE)
    return start, start + pd.Timedelta(days=1)


def time_bin_index(
    timestamps: pd.Series, test_date: str, time_bin_min: int
) -> pd.Series:
    """Map source-day timestamps to fixed bins, clipping small boundary spillover."""
    if 24 * 60 % int(time_bin_min):
        raise ReplayFoundationError("time_bin_min must divide 1440")
    start, _ = day_bounds(test_date)
    count = 24 * 60 // int(time_bin_min)
    elapsed = (timestamps - start).dt.total_seconds()
    index = np.floor(elapsed / (60 * int(time_bin_min)))
    return pd.Series(index, index=timestamps.index).clip(0, count - 1).astype("Int64")


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != CONFIG_KEYS:
        raise ReplayFoundationError(
            f"replay config keys must be exactly {sorted(CONFIG_KEYS)}; got {sorted(config)}"
        )
    if str(config["test_date"]) != "20161031":
        raise ReplayFoundationError("Stage4 S0 is authorized only for Test31")
    if int(config["session_gap_split_min"]) != 90:
        raise ReplayFoundationError(
            "the S0 baseline session split must remain 90 minutes"
        )
    if int(config["time_bin_min"]) != 15:
        raise ReplayFoundationError("the S0 baseline requires fixed 15-minute bins")
    return config


def load_full_test31_orders(
    root: Path, test_date: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    path = root / FULL_ORDERS_REL
    frame = pd.read_parquet(path)
    required = (
        "order_id",
        "driver_id",
        "start_time",
        "end_time",
        "start_lon",
        "start_lat",
        "end_lon",
        "end_lat",
        "time_order_valid",
    )
    _require_columns(frame, required, str(path))
    if frame["order_id"].duplicated().any():
        raise ReplayFoundationError("full Test31 order_id must be unique")
    work = frame[list(required)].copy()
    work["order_id"] = work["order_id"].astype("string")
    work["driver_id"] = work["driver_id"].astype("string")
    work["request_time"] = _timestamp_series(work["start_time"])
    work["arrival_time"] = _timestamp_series(work["end_time"])
    coords = work[["start_lon", "start_lat", "end_lon", "end_lat"]].apply(
        pd.to_numeric, errors="coerce"
    )
    work[["start_lon", "start_lat", "end_lon", "end_lat"]] = coords
    work = add_coordinate_lineage(work)
    finite_coords = np.isfinite(coords.to_numpy(dtype=float)).all(axis=1)
    valid_duration = (
        work["time_order_valid"].fillna(False).astype(bool)
        & work["request_time"].notna()
        & work["arrival_time"].notna()
        & work["arrival_time"].gt(work["request_time"])
    )
    work["valid_session_row"] = (
        valid_duration & finite_coords & work["driver_id"].notna()
    )
    work["realized_service_time_s"] = (
        work["arrival_time"] - work["request_time"]
    ).dt.total_seconds()
    start, end = day_bounds(test_date)
    diagnostics = {
        "source_order_count": int(len(work)),
        "source_unique_driver_count": int(work["driver_id"].nunique()),
        "invalid_session_row_count": int((~work["valid_session_row"]).sum()),
        "request_before_day_count": int(work["request_time"].lt(start).sum()),
        "request_at_or_after_day_end_count": int(work["request_time"].ge(end).sum()),
    }
    return work, diagnostics


def add_coordinate_lineage(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve source GCJ-02 coordinates and derive WGS84 routing coordinates."""
    _require_columns(
        frame, ("start_lon", "start_lat", "end_lon", "end_lat"), "OD coordinates"
    )
    result = frame.copy()
    for prefix in ("start", "end"):
        lon = pd.to_numeric(result[f"{prefix}_lon"], errors="coerce").to_numpy()
        lat = pd.to_numeric(result[f"{prefix}_lat"], errors="coerce").to_numpy()
        result[f"{prefix}_lon_gcj02"] = lon
        result[f"{prefix}_lat_gcj02"] = lat
        wgs_lon, wgs_lat = gcj02_to_wgs84(lon, lat)
        result[f"{prefix}_lon_wgs84"] = wgs_lon
        result[f"{prefix}_lat_wgs84"] = wgs_lat
    return result


def load_replay_orders(root: Path) -> pd.DataFrame:
    files = sorted(root.glob(REPLAY_ORDER_GLOB))
    if not files:
        raise ReplayFoundationError(f"no replay order files match {REPLAY_ORDER_GLOB}")
    columns = ["order_id", "departure_time", "arrival_time", "stage1_core_eligible"]
    frame = pd.concat(
        [pd.read_parquet(path, columns=columns) for path in files], ignore_index=True
    )
    if (
        len(frame) != EXPECTED_REPLAY_ORDERS
        or frame["order_id"].nunique() != EXPECTED_REPLAY_ORDERS
    ):
        raise ReplayFoundationError(
            "replay demand must contain exactly 30,000 unique orders"
        )
    if not frame["stage1_core_eligible"].fillna(False).all():
        raise ReplayFoundationError("replay source contains non-core Stage1 orders")
    frame["order_id"] = frame["order_id"].astype("string")
    frame["request_time"] = _timestamp_series(frame["departure_time"])
    frame["arrival_time"] = _timestamp_series(frame["arrival_time"])
    frame["realized_service_time_s"] = (
        frame["arrival_time"] - frame["request_time"]
    ).dt.total_seconds()
    return frame


def reconstruct_driver_sessions(
    orders: pd.DataFrame,
    session_gap_split_min: int,
    test_date: str,
    time_bin_min: int,
) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    required = (
        "order_id",
        "driver_id",
        "request_time",
        "arrival_time",
        "start_lon",
        "start_lat",
        "end_lon",
        "end_lat",
    )
    _require_columns(orders, required, "session orders")
    work = (
        orders.loc[orders["valid_session_row"].fillna(False)].copy()
        if "valid_session_row" in orders
        else orders.copy()
    )
    work = work.sort_values(["driver_id", "request_time", "order_id"], kind="mergesort")
    if "start_lon_wgs84" not in work.columns:
        work = add_coordinate_lineage(work)
    running_end = work.groupby("driver_id", sort=False)["arrival_time"].cummax()
    previous_running_end = running_end.groupby(work["driver_id"], sort=False).shift()
    gap_s = (work["request_time"] - previous_running_end).dt.total_seconds()
    first = previous_running_end.isna()
    starts_session = first | gap_s.gt(float(session_gap_split_min) * 60.0)
    work["inter_order_gap_s"] = gap_s
    work["session_seq"] = (
        starts_session.groupby(work["driver_id"], sort=False).cumsum().astype(int)
    )

    rows: list[dict[str, Any]] = []
    for (driver_id, sequence), group in work.groupby(
        ["driver_id", "session_seq"], sort=False, observed=True
    ):
        group = group.sort_values(["request_time", "order_id"], kind="mergesort")
        internal = pd.to_numeric(
            group["inter_order_gap_s"].iloc[1:], errors="coerce"
        ).dropna()
        session_start = group["request_time"].iloc[0]
        latest_arrival_index = group["arrival_time"].idxmax()
        latest_arrival = group.loc[latest_arrival_index]
        session_end = latest_arrival["arrival_time"]
        rows.append(
            {
                "session_id": f"{driver_id}__S{int(sequence):03d}",
                "driver_id": str(driver_id),
                "session_start_time": session_start,
                "session_end_time": session_end,
                "first_order_id": str(group["order_id"].iloc[0]),
                "last_order_id": str(latest_arrival["order_id"]),
                "initial_pickup_lon_gcj02": float(group["start_lon_gcj02"].iloc[0]),
                "initial_pickup_lat_gcj02": float(group["start_lat_gcj02"].iloc[0]),
                "initial_pickup_lon_wgs84": float(group["start_lon_wgs84"].iloc[0]),
                "initial_pickup_lat_wgs84": float(group["start_lat_wgs84"].iloc[0]),
                "final_dropoff_lon_gcj02": float(latest_arrival["end_lon_gcj02"]),
                "final_dropoff_lat_gcj02": float(latest_arrival["end_lat_gcj02"]),
                "final_dropoff_lon_wgs84": float(latest_arrival["end_lon_wgs84"]),
                "final_dropoff_lat_wgs84": float(latest_arrival["end_lat_wgs84"]),
                "order_count": int(len(group)),
                "session_span_s": float((session_end - session_start).total_seconds()),
                "max_internal_gap_s": float(internal.max()) if len(internal) else 0.0,
                "mean_internal_gap_s": float(internal.mean()) if len(internal) else 0.0,
                "median_internal_gap_s": float(internal.median())
                if len(internal)
                else 0.0,
            }
        )
    sessions = (
        pd.DataFrame(rows)
        .sort_values(["session_start_time", "session_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    sessions["start_15min_bin"] = time_bin_index(
        sessions["session_start_time"], test_date, time_bin_min
    ).astype(int)
    sessions["end_15min_bin"] = time_bin_index(
        sessions["session_end_time"], test_date, time_bin_min
    ).astype(int)
    non_first_gaps = gap_s.loc[~first].dropna()
    diagnostics = {
        "unique_drivers": int(work["driver_id"].nunique()),
        "session_count": int(len(sessions)),
        "session_split_count": int(starts_session.sum() - work["driver_id"].nunique()),
        "negative_gap_count": int(non_first_gaps.lt(0).sum()),
    }
    return sessions, non_first_gaps.reset_index(drop=True), diagnostics


def _active_supply(
    sessions: pd.DataFrame, bin_starts: pd.Series, bin_ends: pd.Series
) -> np.ndarray:
    starts = sessions["session_start_time"].array
    ends = sessions["session_end_time"].array
    return np.asarray(
        [
            int(((starts < end) & (ends > start)).sum())
            for start, end in zip(bin_starts, bin_ends)
        ],
        dtype=np.int64,
    )


def build_scaling_profile(
    full_orders: pd.DataFrame,
    replay_orders: pd.DataFrame,
    sessions: pd.DataFrame,
    test_date: str,
    time_bin_min: int,
) -> pd.DataFrame:
    bin_count = 24 * 60 // int(time_bin_min)
    start, _ = day_bounds(test_date)
    profile = pd.DataFrame({"time_bin_index": np.arange(bin_count, dtype=np.int64)})
    profile["time_bin_start"] = start + pd.to_timedelta(
        profile["time_bin_index"] * int(time_bin_min), unit="m"
    )
    profile["time_bin_end"] = profile["time_bin_start"] + pd.Timedelta(
        minutes=time_bin_min
    )
    full_bin = time_bin_index(full_orders["request_time"], test_date, time_bin_min)
    replay_bin = time_bin_index(replay_orders["request_time"], test_date, time_bin_min)
    profile["full_order_count"] = (
        full_bin.value_counts()
        .reindex(range(bin_count), fill_value=0)
        .sort_index()
        .to_numpy()
    )
    profile["replay_order_count"] = (
        replay_bin.value_counts()
        .reindex(range(bin_count), fill_value=0)
        .sort_index()
        .to_numpy()
    )
    profile["demand_scale_ratio"] = np.where(
        profile["full_order_count"].gt(0),
        profile["replay_order_count"] / profile["full_order_count"],
        np.nan,
    )
    profile["full_active_supply"] = _active_supply(
        sessions, profile["time_bin_start"], profile["time_bin_end"]
    )
    raw_target = profile["demand_scale_ratio"] * profile["full_active_supply"]
    profile["target_active_supply"] = np.where(
        raw_target.notna(), np.floor(raw_target + 0.5), 0
    ).astype(np.int64)
    return profile


def _sampling_priority(session_id: str, seed: int) -> str:
    return hashlib.sha256(f"{int(seed)}|{session_id}".encode("utf-8")).hexdigest()


def select_fleet_template(
    sessions: pd.DataFrame, scaling: pd.DataFrame, seed: int
) -> pd.DataFrame:
    ratio_by_bin = scaling.set_index("time_bin_index")["demand_scale_ratio"].to_dict()
    global_ratio = float(
        scaling["replay_order_count"].sum() / scaling["full_order_count"].sum()
    )
    selected: list[pd.DataFrame] = []
    for start_bin, group in sessions.groupby("start_15min_bin", sort=True):
        ratio = ratio_by_bin.get(int(start_bin), np.nan)
        ratio = global_ratio if not np.isfinite(ratio) else float(ratio)
        count = min(len(group), int(np.floor(len(group) * max(ratio, 0.0) + 0.5)))
        ranked = group.copy()
        ranked["_priority"] = ranked["session_id"].map(
            lambda value: _sampling_priority(str(value), seed)
        )
        selected.append(ranked.sort_values(["_priority", "session_id"]).head(count))
    chosen = (
        pd.concat(selected, ignore_index=True) if selected else sessions.iloc[:0].copy()
    )
    chosen = chosen.sort_values(
        ["session_start_time", "session_id"], kind="mergesort"
    ).reset_index(drop=True)
    return pd.DataFrame(
        {
            "fleet_template_id": [
                f"FT_{index:06d}" for index in range(1, len(chosen) + 1)
            ],
            "source_session_id": chosen["session_id"].astype("string"),
            "source_driver_id": chosen["driver_id"].astype("string"),
            "availability_start_time": chosen["session_start_time"],
            "availability_end_time": chosen["session_end_time"],
            "initial_lon_gcj02": chosen["initial_pickup_lon_gcj02"].astype(float),
            "initial_lat_gcj02": chosen["initial_pickup_lat_gcj02"].astype(float),
            "initial_lon_wgs84": chosen["initial_pickup_lon_wgs84"].astype(float),
            "initial_lat_wgs84": chosen["initial_pickup_lat_wgs84"].astype(float),
            "source_order_count": chosen["order_count"].astype(np.int64),
            "source_session_span_s": chosen["session_span_s"].astype(float),
        }
    )


def add_simulated_supply(scaling: pd.DataFrame, fleet: pd.DataFrame) -> pd.DataFrame:
    result = scaling.copy()
    windows = fleet.rename(
        columns={
            "availability_start_time": "session_start_time",
            "availability_end_time": "session_end_time",
        }
    )
    result["simulated_active_supply"] = _active_supply(
        windows, result["time_bin_start"], result["time_bin_end"]
    )
    result["absolute_supply_error"] = (
        result["simulated_active_supply"] - result["target_active_supply"]
    ).abs()
    result["relative_supply_error"] = np.where(
        result["target_active_supply"].gt(0),
        result["absolute_supply_error"] / result["target_active_supply"],
        np.nan,
    )
    return result


def _top_supply_error_bins(
    scaling: pd.DataFrame, error_column: str
) -> list[dict[str, Any]]:
    sample = scaling.loc[np.isfinite(scaling[error_column])].nlargest(5, error_column)
    return [
        {
            "time_bin_index": int(row.time_bin_index),
            "time_bin_start": pd.Timestamp(row.time_bin_start).isoformat(),
            "target_active_supply": int(row.target_active_supply),
            "simulated_active_supply": int(row.simulated_active_supply),
            "absolute_supply_error": float(row.absolute_supply_error),
            "relative_supply_error": float(row.relative_supply_error),
        }
        for row in sample.itertuples(index=False)
    ]


def load_or_build_valhalla_auto_times(
    root: Path,
    replay_orders: pd.DataFrame,
    full_orders: pd.DataFrame,
    output: Path,
    checkpoint_every: int = 500,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute one deterministic Valhalla auto route per replay OD, with resume."""
    cache_path = output / AUTO_ROUTE_CACHE_NAME
    columns = [
        "order_id",
        "routing_coordinate_system",
        "origin_lon_wgs84",
        "origin_lat_wgs84",
        "destination_lon_wgs84",
        "destination_lat_wgs84",
        "valhalla_route_time_s",
        "valhalla_route_distance_m",
        "valhalla_route_status",
        "valhalla_failure_reason",
    ]
    cache_invalidated = False
    if cache_path.is_file():
        candidate = pd.read_parquet(cache_path)
        is_current = (
            set(columns).issubset(candidate.columns)
            and len(candidate) > 0
            and candidate["routing_coordinate_system"]
            .eq(ROUTING_COORDINATE_SYSTEM)
            .all()
        )
        if is_current:
            existing = candidate[columns].drop_duplicates("order_id", keep="last")
        else:
            existing = pd.DataFrame(columns=columns)
            cache_invalidated = len(candidate) > 0
    else:
        existing = pd.DataFrame(columns=columns)
    coordinates = full_orders[
        [
            "order_id",
            "start_lon_wgs84",
            "start_lat_wgs84",
            "end_lon_wgs84",
            "end_lat_wgs84",
        ]
    ]
    pending = replay_orders[["order_id", "request_time"]].merge(
        coordinates, on="order_id", how="left", validate="one_to_one"
    )
    pending = pending.loc[~pending["order_id"].isin(existing["order_id"])].sort_values(
        "order_id", kind="mergesort"
    )
    if len(pending):
        try:
            from valhalla import Actor
        except ImportError as exc:
            raise ReplayFoundationError(
                "Valhalla Python bindings are required"
            ) from exc
        stage3_config = json.loads(
            (root / STAGE3_FINALIZATION_CONFIG_REL).read_text(encoding="utf-8")
        )
        actor = Actor(str(Path(str(stage3_config["valhalla_config"])).resolve()))
        additions: list[dict[str, Any]] = []
        for number, row in enumerate(pending.itertuples(index=False), start=1):
            local_time = pd.Timestamp(row.request_time).tz_convert(TIMEZONE)
            request = {
                "locations": [
                    {
                        "lon": float(row.start_lon_wgs84),
                        "lat": float(row.start_lat_wgs84),
                        "type": "break",
                    },
                    {
                        "lon": float(row.end_lon_wgs84),
                        "lat": float(row.end_lat_wgs84),
                        "type": "break",
                    },
                ],
                "costing": "auto",
                "units": "kilometers",
                "directions_type": "none",
                "date_time": {
                    "type": 1,
                    "value": local_time.strftime("%Y-%m-%dT%H:%M"),
                },
            }
            route_lineage = {
                "order_id": str(row.order_id),
                "routing_coordinate_system": ROUTING_COORDINATE_SYSTEM,
                "origin_lon_wgs84": float(row.start_lon_wgs84),
                "origin_lat_wgs84": float(row.start_lat_wgs84),
                "destination_lon_wgs84": float(row.end_lon_wgs84),
                "destination_lat_wgs84": float(row.end_lat_wgs84),
            }
            try:
                trip = actor.route(request)["trip"]
                summary = trip["summary"]
                if int(trip.get("status", 0)) != 0 or len(trip.get("legs", [])) != 1:
                    raise ReplayFoundationError(
                        "Valhalla did not return one successful leg"
                    )
                additions.append(
                    {
                        **route_lineage,
                        "valhalla_route_time_s": float(summary["time"]),
                        "valhalla_route_distance_m": float(summary["length"]) * 1000.0,
                        "valhalla_route_status": "OK",
                        "valhalla_failure_reason": None,
                    }
                )
            except Exception as exc:
                additions.append(
                    {
                        **route_lineage,
                        "valhalla_route_time_s": np.nan,
                        "valhalla_route_distance_m": np.nan,
                        "valhalla_route_status": "ERROR",
                        "valhalla_failure_reason": f"{type(exc).__name__}:{exc}"[:500],
                    }
                )
            if number % int(checkpoint_every) == 0 or number == len(pending):
                new_rows = pd.DataFrame(additions, columns=columns)
                current = (
                    pd.concat([existing, new_rows], ignore_index=True)
                    if len(existing)
                    else new_rows
                ).drop_duplicates("order_id", keep="last")
                _write_parquet(current[columns], cache_path)
                print(f"Valhalla auto ETA: {number}/{len(pending)}", flush=True)
        existing = pd.read_parquet(cache_path, columns=columns)
    expected_ids = set(replay_orders["order_id"].astype(str))
    observed_ids = set(existing["order_id"].astype(str))
    if observed_ids != expected_ids:
        raise ReplayFoundationError(
            f"auto ETA cache identity mismatch: missing={len(expected_ids - observed_ids)}, "
            f"extra={len(observed_ids - expected_ids)}"
        )
    existing = existing.sort_values("order_id", kind="mergesort").reset_index(drop=True)
    return existing, {
        "valhalla_auto_route_count": int(len(existing)),
        "valhalla_auto_route_status_counts": {
            str(key): int(value)
            for key, value in existing["valhalla_route_status"]
            .value_counts()
            .sort_index()
            .items()
        },
        "valhalla_auto_route_policy": "single_deterministic_auto_route_at_request_time",
        "valhalla_routing_coordinate_system": ROUTING_COORDINATE_SYSTEM,
        "legacy_coordinate_cache_invalidated": bool(cache_invalidated),
    }


def build_eta_calibration(
    orders: pd.DataFrame,
    test_date: str,
    time_bin_min: int,
    min_bin_sample: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = ("request_time", "realized_service_time_s", "valhalla_route_time_s")
    _require_columns(orders, required, "ETA calibration orders")
    work = orders.copy()
    ata = pd.to_numeric(work["realized_service_time_s"], errors="coerce")
    valhalla = pd.to_numeric(work["valhalla_route_time_s"], errors="coerce")
    invalid_ata = ~np.isfinite(ata) | ata.le(0)
    invalid_valhalla = ~np.isfinite(valhalla) | valhalla.le(0)
    valid = ~(invalid_ata | invalid_valhalla) & work["request_time"].notna()
    sample = work.loc[valid, ["request_time"]].copy()
    sample["ratio"] = ata.loc[valid] / valhalla.loc[valid]
    sample["time_bin_index"] = time_bin_index(
        sample["request_time"], test_date, time_bin_min
    ).astype(int)
    bins_per_hour = 60 // int(time_bin_min)
    sample["hour"] = sample["time_bin_index"] // bins_per_hour
    global_median = float(sample["ratio"].median()) if len(sample) else np.nan
    hour_stats = sample.groupby("hour")["ratio"].agg(["size", "median"])
    bin_stats = sample.groupby("time_bin_index")["ratio"].agg(["size", "median"])
    bin_count = 24 * 60 // int(time_bin_min)
    start, _ = day_bounds(test_date)
    rows: list[dict[str, Any]] = []
    for index in range(bin_count):
        count = int(bin_stats.loc[index, "size"]) if index in bin_stats.index else 0
        raw_median = (
            float(bin_stats.loc[index, "median"])
            if index in bin_stats.index
            else np.nan
        )
        hour = index // bins_per_hour
        hour_count = (
            int(hour_stats.loc[hour, "size"]) if hour in hour_stats.index else 0
        )
        hour_median = (
            float(hour_stats.loc[hour, "median"])
            if hour in hour_stats.index
            else np.nan
        )
        if count >= int(min_bin_sample):
            selected, fallback = raw_median, "BIN_MEDIAN"
        elif np.isfinite(hour_median):
            selected, fallback = hour_median, "HOUR_MEDIAN"
        else:
            selected, fallback = global_median, "GLOBAL_MEDIAN"
        bin_start = start + pd.Timedelta(minutes=index * int(time_bin_min))
        rows.append(
            {
                "time_bin_start": bin_start,
                "time_bin_end": bin_start + pd.Timedelta(minutes=int(time_bin_min)),
                "time_bin_index": index,
                "sample_count": count,
                "raw_bin_median_ratio": raw_median,
                "selected_eta_multiplier": selected,
                "fallback_level": fallback,
                "hour_sample_count": hour_count,
                "hour_median_ratio": hour_median,
                "global_median_ratio": global_median,
            }
        )
    calibration = pd.DataFrame(rows)
    diagnostics = {
        "valid_calibration_order_count": int(valid.sum()),
        "invalid_ata_count": int(invalid_ata.sum()),
        "invalid_valhalla_eta_count": int(invalid_valhalla.sum()),
        "ratio_distribution": _distribution(
            sample["ratio"], (0.01, 0.05, 0.5, 0.95, 0.99)
        ),
        "global_median_ratio": global_median,
        "multiplier_distribution": _distribution(
            calibration["selected_eta_multiplier"], (0.0, 0.5, 1.0)
        ),
        "fallback_bin_counts": {
            str(key): int(value)
            for key, value in calibration["fallback_level"]
            .value_counts()
            .sort_index()
            .items()
        },
    }
    return calibration, diagnostics


def build_order_replay_base(
    root: Path, replay_orders: pd.DataFrame, full_orders: pd.DataFrame
) -> pd.DataFrame:
    coordinates = full_orders[
        [
            "order_id",
            "start_lon_gcj02",
            "start_lat_gcj02",
            "start_lon_wgs84",
            "start_lat_wgs84",
            "end_lon_gcj02",
            "end_lat_gcj02",
            "end_lon_wgs84",
            "end_lat_wgs84",
        ]
    ].rename(
        columns={
            "start_lon_gcj02": "pickup_lon_gcj02",
            "start_lat_gcj02": "pickup_lat_gcj02",
            "start_lon_wgs84": "pickup_lon_wgs84",
            "start_lat_wgs84": "pickup_lat_wgs84",
            "end_lon_gcj02": "dropoff_lon_gcj02",
            "end_lat_gcj02": "dropoff_lat_gcj02",
            "end_lon_wgs84": "dropoff_lon_wgs84",
            "end_lat_wgs84": "dropoff_lat_wgs84",
        }
    )
    base = replay_orders[["order_id", "request_time", "realized_service_time_s"]].merge(
        coordinates, on="order_id", how="left", validate="one_to_one"
    )
    descriptor = pd.read_parquet(
        root / ORIGINAL_DESCRIPTOR_REL,
        columns=["order_id", "predicted_route_time_p50_s"],
    ).drop_duplicates("order_id")
    base = base.merge(descriptor, on="order_id", how="left", validate="one_to_one")
    base = base.rename(
        columns={"predicted_route_time_p50_s": "predicted_service_time_s"}
    )
    interface_columns = [
        "order_id",
        "profile_id",
        "selected_route_type",
        "hard_state",
        "evidence_complete",
        "selected_service_time_p50_s",
        "rho_static",
        "rho_dynamic",
        "rho_speed",
    ]
    interface = pd.read_parquet(root / STAGE3_INTERFACE_REL, columns=interface_columns)
    result = base.merge(interface, on="order_id", how="inner", validate="one_to_many")
    if len(result) != EXPECTED_REPLAY_ORDERS * 3:
        raise ReplayFoundationError("replay base must contain 30k orders x 3 profiles")
    ordered = [
        "order_id",
        "request_time",
        "pickup_lon_gcj02",
        "pickup_lat_gcj02",
        "pickup_lon_wgs84",
        "pickup_lat_wgs84",
        "dropoff_lon_gcj02",
        "dropoff_lat_gcj02",
        "dropoff_lon_wgs84",
        "dropoff_lat_wgs84",
        "realized_service_time_s",
        "predicted_service_time_s",
        *interface_columns[1:],
    ]
    return (
        result[ordered]
        .sort_values(["request_time", "order_id", "profile_id"])
        .reset_index(drop=True)
    )


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_report(root: Path, summary: Mapping[str, Any]) -> None:
    driver = summary["driver_sessions"]
    fleet = summary["fleet_scaling"]
    eta = summary["pickup_eta_calibration"]
    source_driver_count = summary["source_diagnostics"]["source_unique_driver_count"]
    valid_driver_count = driver["unique_drivers"]
    lines = [
        "# Stage4 S0 Replay Foundation Summary",
        "",
        "## S0 correction comparison",
        "",
        "| Metric | Before correction | After correction |",
        "|---|---:|---:|",
        f"| Session count | {summary['correction_comparison']['before']['session_count']:,} | {summary['correction_comparison']['after']['session_count']:,} |",
        f"| Fleet size | {summary['correction_comparison']['before']['fleet_size']:,} | {summary['correction_comparison']['after']['fleet_size']:,} |",
        f"| Supply-fit MAE | {summary['correction_comparison']['before']['supply_fit_mae']:.6f} | {summary['correction_comparison']['after']['supply_fit_mae']:.6f} |",
        f"| Global beta | {summary['correction_comparison']['before']['global_beta']:.6f} | {summary['correction_comparison']['after']['global_beta']:.6f} |",
        "",
        "The pre-correction beta is retained only as lineage and is scientifically invalid because the legacy Valhalla requests interpreted GCJ-02 as WGS84.",
        "",
        "## Driver/session reconstruction",
        "",
        f"- Full Test31 source drivers: {source_driver_count:,}",
        f"- Drivers with valid reconstructed sessions: {valid_driver_count:,}",
        f"- Sessions: {driver['session_count']:,}",
        f"- 90-minute splits: {driver['session_split_count']:,}",
        f"- Negative inter-order gaps recorded: {driver['negative_gap_count']:,}",
        f"- Session duration (s): `{driver['session_duration_s']}`",
        f"- Orders/session: `{driver['orders_per_session']}`",
        f"- Inter-order gap (s): `{driver['inter_order_gap_s']}`",
        "",
        "Effective gaps use the previous running-maximum arrival time. Session end and final drop-off come from the order with the maximum arrival time. These sessions are effective observed service episodes, not true online or employment shifts.",
        "",
        "## 15-minute fleet scaling",
        "",
        f"- Full demand total: {fleet['full_demand_total']:,}",
        f"- Replay demand total: {fleet['replay_demand_total']:,}",
        f"- Full active supply: `{fleet['full_active_supply']}`",
        f"- Target replay supply: `{fleet['target_active_supply']}`",
        f"- Selected replay fleet templates: {fleet['selected_fleet_template_count']:,}",
        f"- Supply-fit MAE: {fleet['mean_absolute_error']:.3f}",
        f"- Supply-fit maximum absolute error: {fleet['max_absolute_error']:.3f}",
        f"- Mean relative error where target > 0: {fleet['mean_relative_error']:.4%}",
        f"- Exactly matched bins: {fleet['exact_match_bin_count']}/96",
        f"- Top-5 absolute-error bins: `{fleet['top_5_absolute_error_bins']}`",
        f"- Top-5 relative-error bins: `{fleet['top_5_relative_error_bins']}`",
        "",
        "Target supply uses deterministic nearest-integer rounding (`floor(x + 0.5)`). Complete sessions are selected within start-time bins by a seed-bound SHA-256 priority; no fleet optimizer is used.",
        "",
        "## Pickup ETA calibration",
        "",
        f"- Valid calibration orders: {eta['valid_calibration_order_count']:,}",
        f"- Invalid ATA rows: {eta['invalid_ata_count']:,}",
        f"- Invalid Valhalla ETA rows: {eta['invalid_valhalla_eta_count']:,}",
        f"- ATA/Valhalla ratio: `{eta['ratio_distribution']}`",
        f"- Global median ratio: {eta['global_median_ratio']:.6f}",
        f"- Selected multiplier min/p50/max: `{eta['multiplier_distribution']}`",
        f"- Fallback-bin counts: `{eta['fallback_bin_counts']}`",
        "",
        "## Produced files",
        "",
        "- `stage4/input/replay_foundation/full_test31_driver_sessions.parquet`",
        "- `stage4/input/replay_foundation/replay_fleet_template.parquet`",
        "- `stage4/input/replay_foundation/fleet_scaling_15min.parquet`",
        "- `stage4/input/replay_foundation/pickup_eta_calibration_15min.parquet`",
        "- `stage4/input/replay_foundation/stage4_order_replay_base.parquet`",
        "- `stage4/input/replay_foundation/stage4_s0_summary.json`",
        "- `stage4/input/replay_foundation/historical_valhalla_auto_eta.parquet`",
        "",
        "## Input selection",
        "",
        "The Stage0 Test31 candidate manifest is the canonical full-order activity source because it contains all 105,460 source orders with driver, timestamps, and OD coordinates. Manifest GCJ-02 coordinates are preserved for lineage and converted with `stage0.v6.coordinates.gcj02_to_wgs84`; Valhalla and every future vehicle-to-pickup route must use only the explicit WGS84 fields. The Stage1 frozen Test31 order base defines the exact 30,000 replay orders. No independent canonical auto-route ETA product existed, so S0 computes exactly one deterministic Valhalla `auto` route per replay OD at request time using the same frozen config and tiles as Stage3. Stage1 trace-route elapsed values are not used because they inherit observed trajectory timing. Frozen Stage3 descriptors and the final Stage3→Stage4 interface supply decision-time service predictions and per-profile capability fields.",
        "",
        "## Known limitations",
        "",
        "- Sessions represent observed service episodes; unseen idle/online drivers are not inferred.",
        "- Fleet scaling samples complete sessions by start-time bin and may leave a small 15-minute supply error.",
        "- ETA calibration is a day-specific replay traffic calibration using Test31 aggregate time-of-day medians; it is not a strict out-of-sample decision-time ETA predictor.",
        "- Small source-day timestamp spillover beyond local midnight is clipped to the final Test31 bin and counted in the local summary JSON.",
        "- Existing legacy Stage4 dispatch/simulator code was not invoked or modified by S0.",
        "",
        "`ROLLING_DISPATCH = NOT STARTED`",
        "",
    ]
    path = root / REPORT_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build(root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    config_file = (
        Path(config_path).resolve()
        if config_path is not None
        else root / "stage4/config/replay_foundation.json"
    )
    config = load_config(config_file)
    test_date = str(config["test_date"])
    time_bin_min = int(config["time_bin_min"])

    full_orders, source_diagnostics = load_full_test31_orders(root, test_date)
    replay_orders = load_replay_orders(root)
    sessions, gaps, session_diagnostics = reconstruct_driver_sessions(
        full_orders,
        int(config["session_gap_split_min"]),
        test_date,
        time_bin_min,
    )
    scaling = build_scaling_profile(
        full_orders, replay_orders, sessions, test_date, time_bin_min
    )
    fleet = select_fleet_template(sessions, scaling, int(config["fleet_sampling_seed"]))
    scaling = add_simulated_supply(scaling, fleet)

    output = root / OUTPUT_REL
    output.mkdir(parents=True, exist_ok=True)
    valhalla_times, route_time_diagnostics = load_or_build_valhalla_auto_times(
        root, replay_orders, full_orders, output
    )
    eta_orders = replay_orders[
        ["order_id", "request_time", "realized_service_time_s"]
    ].merge(valhalla_times, on="order_id", how="left", validate="one_to_one")
    calibration, eta_diagnostics = build_eta_calibration(
        eta_orders,
        test_date,
        time_bin_min,
        int(config["pickup_eta_min_bin_sample"]),
    )
    replay_base = build_order_replay_base(root, replay_orders, full_orders)

    _write_parquet(sessions, output / "full_test31_driver_sessions.parquet")
    _write_parquet(fleet, output / "replay_fleet_template.parquet")
    _write_parquet(scaling, output / "fleet_scaling_15min.parquet")
    _write_parquet(calibration, output / "pickup_eta_calibration_15min.parquet")
    _write_parquet(replay_base, output / "stage4_order_replay_base.parquet")

    positive_target = scaling["target_active_supply"].gt(0)
    summary = {
        "phase_status": "STAGE4_S0_REPLAY_FOUNDATION_COMPLETE",
        "rolling_dispatch_started": False,
        "test_date": test_date,
        "config": config,
        "input_sources": {
            "full_test31_orders": str(FULL_ORDERS_REL).replace("\\", "/"),
            "replay_order_glob": REPLAY_ORDER_GLOB,
            "valhalla_auto_eta": str(OUTPUT_REL / AUTO_ROUTE_CACHE_NAME).replace(
                "\\", "/"
            ),
            "valhalla_config_source": str(STAGE3_FINALIZATION_CONFIG_REL).replace(
                "\\", "/"
            ),
            "stage3_interface": str(STAGE3_INTERFACE_REL).replace("\\", "/"),
            "original_descriptor": str(ORIGINAL_DESCRIPTOR_REL).replace("\\", "/"),
        },
        "source_diagnostics": {**source_diagnostics, **route_time_diagnostics},
        "coordinate_policy": {
            "source_coordinate_system": "GCJ02",
            "valhalla_coordinate_system": "WGS84",
            "conversion": "stage0.v6.coordinates.gcj02_to_wgs84",
        },
        "driver_sessions": {
            **session_diagnostics,
            "sessions_per_driver": _distribution(
                sessions.groupby("driver_id").size(), (0.5, 0.9, 0.95, 0.99)
            ),
            "session_duration_s": _distribution(
                sessions["session_span_s"], (0.5, 0.9, 0.95, 0.99)
            ),
            "orders_per_session": _distribution(
                sessions["order_count"], (0.5, 0.9, 0.95, 0.99)
            ),
            "inter_order_gap_s": _distribution(gaps, (0.5, 0.9, 0.95, 0.99)),
        },
        "fleet_scaling": {
            "full_demand_total": int(scaling["full_order_count"].sum()),
            "replay_demand_total": int(scaling["replay_order_count"].sum()),
            "full_active_supply": _distribution(
                scaling["full_active_supply"], (0.0, 0.5, 1.0)
            ),
            "target_active_supply": _distribution(
                scaling["target_active_supply"], (0.0, 0.5, 1.0)
            ),
            "selected_fleet_template_count": int(len(fleet)),
            "mean_absolute_error": float(scaling["absolute_supply_error"].mean()),
            "max_absolute_error": float(scaling["absolute_supply_error"].max()),
            "mean_relative_error": float(
                scaling.loc[positive_target, "relative_supply_error"].mean()
            ),
            "exact_match_bin_count": int(scaling["absolute_supply_error"].eq(0).sum()),
            "zero_full_demand_bin_count": int(scaling["full_order_count"].eq(0).sum()),
            "top_5_absolute_error_bins": _top_supply_error_bins(
                scaling, "absolute_supply_error"
            ),
            "top_5_relative_error_bins": _top_supply_error_bins(
                scaling, "relative_supply_error"
            ),
        },
        "pickup_eta_calibration": eta_diagnostics,
        "correction_comparison": {
            "before": PRE_CORRECTION_BASELINE,
            "after": {
                "session_count": int(len(sessions)),
                "fleet_size": int(len(fleet)),
                "supply_fit_mae": float(scaling["absolute_supply_error"].mean()),
                "global_beta": float(eta_diagnostics["global_median_ratio"]),
            },
        },
        "product_row_counts": {
            "full_test31_driver_sessions": int(len(sessions)),
            "replay_fleet_template": int(len(fleet)),
            "fleet_scaling_15min": int(len(scaling)),
            "pickup_eta_calibration_15min": int(len(calibration)),
            "stage4_order_replay_base": int(len(replay_base)),
            "historical_valhalla_auto_eta": int(len(valhalla_times)),
        },
    }
    ready = _json_ready(summary)
    output.mkdir(parents=True, exist_ok=True)
    (output / "stage4_s0_summary.json").write_text(
        json.dumps(ready, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_report(root, ready)
    return ready


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
