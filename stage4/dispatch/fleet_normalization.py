"""Active vehicle-hours normalization for the Stage4 mixed-fleet baseline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stage4.fleetpy_adapter.mixed_fleet_adapter import VehicleFixture
from stage4.fleetpy_adapter.upstream import FleetPyCompatibilityError

TIMEZONE = "Asia/Shanghai"
SCALING_REL = Path("stage4/input/replay_foundation/fleet_scaling_15min.parquet")
FLEET_REL = Path("stage4/input/replay_foundation/replay_fleet_template.parquet")


@dataclass
class FleetScenario:
    scenario_fleet: pd.DataFrame
    native_fixtures: list[VehicleFixture]
    accounting: dict


def _priority(namespace: str, seed: int, value: str) -> str:
    return hashlib.sha256(
        f"{namespace}|{int(seed)}|{value}".encode("utf-8")
    ).hexdigest()


def exact_baseline_vehicle_hours(root: str | Path) -> float:
    """Return exact continuous hours in the frozen replay fleet template."""
    path = Path(root).resolve() / FLEET_REL
    template = pd.read_parquet(
        path,
        columns=[
            "source_session_id",
            "availability_start_time",
            "availability_end_time",
        ],
    )
    if template["source_session_id"].astype(str).duplicated().any():
        raise FleetPyCompatibilityError("source_session_id must be unique")
    start = pd.to_datetime(template["availability_start_time"], utc=True)
    end = pd.to_datetime(template["availability_end_time"], utc=True)
    duration_hours = (end - start).dt.total_seconds() / 3600.0
    if not np.isfinite(duration_hours).all() or (duration_hours <= 0.0).any():
        raise FleetPyCompatibilityError(
            "non-positive or non-finite HV session duration"
        )
    return float(duration_hours.sum())


def build_fleet_scenario(
    root: str | Path,
    *,
    benchmark_start: pd.Timestamp,
    simulation_end: pd.Timestamp,
    requested_q_a: float,
    seed: int,
    max_hv_hour_error_pct: float = 2.0,
) -> FleetScenario:
    """Build one deterministic qA scenario without optimizing the supply curve."""
    root = Path(root).resolve()
    scaling = pd.read_parquet(root / SCALING_REL)
    if len(scaling) != 96 or "simulated_active_supply" not in scaling:
        raise FleetPyCompatibilityError("frozen fleet scaling must contain 96 bins")
    h_base_15min_equivalent = 0.25 * float(scaling["simulated_active_supply"].sum())
    h_base_exact = exact_baseline_vehicle_hours(root)
    requested_av_hours = float(requested_q_a) * h_base_exact
    n_av = int(round(requested_av_hours / 24.0))
    achieved_av_hours = 24.0 * n_av
    achieved_q_a = achieved_av_hours / h_base_exact
    raw_hv_residual_hours = h_base_exact - achieved_av_hours
    target_hv_hours = max(raw_hv_residual_hours, 0.0)
    template = pd.read_parquet(root / FLEET_REL).copy()
    for column in ("availability_start_time", "availability_end_time"):
        template[column] = pd.to_datetime(template[column], utc=True).dt.tz_convert(
            TIMEZONE
        )
    if template["source_session_id"].astype(str).duplicated().any():
        raise FleetPyCompatibilityError("source_session_id must be unique")
    template["vehicle_hours"] = (
        template["availability_end_time"] - template["availability_start_time"]
    ).dt.total_seconds() / 3600.0
    if (template["vehicle_hours"] <= 0).any():
        raise FleetPyCompatibilityError("non-positive HV service-session duration")
    total_template_hours = float(template["vehicle_hours"].sum())
    if not np.isclose(total_template_hours, h_base_exact, rtol=0.0, atol=1e-9):
        raise FleetPyCompatibilityError("exact baseline hours disagree with template")
    if target_hv_hours > total_template_hours:
        raise FleetPyCompatibilityError("insufficient HV service-session hours")
    fraction = target_hv_hours / total_template_hours
    template["start_bin_15m"] = (
        template["availability_start_time"].dt.hour * 60
        + template["availability_start_time"].dt.minute
    ) // 15
    template["_priority"] = (
        template["source_session_id"]
        .astype(str)
        .map(lambda value: _priority("HV", seed, value))
    )
    selected_indices: list[int] = []
    for _, group in template.groupby("start_bin_15m", sort=True):
        count = int(round(len(group) * fraction))
        ordered = group.sort_values(
            ["_priority", "source_session_id"], kind="mergesort"
        )
        selected_indices.extend(ordered.head(count).index.tolist())
    selected_hv = template.loc[selected_indices].copy()
    achieved_hv_hours = float(selected_hv["vehicle_hours"].sum())
    hv_hour_error_pct = (
        abs(achieved_hv_hours - target_hv_hours) / target_hv_hours * 100.0
        if target_hv_hours > 0.0
        else 0.0
    )
    if hv_hour_error_pct > float(max_hv_hour_error_pct):
        raise FleetPyCompatibilityError(
            f"HV vehicle-hour error {hv_hour_error_pct:.3f}% exceeds tolerance"
        )

    near_start = template.loc[
        template["availability_start_time"].le(
            benchmark_start + pd.Timedelta(minutes=30)
        )
        & template["availability_end_time"].gt(benchmark_start)
    ].copy()
    av_pool_rule = "ACTIVE_OR_STARTING_WITHIN_30_MINUTES"
    if len(near_start) < n_av:
        near_start = template.copy()
        av_pool_rule = "FULL_S0_INITIAL_POSITION_SUPPORT_FALLBACK"
    near_start["_av_priority"] = (
        near_start["source_session_id"]
        .astype(str)
        .map(lambda value: _priority("AV", seed, value))
    )
    av_source = near_start.sort_values(
        ["_av_priority", "source_session_id"], kind="mergesort"
    ).head(n_av)
    if len(av_source) != n_av:
        raise FleetPyCompatibilityError("insufficient AV initial-position support")

    scenario_rows: list[dict] = []
    native_fixtures: list[VehicleFixture] = []
    native_hv = selected_hv.loc[
        selected_hv["availability_start_time"].lt(simulation_end)
        & selected_hv["availability_end_time"].gt(benchmark_start)
    ].sort_values(["availability_start_time", "source_session_id"], kind="mergesort")
    native_id = 0
    native_session_ids = set(native_hv["source_session_id"].astype(str))
    for row_index, row in enumerate(
        selected_hv.sort_values(
            ["availability_start_time", "source_session_id"], kind="mergesort"
        ).itertuples(index=False)
    ):
        vehicle_id = f"HV_S3_{row_index:05d}"
        scenario_rows.append(
            {
                "vehicle_id": vehicle_id,
                "vehicle_type": "HV",
                "source_driver_id": str(row.source_driver_id),
                "source_session_id": str(row.source_session_id),
                "initial_lon_wgs84": float(row.initial_lon_wgs84),
                "initial_lat_wgs84": float(row.initial_lat_wgs84),
                "availability_start_time": row.availability_start_time,
                "availability_end_time": row.availability_end_time,
                "vehicle_hours": float(row.vehicle_hours),
                "native_benchmark_vehicle": str(row.source_session_id)
                in native_session_ids,
                "initial_position_rule": "FROZEN_S0_SESSION_TEMPLATE",
            }
        )
    hv_vehicle_id = {
        row["source_session_id"]: row["vehicle_id"] for row in scenario_rows
    }
    for row in native_hv.itertuples(index=False):
        native_fixtures.append(
            VehicleFixture(
                vehicle_id=hv_vehicle_id[str(row.source_session_id)],
                native_id=native_id,
                vehicle_type="HV",
                initial_lon_wgs84=float(row.initial_lon_wgs84),
                initial_lat_wgs84=float(row.initial_lat_wgs84),
                availability_start_time=max(
                    row.availability_start_time, benchmark_start
                ),
                availability_end_time=row.availability_end_time,
                source_session_id=str(row.source_session_id),
                av_source_session_end_inherited=False,
            )
        )
        native_id += 1

    day_start = benchmark_start.normalize()
    day_end = day_start + pd.Timedelta(days=1)
    for av_index, row in enumerate(av_source.itertuples(index=False)):
        vehicle_id = f"AV_S3_{av_index:04d}"
        scenario_rows.append(
            {
                "vehicle_id": vehicle_id,
                "vehicle_type": "AV",
                "source_driver_id": str(row.source_driver_id),
                "source_session_id": str(row.source_session_id),
                "initial_lon_wgs84": float(row.initial_lon_wgs84),
                "initial_lat_wgs84": float(row.initial_lat_wgs84),
                "availability_start_time": day_start,
                "availability_end_time": day_end,
                "vehicle_hours": 24.0,
                "native_benchmark_vehicle": True,
                "initial_position_rule": av_pool_rule,
            }
        )
        native_fixtures.append(
            VehicleFixture(
                vehicle_id=vehicle_id,
                native_id=native_id,
                vehicle_type="AV",
                initial_lon_wgs84=float(row.initial_lon_wgs84),
                initial_lat_wgs84=float(row.initial_lat_wgs84),
                availability_start_time=benchmark_start,
                availability_end_time=simulation_end,
                source_session_id=str(row.source_session_id),
                av_source_session_end_inherited=False,
            )
        )
        native_id += 1

    scenario_fleet = pd.DataFrame(scenario_rows).sort_values(
        ["vehicle_type", "vehicle_id"], kind="mergesort"
    )
    accounting = {
        "h_base_exact": h_base_exact,
        "h_base_15min_equivalent": h_base_15min_equivalent,
        "requested_q_a": float(requested_q_a),
        "achieved_q_a": achieved_q_a,
        "av_count": n_av,
        "requested_av_vehicle_hours": requested_av_hours,
        "achieved_av_vehicle_hours": achieved_av_hours,
        "raw_hv_residual_vehicle_hours": raw_hv_residual_hours,
        "target_hv_vehicle_hours": target_hv_hours,
        "achieved_hv_vehicle_hours": achieved_hv_hours,
        "vehicle_hour_error_pct": hv_hour_error_pct,
        "selected_hv_session_count": int(len(selected_hv)),
        "native_benchmark_hv_count": int(len(native_hv)),
        "native_benchmark_av_count": n_av,
        "hv_sampling_fraction": fraction,
        "av_initial_position_rule": av_pool_rule,
        "hv_unit_semantics": "EFFECTIVE_HV_SERVICE_SESSION_TEMPLATE",
    }
    return FleetScenario(scenario_fleet, native_fixtures, accounting)
