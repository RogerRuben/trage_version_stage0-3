"""Common frozen service-session slots for the controlled replay environment.

The target qA weights the part of each slot inside the benchmark calendar day.
Only labels change across qA: source windows, coordinates and native identities
remain intact, including source windows after midnight.
"""

from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from stage4.dispatch.fleet_normalization import FLEET_REL, TIMEZONE, FleetScenario
from stage4.fleetpy_adapter.mixed_fleet_adapter import VehicleFixture
from stage4.fleetpy_adapter.upstream import FleetPyCompatibilityError

AVAILABILITY_POLICY = "STOP_ADMISSION_FINISH_COMMITTED"
LABEL_NAMESPACE = "COMMON_SLOT"
_NS_PER_SECOND = 1_000_000_000
_NS_PER_HOUR = 3600 * _NS_PER_SECOND


def _slot_priority(slot_id: str, seed: int) -> str:
    return hashlib.sha256(
        f"{LABEL_NAMESPACE}|{int(seed)}|{slot_id}".encode("utf-8")
    ).hexdigest()


def _nearest_prefix(weights_ns: list[int], target: Fraction) -> int:
    """Return the closest positive-weight prefix; exact ties prefer fewer slots."""
    total_ns = sum(weights_ns)
    target_numerator = target.numerator * total_ns
    best_count, prefix_ns = 0, 0
    best_error = target_numerator
    for count, weight_ns in enumerate(weights_ns, start=1):
        prefix_ns += weight_ns
        error = abs(prefix_ns * target.denominator - target_numerator)
        if error < best_error:
            best_count, best_error = count, error
    return best_count


def _supply_totals(weights_ns: np.ndarray, av: np.ndarray) -> dict:
    """Sum exact overlap weights before converting the small ledger to hours."""
    base_ns = int(weights_ns.sum())
    av_ns = int(weights_ns[av].sum())
    hv_ns = int(weights_ns[~av].sum())
    total_ns = hv_ns + av_ns
    return {
        "baseline_vehicle_hours": base_ns / _NS_PER_HOUR,
        "hv_vehicle_hours": hv_ns / _NS_PER_HOUR,
        "av_vehicle_hours": av_ns / _NS_PER_HOUR,
        "total_vehicle_hours": total_ns / _NS_PER_HOUR,
        "total_supply_error_nanoseconds": total_ns - base_ns,
        "total_vehicle_hour_error": (total_ns - base_ns) / _NS_PER_HOUR,
    }


def build_controlled_fleet(
    root: str | Path,
    *,
    benchmark_start: pd.Timestamp,
    simulation_end: pd.Timestamp,
    requested_q_a: float,
    seed: int = 20260824,
) -> FleetScenario:
    """Relabel every frozen slot, retaining its source window and initial place.

    Duration quartiles use all slots' overlap with local [00:00, 24:00).
    Linear quartile boundaries put equal durations in the same, lower quartile.
    Positive-weight slots use nearest prefixes of a fixed hash order in each
    start-hour/quartile stratum. Zero-day-weight slots use that hash as a uniform
    draw. No simulation outcome, profile or dispatch algorithm enters labeling.

    ``native_fixtures`` contains the complete template, in stable identity order;
    ``native_benchmark_vehicle`` only reports overlap with the requested horizon.
    The horizon never clips a source slot's admission window.
    """
    q_a = float(requested_q_a)
    if not np.isfinite(q_a) or not 0.0 <= q_a <= 1.0:
        raise FleetPyCompatibilityError("requested_q_a must be finite and in [0, 1]")
    start, end = pd.Timestamp(benchmark_start), pd.Timestamp(simulation_end)
    if start.tzinfo is None or end.tzinfo is None or pd.isna(start) or pd.isna(end):
        raise FleetPyCompatibilityError("controlled replay horizons must be timezone-aware")
    start, end = start.tz_convert(TIMEZONE), end.tz_convert(TIMEZONE)
    if end <= start:
        raise FleetPyCompatibilityError("simulation_end must follow benchmark_start")
    day_start = start.normalize()
    day_end = day_start + pd.Timedelta(days=1)

    template = pd.read_parquet(Path(root).resolve() / FLEET_REL).copy()
    required_columns = {
        "source_session_id", "source_driver_id", "availability_start_time",
        "availability_end_time", "initial_lon_wgs84", "initial_lat_wgs84",
    }
    missing = required_columns.difference(template.columns)
    if missing or template.empty:
        raise FleetPyCompatibilityError(
            f"frozen common-session template is empty or missing columns: {sorted(missing)}"
        )
    if template["source_session_id"].isna().any():
        raise FleetPyCompatibilityError("source_session_id must not be missing")
    template["source_session_id"] = template["source_session_id"].astype(str)
    if template["source_session_id"].duplicated().any():
        raise FleetPyCompatibilityError("source_session_id must be unique")
    for column in ("availability_start_time", "availability_end_time"):
        template[column] = pd.to_datetime(template[column], utc=True).dt.tz_convert(TIMEZONE)
        if template[column].isna().any():
            raise FleetPyCompatibilityError("source slot admission windows must be finite")
    if not np.isfinite(template[["initial_lon_wgs84", "initial_lat_wgs84"]].to_numpy()).all():
        raise FleetPyCompatibilityError("source slot initial coordinates must be finite")
    template = template.sort_values(
        ["availability_start_time", "source_session_id"], kind="mergesort"
    ).reset_index(drop=True)
    starts_ns = template["availability_start_time"].to_numpy(dtype="datetime64[ns]").view(np.int64)
    ends_ns = template["availability_end_time"].to_numpy(dtype="datetime64[ns]").view(np.int64)
    durations_ns = ends_ns - starts_ns
    if (durations_ns <= 0).any():
        raise FleetPyCompatibilityError("source slots must have positive durations")
    day_ns = np.maximum(
        0, np.minimum(ends_ns, day_end.value) - np.maximum(starts_ns, day_start.value)
    )
    day_total_ns = int(day_ns.sum())
    if day_total_ns <= 0:
        raise FleetPyCompatibilityError("frozen slots have no benchmark-day vehicle hours")

    quartiles_ns = np.quantile(day_ns, [0.25, 0.5, 0.75], method="linear")
    template["slot_id"] = template["source_session_id"]
    template["native_id"] = np.arange(len(template), dtype=np.int64)
    template["vehicle_id"] = [f"COMMON_SLOT_{index:05d}" for index in range(len(template))]
    template["start_hour"] = template["availability_start_time"].dt.hour
    template["duration_quartile"] = np.searchsorted(quartiles_ns, day_ns, side="left")
    template["_day_ns"] = day_ns
    template["_priority"] = template["slot_id"].map(lambda value: _slot_priority(value, seed))

    # A decimal target avoids changing a mathematically tied prefix through
    # floating-point rounding. The same order yields nested labels for all qA.
    target = Fraction(str(q_a))
    av = np.zeros(len(template), dtype=bool)
    strata = []
    positive = template.loc[template["_day_ns"].gt(0)]
    for (hour, quartile), group in positive.groupby(["start_hour", "duration_quartile"], sort=True):
        ordered = group.sort_values(["_priority", "slot_id"], kind="mergesort")
        weights = [int(value) for value in ordered["_day_ns"]]
        count = _nearest_prefix(weights, target)
        av[ordered.index[:count]] = True
        base_ns, selected_ns = sum(weights), sum(weights[:count])
        strata.append({
            "start_hour": int(hour),
            "duration_quartile": int(quartile),
            "slot_count": int(len(ordered)),
            "av_slot_count": count,
            "baseline_vehicle_hours": base_ns / _NS_PER_HOUR,
            "target_av_vehicle_hours": q_a * base_ns / _NS_PER_HOUR,
            "av_vehicle_hours": selected_ns / _NS_PER_HOUR,
            "av_vehicle_hour_error": (selected_ns - q_a * base_ns) / _NS_PER_HOUR,
        })
    zero_day = day_ns == 0
    for index in template.index[zero_day]:
        hash_value = int(template.at[index, "_priority"], 16)
        av[index] = hash_value * target.denominator < target.numerator * (1 << 256)

    template["vehicle_type"] = np.where(av, "AV", "HV")
    template["vehicle_hours"] = durations_ns / _NS_PER_HOUR
    template["day_vehicle_hours"] = day_ns / _NS_PER_HOUR
    template["availability_policy"] = AVAILABILITY_POLICY
    template["window_inherited_from_source"] = True
    template["av_source_session_end_inherited"] = av
    template["initial_position_rule"] = "FROZEN_S0_SESSION_TEMPLATE"
    overlap = (starts_ns < end.value) & (ends_ns > start.value)
    template["native_benchmark_vehicle"] = overlap
    scenario_fleet = template.drop(columns=["_day_ns", "_priority"])
    fixtures = [
        VehicleFixture(
            vehicle_id=row.vehicle_id,
            native_id=int(row.native_id),
            vehicle_type=row.vehicle_type,
            initial_lon_wgs84=float(row.initial_lon_wgs84),
            initial_lat_wgs84=float(row.initial_lat_wgs84),
            availability_start_time=row.availability_start_time,
            availability_end_time=row.availability_end_time,
            source_session_id=row.source_session_id,
            av_source_session_end_inherited=bool(row.av_source_session_end_inherited),
            availability_policy=AVAILABILITY_POLICY,
        )
        for row in scenario_fleet.itertuples(index=False)
    ]

    hourly_supply = []
    for hour in range(24):
        left = day_start + pd.Timedelta(hours=hour)
        right = left + pd.Timedelta(hours=1)
        weights_ns = np.maximum(0, np.minimum(ends_ns, right.value) - np.maximum(starts_ns, left.value))
        record = _supply_totals(weights_ns, av)
        base_hours = record["baseline_vehicle_hours"]
        realized_q = record["av_vehicle_hours"] / base_hours if base_hours else None
        record.update({
            "hour": hour,
            "start_time": left.isoformat(),
            "end_time": right.isoformat(),
            "requested_q_a": q_a,
            "realized_q_a": realized_q,
            "q_a_error": realized_q - q_a if realized_q is not None else None,
            "q_a_error_pp": 100.0 * (realized_q - q_a) if realized_q is not None else None,
            "target_av_vehicle_hours": q_a * base_hours,
            "av_vehicle_hour_error": record["av_vehicle_hours"] - q_a * base_hours,
        })
        hourly_supply.append(record)

    day_supply = _supply_totals(day_ns, av)
    before_ns = np.maximum(0, np.minimum(ends_ns, day_start.value) - starts_ns)
    spillover_ns = np.maximum(0, ends_ns - np.maximum(starts_ns, day_end.value))
    before_supply = _supply_totals(before_ns, av)
    spillover_supply = _supply_totals(spillover_ns, av)
    before_supply.update({"slot_count": int(np.count_nonzero(before_ns)), "end_time": day_start.isoformat()})
    spillover_supply.update({
        "slot_count": int(np.count_nonzero(spillover_ns)),
        "start_time": day_end.isoformat(),
        "end_time": pd.Timestamp(int(max(day_end.value, ends_ns.max())), tz="UTC").tz_convert(TIMEZONE).isoformat(),
    })
    h_base = day_supply["baseline_vehicle_hours"]
    hv_hours, av_hours = day_supply["hv_vehicle_hours"], day_supply["av_vehicle_hours"]
    target_hv_hours = (1.0 - q_a) * h_base
    label_digest = hashlib.sha256()
    for row in scenario_fleet.itertuples(index=False):
        label_digest.update(f"{row.slot_id}|{row.vehicle_type}\n".encode("utf-8"))
    accounting = {
        "fleet_environment": "COMMON_SESSION_SLOTS",
        "availability_policy": AVAILABILITY_POLICY,
        "seed": int(seed),
        "label_namespace": LABEL_NAMESPACE,
        "label_rule": "START_HOUR_X_GLOBAL_DAY_DURATION_QUARTILE_NEAREST_HASH_PREFIX",
        "duration_quartile_edges_s": (quartiles_ns / _NS_PER_SECOND).tolist(),
        "duration_quartile_method": "GLOBAL_LINEAR_QUANTILES_EQUAL_DURATION_TO_LOWER_QUARTILE",
        "day_start_time": day_start.isoformat(),
        "day_end_time": day_end.isoformat(),
        "q_a_denominator": "PLANNED_ADMISSION_VEHICLE_HOURS_WITHIN_LOCAL_CALENDAR_DAY",
        "h_base_exact": h_base,
        "h_base_day_exact": h_base,
        "h_base_full_template_exact": int(durations_ns.sum()) / _NS_PER_HOUR,
        "requested_q_a": q_a,
        "achieved_q_a": av_hours / h_base,
        "q_a_realized": av_hours / h_base,
        "q_a_error": av_hours / h_base - q_a,
        "q_a_error_pp": 100.0 * (av_hours / h_base - q_a),
        "requested_av_vehicle_hours": q_a * h_base,
        "achieved_av_vehicle_hours": av_hours,
        "raw_hv_residual_vehicle_hours": hv_hours,
        "target_hv_vehicle_hours": target_hv_hours,
        "achieved_hv_vehicle_hours": hv_hours,
        "vehicle_hour_error_pct": abs(hv_hours - target_hv_hours) / target_hv_hours * 100.0 if target_hv_hours else 0.0,
        "total_planned_vehicle_hours": day_supply["total_vehicle_hours"],
        "total_supply_error_nanoseconds": day_supply["total_supply_error_nanoseconds"],
        "total_vehicle_hour_error": day_supply["total_vehicle_hour_error"],
        "slot_count": int(len(template)),
        "av_count": int(av.sum()),
        "selected_hv_session_count": int((~av).sum()),
        "native_fixture_count": int(len(fixtures)),
        "native_benchmark_hv_count": int((overlap & ~av).sum()),
        "native_benchmark_av_count": int((overlap & av).sum()),
        "zero_day_slot_count": int(zero_day.sum()),
        "zero_day_av_slot_count": int((zero_day & av).sum()),
        "av_initial_position_rule": "FROZEN_S0_SESSION_TEMPLATE",
        "hv_unit_semantics": "EFFECTIVE_COMMON_SERVICE_SESSION_SLOT",
        "type_label_sha256": label_digest.hexdigest(),
        "hourly_supply": hourly_supply,
        "strata": strata,
        "pre_day_supply": before_supply,
        "spillover_supply": spillover_supply,
    }
    return FleetScenario(scenario_fleet, fixtures, accounting)
