"""Training-prior idle movement policies for Simulator v3.

The manager deliberately separates the two vehicle types:

* HV destinations are sampled from a training-day empirical transition table
  ``P(destination_zone | origin_zone, time_bin, idle_duration_bin)``.
* AV destinations solve a small zone-level minimum-cost assignment from the
  current idle AV distribution to shortages defined by a training-only demand
  prior.

This module only proposes :class:`VehiclePlan` objects.  Physical movement,
time consumption, cost accumulation, and coordinate mutation remain the sole
responsibility of ``VehicleExecutor`` through normal ``VehicleLeg`` execution.
No test-day future demand is accepted by this policy.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .entities import PlanStop, VehiclePlan, VehicleState
from .enums import StopType


EARTH_M = 6_371_000.0


@dataclass
class IdleMovementDecision:
    vehicle: VehicleState
    plan: VehiclePlan
    movement_reason: str


@dataclass(frozen=True)
class IdleMovementPolicy:
    interval_sec: int = 300
    max_share_per_epoch: float = 0.02
    min_hv_idle_sec: float = 300.0
    shortage_weight_km: float = 2.0
    random_seed: int = 41_031


def _as_frame(value: pd.DataFrame | Path | str | None) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lon1, lat1 = map(math.radians, a)
    lon2, lat2 = map(math.radians, b)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2.0 * EARTH_M * math.asin(min(1.0, math.sqrt(h)))


def _duration_bin(idle_sec: float) -> str:
    minutes = max(0.0, float(idle_sec)) / 60.0
    if minutes < 5:
        return "00_05"
    if minutes < 15:
        return "05_15"
    if minutes < 30:
        return "15_30"
    if minutes < 60:
        return "30_60"
    return "60_plus"


class IdleMovementManager:
    """Produce auditable HV reposition and AV rebalance plans.

    Parameters may be dataframes (convenient for tests) or persisted parquet /
    CSV paths.  Empty priors are rejected when the corresponding policy is
    invoked; there is intentionally no hash-neighbour fallback.
    """

    REQUIRED_HV_COLUMNS = {
        "origin_zone",
        "destination_zone",
        "time_bin",
        "idle_duration_bin",
        "transition_probability",
        "sample_count",
    }
    REQUIRED_AV_COLUMNS = {"zone", "time_bin", "forecast_demand"}

    def __init__(
        self,
        zone_system: dict,
        hv_transition_table: pd.DataFrame | Path | str | None = None,
        av_demand_prior: pd.DataFrame | Path | str | None = None,
        *,
        policy: IdleMovementPolicy | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.zone_system = dict(zone_system)
        self.policy = policy or IdleMovementPolicy()
        self.interval_sec = int(self.policy.interval_sec)
        self.max_share_per_epoch = float(self.policy.max_share_per_epoch)
        self.hv_transitions = _as_frame(hv_transition_table)
        self.av_demand_prior = _as_frame(av_demand_prior)
        self._validate_prior_schema()
        self.rng = rng if rng is not None else np.random.default_rng(self.policy.random_seed)
        self.records: list[dict] = []
        self._idle_since: dict[str, pd.Timestamp] = {}

    @classmethod
    def from_data_root(
        cls,
        zone_system: dict,
        data_root: Path | str,
        *,
        policy: IdleMovementPolicy | None = None,
        rng: np.random.Generator | None = None,
    ) -> "IdleMovementManager":
        root = Path(data_root)
        return cls(
            zone_system,
            root / "hv_idle_zone_transition.parquet",
            root / "av_rebalancing_demand_prior.parquet",
            policy=policy,
            rng=rng,
        )

    def _validate_prior_schema(self) -> None:
        if len(self.hv_transitions):
            missing = self.REQUIRED_HV_COLUMNS - set(self.hv_transitions.columns)
            if missing:
                raise ValueError(f"HV transition table missing columns: {sorted(missing)}")
            if bool(self.hv_transitions.get("uses_test_day_future_demand", pd.Series(False)).fillna(False).any()):
                raise ValueError("HV transition table declares test-day future-demand use.")
        if len(self.av_demand_prior):
            missing = self.REQUIRED_AV_COLUMNS - set(self.av_demand_prior.columns)
            if missing:
                raise ValueError(f"AV demand prior missing columns: {sorted(missing)}")
            if bool(self.av_demand_prior.get("uses_test_day_future_demand", pd.Series(False)).fillna(False).any()):
                raise ValueError("AV demand prior declares test-day future-demand use.")

    def should_run(self, now: pd.Timestamp) -> bool:
        seconds = int(pd.Timestamp(now).timestamp())
        return seconds % self.interval_sec == 0

    def zone_center(self, zone: str) -> tuple[float, float]:
        match = re.fullmatch(r"z(-?\d+)_(-?\d+)", str(zone))
        grid = float(self.zone_system.get("grid_size", 0.02))
        min_lon = float(self.zone_system.get("min_lon", 108.9))
        min_lat = float(self.zone_system.get("min_lat", 34.2))
        if not match:
            raise ValueError(f"Invalid operational zone: {zone!r}")
        x = int(match.group(1))
        y = int(match.group(2))
        return min_lon + (x + 0.5) * grid, min_lat + (y + 0.5) * grid

    @staticmethod
    def time_bin(now: pd.Timestamp) -> int:
        stamp = pd.Timestamp(now)
        return int(stamp.hour * 2 + stamp.minute // 30)

    def _eligible_limit(self, count: int) -> int:
        if count <= 0:
            return 0
        return min(count, max(1, int(math.ceil(count * self.max_share_per_epoch))))

    def _hv_distribution(
        self,
        origin_zone: str,
        time_bin: int,
        idle_duration_bin: str,
    ) -> tuple[pd.DataFrame, str]:
        if self.hv_transitions.empty:
            raise RuntimeError("HV idle repositioning requires a training-day transition table.")
        table = self.hv_transitions
        masks = [
            (
                table["origin_zone"].astype(str).eq(str(origin_zone))
                & table["time_bin"].astype(int).eq(int(time_bin))
                & table["idle_duration_bin"].astype(str).eq(str(idle_duration_bin)),
                "origin_time_idle",
            ),
            (
                table["origin_zone"].astype(str).eq(str(origin_zone))
                & table["time_bin"].astype(int).eq(int(time_bin)),
                "origin_time",
            ),
            (table["origin_zone"].astype(str).eq(str(origin_zone)), "origin"),
            (table["time_bin"].astype(int).eq(int(time_bin)), "time"),
        ]
        for mask, source in masks:
            subset = table.loc[mask]
            if len(subset):
                grouped = subset.groupby("destination_zone", as_index=False).agg(
                    transition_probability=("transition_probability", "sum"),
                    sample_count=("sample_count", "sum"),
                )
                return grouped, source
        grouped = table.groupby("destination_zone", as_index=False).agg(
            transition_probability=("transition_probability", "sum"),
            sample_count=("sample_count", "sum"),
        )
        return grouped, "global"

    def _sample_hv_target(
        self,
        origin_zone: str,
        now: pd.Timestamp,
        idle_sec: float,
    ) -> tuple[str, str, float, int]:
        duration_bin = _duration_bin(idle_sec)
        distribution, source = self._hv_distribution(origin_zone, self.time_bin(now), duration_bin)
        # Remaining in the same zone is a valid empirical outcome, but it does
        # not require a physical movement plan.  It is retained in sampling so
        # reposition frequency remains data-driven.
        weights = distribution["transition_probability"].astype(float).clip(lower=0).to_numpy()
        if weights.sum() <= 0:
            weights = distribution["sample_count"].astype(float).clip(lower=0).to_numpy()
        if weights.sum() <= 0:
            raise ValueError("HV transition distribution contains no positive mass.")
        weights = weights / weights.sum()
        position = int(self.rng.choice(len(distribution), p=weights))
        row = distribution.iloc[position]
        return str(row.destination_zone), source, float(weights[position]), int(row.sample_count)

    def _build_hv_plans(self, vehicles: list[VehicleState], now: pd.Timestamp) -> list[IdleMovementDecision]:
        current_idle_ids = {vehicle.vehicle_id for vehicle in vehicles}
        # A vehicle disappearing from the incremental idle index has either
        # started a task/movement or gone offline.  Its next idle spell must
        # therefore start a fresh duration clock.
        for vehicle_id in list(self._idle_since):
            if vehicle_id not in current_idle_ids:
                self._idle_since.pop(vehicle_id, None)
        for vehicle in vehicles:
            self._idle_since.setdefault(vehicle.vehicle_id, now)
        eligible = [
            vehicle
            for vehicle in sorted(vehicles, key=lambda item: item.vehicle_id)
            if (now - self._idle_since[vehicle.vehicle_id]).total_seconds() >= self.policy.min_hv_idle_sec
        ]
        if not eligible:
            return []
        order = self.rng.permutation(len(eligible))
        selected = [eligible[int(i)] for i in order[: self._eligible_limit(len(eligible))]]
        decisions: list[IdleMovementDecision] = []
        for vehicle in selected:
            idle_sec = max(0.0, (now - self._idle_since[vehicle.vehicle_id]).total_seconds())
            target_zone, source, probability, sample_count = self._sample_hv_target(vehicle.current_zone, now, idle_sec)
            if target_zone == vehicle.current_zone:
                self.records.append({
                    "vehicle_id": vehicle.vehicle_id,
                    "vehicle_type": "HV",
                    "movement_time": str(now),
                    "origin_zone": vehicle.current_zone,
                    "target_zone": target_zone,
                    "movement_reason": "empirical_idle_repositioning_stay",
                    "policy_source": source,
                    "training_sample_count": sample_count,
                    "sampled_probability": probability,
                    "idle_duration_sec": idle_sec,
                    "plan_created": False,
                })
                continue
            decision = self._movement_plan(
                vehicle,
                now,
                target_zone,
                StopType.HV_REPOSITION,
                "empirical_idle_repositioning",
            )
            decisions.append(decision)
            self._idle_since.pop(vehicle.vehicle_id, None)
            self.records.append({
                "vehicle_id": vehicle.vehicle_id,
                "vehicle_type": "HV",
                "movement_time": str(now),
                "origin_zone": vehicle.current_zone,
                "target_zone": target_zone,
                "movement_reason": decision.movement_reason,
                "policy_source": source,
                "training_sample_count": sample_count,
                "sampled_probability": probability,
                "idle_duration_sec": idle_sec,
                "plan_created": True,
            })
        return decisions

    def _av_targets(self, vehicles: list[VehicleState], now: pd.Timestamp) -> list[tuple[VehicleState, str, dict]]:
        if self.av_demand_prior.empty:
            raise RuntimeError("AV rebalancing requires a training-only demand prior.")
        prior = self.av_demand_prior
        subset = prior.loc[prior["time_bin"].astype(int).eq(self.time_bin(now))].copy()
        source = "time_bin"
        if subset.empty:
            subset = prior.groupby("zone", as_index=False).agg(forecast_demand=("forecast_demand", "mean"))
            source = "global_time_fallback"
        else:
            subset = subset.groupby("zone", as_index=False).agg(forecast_demand=("forecast_demand", "sum"))
        subset["forecast_demand"] = subset["forecast_demand"].astype(float).clip(lower=0.0)
        if subset["forecast_demand"].sum() <= 0 or not vehicles:
            return []

        zones = subset["zone"].astype(str).tolist()
        demand = dict(zip(zones, subset["forecast_demand"].astype(float)))
        current: dict[str, int] = {}
        by_zone: dict[str, list[VehicleState]] = {}
        for vehicle in sorted(vehicles, key=lambda item: item.vehicle_id):
            current[vehicle.current_zone] = current.get(vehicle.current_zone, 0) + 1
            by_zone.setdefault(vehicle.current_zone, []).append(vehicle)

        total = len(vehicles)
        raw_targets = {zone: total * demand[zone] / sum(demand.values()) for zone in zones}
        targets = {zone: int(math.floor(value)) for zone, value in raw_targets.items()}
        remainder = total - sum(targets.values())
        ranked_fraction = sorted(zones, key=lambda zone: (-(raw_targets[zone] - targets[zone]), zone))
        for zone in ranked_fraction[:remainder]:
            targets[zone] += 1

        deficits = {zone: max(0, targets.get(zone, 0) - current.get(zone, 0)) for zone in zones}
        sources: list[VehicleState] = []
        for zone, items in sorted(by_zone.items()):
            surplus = max(0, len(items) - targets.get(zone, 0))
            sources.extend(items[:surplus])
        limit = self._eligible_limit(len(vehicles))
        if not sources or not any(deficits.values()) or limit <= 0:
            return []
        sources = sources[:limit]
        slots = [zone for zone in sorted(deficits) for _ in range(deficits[zone])]
        if not slots:
            return []

        cost = np.empty((len(sources), len(slots)), dtype=float)
        max_deficit = max(deficits.values()) or 1
        for row_idx, vehicle in enumerate(sources):
            origin = self.zone_center(vehicle.current_zone)
            for col_idx, zone in enumerate(slots):
                distance_km = _haversine_m(origin, self.zone_center(zone)) / 1000.0
                shortage_benefit = self.policy.shortage_weight_km * deficits[zone] / max_deficit
                cost[row_idx, col_idx] = distance_km - shortage_benefit

        assignments: list[tuple[int, int]] = []
        try:
            from scipy.optimize import linear_sum_assignment

            rows, cols = linear_sum_assignment(cost)
            assignments = list(zip(rows.tolist(), cols.tolist()))
            solver = "scipy_linear_sum_assignment"
        except ImportError:
            # Deterministic sparse fallback: repeatedly take the cheapest
            # remaining source-slot pair.  This is only used when scipy is not
            # installed and is explicitly recorded for audit.
            remaining_rows = set(range(len(sources)))
            remaining_cols = set(range(len(slots)))
            while remaining_rows and remaining_cols:
                pair = min((cost[i, j], i, j) for i in remaining_rows for j in remaining_cols)
                _, i, j = pair
                assignments.append((i, j))
                remaining_rows.remove(i)
                remaining_cols.remove(j)
            solver = "deterministic_greedy_fallback"

        results: list[tuple[VehicleState, str, dict]] = []
        for row_idx, col_idx in assignments:
            vehicle = sources[row_idx]
            target_zone = slots[col_idx]
            if target_zone == vehicle.current_zone:
                continue
            results.append((vehicle, target_zone, {
                "policy_source": source,
                "solver": solver,
                "forecast_demand": demand[target_zone],
                "target_supply": targets[target_zone],
                "available_supply_before": current.get(target_zone, 0),
                "shortage_before": deficits[target_zone],
                "assignment_cost": float(cost[row_idx, col_idx]),
            }))
        return results

    def _build_av_plans(self, vehicles: list[VehicleState], now: pd.Timestamp) -> list[IdleMovementDecision]:
        decisions: list[IdleMovementDecision] = []
        for vehicle, target_zone, metadata in self._av_targets(vehicles, now):
            decision = self._movement_plan(
                vehicle,
                now,
                target_zone,
                StopType.AV_REBALANCE,
                "platform_training_demand_rebalancing",
            )
            decisions.append(decision)
            self.records.append({
                "vehicle_id": vehicle.vehicle_id,
                "vehicle_type": "AV",
                "movement_time": str(now),
                "origin_zone": vehicle.current_zone,
                "target_zone": target_zone,
                "movement_reason": decision.movement_reason,
                "plan_created": True,
                **metadata,
            })
        return decisions

    def _movement_plan(
        self,
        vehicle: VehicleState,
        now: pd.Timestamp,
        target_zone: str,
        stop_type: StopType,
        reason: str,
    ) -> IdleMovementDecision:
        lon, lat = self.zone_center(target_zone)
        version = vehicle.plan_version + 1
        plan = VehiclePlan(
            vehicle_id=vehicle.vehicle_id,
            plan_version=version,
            stops=[
                PlanStop(
                    stop_id=f"{vehicle.vehicle_id}:{version}:{stop_type.value}:{target_zone}",
                    stop_type=stop_type,
                    request_id=None,
                    lon=lon,
                    lat=lat,
                    zone=target_zone,
                    earliest_start=None,
                    latest_start=None,
                    planned_arrival=now,
                    planned_departure=now,
                    locked=False,
                )
            ],
            created_time=now,
            trigger=reason,
            feasible=True,
            objective_value=0.0,
        )
        return IdleMovementDecision(vehicle, plan, reason)

    def build_plans(
        self,
        vehicles: list[VehicleState],
        now: pd.Timestamp,
        vehicle_type: str,
    ) -> list[IdleMovementDecision]:
        if not self.should_run(now):
            return []
        if vehicle_type == "HV":
            return self._build_hv_plans(vehicles, now)
        if vehicle_type == "AV":
            return self._build_av_plans(vehicles, now)
        raise ValueError(f"Unsupported idle-movement vehicle type: {vehicle_type!r}")
