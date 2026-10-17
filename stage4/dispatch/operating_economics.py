"""Scenario operating contributions, not enterprise profit or an AV-use bonus.

The same posted fare applies to both types. Fixed online/asset costs are outside
the fixed-supply dispatch problem. Physical kilometres are charged once; time
opportunity is represented by the shared continuation, not a second penalty.
"""
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from functools import cached_property, lru_cache
import hashlib
import json
from math import isfinite


MODE = "PLATFORM_OPERATING_CONTRIBUTION_V1"
MONEY_SCALE = 100  # integer cents, BEFORE exact scenario probability weights
ORDER_LEDGER_COLUMNS = ("native_request_id", "native_vehicle_id", "order_id", "profile_id",
    "fare_cents", "driver_payout_cents", "variable_cost_cents", "contribution_cents",
    "service_distance_m", "pickup_distance_m", "economic_config_sha256")
RELOCATION_LEDGER_COLUMNS = ("native_vehicle_id", "start_time_s", "profile_id",
    "empty_distance_m", "cost_cents")


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"{name} must be an explicit finite number")
    result = float(value)
    if not isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def _cents(value):
    return int((Decimal(str(value)) * MONEY_SCALE).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class OperatingEconomics:
    fare_base_cny: float
    fare_per_km_cny: float
    hv_platform_share: float
    av_variable_per_km_cny: float
    hv_directed_reposition_compensation_per_km_cny: float = 0.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            _number(getattr(self, name), name)
        if not 0 < self.hv_platform_share < 1:
            raise ValueError("HV platform share must be strictly between zero and one")
        if not self.fare_base_cny and not self.fare_per_km_cny:
            raise ValueError("posted fare cannot be identically zero")
        if not self.av_variable_per_km_cny:
            raise ValueError("AV variable operating cost must be positive")

    @classmethod
    def from_config(cls, cfg):
        raw = cfg.get("dispatch_economics")
        if raw is None:
            return None
        if not isinstance(raw, dict) or raw.get("mode") != MODE:
            raise ValueError("unknown dispatch economics; no silent legacy-cost fallback")
        # Missing/unassigned values are not zero. Non-core expense categories
        # are explicitly outside this model, not silently parsed as free.
        return cls(**{name: raw[name] for name in cls.__dataclass_fields__})

    @cached_property
    def fingerprint(self):
        payload = {name: str(getattr(self, name)) for name in self.__dataclass_fields__}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @lru_cache(maxsize=32768)
    def fare_cents(self, service_distance_m):
        distance = _number(service_distance_m, "decision-time service route distance", positive=True)
        return _cents(Decimal(str(self.fare_base_cny))
            + Decimal(str(self.fare_per_km_cny)) * Decimal(str(distance)) / 1000)

    def movement_cost_cents(self, profile, distance_m):
        distance = _number(distance_m, "physical empty distance")
        if profile not in ("C", "HV"):
            raise ValueError("economic native scope is the existing C/HV fleet")
        rate = (self.av_variable_per_km_cny if profile == "C"
            else self.hv_directed_reposition_compensation_per_km_cny)
        return _cents(Decimal(str(rate)) * Decimal(str(distance)) / 1000)

    @lru_cache(maxsize=32768)
    def _quote_values(self, service_distance_m, profile, pickup_distance_m):
        distance = _number(service_distance_m, "decision-time service route distance", positive=True)
        pickup = _number(pickup_distance_m, "pickup route distance")
        fare = self.fare_cents(distance)
        if profile == "HV":
            kept = _cents(Decimal(fare) / MONEY_SCALE * Decimal(str(self.hv_platform_share)))
            payout, cost = fare-kept, 0
        elif profile == "C":
            payout = 0
            cost = _cents(Decimal(str(self.av_variable_per_km_cny))
                * (Decimal(str(distance)) + Decimal(str(pickup))) / 1000)
        else:
            raise ValueError("economic native scope is the existing C/HV fleet")
        return fare, payout, cost, fare-payout-cost

    def quote(self, task, profile, pickup_distance_m):
        distance=_number(task.service_distance_m,"decision-time service route distance",positive=True)
        pickup=_number(pickup_distance_m,"pickup route distance")
        fare, payout, cost, contribution = self._quote_values(distance, profile, pickup)
        return dict(fare_cents=fare, driver_payout_cents=payout,
            variable_cost_cents=cost, contribution_cents=contribution,
            service_distance_m=distance, pickup_distance_m=pickup,
            profile_id=profile, economic_config_sha256=self.fingerprint)

    def action_cents(self, action, root, task=None):
        if action.kind == "SERVE":
            if task is None:
                task = (root.context or {}).get("task")
            if task is None or task.request_id != action.request_id:
                raise ValueError("SERVE economics must bind the same revealed task")
            return self.quote(task, root.profile_id, action.empty_distance_m)["contribution_cents"]
        if action.kind == "RELOCATE":
            return -self.movement_cost_cents(root.profile_id, action.empty_distance_m)
        return 0  # WAIT/BUSY earn no repeated already-committed fare

    def path_cents(self, tasks, profile, path):
        return sum(self._quote_values(tasks[leg[0]].service_distance_m, profile, leg[4])[3] for leg in path)


class OperatingLedger:
    """Settle each actually committed order once; never sum rolling plan values."""
    def __init__(self):
        self.orders = {}
        self.moves = {}

    def commit(self, request_id, vehicle_id, order_id, quote):
        key = int(request_id)
        row = dict(quote, native_request_id=key, native_vehicle_id=int(vehicle_id), order_id=str(order_id))
        old = self.orders.get(key)
        if old is not None and old != row:
            raise ValueError("a committed request received two financial allocations")
        self.orders[key] = row

    def move(self, vehicle_id, start_s, profile, distance_m, model):
        key = int(vehicle_id), float(start_s)
        row = dict(native_vehicle_id=key[0], start_time_s=key[1], profile_id=profile,
            empty_distance_m=float(distance_m), cost_cents=model.movement_cost_cents(profile, distance_m))
        old = self.moves.get(key)
        if old is not None and old != row:
            raise ValueError("one physical relocation received two financial allocations")
        self.moves[key] = row

    def settle_moves(self, rows, model):
        """Post-execution accounting only: interrupted moves charge travelled km.

        This method is never called by a decision or a chain oracle. The final
        runner calls it after physical drain, so unfinished moves are errors.
        """
        settled={}
        for row in rows:
            if row.get("actual_distance_m") is None or row.get("status")=="IN_PROGRESS":
                raise ValueError("completed economic report requires drained actual relocation distances")
            key=int(row["native_vehicle_id"]),float(row["start_time_s"])
            if key in settled: raise ValueError("duplicate executed relocation in settlement")
            profile="HV" if row["vehicle_type"]=="HV" else "C"
            distance=_number(row["actual_distance_m"],"executed relocation distance")
            settled[key]=dict(native_vehicle_id=key[0],start_time_s=key[1],profile_id=profile,
                empty_distance_m=distance,cost_cents=model.movement_cost_cents(profile,distance))
        self.moves=settled

    def summary(self, completed_ids):
        completed = set(map(int, completed_ids))
        rows = [row for rid, row in self.orders.items() if rid in completed]
        relocation = sum(row["cost_cents"] for row in self.moves.values())
        return dict(scope="SCENARIO_CONTRIBUTION_FROM_COMMITTED_ROUTE_DISTANCES_NOT_OBSERVED_CASH_PROFIT",
            money_unit="CNY", rounding="HALF_UP_PER_ORDER_TO_CENT",
            committed_orders=len(self.orders), completed_orders=len(rows),
            fare_cny=sum(r["fare_cents"] for r in rows)/MONEY_SCALE,
            driver_payout_cny=sum(r["driver_payout_cents"] for r in rows)/MONEY_SCALE,
            variable_cost_cny=sum(r["variable_cost_cents"] for r in rows)/MONEY_SCALE,
            relocation_cost_cny=relocation/MONEY_SCALE,
            operating_contribution_cny=(sum(r["contribution_cents"] for r in rows)-relocation)/MONEY_SCALE,
            fixed_costs_in_objective=False, realized_travel_times_used_for_decision=False,
            rolling_planning_values_added_to_realized_ledger=False,
            by_profile={p:dict(completed_orders=sum(r["profile_id"]==p for r in rows),
                order_contribution_cny=sum(r["contribution_cents"] for r in rows if r["profile_id"]==p)/MONEY_SCALE,
                relocation_cost_cny=sum(r["cost_cents"] for r in self.moves.values() if r["profile_id"]==p)/MONEY_SCALE,
                operating_contribution_cny=(sum(r["contribution_cents"] for r in rows if r["profile_id"]==p)
                    -sum(r["cost_cents"] for r in self.moves.values() if r["profile_id"]==p))/MONEY_SCALE)
                for p in ("C", "HV")})
