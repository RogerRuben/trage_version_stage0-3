"""Citywide Test31 data contract; no native controller or solver is connected.

The decision snapshot contains only released, uncommitted, unexpired requests.
Frozen M3 P50 predictions are decision inputs. Historical execution durations
live in a separate environment lookup, accessible only after commitment. This
module does not replay movement, infer online shifts, or select dispatch arcs.
"""

from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd
import pyarrow.parquet as pq

from stage4.dispatch.acceptance import passenger_acceptance
from stage4.dispatch.controlled_fleet import AVAILABILITY_POLICY, build_controlled_fleet
from stage4.dispatch.fleet_normalization import FLEET_REL, FleetScenario
from stage4.fleetpy_adapter.test31_demand_adapter import ORDER_BASE_REL, TIMEZONE

CONTRACT_VERSION = "stage4_research_world.v1"
ROUTES_REL = Path("stage4/output/symmetric_flexibility_v1/input/test31_research_routes.parquet")
BASE_CONFIG_REL = Path("stage4/config/symmetric_flexibility_full_day_v1.json")
CONTROLLED_CONFIG_REL = Path("stage4/config/controlled_replay_v2.json")
SOURCE_PROFILE_ID = "M"  # Same order/coordinate rows as the existing common-population replay.
COORDINATE_COLUMNS = (
    "pickup_lon_wgs84", "pickup_lat_wgs84", "dropoff_lon_wgs84", "dropoff_lat_wgs84",
)
DECISION_BASE_COLUMNS = ("order_id", "request_time", *COORDINATE_COLUMNS, "profile_id")
ROUTE_COLUMNS = (
    "order_id", "common_eligible", "predicted_route_time_p50_s",
    "compatible_C", "compatible_M", "compatible_A", "compatible_HV",
)
TRUTH_COLUMNS = (
    "observed_service_time_s", "realized_service_time_s", "observed_arrival_time_s", "arrival_time",
)


def _seconds(value: float) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError("episode time must be finite")
    return value


def _optional_duration(value) -> float | None:
    if pd.isna(value):
        return None
    value = float(value)
    return value if isfinite(value) and value > 0 else None


def _read_projection(path: Path, columns, *, source_profile_id: str | None = None) -> pd.DataFrame:
    """Stream selected columns, retaining only a lightweight one-day table."""
    parts = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=65536, columns=list(columns)):
        frame = batch.to_pandas()
        if source_profile_id is not None:
            frame = frame.loc[frame.profile_id.astype(str).eq(source_profile_id)]
        if len(frame):
            parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


@dataclass(frozen=True)
class DecisionRequest:
    """Allowlisted decision-time information; no realized or observed fields."""

    order_id: str
    release_time_s: float
    expires_at_s: float
    pickup_lon_wgs84: float
    pickup_lat_wgs84: float
    dropoff_lon_wgs84: float
    dropoff_lat_wgs84: float
    predicted_route_time_p50_s: float
    profile_id: str
    compatible_profiles: frozenset[str]
    passenger_accepts_av: bool
    acceptance_source: str

    @property
    def av_compatible(self) -> bool:
        return self.profile_id in self.compatible_profiles


class RequestFeed:
    """Monotone release cursor; reveal_until returns only newly available rows.

    A first jump to 09:00 reveals the requests still pending at that time, and
    accounts for earlier requests as expired. inventory() gives offline counts,
    not future order identities. Requests expire at now >= pickup deadline,
    matching the existing rolling controller's convention.
    """

    def __init__(self, requests: tuple[DecisionRequest, ...], *, original_count: int) -> None:
        self._requests = tuple(sorted(requests, key=lambda r: (r.release_time_s, r.order_id)))
        if len({r.order_id for r in requests}) != len(requests):
            raise ValueError("common-population order_id must be unique")
        self._release_times = tuple(r.release_time_s for r in self._requests)
        self._cursor = 0
        self._now_s = float("-inf")
        self._pending: dict[str, DecisionRequest] = {}
        self._committed: dict[str, DecisionRequest] = {}
        self._completed: set[str] = set()
        self._expired_count = 0
        self._original_count = int(original_count)

    def reveal_until(self, now_s: float) -> tuple[DecisionRequest, ...]:
        now_s = _seconds(now_s)
        if now_s < self._now_s:
            raise ValueError("request feed cannot move backwards; create another episode")
        self._now_s = now_s
        expired = [oid for oid, row in self._pending.items() if now_s >= row.expires_at_s]
        for oid in expired:
            del self._pending[oid]
        self._expired_count += len(expired)
        stop = bisect_right(self._release_times, now_s, lo=self._cursor)
        newly_available = []
        for row in self._requests[self._cursor:stop]:
            if now_s >= row.expires_at_s:
                self._expired_count += 1
            else:
                self._pending[row.order_id] = row
                newly_available.append(row)
        self._cursor = stop
        return tuple(newly_available)

    def pending(self) -> tuple[DecisionRequest, ...]:
        return tuple(self._pending.values())

    def commit(self, order_id: str, *, now_s: float) -> DecisionRequest:
        self.reveal_until(now_s)
        try:
            row = self._pending.pop(str(order_id))
        except KeyError as error:
            raise ValueError("commit requires a released, uncommitted, unexpired request") from error
        self._committed[row.order_id] = row
        return row

    def complete(self, order_id: str, *, now_s: float) -> None:
        self.reveal_until(now_s)
        order_id = str(order_id)
        if order_id not in self._committed or order_id in self._completed:
            raise ValueError("completion requires an unfinished committed request")
        self._completed.add(order_id)

    def was_committed(self, order_id: str) -> bool:
        return str(order_id) in self._committed

    def counts(self) -> dict:
        return {
            "original_requests": self._original_count,
            "common_population_requests": len(self._requests),
            "excluded_common_input": self._original_count - len(self._requests),
            "revealed": self._cursor,
            "unreleased": len(self._requests) - self._cursor,
            "pending": len(self._pending),
            "committed": len(self._committed),
            "completed": len(self._completed),
            "expired": self._expired_count,
        }

    def inventory(self) -> dict:
        return {
            "original_requests": self._original_count,
            "common_population_requests": len(self._requests),
            "excluded_common_input": self._original_count - len(self._requests),
            "compatible_counts": {
                p: sum(p in r.compatible_profiles for r in self._requests) for p in ("C", "M", "A", "HV")
            },
            "passenger_accepts_av_count": sum(r.passenger_accepts_av for r in self._requests),
            "population_filter": "EXISTING_COMMON_INPUT_ONLY_NO_CELL_ROI_OR_PROFILE_FILTER",
        }


@dataclass(frozen=True)
class ExecutionTruth:
    """Historical environment input; missing explicit observations stay None.

    The replay base's realized_service_time_s is a historical arrival-minus-
    departure duration. It is not the arrival of a newly assigned vehicle.
    No P50 value is a fallback for any field in this class.
    """

    order_id: str
    observed_service_time_s: float | None
    realized_service_time_s: float | None
    observed_arrival_time_s: float | None
    status: str


class ExecutionTruthLookup:
    """Environment-only lookup guarded by irreversible request commitment.

    Give decision code an EpisodeSnapshot, not this object or the episode.
    Loading hidden input is separate from querying it: lookup rejects both
    unreleased and released-but-uncommitted identities. It has no future-row
    enumeration API. Missing truth is reported rather than imputed from M3.
    """

    def __init__(self, loader: Callable[[], pd.DataFrame], feed: RequestFeed,
                 benchmark_start: pd.Timestamp, *, source_path: str, source_columns) -> None:
        self._loader = loader
        self._feed = feed
        self._start = benchmark_start
        self._values: dict[str, ExecutionTruth] | None = None
        self._source_path = source_path
        self._source_columns = tuple(source_columns)

    def inventory(self) -> dict:
        return {
            "source_path": self._source_path,
            "source_columns": list(self._source_columns),
            "observed_service_time_s": "AVAILABLE" if "observed_service_time_s" in self._source_columns else "MISSING_EXPLICIT_COLUMN",
            "realized_service_time_s": "AVAILABLE" if "realized_service_time_s" in self._source_columns else "MISSING",
            "observed_arrival_time_s": "AVAILABLE" if {"observed_arrival_time_s", "arrival_time"}.intersection(self._source_columns) else "MISSING_EXPLICIT_COLUMN",
            "lookup_policy": "ENVIRONMENT_ONLY_AFTER_COMMITMENT",
            "p50_fallback": False,
            "historical_realized_lineage": "stage4.replay_foundation.load_replay_orders: Stage1 arrival_time - departure_time",
            "new_vehicle_execution_arrivals": "NOT_GENERATED_BY_THIS_ADAPTER",
        }

    def lookup(self, order_id: str) -> ExecutionTruth:
        order_id = str(order_id)
        if not self._feed.was_committed(order_id):
            raise PermissionError("execution truth is available only to the environment after commitment")
        if self._values is None:
            frame = self._loader()
            if frame.order_id.astype(str).duplicated().any():
                raise ValueError("execution truth order_id must be unique")
            values = {}
            for row in frame.to_dict("records"):
                observed = _optional_duration(row.get("observed_service_time_s", None))
                realized = _optional_duration(row.get("realized_service_time_s", None))
                arrival = row.get("observed_arrival_time_s", None)
                if pd.isna(arrival):
                    arrival = None
                if arrival is None and not pd.isna(row.get("arrival_time", None)):
                    arrival = float((pd.Timestamp(row["arrival_time"]) - self._start).total_seconds())
                if arrival is not None:
                    arrival = _seconds(arrival)
                status = "AVAILABLE_OBSERVED" if observed is not None else "AVAILABLE_REALIZED_ONLY" if realized is not None else "MISSING"
                values[str(row["order_id"])] = ExecutionTruth(str(row["order_id"]), observed, realized, arrival, status)
            self._values = values
        return self._values.get(order_id, ExecutionTruth(order_id, None, None, None, "MISSING"))


@dataclass(frozen=True)
class SupplyPlanRow:
    vehicle_id: str
    native_id: int
    slot_id: str
    vehicle_type: str
    profile_id: str
    activation_s: float
    admission_end_s: float
    initial_lon_wgs84: float
    initial_lat_wgs84: float
    availability_policy: str


@dataclass(frozen=True)
class CommittedPlan:
    order_id: str
    vehicle_id: str
    commit_time_s: float
    predicted_pickup_time_s: float
    predicted_service_end_s: float
    dropoff_lon_wgs84: float
    dropoff_lat_wgs84: float


@dataclass(frozen=True)
class SupplySnapshotRow:
    plan: SupplyPlanRow
    current_lon_wgs84: float
    current_lat_wgs84: float
    location_source: str
    activated: bool
    admission_open: bool
    can_admit: bool
    active_order_id: str | None
    finishing_committed: bool
    availability_state: str


class ControlledSupply:
    """Read-only view of the real controlled builder's common service slots."""

    def __init__(self, scenario: FleetScenario, benchmark_start: pd.Timestamp, profile_id: str) -> None:
        self._accounting = deepcopy(scenario.accounting)
        rows = []
        for row in scenario.scenario_fleet.itertuples(index=False):
            if row.availability_policy != AVAILABILITY_POLICY:
                raise ValueError("research supply requires STOP_ADMISSION_FINISH_COMMITTED")
            rows.append(SupplyPlanRow(
                str(row.vehicle_id), int(row.native_id), str(row.slot_id), str(row.vehicle_type),
                profile_id if row.vehicle_type == "AV" else "HV",
                float((pd.Timestamp(row.availability_start_time) - benchmark_start).total_seconds()),
                float((pd.Timestamp(row.availability_end_time) - benchmark_start).total_seconds()),
                float(row.initial_lon_wgs84), float(row.initial_lat_wgs84), str(row.availability_policy),
            ))
        self._plan = tuple(rows)
        self._by_vehicle = {r.vehicle_id: r for r in rows}
        if len(self._by_vehicle) != len(rows):
            raise ValueError("supply vehicle_id must be unique")

    def plan(self) -> tuple[SupplyPlanRow, ...]:
        return self._plan

    def plan_table(self) -> pd.DataFrame:
        return pd.DataFrame(asdict(r) for r in self._plan)

    def snapshot(self, now_s: float, *, commitments: Mapping[str, CommittedPlan] | None = None,
                 locations: Mapping[str, tuple[float, float]] | None = None) -> tuple[SupplySnapshotRow, ...]:
        now_s = _seconds(now_s)
        commitments, locations = commitments or {}, locations or {}
        result = []
        for plan in self._plan:
            activated = now_s >= plan.activation_s
            admission = activated and now_s < plan.admission_end_s
            commitment = commitments.get(plan.vehicle_id)
            finishing = commitment is not None and now_s >= plan.admission_end_s
            state = ("FINISHING_COMMITTED" if finishing else "COMMITTED" if commitment is not None
                     else "AVAILABLE" if admission else "EXITED" if activated else "NOT_YET_ACTIVE")
            lon, lat = locations.get(plan.vehicle_id, (plan.initial_lon_wgs84, plan.initial_lat_wgs84))
            result.append(SupplySnapshotRow(
                plan, float(lon), float(lat), "OBSERVED_ENVIRONMENT" if plan.vehicle_id in locations else "INITIAL_TEMPLATE_ONLY",
                activated, admission, admission and commitment is None,
                commitment.order_id if commitment is not None else None, finishing, state,
            ))
        return tuple(result)

    def inventory(self) -> dict:
        return {
            "slot_count": len(self._plan),
            "av_count": sum(p.vehicle_type == "AV" for p in self._plan),
            "hv_count": sum(p.vehicle_type == "HV" for p in self._plan),
            "availability_policy": AVAILABILITY_POLICY,
            "unit_semantics": "EFFECTIVE_COMMON_SERVICE_SESSION_SLOT_NOT_TRUE_ONLINE_OR_EMPLOYMENT_SHIFT",
            "locations_without_environment_updates": "INITIAL_TEMPLATE_ONLY",
            "accounting": deepcopy(self._accounting),
        }


@dataclass(frozen=True)
class EpisodeSnapshot:
    """The only object to pass into a decision algorithm."""

    now_s: float
    requests: tuple[DecisionRequest, ...]
    vehicles: tuple[SupplySnapshotRow, ...]


class ResearchEpisode:
    """Data/lifecycle interface, not a multi-vehicle native simulation.

    commit_request records a caller-chosen feasible commitment; it is not an
    assignment algorithm. complete_request and update_location take environment
    observations. Supply snapshots never infer completion from predicted time.
    """

    def __init__(self, requests: RequestFeed, supply: ControlledSupply,
                 execution_truth: ExecutionTruthLookup, *, metadata: dict | None = None) -> None:
        self.requests = requests
        self.supply = supply
        self.execution_truth = execution_truth
        self._metadata = deepcopy(metadata or {})
        self._commitments: dict[str, CommittedPlan] = {}
        self._locations: dict[str, tuple[float, float]] = {}

    def reveal_until(self, now_s: float) -> tuple[DecisionRequest, ...]:
        return self.requests.reveal_until(now_s)

    def snapshot(self, now_s: float) -> EpisodeSnapshot:
        self.requests.reveal_until(now_s)
        return EpisodeSnapshot(float(now_s), self.requests.pending(),
                               self.supply.snapshot(now_s, commitments=self._commitments, locations=self._locations))

    def commit_request(self, order_id: str, vehicle_id: str, *, now_s: float, pickup_eta_s: float = 0.) -> CommittedPlan:
        snapshot = self.snapshot(now_s)
        request = next((r for r in snapshot.requests if r.order_id == str(order_id)), None)
        vehicle = next((v for v in snapshot.vehicles if v.plan.vehicle_id == str(vehicle_id)), None)
        if request is None or vehicle is None or not vehicle.can_admit:
            raise ValueError("commit requires a pending request and vehicle inside its admission window")
        pickup_eta_s = _seconds(pickup_eta_s)
        if pickup_eta_s < 0 or now_s + pickup_eta_s > request.expires_at_s:
            raise ValueError("pickup must fit remaining passenger patience")
        if vehicle.plan.vehicle_type == "AV" and (not request.av_compatible or not request.passenger_accepts_av):
            raise ValueError("AV commitment requires profile compatibility and passenger acceptance")
        self.requests.commit(request.order_id, now_s=now_s)
        plan = CommittedPlan(request.order_id, vehicle.plan.vehicle_id, float(now_s),
                             float(now_s + pickup_eta_s), float(now_s + pickup_eta_s + request.predicted_route_time_p50_s),
                             request.dropoff_lon_wgs84, request.dropoff_lat_wgs84)
        self._commitments[plan.vehicle_id] = plan
        return plan

    def complete_request(self, order_id: str, *, now_s: float) -> None:
        vehicle_id = next((vid for vid, plan in self._commitments.items() if plan.order_id == str(order_id)), None)
        if vehicle_id is None:
            raise ValueError("completion requires an active commitment")
        plan = self._commitments[vehicle_id]
        if _seconds(now_s) < plan.commit_time_s:
            raise ValueError("completion cannot precede commitment")
        self.requests.complete(order_id, now_s=now_s)
        self._locations[vehicle_id] = (plan.dropoff_lon_wgs84, plan.dropoff_lat_wgs84)
        del self._commitments[vehicle_id]

    def update_location(self, vehicle_id: str, lon_wgs84: float, lat_wgs84: float) -> None:
        if str(vehicle_id) not in self.supply._by_vehicle:
            raise KeyError(vehicle_id)
        lon, lat = float(lon_wgs84), float(lat_wgs84)
        if not isfinite(lon) or not isfinite(lat) or not -180 <= lon <= 180 or not -90 <= lat <= 90:
            raise ValueError("environment location must be finite WGS84 coordinates")
        self._locations[str(vehicle_id)] = (lon, lat)

    def counts(self, now_s: float | None = None) -> dict:
        if now_s is not None:
            self.requests.reveal_until(now_s)
        current = self.requests._now_s
        result = self.requests.counts()
        result["now_s"] = current if isfinite(current) else None
        if isfinite(current):
            vehicles = self.supply.snapshot(current, commitments=self._commitments, locations=self._locations)
            result.update(admission_open_slots=sum(v.admission_open for v in vehicles),
                          can_admit_slots=sum(v.can_admit for v in vehicles),
                          finishing_committed_slots=sum(v.finishing_committed for v in vehicles))
        return result

    def inventory(self) -> dict:
        return {
            "contract_version": CONTRACT_VERSION,
            "adapter_status": "READY", "data_contract_status": "READY",
            "native_multi_vehicle_solver_status": "NOT_CONNECTED",
            "native_execution_performed": False,
            "requests": self.requests.inventory(), "supply": self.supply.inventory(),
            "execution_truth": self.execution_truth.inventory(), **deepcopy(self._metadata),
        }


def episode_from_frames(request_frame: pd.DataFrame, routes: pd.DataFrame, fleet: FleetScenario, *,
                        benchmark_start: pd.Timestamp, profile_id: str = "C", patience_s: float = 300.,
                        passenger_acceptance_rate: float = .7, acceptance_seed: int = 20260827,
                        truth_frame: pd.DataFrame | None = None, metadata: dict | None = None) -> ResearchEpisode:
    """Create an isolated episode from projected/synthetic tables, without native actors."""
    if profile_id not in ("C", "M", "A"):
        raise ValueError("profile_id must be C, M or A")
    start = pd.Timestamp(benchmark_start)
    if start.tzinfo is None or pd.isna(start):
        raise ValueError("benchmark_start must be timezone-aware")
    start = start.tz_convert(TIMEZONE)
    patience = _seconds(patience_s)
    if patience <= 0:
        raise ValueError("patience_s must be positive")
    base_columns = ["order_id", "request_time", *COORDINATE_COLUMNS]
    for name, frame, required in (("requests", request_frame, base_columns), ("routes", routes, ROUTE_COLUMNS)):
        missing = set(required).difference(frame.columns)
        if missing:
            raise ValueError(f"{name} table missing columns: {sorted(missing)}")
        if frame.order_id.isna().any() or frame.order_id.astype(str).duplicated().any():
            raise ValueError(f"{name} order_id must be present and unique")
    if set(request_frame.order_id.astype(str)) != set(routes.order_id.astype(str)):
        raise ValueError("requests and common-route tables must have identical order identities")
    # The optimizer never receives a source DataFrame. Project before merging,
    # even when an in-memory test fixture contains execution columns.
    base = request_frame[base_columns].copy()
    base["order_id"] = base.order_id.astype(str)
    route = routes[list(ROUTE_COLUMNS)].copy()
    route["order_id"] = route.order_id.astype(str)
    common = base.merge(route, on="order_id", validate="one_to_one")
    common = common.loc[common.common_eligible.fillna(False)].copy()
    records = []
    for row in common.itertuples(index=False):
        release = float((pd.Timestamp(row.request_time) - start).total_seconds())
        coordinates = tuple(float(getattr(row, c)) for c in COORDINATE_COLUMNS)
        prediction = float(row.predicted_route_time_p50_s)
        if not all(isfinite(v) for v in (release, prediction, *coordinates)) or prediction <= 0:
            raise ValueError("common population contains missing decision inputs; do not silently filter requests")
        acceptance = passenger_acceptance(str(row.order_id), passenger_acceptance_rate, acceptance_seed)
        profiles = frozenset(p for p in ("C", "M", "A", "HV") if bool(getattr(row, f"compatible_{p}")))
        records.append(DecisionRequest(str(row.order_id), release, release + patience, *coordinates,
                                       prediction, profile_id, profiles,
                                       acceptance.passenger_accepts_av, acceptance.acceptance_source))
    feed = RequestFeed(tuple(records), original_count=len(base))
    truth = request_frame if truth_frame is None else truth_frame
    truth_columns = [c for c in TRUTH_COLUMNS if c in truth.columns]
    truth_projection = truth[["order_id", *truth_columns]].copy()
    lookup = ExecutionTruthLookup(lambda: truth_projection.copy(), feed, start,
                                  source_path="IN_MEMORY_FIXTURE", source_columns=truth_columns)
    meta = {"decision_profile_id": profile_id, "patience_s": patience,
            "passenger_acceptance_rate": float(passenger_acceptance_rate), "passenger_acceptance_seed": int(acceptance_seed),
            "benchmark_start": start.isoformat(), "decision_prediction_source": "FROZEN_M3_P50_DECISION_ONLY_NO_REFIT",
            **(metadata or {})}
    return ResearchEpisode(feed, ControlledSupply(fleet, start, profile_id), lookup, metadata=meta)


def load_research_episode(root: str | Path, *, profile_id: str = "C", requested_q_a: float = .1) -> ResearchEpisode:
    """Dry-load existing citywide common demand and actual controlled supply.

    The explicit C/M/A decision profile changes compatibility labels only.
    It never filters demand or changes common slot labeling for the same q/seed.
    The existing M controlled runner remains untouched and is not launched.
    """
    root = Path(root).resolve()
    cfg = json.loads((root / BASE_CONFIG_REL).read_text(encoding="utf-8"))
    controlled = json.loads((root / CONTROLLED_CONFIG_REL).read_text(encoding="utf-8"))
    scenario_path = root / "stage4/output/final_experiments" / cfg["source_scenario"] / "scenario_config.json"
    runtime = json.loads(scenario_path.read_text(encoding="utf-8"))["runtime_configuration"]
    start = pd.Timestamp(runtime["benchmark_start_time"]).tz_convert(TIMEZONE)
    request_path, route_path = root / ORDER_BASE_REL, root / ROUTES_REL
    schema = pq.read_schema(request_path).names
    missing = set(DECISION_BASE_COLUMNS).difference(schema)
    if missing:
        raise ValueError(f"replay base missing decision columns: {sorted(missing)}")
    base = _read_projection(request_path, DECISION_BASE_COLUMNS, source_profile_id=SOURCE_PROFILE_ID)
    base["request_time"] = pd.to_datetime(base.request_time, utc=True).dt.tz_convert(TIMEZONE)
    if not (base.request_time.ge(start) & base.request_time.lt(start + pd.Timedelta(seconds=cfg["measurement_end_s"]))).all():
        raise ValueError("replay base is not the frozen bounded Test31 population")
    routes = _read_projection(route_path, ROUTE_COLUMNS)
    fleet = build_controlled_fleet(root, benchmark_start=start,
                                   simulation_end=start + pd.Timedelta(seconds=cfg["physical_drain_limit_s"]),
                                   requested_q_a=requested_q_a, seed=controlled["supply_seed"])
    episode = episode_from_frames(base, routes, fleet, benchmark_start=start, profile_id=profile_id,
                                  patience_s=cfg["patience_s"], passenger_acceptance_rate=.7,
                                  acceptance_seed=runtime["passenger_acceptance_seed"], metadata={
                                      "source_profile_id": SOURCE_PROFILE_ID,
                                      "source_population": "TEST31_30000_REPLAY_COMMON_POPULATION_CITYWIDE",
                                      "entire_raw_test31_population": False,
                                      "source_config_version": cfg["version"], "controlled_config_version": controlled["version"],
                                      "requested_q_a": float(requested_q_a), "supply_seed": int(controlled["supply_seed"]),
                                      "source_paths": {"requests": str(request_path), "routes": str(route_path),
                                                       "fleet_template": str(root / FLEET_REL)},
                                      "decision_read_columns": {"requests": list(DECISION_BASE_COLUMNS), "routes": list(ROUTE_COLUMNS)},
                                  })
    truth_columns = [c for c in TRUTH_COLUMNS if c in schema]
    episode.execution_truth = ExecutionTruthLookup(
        lambda: _read_projection(request_path, ["order_id", "profile_id", *truth_columns], source_profile_id=SOURCE_PROFILE_ID),
        episode.requests, start, source_path=str(request_path), source_columns=truth_columns,
    )
    return episode
