"""Train-only time-of-day demand balancing for the Stage4 robustness test."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from stage4.fleetpy_adapter.upstream import FleetPyCompatibilityError

from .candidate_graph import SparseValhallaMatrixAdapter, SpatialVehicle


POLICY_NAME = "TRAIN_TOD_DEMAND_BALANCE"
POLICY_VERSION = "stage4_repositioning_r1.1"
TIMEZONE = "Asia/Shanghai"
BIN_MINUTES = 15
TRAIN_DATES = tuple(f"201610{day:02d}" for day in range(9, 25))
REFERENCE_REL = Path(
    "stage4/output/paper_enhancement/repositioning_robustness/"
    "train_tod_demand_reference.parquet"
)
REFERENCE_MANIFEST_REL = REFERENCE_REL.with_suffix(".manifest.json")
PBF_CONFIG_REL = Path("stage3/config/stage3_s1_static_inventory.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def time_bin_index(timestamp: pd.Timestamp) -> int:
    value = pd.Timestamp(timestamp)
    if value.tzinfo is None:
        value = value.tz_localize(TIMEZONE)
    else:
        value = value.tz_convert(TIMEZONE)
    return int(value.hour * (60 // BIN_MINUTES) + value.minute // BIN_MINUTES)


def largest_remainder_quotas(weights: pd.Series, total: int) -> pd.Series:
    """Convert nonnegative weights into exact deterministic integer quotas."""
    if int(total) < 0:
        raise ValueError("quota total must be nonnegative")
    values = pd.to_numeric(weights, errors="coerce").fillna(0.0).astype(float)
    if (values < 0).any() or not np.isfinite(values).all():
        raise ValueError("quota weights must be finite and nonnegative")
    result = pd.Series(0, index=weights.index, dtype="int64")
    if int(total) == 0:
        return result
    weight_sum = float(values.sum())
    if weight_sum <= 0:
        raise ValueError("positive quota total requires positive weights")
    exact = values / weight_sum * int(total)
    floor = np.floor(exact).astype("int64")
    result.loc[:] = floor
    remaining = int(total) - int(result.sum())
    if remaining:
        ranking = pd.DataFrame(
            {
                "index": list(weights.index),
                "fraction": (exact - floor).to_numpy(float),
                "tie": [str(value) for value in weights.index],
            }
        ).sort_values(["fraction", "tie"], ascending=[False, True], kind="mergesort")
        for index in ranking.head(remaining)["index"]:
            result.loc[index] += 1
    if int(result.sum()) != int(total):
        raise AssertionError("largest-remainder quotas do not conserve the total")
    return result


def _train_order_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for date in TRAIN_DATES:
        files.extend(
            sorted(
                (root / f"stage1/input_v1/split=train/date={date}").glob(
                    "bucket=*/order_base.parquet"
                )
            )
        )
    if not files:
        raise FleetPyCompatibilityError("no frozen Train order_base files found")
    observed_dates = {path.parent.parent.name.removeprefix("date=") for path in files}
    if observed_dates != set(TRAIN_DATES):
        raise FleetPyCompatibilityError(
            f"Train demand dates mismatch: {sorted(observed_dates)}"
        )
    if any("split=train" not in path.as_posix() for path in files):
        raise FleetPyCompatibilityError("non-Train file entered demand reference")
    return files


def _frozen_pbf_path(root: Path) -> Path:
    config = json.loads((root / PBF_CONFIG_REL).read_text(encoding="utf-8"))
    pbf = Path(config["paths"]["pbf"])
    if not pbf.is_file():
        raise FleetPyCompatibilityError(f"frozen PBF not found: {pbf}")
    return pbf


def _osm_node_coordinates(pbf: Path, target_ids: set[int]) -> dict[int, tuple[float, float]]:
    import osmium

    found: dict[int, tuple[float, float]] = {}

    class Handler(osmium.SimpleHandler):
        def node(self, node: Any) -> None:
            node_id = int(node.id)
            if node_id in target_ids:
                found[node_id] = (float(node.location.lon), float(node.location.lat))

    Handler().apply_file(str(pbf), locations=False)
    return found


def build_train_demand_reference(root: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the frozen 96-bin pickup-node distribution without Test31 input."""
    root = Path(root).resolve()
    files = _train_order_files(root)
    counts: dict[tuple[int, int], int] = {}
    input_rows = eligible_rows = missing_start_node_rows = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        required = {"start_node", "departure_time", "stage1_core_eligible"}
        if not required.issubset(parquet.schema_arrow.names):
            raise FleetPyCompatibilityError(f"Train order schema incomplete: {path}")
        for batch in parquet.iter_batches(columns=sorted(required), batch_size=32768):
            frame = batch.to_pandas()
            input_rows += len(frame)
            frame = frame.loc[frame["stage1_core_eligible"].fillna(False)].copy()
            eligible_rows += len(frame)
            node = pd.to_numeric(frame["start_node"], errors="coerce")
            missing_start_node_rows += int(node.isna().sum())
            frame = frame.loc[node.notna()].copy()
            node = node.loc[node.notna()].astype("int64")
            timestamp = pd.to_datetime(
                pd.to_numeric(frame["departure_time"], errors="coerce"),
                unit="s",
                utc=True,
            ).dt.tz_convert(TIMEZONE)
            bins = timestamp.dt.hour * 4 + timestamp.dt.minute // BIN_MINUTES
            grouped = pd.DataFrame(
                {"time_bin_index": bins.to_numpy(int), "node_id": node.to_numpy(int)}
            ).value_counts(sort=False)
            for key, value in grouped.items():
                pair = (int(key[0]), int(key[1]))
                counts[pair] = counts.get(pair, 0) + int(value)
    target_ids = {node_id for _, node_id in counts}
    pbf = _frozen_pbf_path(root)
    coordinates = _osm_node_coordinates(pbf, target_ids)
    if set(coordinates) != target_ids:
        missing = sorted(target_ids - set(coordinates))
        raise FleetPyCompatibilityError(
            f"frozen PBF misses {len(missing)} Train pickup nodes; sample={missing[:10]}"
        )
    rows = [
        {
            "time_bin_index": bin_index,
            "clock_time": f"{bin_index // 4:02d}:{(bin_index % 4) * 15:02d}",
            "node_id": node_id,
            "lon_wgs84": coordinates[node_id][0],
            "lat_wgs84": coordinates[node_id][1],
            "train_pickup_count": count,
        }
        for (bin_index, node_id), count in sorted(counts.items())
    ]
    reference = pd.DataFrame(rows)
    totals = reference.groupby("time_bin_index")["train_pickup_count"].transform("sum")
    reference["demand_share"] = reference["train_pickup_count"] / totals
    if set(reference["time_bin_index"]) != set(range(96)):
        raise FleetPyCompatibilityError("Train demand reference does not cover all 96 bins")
    if not np.allclose(reference.groupby("time_bin_index")["demand_share"].sum(), 1.0):
        raise FleetPyCompatibilityError("Train demand weights do not sum to one by bin")
    freeze = json.loads((root / "stage0/docs/stage0_v6_freeze_manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": POLICY_VERSION,
        "policy_name": POLICY_NAME,
        "train_dates": list(TRAIN_DATES),
        "validation_dates_included": [],
        "test31_included": False,
        "source_split": "stage1/input_v1/split=train",
        "source_file_count": len(files),
        "source_relative_paths_sha256": hashlib.sha256(
            "\n".join(path.relative_to(root).as_posix() for path in files).encode("utf-8")
        ).hexdigest(),
        "input_order_rows": input_rows,
        "eligible_order_rows": eligible_rows,
        "orders_with_start_node": eligible_rows - missing_start_node_rows,
        "missing_start_node_rows": missing_start_node_rows,
        "unique_pickup_node_count": len(target_ids),
        "mapped_pickup_node_count": len(coordinates),
        "pbf_path_from_frozen_config": pbf.as_posix(),
        "pbf_config_path": PBF_CONFIG_REL.as_posix(),
        "pbf_sha256": freeze["pbf_sha"],
        "time_bin_minutes": BIN_MINUTES,
        "time_bin_count": 96,
        "spatial_representation": "Stage1 start_node (OSM node ID) + frozen PBF WGS84",
        "reference_row_count": len(reference),
    }
    output = root / REFERENCE_REL
    _atomic_parquet(reference, output)
    manifest["reference_sha256"] = _sha256(output)
    _atomic_json(manifest, root / REFERENCE_MANIFEST_REL)
    return reference, manifest


def load_train_demand_reference(root: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(root).resolve()
    path = root / REFERENCE_REL
    manifest_path = root / REFERENCE_MANIFEST_REL
    if not path.is_file() or not manifest_path.is_file():
        return build_train_demand_reference(root)
    reference = pd.read_parquet(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("test31_included") is not False or manifest.get("train_dates") != list(TRAIN_DATES):
        raise FleetPyCompatibilityError("Train demand reference provenance is invalid")
    if manifest.get("reference_sha256") != _sha256(path):
        raise FleetPyCompatibilityError("Train demand reference hash mismatch")
    return reference, manifest


def _xy(lon: np.ndarray, lat: np.ndarray, reference_lat: float) -> np.ndarray:
    return np.column_stack(
        (lon * 111_320.0 * np.cos(np.deg2rad(reference_lat)), lat * 110_540.0)
    )


@dataclass(frozen=True)
class RepositionProposal:
    native_vehicle_id: int
    vehicle_id: str
    origin_node_id: int
    origin_lon_wgs84: float
    origin_lat_wgs84: float
    target_node_id: int
    target_lon_wgs84: float
    target_lat_wgs84: float


def surplus_to_deficit_plan(
    idle_vehicles: Iterable[dict[str, Any]], target: pd.DataFrame
) -> tuple[list[RepositionProposal], pd.DataFrame]:
    """Map idle AVs to demand nodes and move only deterministic surplus units."""
    idle = pd.DataFrame(list(idle_vehicles))
    target = target.sort_values("node_id", kind="mergesort").reset_index(drop=True)
    quotas = largest_remainder_quotas(
        target.set_index("node_id")["demand_share"], len(idle)
    )
    target = target.assign(target_quota=target["node_id"].map(quotas).astype(int))
    if idle.empty:
        target["current_count"] = 0
        target["surplus_count"] = 0
        target["deficit_count"] = target["target_quota"]
        return [], target
    reference_lat = float(target["lat_wgs84"].mean())
    target_xy = _xy(
        target["lon_wgs84"].to_numpy(float),
        target["lat_wgs84"].to_numpy(float),
        reference_lat,
    )
    idle_xy = _xy(
        idle["lon_wgs84"].to_numpy(float),
        idle["lat_wgs84"].to_numpy(float),
        reference_lat,
    )
    _, nearest = cKDTree(target_xy).query(idle_xy, k=1)
    idle["current_node_id"] = target.iloc[np.asarray(nearest, dtype=int)]["node_id"].to_numpy(int)
    current = idle["current_node_id"].value_counts().astype(int)
    target["current_count"] = target["node_id"].map(current).fillna(0).astype(int)
    target["surplus_count"] = np.maximum(target["current_count"] - target["target_quota"], 0)
    target["deficit_count"] = np.maximum(target["target_quota"] - target["current_count"], 0)
    deficits = {
        int(row.node_id): int(row.deficit_count)
        for row in target.itertuples(index=False)
        if int(row.deficit_count) > 0
    }
    coordinate = target.set_index("node_id")[["lon_wgs84", "lat_wgs84"]].to_dict("index")
    surplus_rows: list[dict[str, Any]] = []
    for node_id, group in idle.sort_values("vehicle_id", kind="mergesort").groupby("current_node_id", sort=True):
        quota = int(quotas.loc[int(node_id)])
        surplus_rows.extend(group.iloc[quota:].to_dict("records"))
    proposals: list[RepositionProposal] = []
    for vehicle in sorted(surplus_rows, key=lambda row: str(row["vehicle_id"])):
        choices = [node_id for node_id, remaining in deficits.items() if remaining > 0]
        if not choices:
            break
        distances = []
        for node_id in choices:
            point = coordinate[node_id]
            dx = (float(vehicle["lon_wgs84"]) - point["lon_wgs84"]) * math.cos(math.radians(reference_lat))
            dy = float(vehicle["lat_wgs84"]) - point["lat_wgs84"]
            distances.append((dx * dx + dy * dy, int(node_id)))
        target_node = min(distances)[1]
        deficits[target_node] -= 1
        point = coordinate[target_node]
        proposals.append(
            RepositionProposal(
                int(vehicle["native_vehicle_id"]),
                str(vehicle["vehicle_id"]),
                int(vehicle["current_node_id"]),
                float(vehicle["lon_wgs84"]),
                float(vehicle["lat_wgs84"]),
                target_node,
                float(point["lon_wgs84"]),
                float(point["lat_wgs84"]),
            )
        )
    return proposals, target


class TrainTODRepositioningManager:
    """Apply the frozen policy after normal dispatch at each 15-minute boundary."""

    def __init__(
        self,
        *,
        bindings: Any,
        runtimes: Iterable[Any],
        network: Any,
        eta_adapter: SparseValhallaMatrixAdapter,
        reference: pd.DataFrame,
        start: pd.Timestamp,
        policy_end: pd.Timestamp,
    ) -> None:
        required = {
            "time_bin_index",
            "node_id",
            "lon_wgs84",
            "lat_wgs84",
            "train_pickup_count",
            "demand_share",
        }
        if not required.issubset(reference.columns):
            raise FleetPyCompatibilityError("repositioning reference schema incomplete")
        self.bindings = bindings
        self.runtime_by_vid = {
            int(runtime.fixture.native_id): runtime for runtime in runtimes
        }
        self.network = network
        self.eta_adapter = eta_adapter
        self.start = pd.Timestamp(start)
        self.policy_end = pd.Timestamp(policy_end)
        self.targets = {
            int(bin_index): group.loc[:, sorted(required)].copy()
            for bin_index, group in reference.groupby("time_bin_index", sort=True)
        }
        if set(self.targets) != set(range(96)):
            raise FleetPyCompatibilityError("repositioning reference must contain 96 bins")
        for target in self.targets.values():
            for row in target.itertuples(index=False):
                self.network.registry.position_for(row.lon_wgs84, row.lat_wgs84)
        self.trip_rows: list[dict[str, Any]] = []
        self.epoch_rows: list[dict[str, Any]] = []
        self.distribution_rows: list[dict[str, Any]] = []
        self.active: dict[int, int] = {}
        self.route_attempts = 0
        self.route_failures = 0
        self.position_failures = 0

    def _timestamp(self, simulation_time: int) -> pd.Timestamp:
        return self.start + pd.Timedelta(seconds=int(simulation_time))

    def _native_free(self, runtime: Any) -> bool:
        native = runtime.native_vehicle
        return native.status == self.bindings.states.IDLE and not native.assigned_route

    def reconcile_arrivals(self, simulation_time: int) -> None:
        timestamp = self._timestamp(simulation_time)
        for vid, row_index in list(self.active.items()):
            runtime = self.runtime_by_vid[vid]
            if not self._native_free(runtime):
                continue
            row = self.trip_rows[row_index]
            lon, lat = self.network.return_position_coordinates(runtime.native_vehicle.pos)
            if not (
                np.isclose(lon, row["target_lon_wgs84"], atol=1e-7)
                and np.isclose(lat, row["target_lat_wgs84"], atol=1e-7)
            ):
                self.position_failures += 1
                raise FleetPyCompatibilityError(
                    f"reposition completion position mismatch for vehicle {vid}"
                )
            row["completion_time"] = timestamp
            row["actual_reposition_time_s"] = float(
                (timestamp - pd.Timestamp(row["start_time"])).total_seconds()
            )
            row["status"] = "COMPLETED"
            runtime.state = "NATIVE_AVAILABLE"
            self.active.pop(vid)

    def before_normal_dispatch(self, simulation_time: int) -> None:
        """Release completed empty movements before passenger candidates are built."""
        self.reconcile_arrivals(simulation_time)

    def _start_trip(
        self,
        proposal: RepositionProposal,
        estimate: Any,
        simulation_time: int,
    ) -> None:
        runtime = self.runtime_by_vid[proposal.native_vehicle_id]
        native = runtime.native_vehicle
        if runtime.fixture.vehicle_type != "AV" or not self._native_free(runtime):
            raise FleetPyCompatibilityError("only idle AVs may be repositioned")
        if runtime.active_order_id is not None:
            raise FleetPyCompatibilityError("assigned AV entered repositioning")
        origin = native.pos
        destination = self.network.registry.position_for(
            proposal.target_lon_wgs84, proposal.target_lat_wgs84
        )
        if (
            int(origin[0]) != int(destination[0])
            and (
                not np.isfinite(estimate.corrected_pickup_eta_s)
                or estimate.corrected_pickup_eta_s <= 0
            )
        ):
            raise FleetPyCompatibilityError("invalid nonzero reposition route")
        self.network.register_vehicle_leg(
            (0, proposal.native_vehicle_id),
            origin,
            destination,
            estimate.corrected_pickup_eta_s,
            estimate.route_distance_m,
        )
        leg = self.bindings.vehicle_route_leg(
            self.bindings.states.REPOSITION, destination, {}
        )
        native.assign_vehicle_plan([leg], int(simulation_time))
        runtime.state = "NATIVE_REPOSITIONING"
        start_time = self._timestamp(simulation_time)
        row = {
            "policy_name": POLICY_NAME,
            "policy_version": POLICY_VERSION,
            "native_vehicle_id": proposal.native_vehicle_id,
            "vehicle_id": proposal.vehicle_id,
            "start_time": start_time,
            "time_bin_index": time_bin_index(start_time),
            "origin_node_id": proposal.origin_node_id,
            "origin_lon_wgs84": proposal.origin_lon_wgs84,
            "origin_lat_wgs84": proposal.origin_lat_wgs84,
            "target_node_id": proposal.target_node_id,
            "target_lon_wgs84": proposal.target_lon_wgs84,
            "target_lat_wgs84": proposal.target_lat_wgs84,
            "route_time_s": float(estimate.corrected_pickup_eta_s),
            "route_distance_m": float(estimate.route_distance_m),
            "completion_time": pd.NaT,
            "actual_reposition_time_s": np.nan,
            "status": "IN_PROGRESS",
            "deadhead_odd_qualified": False,
        }
        self.trip_rows.append(row)
        self.active[proposal.native_vehicle_id] = len(self.trip_rows) - 1

    def after_normal_dispatch(self, control: Any, simulation_time: int) -> None:
        """Reposition only AVs left idle by the completed normal dispatch."""
        timestamp = self._timestamp(simulation_time)
        if int(simulation_time) % (BIN_MINUTES * 60) != 0 or timestamp >= self.policy_end:
            return
        idle: list[dict[str, Any]] = []
        for runtime in sorted(
            self.runtime_by_vid.values(), key=lambda item: item.fixture.vehicle_id
        ):
            if runtime.fixture.vehicle_type != "AV" or not control._available(
                runtime, simulation_time
            ):
                continue
            lon, lat = self.network.return_position_coordinates(runtime.native_vehicle.pos)
            idle.append(
                {
                    "native_vehicle_id": int(runtime.fixture.native_id),
                    "vehicle_id": runtime.fixture.vehicle_id,
                    "lon_wgs84": float(lon),
                    "lat_wgs84": float(lat),
                }
            )
        target = self.targets[time_bin_index(timestamp)]
        proposals, distribution = surplus_to_deficit_plan(idle, target)
        for row in distribution.loc[
            distribution["current_count"].gt(0) | distribution["target_quota"].gt(0)
        ].itertuples(index=False):
            self.distribution_rows.append(
                {
                    "timestamp": timestamp,
                    "time_bin_index": time_bin_index(timestamp),
                    "node_id": int(row.node_id),
                    "lon_wgs84": float(row.lon_wgs84),
                    "lat_wgs84": float(row.lat_wgs84),
                    "current_idle_av_count": int(row.current_count),
                    "target_idle_av_quota": int(row.target_quota),
                    "surplus_count": int(row.surplus_count),
                    "deficit_count": int(row.deficit_count),
                }
            )
        proposals_by_target: dict[int, list[RepositionProposal]] = {}
        for proposal in proposals:
            proposals_by_target.setdefault(proposal.target_node_id, []).append(proposal)
        successes = failures = 0
        for target_node in sorted(proposals_by_target):
            group = proposals_by_target[target_node]
            first = group[0]
            candidates = [
                SpatialVehicle(
                    proposal.vehicle_id,
                    proposal.native_vehicle_id,
                    "AV",
                    proposal.origin_lon_wgs84,
                    proposal.origin_lat_wgs84,
                )
                for proposal in group
            ]
            self.route_attempts += len(group)
            estimates = self.eta_adapter.estimate_many(
                candidates,
                first.target_lon_wgs84,
                first.target_lat_wgs84,
                timestamp,
            )
            for proposal in group:
                estimate = estimates.get(proposal.native_vehicle_id)
                if estimate is None:
                    self.route_failures += 1
                    failures += 1
                    self.trip_rows.append(
                        {
                            "policy_name": POLICY_NAME,
                            "policy_version": POLICY_VERSION,
                            "native_vehicle_id": proposal.native_vehicle_id,
                            "vehicle_id": proposal.vehicle_id,
                            "start_time": timestamp,
                            "time_bin_index": time_bin_index(timestamp),
                            "origin_node_id": proposal.origin_node_id,
                            "origin_lon_wgs84": proposal.origin_lon_wgs84,
                            "origin_lat_wgs84": proposal.origin_lat_wgs84,
                            "target_node_id": proposal.target_node_id,
                            "target_lon_wgs84": proposal.target_lon_wgs84,
                            "target_lat_wgs84": proposal.target_lat_wgs84,
                            "route_time_s": np.nan,
                            "route_distance_m": np.nan,
                            "completion_time": pd.NaT,
                            "actual_reposition_time_s": np.nan,
                            "status": "ROUTING_FAILED_STAYED_IN_PLACE",
                            "deadhead_odd_qualified": False,
                        }
                    )
                    continue
                self._start_trip(proposal, estimate, simulation_time)
                successes += 1
        self.epoch_rows.append(
            {
                "simulation_time_s": int(simulation_time),
                "timestamp": timestamp,
                "time_bin_index": time_bin_index(timestamp),
                "idle_av_after_dispatch": len(idle),
                "surplus_av_selected": len(proposals),
                "reposition_routes_started": successes,
                "reposition_routing_failures": failures,
                "active_reposition_after_epoch": len(self.active),
            }
        )

    def finalize(self, simulation_time: int) -> None:
        self.reconcile_arrivals(simulation_time)
        for vid, row_index in self.active.items():
            self.trip_rows[row_index]["status"] = "IN_PROGRESS_AT_SIMULATION_END"

    def summary(self, *, av_count: int, horizon_seconds: float) -> dict[str, Any]:
        trips = pd.DataFrame(self.trip_rows)
        successful = trips.loc[trips.get("route_time_s", pd.Series(dtype=float)).notna()].copy()
        travel_time = pd.to_numeric(successful.get("route_time_s"), errors="coerce")
        distance = pd.to_numeric(successful.get("route_distance_m"), errors="coerce")
        if len(successful):
            start_seconds = (
                pd.to_datetime(successful["start_time"]) - self.start
            ).dt.total_seconds()
            in_horizon = np.minimum(
                travel_time.to_numpy(float),
                np.maximum(float(horizon_seconds) - start_seconds.to_numpy(float), 0.0),
            ).sum()
        else:
            in_horizon = 0.0
        denominator = float(av_count) * float(horizon_seconds)
        return {
            "policy_name": POLICY_NAME,
            "policy_version": POLICY_VERSION,
            "empty_route_odd_qualification": "OPERATIONAL_ABSTRACTION_NOT_ODD_CERTIFIED",
            "reposition_trip_count": int(len(successful)),
            "reposition_routing_attempt_count": int(self.route_attempts),
            "reposition_routing_failure_count": int(self.route_failures),
            "total_reposition_distance_m": float(distance.sum()) if len(distance) else 0.0,
            "total_reposition_travel_time_s": float(travel_time.sum()) if len(travel_time) else 0.0,
            "mean_reposition_distance_m": float(distance.mean()) if len(distance) else 0.0,
            "mean_reposition_travel_time_s": float(travel_time.mean()) if len(travel_time) else 0.0,
            "av_vehicle_time_repositioning_share": float(in_horizon / denominator) if denominator else 0.0,
            "in_progress_at_simulation_end": int(len(self.active)),
            "position_reconciliation_failure_count": int(self.position_failures),
        }
