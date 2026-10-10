"""Sparse empty-route geometry and frozen research compatibility for replay v2.

One scalar ``auto`` response supplies the time, distance and precision-6 shape.
Its same-engine ``edge_walk`` supplies actual directed identities; no matrix
route, reverse overlay, nearest geometry repair, or historical prediction is
substituted. New routes have no exact frozen M3 input rows, so their complete
traffic partition is explicitly U=1 under the existing research assumption.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from stage3.odd_tod.capability_envelope import parse_route_complex_encounters
from stage3.odd_tod.finalization import DYNAMIC_UNKNOWN_REASON, decode_polyline6
from stage3.odd_tod.intersection_complex import _bearing
from stage3.odd_tod.network_foundation import Stage3S2AError, stage3_edge_uid
from stage3.odd_tod.research_compatibility import evaluate_research_compatibility
from stage4.analysis.flexibility_prepare import MovementClassifier, S2A, S2B
from stage4.fleetpy_adapter.valhalla_time_adapter import TIMEZONE


MAX_CACHE_ENTRIES = 1024
CONTROL_POLICY = "SIGNALIZED_CONFLICT_MOVEMENTS"
TRAFFIC_OUTSIDE_BUDGET = 0.05
UNKNOWN_TRAFFIC = dict(NC=0.0, ML=0.0, MR=0.0, CG=0.0, SC=0.0, U=1.0)
CONTROL_REL = Path("stage4/output/flexibility_dispatch_v1/input/complex_control_overlay.parquet")
BOUNDARY_COLUMNS = ["stage3_edge_uid", "intersection_complex_uid", "boundary_role"]
MOVEMENT_COLUMNS = ["intersection_complex_uid", "incoming_stage3_edge_uid",
    "outgoing_stage3_edge_uid", "route_turn_type", "restriction_enforcement_certified",
    "movement_legality_state"]
CONTROL_COLUMNS = ["intersection_complex_uid", "roundabout_evidence_present",
    "signalized_research", "control_basis"]


def _read_columns(path: Path, columns: list[str]) -> pa.Table:
    """Keep columnar buffers, not one Python object per full-network row."""
    source = pq.ParquetFile(path)
    return pa.Table.from_batches(source.iter_batches(
        batch_size=8192, columns=columns, use_threads=False))


def _select(table: pa.Table, column: str, values) -> pa.Table:
    return table.filter(pc.is_in(table[column], value_set=pa.array(
        list(values), type=table[column].type)))


def _gap_m(left, right) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (*left, *right))
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371008.8 * 2 * math.asin(math.sqrt(min(1.0, max(0.0, a))))


class ControlledEmptyRouter:
    """Bounded, type-neutral scalar routes; share the pickup adapter's Actor."""

    def __init__(self, root: str | Path, eta_adapter) -> None:
        self.root = Path(root).resolve()
        self.eta_adapter = eta_adapter
        self.cache: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
        self._bearing_cache: OrderedDict[str, tuple] = OrderedDict()
        self.cache_hit_count = self.route_call_count = self.trace_call_count = 0
        self.route_request_count = 0
        self._result_counts = Counter()
        self._common_failure_reasons = Counter()
        self._cache_limit = MAX_CACHE_ENTRIES
        edge_path = self.root / S2A / "stage3_full_network_edges.parquet"
        edges = _read_columns(edge_path, ["valhalla_directed_edge_id", "stage3_edge_uid", "auto_routable"])
        if (any(edges[c].null_count for c in edges.column_names)
                or not pc.all(edges["auto_routable"]).as_py()):
            raise Stage3S2AError("frozen full-network identity/routability is incomplete")
        ids = edges["valhalla_directed_edge_id"].to_numpy()
        order = np.argsort(ids, kind="stable")
        self._edge_ids = np.asarray(ids[order], dtype=np.int64)
        if len(self._edge_ids) == 0 or np.any(self._edge_ids < 0) or np.any(np.diff(self._edge_ids) == 0):
            raise Stage3S2AError("frozen directed edge IDs must be unique and nonnegative")
        self._edge_uids = pc.take(edges["stage3_edge_uid"], pa.array(order))
        self._boundary = _read_columns(self.root / S2B / "stage3_edge_complex_boundary_index.parquet", BOUNDARY_COLUMNS)
        self._movements = _read_columns(self.root / S2B / "stage3_route_movement_lookup.parquet", MOVEMENT_COLUMNS)
        self._controls = _read_columns(self.root / CONTROL_REL, CONTROL_COLUMNS)
        if any(t[c].null_count for t in (self._boundary, self._movements, self._controls) for c in t.column_names):
            raise Stage3S2AError("frozen movement/control columns are incomplete")
        self.columnar_evidence_mib = (self._edge_ids.nbytes + self._edge_uids.nbytes
            + self._boundary.nbytes + self._movements.nbytes + self._controls.nbytes) / 2**20

    @staticmethod
    def _copy_result(result: dict[str, Any], *, cache_hit=False) -> dict[str, Any]:
        copied = dict(result)
        copied["points"] = copied["geometry_points"] = list(result["points"])
        copied["edges"] = [dict(edge) for edge in result["edges"]]
        copied["cache_hit"] = cache_hit
        return copied

    def _remember(self, key: tuple, result: dict[str, Any]) -> dict[str, Any]:
        self._record_result(result)
        self.cache[key] = result
        self.cache.move_to_end(key)
        while len(self.cache) > self._cache_limit:
            self.cache.popitem(last=False)
        return self._copy_result(result)

    def _record_result(self, result: dict[str, Any]) -> None:
        state = "supported" if result["common_supported"] else "unsupported"
        self._result_counts[state] += 1
        if state == "unsupported":
            self._common_failure_reasons.update(result["reason_codes"])

    def diagnostics(self) -> dict[str, Any]:
        """Observed adapter counters and the declared new-route U assumption."""
        return dict(route_requests=self.route_request_count,
            scalar_route_calls=self.route_call_count, edge_walk_calls=self.trace_call_count,
            route_cache_hits=self.cache_hit_count, route_cache_entries=len(self.cache),
            route_cache_limit=self._cache_limit, bearing_cache_entries=len(self._bearing_cache),
            common_support_evaluations=dict(self._result_counts),
            common_failure_reason_counts=dict(self._common_failure_reasons),
            columnar_evidence_mib=self.columnar_evidence_mib,
            traffic_policy="EXPLICIT_U_FOR_NEW_ROUTE_WITHOUT_FROZEN_M3_FEATURE_ROWS",
            traffic_partition=dict(UNKNOWN_TRAFFIC), traffic_unknown_share=1.0,
            dynamic_evidence_complete=False, dynamic_reason_codes=[DYNAMIC_UNKNOWN_REASON],
            conservative_control_policy=CONTROL_POLICY, traffic_outside_budget=TRAFFIC_OUTSIDE_BUDGET)

    @staticmethod
    def _base_result() -> dict[str, Any]:
        return dict(points=[], geometry_points=[], raw_time_s=None, duration_s=None,
            distance_m=None, common_supported=False, compatible_profiles=frozenset(),
            reason_codes=(), reasons=(), traffic_unknown_share=1.0,
            traffic_shares=dict(UNKNOWN_TRAFFIC), dynamic_evidence_complete=False,
            dynamic_evidence_state="UNKNOWN_NEW_ROUTE_NO_FROZEN_M3_FEATURE_ROWS",
            dynamic_reason_codes=(DYNAMIC_UNKNOWN_REASON,), edges=[], shape_polyline6=None,
            snapped_origin=None, snapped_target=None, snap_gap_origin_m=None,
            snap_gap_target_m=None, maneuver_types=(), profile_reason_codes={},
            encounter_count=0, control_assumption_count=0, bearing_fallback_count=0)

    @staticmethod
    def _local_timestamp(timestamp) -> pd.Timestamp:
        if isinstance(timestamp, (int, float, np.integer, np.floating)):
            local = pd.Timestamp(timestamp, unit="s", tz="UTC")
        else:
            local = pd.Timestamp(timestamp)
        if pd.isna(local):
            raise Stage3S2AError("empty-route decision time is missing")
        return local.tz_localize(TIMEZONE) if local.tzinfo is None else local.tz_convert(TIMEZONE)

    def _edge_walk(self, actor, shape: str, points, local) -> list[dict[str, Any]]:
        request = dict(shape=[dict(lon=lon, lat=lat) for lon, lat in points],
            costing="auto", shape_match="edge_walk", units="kilometers",
            date_time=dict(type=1, value=local.strftime("%Y-%m-%dT%H:%M")),
            filters=dict(action="include", attributes=["edge.id", "edge.length",
                "edge.begin_shape_index", "edge.end_shape_index", "shape"]))
        self.trace_call_count += 1
        response = actor.trace_attributes(request)
        records = response.get("edges") or []
        if not records:
            raise Stage3S2AError("EDGE_WALK_EMPTY")
        # Indices are attached to the edge_walk response's shape. Retain them
        # independently; physical geometry/time/distance stay the scalar route's.
        trace_shape = response.get("shape", shape)
        trace_points = decode_polyline6(trace_shape)
        parsed = []
        for sequence, edge in enumerate(records):
            identifier = edge.get("id")
            begin, end = edge.get("begin_shape_index"), edge.get("end_shape_index")
            if identifier is None or begin is None or end is None or edge.get("length") is None:
                raise Stage3S2AError("EDGE_WALK_IDENTITY_OR_SHAPE_INDEX_INCOMPLETE")
            originals = (identifier, begin, end)
            identifier, begin, end = (int(value) for value in originals)
            if any(isinstance(value, bool) or float(value) != integer
                    for value, integer in zip(originals, (identifier, begin, end))):
                raise Stage3S2AError("EDGE_WALK_NONINTEGER_IDENTITY_OR_SHAPE_INDEX")
            length = float(edge["length"]) * 1000.0
            if not (0 <= begin <= end < len(trace_points)) or not math.isfinite(length) or length <= 0:
                raise Stage3S2AError("EDGE_WALK_INVALID_SHAPE_INDEX_OR_LENGTH")
            index = int(np.searchsorted(self._edge_ids, identifier))
            if index == len(self._edge_ids) or int(self._edge_ids[index]) != identifier:
                raise Stage3S2AError("EDGE_OUTSIDE_FROZEN_FULL_NETWORK")
            uid = self._edge_uids[index].as_py()
            if uid != stage3_edge_uid(identifier):
                raise Stage3S2AError("FROZEN_FULL_NETWORK_EDGE_NAMESPACE_MISMATCH")
            if (not parsed and begin != 0) or (parsed and begin != parsed[-1]["end_shape_index"]):
                raise Stage3S2AError("EDGE_WALK_SHAPE_COVERAGE_INCOMPLETE")
            parsed.append(dict(route_sequence=sequence, valhalla_directed_edge_id=identifier,
                stage3_edge_uid=uid, begin_shape_index=begin,
                end_shape_index=end, length_m=length, trace_shape_polyline6=trace_shape))
        if parsed[-1]["end_shape_index"] != len(trace_points) - 1:
            raise Stage3S2AError("EDGE_WALK_SHAPE_COVERAGE_INCOMPLETE")
        return parsed

    def _bearings_for(self, edge_uids: set[str]) -> dict[str, tuple]:
        needed = edge_uids - self._bearing_cache.keys()
        found = {uid: self._bearing_cache[uid] for uid in edge_uids if uid in self._bearing_cache}
        if needed:
            source = pq.ParquetFile(self.root / S2A / "stage3_full_network_edges.parquet")
            values = pa.array(list(needed))
            for batch in source.iter_batches(batch_size=4096,
                    columns=["stage3_edge_uid", "geometry"], use_threads=False):
                small = pa.Table.from_batches([batch]).filter(pc.is_in(batch.column(0), value_set=values))
                for row in small.to_pylist():
                    geometry = json.loads(row["geometry"])
                    bearings = ((_bearing(geometry[0], geometry[1]), _bearing(geometry[-2], geometry[-1]))
                        if len(geometry) >= 2 else (None, None))
                    found[row["stage3_edge_uid"]] = bearings
                if needed.issubset(found):
                    break
        for uid, bearings in found.items():
            self._bearing_cache[uid] = bearings
            self._bearing_cache.move_to_end(uid)
        while len(self._bearing_cache) > MAX_CACHE_ENTRIES:
            self._bearing_cache.popitem(last=False)
        return found

    def _classify(self, records, local) -> dict[str, Any]:
        uids = {edge["stage3_edge_uid"] for edge in records}
        boundary = _select(self._boundary, "stage3_edge_uid", uids).to_pandas()
        cids = set(boundary.intersection_complex_uid.astype(str))
        movements = _select(self._movements, "intersection_complex_uid", cids)
        movements = _select(movements, "incoming_stage3_edge_uid", uids)
        movements = _select(movements, "outgoing_stage3_edge_uid", uids).to_pandas()
        controls = _select(self._controls, "intersection_complex_uid", cids).to_pandas()
        if set(controls.intersection_complex_uid.astype(str)) != cids or controls.intersection_complex_uid.duplicated().any():
            raise Stage3S2AError("ROUTE_COMPLEX_CONTROL_IDENTITY_INCOMPLETE")
        typed = pd.DataFrame(dict(date=local.strftime("%Y%m%d"), order_id="EMPTY_ROUTE",
            route_sequence=edge["route_sequence"], resolved_stage3_edge_uid=edge["stage3_edge_uid"],
            route_token_type="FULL_NETWORK_EDGE") for edge in records)
        encounters = parse_route_complex_encounters(typed, boundary, movements)
        classifier = MovementClassifier.__new__(MovementClassifier)
        classifier.complexes = controls.set_index("intersection_complex_uid").to_dict("index")
        classifier.lookup = {}
        for row in movements.itertuples(index=False):
            record = row._asdict()
            # Frozen S2B uses CERTIFIED_PROHIBITED; the research classifier's
            # predicate uses PROHIBITED. Both mean the same certified ban.
            if record["movement_legality_state"] == "CERTIFIED_PROHIBITED":
                record["movement_legality_state"] = "PROHIBITED"
            classifier.lookup[tuple(str(record[c]) for c in MOVEMENT_COLUMNS[:3])] = SimpleNamespace(**record)
        missing_bearings = set()
        for row in encounters.itertuples(index=False):
            key = (str(row.intersection_complex_uid), str(row.incoming_stage3_edge_uid), str(row.outgoing_stage3_edge_uid))
            known = classifier.lookup.get(key)
            if known is None or known.route_turn_type == "UNKNOWN":
                missing_bearings.update(key[1:])
        classifier.bearings = self._bearings_for(missing_bearings)
        classified, summary = classifier.summarize(encounters)
        valid, research_movements = classified.get("EMPTY_ROUTE", (True, ()))
        if not valid:
            raise Stage3S2AError("ROUTE_MOVEMENT_UNRESOLVED")
        prohibited = any(m.certified_prohibited for m in research_movements)
        evaluations = {profile: evaluate_research_compatibility(profile, research_movements,
            dict(UNKNOWN_TRAFFIC), direction_routable=True,
            outside_budget=TRAFFIC_OUTSIDE_BUDGET, conservative_control_policy=CONTROL_POLICY)
            for profile in ("HV", "C", "M", "A")}
        if evaluations["HV"].compatible != evaluations["A"].compatible:
            raise Stage3S2AError("A_HV_RESEARCH_COMPATIBILITY_MISMATCH")
        reasons = ("CERTIFIED_COMMON_PROHIBITION",) if prohibited else ()
        small_summary = summary.iloc[0].to_dict() if len(summary) else {}
        return dict(common_supported=not prohibited,
            compatible_profiles=frozenset(k for k, value in evaluations.items() if value.compatible),
            reason_codes=reasons, reasons=reasons,
            profile_reason_codes={k: value.reason_codes for k, value in evaluations.items()},
            maneuver_types=tuple(m.maneuver for m in research_movements),
            encounter_count=int(small_summary.get("encounter_count", 0)),
            control_assumption_count=int(small_summary.get("control_assumption_count", 0)),
            bearing_fallback_count=int(small_summary.get("bearing_fallback_count", 0)))

    def route(self, origin_lon, origin_lat, dest_lon, dest_lat, timestamp) -> dict[str, Any]:
        self.route_request_count += 1
        result = self._base_result()
        key = None
        try:
            coordinates = tuple(float(v) for v in (origin_lon, origin_lat, dest_lon, dest_lat))
            if not all(math.isfinite(v) for v in coordinates) or not (
                    -180 <= coordinates[0] <= 180 and -90 <= coordinates[1] <= 90
                    and -180 <= coordinates[2] <= 180 and -90 <= coordinates[3] <= 90):
                raise Stage3S2AError("INVALID_EMPTY_ROUTE_WGS84")
            local = self._local_timestamp(timestamp)
            key = (*coordinates, local.strftime("%Y-%m-%dT%H:%M"))
            if key in self.cache:
                self.cache_hit_count += 1
                self.cache.move_to_end(key)
                return self._copy_result(self.cache[key], cache_hit=True)
            if coordinates[:2] == coordinates[2:]:
                # No physical move exists. It has no edge/movement exposure;
                # zero is accepted only for an exactly coincident OD.
                points = [coordinates[:2], coordinates[2:]]
                result.update(points=points, geometry_points=points, raw_time_s=0.0,
                    duration_s=0.0, distance_m=0.0, common_supported=True,
                    compatible_profiles=frozenset(("HV", "C", "M", "A")),
                    snapped_origin=coordinates[:2], snapped_target=coordinates[2:],
                    snap_gap_origin_m=0.0, snap_gap_target_m=0.0, coincident_od=True)
                return self._remember(key, result)
            _, beta = self.eta_adapter.beta_for(local)
            beta = float(beta)
            if not math.isfinite(beta) or beta <= 0:
                raise Stage3S2AError("INVALID_FROZEN_EMPTY_ROUTE_BETA")
            actor = self.eta_adapter.actor
            request = dict(locations=[dict(lon=coordinates[0], lat=coordinates[1], type="break"),
                dict(lon=coordinates[2], lat=coordinates[3], type="break")], costing="auto",
                units="kilometers", directions_type="none",
                date_time=dict(type=1, value=local.strftime("%Y-%m-%dT%H:%M")))
            self.route_call_count += 1
            trip = actor.route(request)["trip"]
            if int(trip.get("status", 0)) != 0 or len(trip.get("legs", [])) != 1:
                raise Stage3S2AError("SCALAR_EMPTY_ROUTE_NOT_ONE_SUCCESSFUL_LEG")
            raw_time = float(trip["summary"]["time"])
            distance = float(trip["summary"]["length"]) * 1000.0
            duration = raw_time * beta
            if not all(math.isfinite(v) and v > 0 for v in (raw_time, distance, duration)):
                raise Stage3S2AError("SCALAR_EMPTY_ROUTE_TIME_OR_DISTANCE_INVALID")
            shape = str(trip["legs"][0]["shape"])
            points = decode_polyline6(shape)
            if not all(math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180
                    and -90 <= lat <= 90 for lon, lat in points) or not any(p != points[0] for p in points[1:]):
                raise Stage3S2AError("SCALAR_EMPTY_ROUTE_GEOMETRY_INVALID")
            result.update(points=points, geometry_points=points, shape_polyline6=shape,
                raw_time_s=raw_time, duration_s=duration, distance_m=distance,
                beta=beta, snapped_origin=points[0], snapped_target=points[-1],
                snap_gap_origin_m=_gap_m(coordinates[:2], points[0]),
                snap_gap_target_m=_gap_m(coordinates[2:], points[-1]), coincident_od=False)
            edges = self._edge_walk(actor, shape, points, local)
            result["edges"] = edges
            result.update(self._classify(edges, local))
        except Exception as exc:
            code = str(exc) if isinstance(exc, Stage3S2AError) else "EMPTY_ROUTE_EXCEPTION"
            result.update(common_supported=False, compatible_profiles=frozenset(),
                reason_codes=(code,), reasons=(code,), failure_detail=f"{type(exc).__name__}:{exc}")
        if key is not None:
            return self._remember(key, result)
        self._record_result(result)
        return self._copy_result(result)
