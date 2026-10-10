"""Typed, routed Scheme-A connections and model-selected native idle moves.

Historical customer bodies are frozen typed routes, not hypothetical new M3
predictions. Only the new empty connector uses the existing explicit U traffic
assumption. Geometry is routed at the declared snapshot; scalar auto time is
retimed with the frozen pickup multiplier at its predicted departure time.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict, deque
from copy import deepcopy
import hashlib
from math import isfinite
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from stage3.odd_tod.intersection_complex import _bearing
from stage4.analysis.capability_chain_instances import IDENTITY_COLUMNS, JoinEvidence
from stage4.analysis.flexibility_prepare import S3
from stage4.dispatch.city_pickup_support import (
    OVERLAY_REL, TOPOLOGY_COLUMNS, _SmallJoinRouter, _matched_batches,
)
from stage4.dispatch.controlled_movement import CommonIdleMovementManager
from stage4.dispatch.controlled_routes import S2A, _gap_m, _select
from stage4.dispatch.candidate_graph import SpatialVehicle
from stage4.fleetpy_adapter.valhalla_time_adapter import TIMEZONE


ROUTE_CACHE_LIMIT = 1024
BODY_CACHE_LIMIT = 256
INTERFACE_SNAP_TOLERANCE_M = 80.0
GEOMETRY_ROWGROUP_CACHE_BYTES = 64 * 2**20
CERTIFICATE_CACHE_BYTES = 32 * 2**20
CERTIFICATE_CACHE_ENTRIES = 50000
JOIN_STATIC_CACHE_BYTES = 16 * 2**20


def _value(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _point(value):
    point = tuple(map(float, value))
    if (len(point) != 2 or not all(map(isfinite, point))
            or not -180 <= point[0] <= 180 or not -90 <= point[1] <= 90):
        raise ValueError("Scheme-A positions must be finite WGS84 (lon,lat) pairs")
    return point


def _ranges(table):
    """One range per selected order, never one Python row per route token."""
    if not len(table):
        return {}
    encoded = pc.run_end_encode(table["order_id"].combine_chunks())
    result, start = {}, 0
    for order, end in zip(encoded.values.to_pylist(), encoded.run_ends.to_pylist()):
        result[str(order)] = (start, int(end))
        start = int(end)
    return result


class _IndexedJoinRouter(_SmallJoinRouter):
    """Slim UID index and bounded Arrow buffers, no full geometry row objects.

The frozen file currently has one row group. A Parquet reader must decode that
group to fetch any of its rows; retaining a byte-bounded columnar buffer avoids
doing so for every connector. Only requested feature rows become persistent
selected evidence or Python geometry objects.
"""

    def __init__(self, previous):
        super().__init__(previous.base)
        self._features, self._feature_ids = previous._features, set(previous._feature_ids)
        self.geometry_cache = previous.geometry_cache
        self.projection_scan_count = previous.projection_scan_count
        self.projection_time_s = previous.projection_time_s
        self.source = pq.ParquetFile(self.base.root / S2A / "stage3_full_network_edges.parquet")
        tables = []
        for rg in range(self.source.metadata.num_row_groups):
            uid = self.source.read_row_group(rg, columns=["stage3_edge_uid"], use_threads=False)
            tables.append(uid.append_column("row_group", pa.array(np.full(len(uid), rg, np.int32))))
        self.index = pa.concat_tables(tables)
        self.row_groups = OrderedDict()
        self.row_group_bytes = 0
        self.row_group_reads, self.row_group_hits = 0, 0
        self.missing_uids = OrderedDict()
        self.join_static_cache = OrderedDict()
        self.join_static_bytes = 0
        self.join_static_hits, self.join_static_misses = 0, 0
        self.join_parse_hits, self.join_parse_misses = 0, 0

    def prepare_geometry(self, uids):
        wanted = {str(uid) for uid in uids if uid is not None}
        needed = wanted - self._feature_ids - self.missing_uids.keys()
        if not needed:
            return
        started = perf_counter()
        values = pa.array(sorted(needed), type=self.index["stage3_edge_uid"].type)
        matched = self.index.filter(pc.is_in(self.index["stage3_edge_uid"], value_set=values))
        groups = pc.unique(matched["row_group"]).to_pylist()
        pieces = [self._features] if self._features is not None else []
        found = set()
        for rg in groups:
            rg = int(rg)
            if rg in self.row_groups:
                table = self.row_groups[rg]
                self.row_groups.move_to_end(rg)
                self.row_group_hits += 1
            else:
                table = self.source.read_row_group(rg, columns=TOPOLOGY_COLUMNS, use_threads=False)
                self.row_group_reads += 1
                self.projection_scan_count += 1
                if table.nbytes <= GEOMETRY_ROWGROUP_CACHE_BYTES:
                    while self.row_groups and self.row_group_bytes + table.nbytes > GEOMETRY_ROWGROUP_CACHE_BYTES:
                        _, old = self.row_groups.popitem(last=False)
                        self.row_group_bytes -= old.nbytes
                    self.row_groups[rg] = table
                    self.row_group_bytes += table.nbytes
            small = table.filter(pc.is_in(table["stage3_edge_uid"], value_set=values))
            if len(small):
                pieces.append(small)
                found.update(small["stage3_edge_uid"].to_pylist())
        if found:
            self._features = pa.concat_tables(pieces)
            self._feature_ids.update(found)
        for uid in needed - found:
            self.missing_uids[uid] = True
        while len(self.missing_uids) > ROUTE_CACHE_LIMIT:
            self.missing_uids.popitem(last=False)
        self.projection_time_s += perf_counter() - started

    def small_geometry(self, uids):
        result = super().small_geometry(uids)
        while len(self.geometry_cache) > BODY_CACHE_LIMIT:
            self.geometry_cache.popitem(last=False)
        return result

    def join_static_inputs(self, uids):
        """Frozen geometry/control slices, not cached dynamic interface tests.

        Identical UID sets need no repeated whole-table Arrow selection or
        geometry decode. Parser bearings/geometry maps are copied because
        historical reversal adds virtual identities to those maps.
        """
        key = "INPUTS", tuple(sorted(uids))
        cached = self.join_static_cache.get(key)
        if cached is None:
            self.join_static_misses += 1
            geometry = self.small_geometry(uids)
            boundary = _select(self._boundary, "stage3_edge_uid", uids).to_pandas()
            cids = set(boundary.intersection_complex_uid.astype(str))
            movements = _select(self._movements, "intersection_complex_uid", cids)
            movements = _select(_select(movements, "incoming_stage3_edge_uid", uids),
                                "outgoing_stage3_edge_uid", uids).to_pandas()
            controls = _select(self._controls, "intersection_complex_uid", cids).to_pandas()
            bearings = {}
            for uid, record in geometry.items():
                points = record["geometry"]
                bearings[uid] = ((_bearing(points[0], points[1]), _bearing(points[-2], points[-1]))
                                 if len(points) >= 2 else (None, None))
            cached = geometry, boundary, movements, controls, bearings
            self._remember_join_data(key, cached)
        else:
            self.join_static_hits += 1
            self.join_static_cache.move_to_end(key)
            cached, _ = cached
        geometry, boundary, movements, controls, bearings = cached
        return dict(geometry), boundary, movements, controls, dict(bearings)

    def _remember_join_data(self, key, value):
        size = SchemeACityConnectors._certificate_size(key, value)
        if size > JOIN_STATIC_CACHE_BYTES:
            return
        old = self.join_static_cache.pop(key, None)
        if old is not None:
            self.join_static_bytes -= old[1]
        while self.join_static_cache and (len(self.join_static_cache) >= 128
                or self.join_static_bytes + size > JOIN_STATIC_CACHE_BYTES):
            _, (_, old_size) = self.join_static_cache.popitem(last=False)
            self.join_static_bytes -= old_size
        self.join_static_cache[key] = value, size
        self.join_static_bytes += size

    @staticmethod
    def join_parse_key(pieces, overlay, policy):
        columns = ["date", "canonical_edge_uid", "route_token_type", "resolved_stage3_edge_uid"]
        sequence = tuple((label, tuple(frame[columns].itertuples(index=False, name=None)))
                         for label, frame in pieces)
        return "PARSE", hashlib.sha256(repr((sequence, tuple(sorted(overlay.items())), policy))
                                     .encode("utf-8")).digest()

    def join_parse_get(self, key):
        cached = self.join_static_cache.get(key)
        if cached is None:
            self.join_parse_misses += 1
            return None
        self.join_parse_hits += 1
        self.join_static_cache.move_to_end(key)
        return deepcopy(cached[0])

    def join_parse_put(self, key, result):
        self._remember_join_data(key, deepcopy(result))


class SchemeACityConnectors:
    """Callable scalar/identity/seam support shared by layout and chain plans.

The decision caller supplies RouteState(position, context) and either a
ChainTask(pickup, dropoff, source_date, source_order_id, source_kind) or a site
dictionary. Test31 task identities are accessible only after native activation.
Forecast identities must belong to the supplied earlier-history templates.
Site dictionaries with require_geometry=True obtain an executable routed
record; hypothetical links retain no dense geometry in their certificate.
Returned route geometries are suitable for a selected current MOVE; they are
not permission to execute a scenario's hypothetical future customer path.
"""

    def __init__(self, control, empty_router, current_validator, forecast_templates,
            *, geometry_timestamp):
        self.control = control
        self.router = empty_router
        self.current_validator = current_validator
        if control.eta_adapter is not empty_router.eta_adapter:
            raise ValueError("Scheme-A connectors must share the existing ETA adapter")
        stamp = pd.Timestamp(geometry_timestamp)
        if pd.isna(stamp):
            raise ValueError("Scheme-A connector geometry snapshot must be explicit")
        self.geometry_timestamp = (stamp.tz_localize(TIMEZONE) if stamp.tzinfo is None
                                   else stamp.tz_convert(TIMEZONE))
        self._geometry_key = self.geometry_timestamp.isoformat()
        self.actual_evidence = current_validator.seam_evidence
        self.join_router = self.actual_evidence.router
        topology = Path(empty_router.root) / S2A / "stage3_full_network_edges.parquet"
        if isinstance(self.join_router, _SmallJoinRouter) and topology.is_file():
            if not isinstance(self.join_router, _IndexedJoinRouter):
                self.join_router = _IndexedJoinRouter(self.join_router)
                self.actual_evidence.router = self.join_router
            # The scalar router's unknown-turn bearing fallback shares this
            # bounded geometry access, not another full-parquet per-edge scan.
            self.router._bearings_for = self.join_router._bearings_for
        self.route_cache = OrderedDict()
        self.body_cache = OrderedDict()
        self.native_context_cache = OrderedDict()
        self.native_signature_cache = OrderedDict()
        self._native_scope = None
        self._prefetched_hv = None
        self.certificate_cache = OrderedDict()
        self.certificate_bytes = 0
        self.train_tables, self.train_ranges = {}, {}
        self.train_pairs = set()
        self.train_overlay = None
        self.counts, self.reasons, self.timings = Counter(), Counter(), Counter()
        self._known_count, self._known_orders = -1, {}
        self._prepare_train(forecast_templates)

    def _prepare_train(self, templates):
        started = perf_counter()
        if isinstance(templates, pd.DataFrame):
            date_col = "source_date" if "source_date" in templates else "date"
            order_col = "source_order_id" if "source_order_id" in templates else "order_id"
            pairs = {(str(date), str(order)) for date, order
                in templates[[date_col, order_col]].itertuples(index=False, name=None)}
            selected = templates
            if "compatible_C" in selected:
                selected = selected.loc[selected.compatible_C.astype(bool)]
            identity_pairs = {(str(date), str(order)) for date, order
                in selected[[date_col, order_col]].itertuples(index=False, name=None)}
        else:
            templates = list(templates)
            pairs = {(str(_value(task, "source_date")), str(_value(task, "source_order_id")))
                for task in templates}
            identity_pairs = {(str(_value(task, "source_date")), str(_value(task, "source_order_id")))
                for task in templates if "C" in _value(task, "compatible_profiles", ())}
        if any(not "20161009" <= date <= "20161024" for date, _ in pairs):
            raise ValueError("Scheme-A forecast identities must come from the earlier Train period")
        self.train_pairs = pairs
        needed_geometry, reverse_keys = set(), set()
        root = Path(self.router.root)
        for date in sorted({date for date, _ in identity_pairs}):
            wanted = {order for day, order in identity_pairs if day == date}
            path = root / S3 / f"cache/train/date={date}/identity.parquet"
            pieces = list(_matched_batches(path, IDENTITY_COLUMNS, "order_id", wanted))
            self.counts["train_identity_projection_scans"] += 1
            if not pieces:
                self.reasons["TRAIN_TEMPLATE_IDENTITY_MISSING"] += len(wanted)
                continue
            table = pa.concat_tables(pieces).sort_by([
                ("order_id", "ascending"), ("route_sequence", "ascending")])
            self.train_tables[date] = table
            self.train_ranges[date] = _ranges(table)
            self.counts["train_identity_orders_loaded"] += len(self.train_ranges[date])
            self.counts["train_identity_tokens_loaded"] += len(table)
            self.reasons["TRAIN_TEMPLATE_IDENTITY_MISSING"] += len(wanted - self.train_ranges[date].keys())
            needed_geometry.update(pc.unique(table["resolved_stage3_edge_uid"]).drop_null().to_pylist())
            reverse = table.filter(pc.equal(table["route_token_type"], "HISTORICAL_REVERSE_OVERLAY"))
            reverse_keys.update(pc.unique(reverse["canonical_edge_uid"]).drop_null().to_pylist())
        if reverse_keys:
            parts = list(_matched_batches(root / OVERLAY_REL,
                ["canonical_edge_uid", "physical_forward_stage3_edge_uid"],
                "canonical_edge_uid", reverse_keys))
            self.counts["train_overlay_projection_scans"] += 1
            if parts:
                self.train_overlay = pa.concat_tables(parts)
                needed_geometry.update(pc.unique(self.train_overlay["physical_forward_stage3_edge_uid"])
                    .drop_null().to_pylist())
        self.join_router.prepare_geometry(needed_geometry)
        self.timings["train_identity_geometry_preparation_s"] += perf_counter() - started

    def _known(self):
        records = self.control.request_by_rid
        if len(records) != self._known_count:
            self._known_orders = {str(record.order_id): record for record in records.values()}
            self._known_count = len(records)
        return self._known_orders

    def _task_metadata(self, task):
        date = str(_value(task, "source_date"))
        order = str(_value(task, "source_order_id"))
        kind = str(_value(task, "source_kind", ""))
        if kind == "FORECAST":
            if (date, order) not in self.train_pairs:
                return None, "FORECAST_SOURCE_NOT_IN_DECLARED_EARLIER_TEMPLATES"
        elif date == "20161031" and kind in (
                "ACTUAL_PENDING", "ACTUAL_COMMITTED", "ACTUAL_BOOKED", "ACTUAL_COMPLETED"):
            record = self._known().get(order)
            if record is None or float(record.sim_time_s) > float(self.control.sim_time):
                return None, "TEST31_CUSTOMER_NOT_YET_REVEALED"
        else:
            return None, "CUSTOMER_SOURCE_PROVENANCE_INVALID"
        return (date, order), None

    def _body(self, task):
        key, reason = self._task_metadata(task)
        if reason:
            return None, reason
        if key in self.body_cache:
            self.body_cache.move_to_end(key)
            self.counts["customer_body_cache_hits"] += 1
            return self.body_cache[key], None
        date, order = key
        if date == "20161031":
            evidence = self.actual_evidence
            if order in evidence.tokens:
                frame = evidence.tokens[order]
            else:
                bounds = getattr(evidence, "_private_ranges", {}).get(order)
                table = getattr(evidence, "_private_identity", None)
                if bounds is None or table is None:
                    return None, "TEST31_IDENTITY_NOT_IN_PRIVATE_PREPROJECTED_STORE"
                first, end = bounds
                frame = table.slice(first, end - first).to_pandas()
        else:
            bounds = self.train_ranges.get(date, {}).get(order)
            if bounds is None:
                return None, "TRAIN_TEMPLATE_IDENTITY_MISSING"
            first, end = bounds
            frame = self.train_tables[date].slice(first, end - first).to_pandas()
        # Whole selected body: no 8-edge truncation of complex INTERNAL runs.
        frame = frame.sort_values("route_sequence", kind="stable").reset_index(drop=True)
        self.body_cache[key] = frame
        while len(self.body_cache) > BODY_CACHE_LIMIT:
            self.body_cache.popitem(last=False)
        self.counts["customer_body_views_materialized"] += 1
        return frame, None

    @staticmethod
    def _coords(task):
        first, last = _point(_value(task, "pickup")), _point(_value(task, "dropoff"))
        return dict(start_lon_wgs84=first[0], start_lat_wgs84=first[1],
            end_lon_wgs84=last[0], end_lat_wgs84=last[1])

    def _completed_task(self, record):
        return SimpleNamespace(source_date="20161031", source_order_id=str(record.order_id),
            source_kind="ACTUAL_COMPLETED", pickup=(record.pickup_lon_wgs84, record.pickup_lat_wgs84),
            dropoff=(record.dropoff_lon_wgs84, record.dropoff_lat_wgs84))

    def _native_source(self, context, origin):
        c = self.control
        vid, now = int(context["native_vehicle_id"]), float(context["timestamp_s"])
        if abs(now - float(c.sim_time)) > 1e-8:
            return None, None, "NATIVE_DEPARTURE_CONTEXT_NOT_CURRENT"
        runtime = c.runtime_by_vid[vid]
        current = _point(c.routing_engine.return_position_coordinates(runtime.native_vehicle.pos))
        if _gap_m(current, origin) > 1e-5:
            return None, None, "NATIVE_DEPARTURE_POSITION_MISMATCH"
        key = (vid, now, current, len(c.assignment_rows))
        scope = now, len(c.assignment_rows)
        if self._native_scope != scope:
            self.native_context_cache.clear()
            self.native_signature_cache.clear()
            self._native_scope = scope
        if key in self.native_context_cache:
            self.native_context_cache.move_to_end(key)
            return self.native_context_cache[key]
        previous, finished_s = None, -1.
        for row in reversed(c.assignment_rows):
            if int(row["native_vehicle_id"]) == vid and (row.get("completed", False)
                    or int(row["native_request_id"]) in c.completed_rids):
                previous = c.request_by_rid.get(int(row["native_request_id"]))
                finish = row.get("service_end_time")
                if finish is not None and not pd.isna(finish):
                    finished_s = float((pd.Timestamp(finish) - c._timestamp(0)).total_seconds())
                break
        routed_context = self.actual_evidence._idle_context(c, runtime, now, finished_s)
        saved = getattr(c, "city_departure_contexts", {}).get(vid)
        if routed_context is None and saved is not None:
            if (_point(saved.get("point", current)) == current
                    and float(saved.get("source_move_start_s", float("inf"))) >= finished_s):
                routed_context = saved
        if routed_context is not None:
            frame = self._empty_context_tokens(routed_context)
            result = (frame, dict(start_lon_wgs84=origin[0], start_lat_wgs84=origin[1],
                end_lon_wgs84=origin[0], end_lat_wgs84=origin[1]), None)
        elif previous is not None and runtime.native_vehicle.pos == previous.dropoff_position:
            task = self._completed_task(previous)
            frame, reason = self._body(task)
            result = (frame, self._coords(task), reason)
        elif previous is not None or vid in getattr(getattr(c, "repositioning_manager", None), "active", {}):
            result = (None, None, "PREVIOUS_DIRECTION_CONTEXT_UNAVAILABLE")
        else:
            result = (None, None, None)  # Genuine initial admission has no fabricated incoming edge.
        self.native_context_cache[key] = result
        while len(self.native_context_cache) > BODY_CACHE_LIMIT:
            self.native_context_cache.popitem(last=False)
        return result

    @staticmethod
    def _empty_context_tokens(context):
        uids = context.get("edge_uids")
        if uids is None:
            uids = [edge["stage3_edge_uid"] for edge in context.get("edges", ())]
        return pd.DataFrame([dict(date="20161031", order_id="EMPTY_SOURCE",
            route_sequence=i, canonical_edge_uid=None, route_token_type="FULL_NETWORK_EDGE",
            resolved_stage3_edge_uid=uid) for i, uid in enumerate(uids)], columns=IDENTITY_COLUMNS)

    def _source(self, origin_state):
        origin, context = _point(_value(origin_state, "position")), _value(origin_state, "context")
        if context is None:
            return None, None, None
        kind = context.get("kind")
        if kind == "NATIVE_VEHICLE":
            return self._native_source(context, origin)
        if kind == "CUSTOMER":
            task = context["task"]
            if _gap_m(_point(_value(task, "dropoff")), origin) > 1e-5:
                return None, None, "CUSTOMER_DEPARTURE_POSITION_MISMATCH"
            body, reason = self._body(task)
            return body, self._coords(task), reason
        if kind == "EMPTY_ARRIVAL":
            if _gap_m(_point(context["point"]), origin) > 1e-5:
                return None, None, "EMPTY_ARRIVAL_POSITION_MISMATCH"
            frame = self._empty_context_tokens(context)
            if not len(frame):
                # A coincident move retains its prior context, never invents edges.
                previous = context.get("previous_context")
                return self._source(SimpleNamespace(position=origin, context=previous))
            return frame, dict(start_lon_wgs84=origin[0], start_lat_wgs84=origin[1],
                end_lon_wgs84=origin[0], end_lat_wgs84=origin[1]), None
        return None, None, "DEPARTURE_CONTEXT_KIND_UNSUPPORTED"

    def _join(self, origin_state, target, routed):
        source, source_coords, reason = self._source(origin_state)
        if reason:
            return dict(supported=False, compatible_profiles=[], reason_codes=[reason])
        following, target_coords = None, None
        if not isinstance(target, dict) or "site_id" not in target:
            following, reason = self._body(target)
            if reason:
                return dict(supported=False, compatible_profiles=[], reason_codes=[reason])
            target_coords = self._coords(target)
        evidence = JoinEvidence.__new__(JoinEvidence)
        evidence.router = self.join_router
        evidence.tokens = {}
        coords = {}
        source_key = target_key = None
        if source is not None and len(source):
            source_key = "SCHEME_A_PREVIOUS"
            evidence.tokens[source_key] = source
            coords[source_key] = source_coords
        if following is not None:
            target_key = "SCHEME_A_FOLLOWING"
            evidence.tokens[target_key] = following
            coords[target_key] = target_coords
        evidence.selected = pd.DataFrame.from_dict(coords, orient="index")
        evidence.simulation_date = "20161031"
        reverse_keys = set()
        for frame in evidence.tokens.values():
            reverse_keys.update(frame.loc[frame.route_token_type.eq("HISTORICAL_REVERSE_OVERLAY"),
                "canonical_edge_uid"].dropna().astype(str))
        overlays = []
        for table in (self.train_overlay, getattr(self.actual_evidence, "_private_overlay", None)):
            if table is not None and reverse_keys:
                selected = table.filter(pc.is_in(table["canonical_edge_uid"],
                    value_set=pa.array(sorted(reverse_keys), type=table["canonical_edge_uid"].type)))
                if len(selected):
                    overlays.append(selected.to_pandas())
        if not overlays and reverse_keys and len(getattr(self.actual_evidence, "overlay", ())):
            overlays = [self.actual_evidence.overlay.loc[
                self.actual_evidence.overlay.canonical_edge_uid.isin(reverse_keys)]]
        evidence.overlay = (pd.concat(overlays, ignore_index=True).drop_duplicates("canonical_edge_uid")
            if overlays else pd.DataFrame(columns=["canonical_edge_uid", "physical_forward_stage3_edge_uid"]))
        evidence.overlay_forward = dict(zip(evidence.overlay.canonical_edge_uid,
            evidence.overlay.physical_forward_stage3_edge_uid))
        if not reverse_keys <= evidence.overlay_forward.keys():
            return dict(supported=False, compatible_profiles=[], reason_codes=["HISTORICAL_REVERSE_OVERLAY_MISSING"])
        result = evidence.evaluate(source_key, target_key, routed, INTERFACE_SNAP_TOLERANCE_M)
        # Python geometry views are bounded independently of columnar buffers.
        cache = getattr(self.join_router, "geometry_cache", None)
        if cache is not None:
            while len(cache) > BODY_CACHE_LIMIT:
                cache.popitem(last=False)
        return result

    def _reject(self, reasons, *, routed=None):
        reasons = sorted(set(reasons or ["CONNECTOR_SUPPORT_UNAVAILABLE"]))
        self.counts["rejected_connections"] += 1
        self.reasons.update(reasons)
        return dict(supported=False, compatible_profiles=(), travel_time_s=None,
            empty_distance_m=None, arrival_context=None, reason_codes=reasons, routed=routed)

    def _certificate_source_key(self, origin_state, *, profile=None):
        """Cache only exact immutable evidence, never a vehicle ID alone."""
        origin, context = _point(_value(origin_state, "position")), _value(origin_state, "context")
        if context is None:
            return ("INITIAL", origin), None
        kind = context.get("kind")
        if kind == "NATIVE_VEHICLE":
            if profile != "C":
                return None, None
            try:
                frame, coords, reason = self._native_source(context, origin)
            except (KeyError, AttributeError, ValueError):
                return None, None
            if reason:
                # Preserve the old route-before-seam rejection priority.
                return None, None
            cache_key = (int(context["native_vehicle_id"]), float(context["timestamp_s"]),
                         origin, len(self.control.assignment_rows))
            signature = self.native_signature_cache.get(cache_key)
            if signature is None:
                rows = tuple(frame[IDENTITY_COLUMNS].itertuples(index=False, name=None)) \
                    if frame is not None else ()
                source = rows, tuple(sorted((coords or {}).items()))
                signature = hashlib.sha256(repr(source).encode("utf-8")).digest()
                self.native_signature_cache[cache_key] = signature
                while len(self.native_signature_cache) > BODY_CACHE_LIMIT:
                    self.native_signature_cache.popitem(last=False)
            else:
                self.native_signature_cache.move_to_end(cache_key)
            self.counts["native_exact_prefix_certificate_keys"] += 1
            return ("NATIVE_IMMUTABLE_PREFIX", origin, signature), None
        if kind == "CUSTOMER":
            task = context["task"]
            identity, reason = self._task_metadata(task)
            if reason:
                return None, reason
            pickup, dropoff = _point(_value(task, "pickup")), _point(_value(task, "dropoff"))
            if _gap_m(dropoff, origin) > 1e-5:
                return None, "CUSTOMER_DEPARTURE_POSITION_MISMATCH"
            return ("CUSTOMER", *identity, pickup, dropoff, origin), None
        if kind == "EMPTY_ARRIVAL":
            if _gap_m(_point(context["point"]), origin) > 1e-5:
                return None, "EMPTY_ARRIVAL_POSITION_MISMATCH"
            edges = context.get("edge_uids")
            if edges is None:
                edges = tuple(edge["stage3_edge_uid"] for edge in context.get("edges", ()))
            edges = tuple(edges)
            if not edges:
                return self._certificate_source_key(SimpleNamespace(
                    position=origin, context=context.get("previous_context")))
            return ("EMPTY_ARRIVAL", origin, edges), None
        return None, "DEPARTURE_CONTEXT_KIND_UNSUPPORTED"

    def _certificate_key(self, mode, profile, origin, source_key, target_key):
        if source_key is None:
            return None
        full = self._geometry_key, mode, profile, origin, source_key, target_key
        # Complete evidence is hashed, not truncated or rounded. Keeping its
        # digest rather than repeated long edge-ID tuples improves the hit
        # rate within the SAME 32-MiB conservative allocation ceiling.
        return hashlib.sha256(repr(full).encode("utf-8")).digest()

    @staticmethod
    def _certificate_size(key, certificate):
        """Conservative Python allocation size, including nested context tokens."""
        seen = set()
        def size(value):
            identity = id(value)
            if identity in seen:
                return 0
            seen.add(identity)
            count = sys.getsizeof(value)
            if isinstance(value, dict):
                count += sum(size(k) + size(v) for k, v in value.items())
            elif isinstance(value, (tuple, list, set, frozenset)):
                count += sum(map(size, value))
            elif isinstance(value, pd.DataFrame):
                count += int(value.memory_usage(index=True, deep=True).sum()) + 4096
            elif isinstance(value, np.ndarray):
                count += value.nbytes
            elif hasattr(value, "__dict__"):
                count += size(vars(value))
            return count
        # Allow for the OrderedDict node and tuple/size bookkeeping as well.
        return size(key) + size(certificate) + 160

    def _remember_certificate(self, key, certificate):
        if key is None:
            return
        byte_count = self._certificate_size(key, certificate)
        if byte_count > CERTIFICATE_CACHE_BYTES:
            self.counts["certificate_too_large_not_cached"] += 1
            return
        while self.certificate_cache and (len(self.certificate_cache) >= CERTIFICATE_CACHE_ENTRIES
                or self.certificate_bytes + byte_count > CERTIFICATE_CACHE_BYTES):
            _, (_, old_bytes) = self.certificate_cache.popitem(last=False)
            self.certificate_bytes -= old_bytes
            self.counts["certificate_cache_evictions"] += 1
        self.certificate_cache[key] = (certificate, byte_count)
        self.certificate_bytes += byte_count

    def _static_certificate(self, origin_state, target, origin, target_point, profile, site):
        key = (*origin, *target_point)
        if key in self.route_cache:
            route = self.route_cache[key]
            self.route_cache.move_to_end(key)
            self.counts["route_cache_hits"] += 1
        else:
            started = perf_counter()
            route = self.router.route(*origin, *target_point, self.geometry_timestamp)
            self.timings["scalar_routing_time_s"] += perf_counter() - started
            self.counts["scalar_route_evaluations"] += 1
            self.route_cache[key] = route
            while len(self.route_cache) > ROUTE_CACHE_LIMIT:
                self.route_cache.popitem(last=False)
        certificate = dict(supported=False, compatible_profiles=(), reason_codes=(),
            raw_time_s=None, distance_m=None, arrival_context=None,
            connection_evidence_state=None, coincident_od=bool(route.get("coincident_od", False)))
        def failed(reasons):
            certificate["reason_codes"] = tuple(sorted(set(reasons or ["CONNECTOR_SUPPORT_UNAVAILABLE"])))
            return certificate, route
        raw, distance = route.get("raw_time_s"), route.get("distance_m")
        if (raw is None or distance is None or not all(isfinite(float(v)) and float(v) >= 0
                for v in (raw, distance)) or not route.get("points")):
            return failed(route.get("reason_codes") or ["SCALAR_AUTO_GEOMETRY_TIME_UNAVAILABLE"])
        raw, distance = float(raw), float(distance)
        certificate.update(raw_time_s=raw, distance_m=distance)
        if not certificate["coincident_od"] and min(raw, distance) <= 0:
            return failed(["SCALAR_AUTO_NONCOINCIDENT_ZERO_TIME_OR_DISTANCE"])
        if "CERTIFIED_COMMON_PROHIBITION" in route.get("reason_codes", ()):
            return failed(["CERTIFIED_COMMON_PROHIBITION"])
        if site and (not route["common_supported"] or profile not in route["compatible_profiles"]):
            return failed(route.get("reason_codes")
                or ["OPTIONAL_MOVE_COMMON_OR_PROFILE_SUPPORT_UNAVAILABLE"])
        if profile == "C":
            if not route["common_supported"]:
                return failed(route.get("reason_codes"))
            if "C" not in route["compatible_profiles"]:
                return failed(route.get("profile_reason_codes", {}).get("C")
                    or ["EMPTY_CONNECTOR_OUTSIDE_C_PROFILE"])
            if any(route.get(name) is None or not isfinite(float(route[name]))
                    or float(route[name]) > INTERFACE_SNAP_TOLERANCE_M
                    for name in ("snap_gap_origin_m", "snap_gap_target_m")):
                return failed(["SNAP_GAP_EXCEEDS_COMMON_INSTANCE_TOLERANCE"])
            started = perf_counter()
            join = self._join(origin_state, target, route)
            self.timings["directional_seam_time_s"] += perf_counter() - started
            self.counts["C_directional_seams_examined"] += 1
            if not join["supported"]:
                return failed(join.get("reason_codes"))
            if "C" not in join.get("compatible_profiles", ()):
                return failed(join.get("C_reason_codes") or ["CONNECTOR_SEAM_OUTSIDE_C_PROFILE"])
            self.counts["C_directional_seams_passed"] += 1
        else:
            join = None
        arrival = None
        if site:
            edges = tuple(edge["stage3_edge_uid"] for edge in route.get("edges", ()))
            arrival = dict(kind="EMPTY_ARRIVAL", edge_uids=edges, point=target_point,
                geometry_timestamp=self._geometry_key, traffic_unknown_share=1.0,
                previous_context=_value(origin_state, "context") if not edges else None,
                directional_evidence="FULL_SELECTED_EMPTY_ROUTE_NOT_FABRICATED_CUSTOMER")
        certificate.update(supported=True, compatible_profiles=(profile,), arrival_context=arrival,
            connection_evidence_state="C_DIRECTED_SEAM_CERTIFIED" if profile == "C" else "HV_AUTO_ROUTABLE")
        # Keep only scalar/static support in the LRU. A full current executable
        # route is returned separately and never copied into this certificate.
        route = dict(route, seam=join)
        return certificate, route

    def _hv_auto_od_certificate(self, origin, target_point, stamp):
        """Existing certified sparse AUTO OD for hypothetical HV customer links.

No geometry is invented and no C direction/control certificate is implied.
This branch is enabled explicitly by the parent protocol, never for site MOVE.
"""
        started = perf_counter()
        self.counts["HV_future_auto_OD_evaluations"] += 1
        certificate = dict(supported=False, compatible_profiles=(), reason_codes=(),
            raw_time_s=None, distance_m=None, arrival_context=None,
            connection_evidence_state="HV_STATIC_AUTO_OD_ROUTABLE",
            coincident_od=origin == target_point, allow_zero_auto_od=True)
        try:
            request_key = origin, target_point, pd.Timestamp(stamp).isoformat()
            prepared = (self._prefetched_hv.get(request_key) if self._prefetched_hv is not None else None)
            if prepared:
                estimate = prepared.popleft()
                self.counts["HV_future_batch_answers_consumed"] += 1
            else:
                vehicle = SpatialVehicle(vehicle_id="SCHEME_A_HV_FORECAST", native_vehicle_id=0,
                    vehicle_type="HV", lon_wgs84=origin[0], lat_wgs84=origin[1])
                answers = self.control.eta_adapter.estimate_many([vehicle], *target_point, stamp)
                estimate = answers.get(0)
            if estimate is None:
                certificate["reason_codes"] = ("HV_AUTO_OD_ROUTING_RESULT_MISSING",)
                return certificate, None
            raw, distance = float(estimate.valhalla_time_s), float(estimate.route_distance_m)
            if not all(isfinite(value) and value >= 0 for value in (raw, distance)):
                certificate["reason_codes"] = ("HV_AUTO_OD_ROUTING_RESULT_INVALID",)
                return certificate, None
            self.counts["HV_future_auto_OD_adapter_cache_hits"] += int(bool(estimate.cache_hit))
            certificate.update(supported=True, compatible_profiles=("HV",),
                raw_time_s=raw, distance_m=distance)
            return certificate, None
        except Exception as exc:
            self.counts[f"HV_future_auto_OD_exception_{type(exc).__name__}"] += 1
            certificate["reason_codes"] = ("HV_AUTO_OD_ROUTING_EXCEPTION",)
            return certificate, None
        finally:
            self.timings["HV_future_auto_OD_time_s"] += perf_counter() - started

    def evaluate_many(self, queries):
        """Route one label's sparse HV candidates in original logical order.

        The existing epoch adapter preserves sequential cache-fill and rounded
        alias semantics. A mixed/cached certificate batch uses the scalar path
        instead of prefetching speculative arcs. C and MOVE never use ETA-only
        evidence. At most the frozen Top-3 inputs are queued by the generator.
        """
        queries = tuple(queries)
        eta = self.control.eta_adapter
        if (not queries or len(queries) > 3 or self._prefetched_hv is not None
                or not getattr(self.control, "config", {}).get("scheme_a_fast_forecast_hv", False)
                or not hasattr(eta, "estimate_epoch")):
            return [self(*query) for query in queries]
        batches, keys, seen_certificates = [], [], set()
        eligible = True
        try:
            for state, target, departure, profile in queries:
                if profile != "HV" or isinstance(target, dict) or not isfinite(float(departure)) or departure < 0:
                    eligible = False
                    break
                origin, point = _point(_value(state, "position")), _point(_value(target, "pickup"))
                source_key, reason = self._certificate_source_key(state, profile="HV")
                identity, target_reason = self._task_metadata(target)
                if reason or target_reason or "HV" not in _value(target, "compatible_profiles", ()):
                    eligible = False
                    break
                target_key = ("CUSTOMER", *identity, point, _point(_value(target, "dropoff")))
                key = self._certificate_key("AUTO_OD", "HV", origin, source_key, target_key)
                if key is not None and key in self.certificate_cache:
                    eligible = False
                    break
                stamp = self.control._timestamp(float(departure))
                _, beta = eta.beta_for(stamp)
                if not isfinite(float(beta)) or beta <= 0 or self.geometry_timestamp.isoformat() != self._geometry_key:
                    eligible = False
                    break
                if key is not None and key in seen_certificates:
                    continue  # Later call hits the first static certificate.
                seen_certificates.add(key)
                vehicle = SpatialVehicle(vehicle_id="SCHEME_A_HV_FORECAST", native_vehicle_id=0,
                    vehicle_type="HV", lon_wgs84=origin[0], lat_wgs84=origin[1])
                batches.append(([vehicle], *point, stamp))
                keys.append((origin, point, pd.Timestamp(stamp).isoformat()))
        except (ValueError, TypeError, KeyError):
            eligible = False
        if not eligible:
            return [self(*query) for query in queries]
        started = perf_counter()
        failed = False
        try:
            results = eta.estimate_epoch(batches)
            if len(results) != len(batches):
                raise ValueError("HV sparse epoch dropped a batch result")
        except Exception:
            self.counts["HV_future_batch_adapter_exception_scalar_fallback"] += 1
            failed = True
        finally:
            elapsed = perf_counter() - started
            self.timings["HV_future_batch_queue_time_s"] += elapsed
            self.timings["HV_future_auto_OD_time_s"] += elapsed
            self.timings["total_connection_time_s"] += elapsed
        if failed:
            return [self(*query) for query in queries]
        self.counts["HV_future_batch_calls"] += 1
        self.counts["HV_future_batch_arc_inputs"] += len(batches)
        prepared = defaultdict(deque)
        for key, result in zip(keys, results):
            prepared[key].append(result.get(0))
        self._prefetched_hv = prepared
        try:
            return [self(*query) for query in queries]
        finally:
            self._prefetched_hv = None

    def __call__(self, origin_state, target, departure_s, profile_id):
        started = perf_counter()
        self.counts["examined_connections"] += 1
        try:
            profile = str(profile_id)
            if profile not in ("HV", "C"):
                return self._reject(["OUTSIDE_FIXED_HV_C_COMPARISON"])
            departure = float(departure_s)
            if not isfinite(departure) or departure < 0:
                return self._reject(["CONNECTOR_DEPARTURE_TIME_INVALID"])
            origin = _point(_value(origin_state, "position"))
            site = isinstance(target, dict) and "site_id" in target
            target_point = _point(target["position"] if site else _value(target, "pickup"))
            source_key, reason = self._certificate_source_key(origin_state, profile=profile)
            if reason:
                return self._reject([reason])
            identity = None
            if not site:
                identity, reason = self._task_metadata(target)
                if reason:
                    return self._reject([reason])
                if profile not in _value(target, "compatible_profiles", ()):
                    return self._reject(["CUSTOMER_BODY_OUTSIDE_PROFILE"])
            if self.geometry_timestamp.isoformat() != self._geometry_key:
                raise ValueError("the connector geometry snapshot is fixed for the certificate lifetime")
            stamp = self.control._timestamp(departure)
            bin_index, beta = self.control.eta_adapter.beta_for(stamp)
            beta = float(beta)
            if not isfinite(beta) or beta <= 0:
                return self._reject(["CONNECTOR_DEPARTURE_BETA_INVALID"])
            require_geometry = bool(site and target.get("require_geometry", False))
            fast_hv = bool(profile == "HV" and not site
                and getattr(self.control, "config", {}).get("scheme_a_fast_forecast_hv", False)
                and hasattr(self.control.eta_adapter, "estimate_many"))
            target_key = (("SITE", str(target["site_id"]), target_point) if site else
                ("CUSTOMER", *identity, target_point, _point(_value(target, "dropoff"))))
            key = (self._certificate_key("AUTO_OD" if fast_hv else "TYPED_SCALAR",
                profile, origin, source_key, target_key) if not require_geometry else None)
            if key is not None and key in self.certificate_cache:
                certificate, _ = self.certificate_cache[key]
                self.certificate_cache.move_to_end(key)
                self.counts["certificate_cache_hits"] += 1
                route = None
            else:
                self.counts["certificate_cache_misses" if key is not None else "certificate_cache_bypassed"] += 1
                certificate, route = (self._hv_auto_od_certificate(origin, target_point, stamp)
                    if fast_hv else self._static_certificate(origin_state, target, origin, target_point, profile, site))
                self._remember_certificate(key, certificate)
            if not certificate["supported"]:
                return self._reject(certificate["reason_codes"], routed=route)
            raw, distance = certificate["raw_time_s"], certificate["distance_m"]
            duration = raw * beta
            if (not certificate["coincident_od"] and not certificate.get("allow_zero_auto_od", False)
                    and min(raw, duration, distance) <= 0):
                return self._reject(["SCALAR_AUTO_NONCOINCIDENT_ZERO_TIME_OR_DISTANCE"], routed=route)
            if not site:
                deadline = _value(target, "deadline_s")
                if deadline is not None and departure + duration > float(deadline) + 1e-9:
                    return self._reject(["CONNECTOR_EXCEEDS_ORIGINAL_PICKUP_DEADLINE"], routed=route)
            routed = dict(route, duration_s=duration, beta=beta, time_bin_index=int(bin_index),
                origin_wgs84=origin, target_wgs84=target_point, departure_s=departure,
                geometry_timestamp=self.geometry_timestamp.isoformat(), decision_timestamp=pd.Timestamp(stamp).isoformat(),
                connection_evidence_state=certificate["connection_evidence_state"], supported=True) if require_geometry and route is not None else None
            self.counts["supported_connections"] += 1
            self.counts[f"{profile}_connections_passed"] += 1
            return dict(supported=True, compatible_profiles=(profile,), travel_time_s=duration,
                empty_distance_m=distance, arrival_context=certificate["arrival_context"],
                reason_codes=[], routed=routed,
                connection_evidence_state=certificate["connection_evidence_state"])
        finally:
            self.timings["total_connection_time_s"] += perf_counter() - started
            while len(self.actual_evidence.tokens) > BODY_CACHE_LIMIT:
                self.actual_evidence.tokens.popitem(last=False)

    def diagnostics(self):
        identity_bytes = sum(table.nbytes for table in self.train_tables.values())
        overlay_bytes = self.train_overlay.nbytes if self.train_overlay is not None else 0
        features = getattr(self.join_router, "_features", None)
        return dict(counts=dict(self.counts), rejection_reason_counts=dict(self.reasons),
            timings_s=dict(self.timings), route_cache_entries=len(self.route_cache),
            route_cache_limit=ROUTE_CACHE_LIMIT, body_view_cache_entries=len(self.body_cache),
            body_view_cache_limit=BODY_CACHE_LIMIT, native_context_cache_entries=len(self.native_context_cache),
            train_identity_arrow_mib=identity_bytes / 2**20, train_overlay_arrow_mib=overlay_bytes / 2**20,
            shared_geometry_arrow_mib=features.nbytes / 2**20 if features is not None else 0.,
            geometry_object_cache_entries=len(getattr(self.join_router, "geometry_cache", ())),
            actual_body_view_cache_entries=len(self.actual_evidence.tokens),
            certificate_cache_entries=len(self.certificate_cache),
            certificate_cache_key_format="SHA256_COMPLETE_EXACT_CONTEXT_NOT_COORDINATE_ROUNDING",
            native_signature_cache_entries=len(self.native_signature_cache),
            certificate_cache_entry_limit=CERTIFICATE_CACHE_ENTRIES,
            certificate_cache_mib=self.certificate_bytes / 2**20,
            certificate_cache_byte_limit_mib=CERTIFICATE_CACHE_BYTES / 2**20,
            geometry_uid_rowgroup_index_mib=getattr(self.join_router, "index", pa.table({})).nbytes / 2**20,
            geometry_rowgroup_cache_mib=getattr(self.join_router, "row_group_bytes", 0) / 2**20,
            geometry_rowgroup_cache_limit_mib=GEOMETRY_ROWGROUP_CACHE_BYTES / 2**20,
            geometry_rowgroup_reads=getattr(self.join_router, "row_group_reads", None),
            geometry_rowgroup_cache_hits=getattr(self.join_router, "row_group_hits", None),
            join_static_cache_entries=len(getattr(self.join_router, "join_static_cache", ())),
            join_static_cache_mib=getattr(self.join_router, "join_static_bytes", 0) / 2**20,
            join_static_cache_limit_mib=JOIN_STATIC_CACHE_BYTES / 2**20,
            join_static_cache_hits=getattr(self.join_router, "join_static_hits", 0),
            join_static_cache_misses=getattr(self.join_router, "join_static_misses", 0),
            join_parse_cache_hits=getattr(self.join_router, "join_parse_hits", 0),
            join_parse_cache_misses=getattr(self.join_router, "join_parse_misses", 0),
            geometry_timestamp=self.geometry_timestamp.isoformat(),
            eta_beta_time="ACTUAL_OR_PREDICTED_CONNECTOR_DEPARTURE",
            traffic_policy="EXPLICIT_U_FOR_NEW_EMPTY_CONNECTOR_NO_M3_FABRICATION",
            full_customer_token_views=True, arbitrary_terminal_edge_truncation=False,
            HV_semantics="AUTO_ROUTABLE_WITHOUT_C_CAPABILITY_OR_SEAM_GATE")


class SelectedMovementManager(CommonIdleMovementManager):
    """Native accounting/hooks are shared; only master-selected moves execute.

The historical deficit heuristic is deliberately dormant in this subclass.
It does not silently replace, extend, reroute or reschedule a selected action.
"""

    def __init__(self, control, reference, empty_router, **kwargs):
        super().__init__(control, reference, empty_router, **kwargs)
        self._queued, self._queued_time = [], None

    def queue_actions(self, moves, now):
        now = float(now)
        moves = list(moves)
        if moves and (now % 900 or not 0 <= now < self.day_end_s):
            raise ValueError("selected idle moves retain the frozen 900-second decision times")
        if self._queued:
            raise ValueError("previous selected idle moves have not been consumed")
        if len(moves) > self.max_moves:
            raise ValueError("selected idle movement cap exceeded; no silent truncation")
        vids = [int(move.get("native_vehicle_id", move["runtime"].fixture.native_id
            if "runtime" in move else -1)) for move in moves]
        if len(set(vids)) != len(vids) or -1 in vids:
            raise ValueError("selected idle moves require unique existing native vehicles")
        self._queued, self._queued_time = moves, now
        self.counts["MODEL_SELECTED_QUEUED"] += len(moves)

    def after_normal_dispatch(self, control, simulation_time):
        if control is not self.control:
            raise ValueError("selected movement manager attached to a different controller")
        if self._queued_time is not None and self._queued_time != float(simulation_time):
            if self._queued:
                raise ValueError("selected idle movement queue belongs to a different native epoch")
            self._queued_time = None
        if simulation_time % 900 or not 0 <= simulation_time < self.day_end_s:
            return
        started_at = perf_counter()
        moves, self._queued = self._queued, []
        self._queued_time = None
        started = 0
        for move in moves:
            runtime = move.get("runtime")
            vid = int(move.get("native_vehicle_id", runtime.fixture.native_id if runtime is not None else -1))
            if runtime is None:
                runtime = control.runtime_by_vid[vid]
            if vid in self.active or not control._available(runtime, simulation_time):
                self.counts["SELECTED_MOVE_NOT_IDLE_AFTER_DISPATCH"] += 1
                continue
            result = move.get("routed", move.get("route"))
            if result is None:
                raise ValueError("selected idle move requires its already-validated scalar routed record")
            if "routed" in result:
                result = result["routed"]
            if not result.get("supported", result.get("common_supported", False)):
                raise ValueError("selected idle move has no supported route record")
            kind = "HV" if runtime.fixture.vehicle_type == "HV" else control.config["profile_id"]
            if kind == "C" and result.get("connection_evidence_state") != "C_DIRECTED_SEAM_CERTIFIED":
                raise ValueError("selected C idle move lacks departure seam certification")
            origin = _point(self.network.return_position_coordinates(runtime.native_vehicle.pos))
            target = _point(move["position"])
            if ("origin_wgs84" not in result or "target_wgs84" not in result
                    or _gap_m(origin, result["origin_wgs84"]) > 1e-5
                    or _gap_m(target, result["target_wgs84"]) > 1e-5
                    or abs(float(result.get("departure_s", -1)) - float(simulation_time)) > 1e-8):
                raise ValueError("selected move route does not bind the present native origin/target/time")
            duration, distance = float(result["duration_s"]), float(result["distance_m"])
            if (_gap_m(origin, target) > self.radius + 1e-6 or not all(isfinite(v) and v > 0
                    for v in (duration, distance)) or duration > self.max_eta + 1e-9):
                raise ValueError("selected move violates the original radius/duration limits")
            if simulation_time + duration > control.fixture_windows_s[vid][1] + 1e-9:
                raise ValueError("optional selected move crosses the original shift end")
            destination = self.network.registry.position_for(*target)
            native_origin = runtime.native_vehicle.pos
            self.network.register_vehicle_leg((0, vid), native_origin, destination, duration, distance)
            self.geometry.register(vid, native_origin, destination, result["points"])
            row = dict(native_vehicle_id=vid, vehicle_id=runtime.fixture.vehicle_id,
                vehicle_type=runtime.fixture.vehicle_type, start_time_s=int(simulation_time),
                target_node_id=int(move.get("node_id", destination[0])), target_site_id=str(move.get("site_id", "")),
                destination_position=destination, planned_duration_s=duration, planned_distance_m=distance,
                status="IN_PROGRESS", actual_duration_s=None, actual_distance_m=None,
                traffic_unknown_share=float(result["traffic_unknown_share"]),
                start_lon_wgs84=origin[0], start_lat_wgs84=origin[1],
                shape_polyline6=result.get("shape_polyline6"), geometry_point_count=len(result["points"]),
                snap_gap_origin_m=float(result.get("snap_gap_origin_m", 0.)),
                snap_gap_target_m=float(result.get("snap_gap_target_m", 0.)),
                decision_source="SCHEME_A_SHARED_MASTER_SELECTED_CURRENT_MOVE",
                connection_evidence_state=result.get("connection_evidence_state"))
            self.active[vid] = len(self.rows)
            self.rows.append(row)
            leg = self.bindings.vehicle_route_leg(self.bindings.states.REPOSITION, destination, {})
            runtime.native_vehicle.assign_vehicle_plan([leg], int(simulation_time))
            runtime.state = "NATIVE_REPOSITIONING"
            started += 1
            self.counts["STARTED"] += 1
        self.epochs.append(dict(simulation_time_s=simulation_time, routing_candidates=0,
            model_selected=len(moves), started=started, automatic_greedy_moves=0))
        self.total_decision_time_s += perf_counter() - started_at

    def summary(self):
        return dict(super().summary(), control_mode="MASTER_SELECTED_ONLY",
            automatic_historical_deficit_rule_active=False,
            selected_queue_pending=len(self._queued))
