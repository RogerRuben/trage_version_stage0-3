"""Sparse, executable C pickup checks before the city short-window solver.

Route geometry and raw auto time use the declared window-start snapshot. The
frozen pickup multiplier and passenger's original deadline use the actual
decision time. Only already-built current arcs are examined; no vehicle/order
Cartesian product, new traffic forecast, or historical-route imputation exists.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import replace
import json
from math import isfinite
from pathlib import Path
from time import perf_counter

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from stage3.odd_tod.intersection_complex import _bearing
from stage3.odd_tod.finalization import decode_polyline6
from stage4.analysis.capability_chain_instances import IDENTITY_COLUMNS, JoinEvidence
from stage4.dispatch.controlled_routes import MAX_CACHE_ENTRIES, S2A
from stage4.dispatch.controlled_movement import segment_lengths
from stage4.fleetpy_adapter.valhalla_time_adapter import PickupEstimate, TIMEZONE


TEST31_IDENTITY = Path("stage3/output/odd_tod/s4/test31_route_identity_resolution.parquet")
OVERLAY_REL = S2A / "stage3_historical_direction_overlay.parquet"
TOPOLOGY_COLUMNS = ["stage3_edge_uid", "geometry", "from_stage3_node_uid", "to_stage3_node_uid"]


def _matched_batches(path, columns, key, values, *, batch_size=8192):
    """Filter Arrow batches before any conversion to Python customer rows."""
    wanted = pa.array(sorted(values), type=pa.string())
    source = pq.ParquetFile(path)
    for batch in source.iter_batches(batch_size=batch_size, columns=columns, use_threads=False):
        table = pa.Table.from_batches([batch])
        small = table.filter(pc.is_in(table[key], value_set=wanted))
        if len(small):
            yield small


class _SmallJoinRouter:
    """The JoinEvidence router contract without a second whole geometry table."""

    def __init__(self, router):
        self.base = router
        self.geometry_cache = OrderedDict()
        self._features = None
        self._feature_ids = set()
        self.projection_scan_count = 0
        self.projection_time_s = 0.

    def __getattr__(self, name):
        return getattr(self.base, name)

    def prepare_geometry(self, uids):
        """One batch projection for all required edges, held as Arrow buffers."""
        uids = {str(uid) for uid in uids if uid is not None}
        needed = uids - self._feature_ids
        if needed:
            projection_started = perf_counter()
            self.projection_scan_count += 1
            path = self.base.root / S2A / "stage3_full_network_edges.parquet"
            pieces = [self._features] if self._features is not None else []
            found_ids = set()
            for small in _matched_batches(path, TOPOLOGY_COLUMNS, "stage3_edge_uid", needed, batch_size=4096):
                pieces.append(small)
                found_ids.update(small["stage3_edge_uid"].to_pylist())
                if needed.issubset(found_ids):
                    break
            if pieces:
                self._features = pa.concat_tables(pieces)
                self._feature_ids.update(found_ids)
            self.projection_time_s += perf_counter() - projection_started

    def small_geometry(self, uids):
        uids = {str(uid) for uid in uids if uid is not None}
        found = {uid: self.geometry_cache[uid] for uid in uids if uid in self.geometry_cache}
        needed = uids - found.keys()
        self.prepare_geometry(needed)
        if needed and self._features is not None:
            small = self._features.filter(pc.is_in(self._features["stage3_edge_uid"], value_set=pa.array(sorted(needed))))
            for row in small.to_pylist():
                found[row["stage3_edge_uid"]] = dict(uid=row["stage3_edge_uid"],
                    geometry=json.loads(row["geometry"]), from_node=row["from_stage3_node_uid"],
                    to_node=row["to_stage3_node_uid"])
        for uid, record in found.items():
            self.geometry_cache[uid] = record
            self.geometry_cache.move_to_end(uid)
        while len(self.geometry_cache) > MAX_CACHE_ENTRIES:
            self.geometry_cache.popitem(last=False)
        return found

    def _bearings_for(self, uids):
        result = {}
        for uid, row in self.small_geometry(uids).items():
            geometry = row["geometry"]
            result[uid] = ((_bearing(geometry[0], geometry[1]), _bearing(geometry[-2], geometry[-1]))
                if len(geometry) >= 2 else (None, None))
        return result


class Test31PickupJoinEvidence(JoinEvidence):
    """Lazy revealed Test31 tokens; reuse the existing cross-part action test.

The 80 m interface snap rule is the existing capability-chain instance rule,
not a new capability threshold. Callers may pass that frozen config value.
No Actor or data table is constructed until a candidate needs its evidence.
"""

    def __init__(self, empty_router, *, interface_snap_tolerance_m=80.0):
        self.router = _SmallJoinRouter(empty_router)
        self.root = Path(empty_router.root)
        self.tolerance = float(interface_snap_tolerance_m)
        if not isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("invalid frozen interface snap tolerance")
        self.tokens = OrderedDict()
        self.overlay = pd.DataFrame(columns=["canonical_edge_uid", "physical_forward_stage3_edge_uid"])
        self.overlay_forward = {}
        self.selected = pd.DataFrame()
        self.simulation_date = "20161031"
        self._sources = {}
        self._source_reasons = {}
        self.identity_scan_count = 0
        self.identity_loaded_order_count = 0
        self._private_identity = None
        self._private_ranges = {}
        self._private_overlay = None
        self._store_summary = None
        self.identity_read_time_s = 0.
        self.departure_context_routing_time_s = 0.
        self.departure_context_route_count = 0
        self._idle_routes = OrderedDict()

    def prepare_private_identity_store(self, scope_window_order_ids):
        """Environment-only feature store; the policy is never given scope IDs.

        Precomputing frozen geometry/identity is allowed; runtime lookup below
        still requires a revealed request or an already-completed customer.
        No future request times, actual durations, or demand outcomes are stored.
        """
        wanted = {str(order) for order in scope_window_order_ids}
        setup_started = perf_counter()
        self._private_ranges, self._private_overlay = {}, None
        self.identity_scan_count += 1
        read_started = perf_counter()
        parts = list(_matched_batches(self.root / TEST31_IDENTITY, IDENTITY_COLUMNS, "order_id", wanted))
        self.identity_read_time_s += perf_counter() - read_started
        if parts:
            self._private_identity = pa.concat_tables(parts).sort_by([("order_id", "ascending"), ("route_sequence", "ascending")])
            orders = self._private_identity["order_id"].to_pylist()
            start = 0
            for end in range(1, len(orders) + 1):
                if end == len(orders) or orders[end] != orders[start]:
                    self._private_ranges[str(orders[start])] = (start, end)
                    start = end
        else:
            schema = pq.ParquetFile(self.root / TEST31_IDENTITY).schema_arrow
            self._private_identity = pa.Table.from_batches([], schema=pa.schema([schema.field(c) for c in IDENTITY_COLUMNS]))
        reverse_table = self._private_identity.filter(pc.equal(self._private_identity["route_token_type"], "HISTORICAL_REVERSE_OVERLAY"))
        reverse = set(reverse_table["canonical_edge_uid"].drop_null().to_pylist())
        if reverse:
            overlays = list(_matched_batches(self.root / OVERLAY_REL,
                ["canonical_edge_uid", "physical_forward_stage3_edge_uid"], "canonical_edge_uid", reverse))
            self._private_overlay = pa.concat_tables(overlays) if overlays else None
        uids = set(self._private_identity["resolved_stage3_edge_uid"].drop_null().to_pylist())
        if self._private_overlay is not None:
            uids.update(self._private_overlay["physical_forward_stage3_edge_uid"].drop_null().to_pylist())
        self.router.prepare_geometry(uids)
        self._store_summary = dict(store_role="ENVIRONMENT_PRIVATE_FROZEN_IDENTITY_GEOMETRY_NOT_FUTURE_OUTCOMES",
            stored_order_count=len(self._private_ranges), stored_token_count=len(self._private_identity),
            missing_scope_order_count=len(wanted - self._private_ranges.keys()),
            identity_arrow_mib=self._private_identity.nbytes / 2**20,
            geometry_arrow_mib=self.router._features.nbytes / 2**20 if self.router._features is not None else 0.,
            setup_time_s=perf_counter() - setup_started, identity_read_filter_time_s=self.identity_read_time_s,
            geometry_projection_time_s=self.router.projection_time_s,
            policy_has_scope_order_ids=False, runtime_lookup_requires_revealed_or_completed=True)
        return dict(self._store_summary)

    def _idle_context(self, c, runtime, now, previous_finished_s):
        """Recover only an actually issued idle route, then keep its driven prefix."""
        manager = getattr(c, "repositioning_manager", None)
        if manager is None:
            return None
        vid = int(runtime.fixture.native_id)
        index = getattr(manager, "active", {}).get(vid)
        if index is not None:
            row = manager.rows[index]
            position = runtime.native_vehicle.pos
            fraction = float(position[2]) if len(position) >= 3 and position[2] is not None else 0.
        else:
            row = next((item for item in reversed(manager.rows)
                if int(item["native_vehicle_id"]) == vid and item.get("status") == "COMPLETED"
                and float(item.get("end_time_s", float("inf"))) <= float(now)
                and float(item.get("end_time_s", -1.)) >= previous_finished_s
                and item["destination_position"] == runtime.native_vehicle.pos), None)
            fraction = 1.
        if row is None or fraction <= 0 or not row.get("shape_polyline6"):
            return None
        target = tuple(c.routing_engine.return_position_coordinates(row["destination_position"]))
        origin = (float(row["start_lon_wgs84"]), float(row["start_lat_wgs84"]))
        key = (vid, int(row["start_time_s"]), row["shape_polyline6"])
        routed = self._idle_routes.get(key)
        if routed is None:
            route_started = perf_counter()
            routed = self.router.base.route(*origin, *target, c._timestamp(row["start_time_s"]))
            self.departure_context_routing_time_s += perf_counter() - route_started
            self.departure_context_route_count += 1
            self._idle_routes[key] = routed
            while len(self._idle_routes) > MAX_CACHE_ENTRIES:
                self._idle_routes.popitem(last=False)
        else:
            self._idle_routes.move_to_end(key)
        if not routed["common_supported"] or routed.get("shape_polyline6") != row["shape_polyline6"]:
            return None  # Never replace the route that was actually issued.
        points = routed["points"]
        trace_shape = routed["edges"][0].get("trace_shape_polyline6") if routed["edges"] else None
        trace_points = decode_polyline6(trace_shape) if trace_shape else points
        if trace_points != points:
            return None  # Shape indices cannot certify a different geometry's prefix.
        from stage4.dispatch.controlled_routes import _gap_m
        leading, trailing = _gap_m(origin, points[0]), _gap_m(points[-1], target)
        cumulative = [0.]
        for length in segment_lengths(points):
            cumulative.append(cumulative[-1] + float(length))
        travelled = min(1., max(0., fraction)) * (leading + cumulative[-1] + trailing)
        prefix = [edge for edge in routed["edges"]
            if leading + cumulative[int(edge["begin_shape_index"])] < travelled]
        if not prefix:
            return None
        current = tuple(c.routing_engine.return_position_coordinates(runtime.native_vehicle.pos))
        return dict(edges=prefix, point=current, evidence_kind="ACTUAL_ROUTED_ARRIVAL_OR_PREFIX",
            source_move_start_s=int(row["start_time_s"]), source_progress_fraction=fraction,
            original_shape_verified=True)

    def prepare(self, c, arcs, now):
        """Observe only current candidates and customers already completed."""
        requests, by_vid = {}, {}
        for arc in arcs:
            if arc.vehicle_type == "AV":
                runtime, request, _ = arc.payload
                if float(request.sim_time_s) > float(now):
                    raise ValueError("city pickup evidence queried an unrevealed request")
                requests[str(request.order_id)] = request
                by_vid[int(arc.vehicle_id)] = runtime
        completed, completed_times = {}, {}
        completed_rids = getattr(c, "completed_rids", set())
        for row in reversed(getattr(c, "assignment_rows", ())):
            vid, rid = int(row["native_vehicle_id"]), int(row["native_request_id"])
            if vid in by_vid and vid not in completed and (row.get("completed", False) or rid in completed_rids):
                completed[vid] = c.request_by_rid.get(rid)
                finish = row.get("service_end_time")
                completed_times[vid] = (float((pd.Timestamp(finish) - c._timestamp(0)).total_seconds())
                    if finish is not None and not pd.isna(finish) else -1.)
        self._sources, self._source_reasons = {}, {}
        contexts = getattr(c, "city_departure_contexts", None)
        if contexts is None:
            contexts = {}
            c.city_departure_contexts = contexts
        context_tokens, context_rows = {}, {}
        for vid, runtime in by_vid.items():
            current = tuple(c.routing_engine.return_position_coordinates(runtime.native_vehicle.pos))
            context = self._idle_context(c, runtime, now, completed_times.get(vid, -1.))
            if context is not None:
                contexts[vid] = context
            else:
                context = contexts.get(vid)
                if context is not None and (tuple(context.get("point", ())) != current
                        or ("source_move_start_s" in context
                            and float(context["source_move_start_s"]) < completed_times.get(vid, -1.))):
                    # A prior prefix is not the present incoming direction after
                    # a new customer or subsequent motion. Try the actual last
                    # completed customer at its current stop below.
                    del contexts[vid]
                    context = None
            if context is not None:
                if (context.get("evidence_kind") != "ACTUAL_ROUTED_ARRIVAL_OR_PREFIX"
                        or tuple(context.get("point", ())) != current or not context.get("edges")):
                    self._source_reasons[vid] = "PREVIOUS_DIRECTION_CONTEXT_UNAVAILABLE"
                    continue
                key = f"CITY_DEPARTURE:{vid}"
                context_tokens[key] = pd.DataFrame([dict(date=self.simulation_date, order_id=key,
                    route_sequence=i, canonical_edge_uid=None, route_token_type="FULL_NETWORK_EDGE",
                    resolved_stage3_edge_uid=edge["stage3_edge_uid"])
                    for i, edge in enumerate(context["edges"])], columns=IDENTITY_COLUMNS)
                context_rows[key] = dict(start_lon_wgs84=current[0], start_lat_wgs84=current[1],
                    end_lon_wgs84=current[0], end_lat_wgs84=current[1])
                self._sources[vid] = key
                continue
            previous = completed.get(vid)
            manager = getattr(c, "repositioning_manager", None)
            if previous is None:
                if manager is not None and vid in getattr(manager, "active", {}):
                    self._source_reasons[vid] = "PREVIOUS_DIRECTION_CONTEXT_UNAVAILABLE"
                else:
                    self._sources[vid] = None  # Cold start supplies no fabricated incoming edge.
            elif runtime.native_vehicle.pos == previous.dropoff_position:
                order = str(previous.order_id)
                self._sources[vid] = order
                requests[order] = previous
            else:
                self._source_reasons[vid] = "PREVIOUS_DIRECTION_CONTEXT_UNAVAILABLE"
        # Only this round's known orders become Python DataFrames. The source
        # contains 2.1M tokens, but batches outside these order IDs are discarded.
        needed = set(requests) - self.tokens.keys()
        if needed:
            if self._private_identity is not None:
                for order in needed:
                    bounds = self._private_ranges.get(order)
                    if bounds is not None:
                        first, end = bounds
                        self.tokens[order] = self._private_identity.slice(first, end - first).to_pandas()
                        self.identity_loaded_order_count += 1
            else:
                pieces = {}
                self.identity_scan_count += 1
                read_started = perf_counter()
                for small in _matched_batches(self.root / TEST31_IDENTITY, IDENTITY_COLUMNS, "order_id", needed):
                    for order, group in small.to_pandas().groupby("order_id", sort=False):
                        pieces.setdefault(str(order), []).append(group)
                for order, frames in pieces.items():
                    self.tokens[order] = pd.concat(frames, ignore_index=True).sort_values("route_sequence", kind="stable")
                    self.identity_loaded_order_count += 1
                self.identity_read_time_s += perf_counter() - read_started
        for order in requests:
            if order in self.tokens:
                self.tokens.move_to_end(order)
        for key in list(self.tokens):
            if key.startswith("CITY_DEPARTURE:"):
                del self.tokens[key]
        self.tokens.update(context_tokens)
        while len(self.tokens) > MAX_CACHE_ENTRIES:
            self.tokens.popitem(last=False)
        rows = {order: dict(start_lon_wgs84=request.pickup_lon_wgs84,
            start_lat_wgs84=request.pickup_lat_wgs84, end_lon_wgs84=request.dropoff_lon_wgs84,
            end_lat_wgs84=request.dropoff_lat_wgs84) for order, request in requests.items()}
        rows.update(context_rows)
        self.selected = pd.DataFrame.from_dict(rows, orient="index")
        reverse = {str(row.canonical_edge_uid) for order in rows if order in self.tokens
            for row in self.tokens[order].loc[lambda f: f.route_token_type.eq("HISTORICAL_REVERSE_OVERLAY")].itertuples(index=False)}
        if reverse:
            if self._private_identity is not None:
                parts = ([self._private_overlay.filter(pc.is_in(self._private_overlay["canonical_edge_uid"],
                    value_set=pa.array(sorted(reverse)))).to_pandas()] if self._private_overlay is not None else [])
            else:
                parts = [small.to_pandas() for small in _matched_batches(self.root / OVERLAY_REL,
                    ["canonical_edge_uid", "physical_forward_stage3_edge_uid"], "canonical_edge_uid", reverse)]
            self.overlay = pd.concat(parts, ignore_index=True) if parts else self.overlay.iloc[:0]
        else:
            self.overlay = self.overlay.iloc[:0]
        self.overlay_forward = dict(zip(self.overlay.canonical_edge_uid, self.overlay.physical_forward_stage3_edge_uid))

    def prepare_routes(self, routed_results):
        uids = {edge["stage3_edge_uid"] for routed in routed_results for edge in routed["edges"]}
        for order in self.selected.index:
            if order in self.tokens:
                uids.update(self.tokens[order].resolved_stage3_edge_uid.dropna())
        uids.update(value for value in self.overlay_forward.values() if pd.notna(value))
        self.router.prepare_geometry(uids)

    def diagnostics(self):
        return dict(private_feature_store=dict(self._store_summary or {"prepared": False}),
            identity_scan_count=self.identity_scan_count, identity_loaded_order_count=self.identity_loaded_order_count,
            identity_read_filter_time_s=self.identity_read_time_s,
            geometry_projection_scan_count=self.router.projection_scan_count,
            geometry_projection_time_s=self.router.projection_time_s,
            geometry_feature_rows=len(self.router._feature_ids), token_cache_entries=len(self.tokens),
            geometry_object_cache_entries=len(self.router.geometry_cache),
            idle_context_route_count=self.departure_context_route_count,
            idle_context_routing_time_s=self.departure_context_routing_time_s)

    def evaluate_pickup(self, c, runtime, request, routed, now):
        del c, now
        vid = int(runtime.fixture.native_id)
        if vid in self._source_reasons:
            return dict(supported=False, reason_codes=[self._source_reasons[vid]], compatible_profiles=[])
        source, target = self._sources.get(vid), str(request.order_id)
        if target not in self.tokens or (source is not None and source not in self.tokens):
            return dict(supported=False, reason_codes=["CURRENT_CUSTOMER_ROUTE_IDENTITY_MISSING"], compatible_profiles=[])
        result = super().evaluate(source, target, routed, self.tolerance)
        result.update(pickup_to_customer_checked=True, previous_to_pickup_checked=source is not None,
            departure_context="COLD_START_NO_PRIOR_CUSTOMER" if source is None else "ACTUAL_ROUTED_CONTEXT_OR_COMPLETED_CUSTOMER")
        return result


class CityPickupSupportValidator:
    """Callable current-arc gate with a routing budget separate from the solver."""

    def __init__(self, empty_router, *, geometry_timestamp, routing_budget_s,
            seam_evidence=None, interface_snap_tolerance_m=80.0):
        self.empty_router = empty_router
        stamp = pd.Timestamp(geometry_timestamp)
        self.geometry_timestamp = stamp.tz_localize(TIMEZONE) if stamp.tzinfo is None else stamp.tz_convert(TIMEZONE)
        self.routing_budget_s = float(routing_budget_s)
        if pd.isna(stamp) or not isfinite(self.routing_budget_s) or self.routing_budget_s < 0:
            raise ValueError("pickup snapshot time and routing budget must be explicit and finite")
        self.seam_evidence = seam_evidence if seam_evidence is not None else Test31PickupJoinEvidence(
            empty_router, interface_snap_tolerance_m=interface_snap_tolerance_m)
        self.cache = OrderedDict()
        self.pickup_paths = {}
        self.last_diagnostics = {}
        self._counts = Counter()
        self._reason_counts = Counter()
        self._timings = Counter()

    def prepare_private_identity_store(self, scope_window_order_ids):
        return self.seam_evidence.prepare_private_identity_store(scope_window_order_ids)

    def diagnostics(self):
        return dict(counts=dict(self._counts), rejection_reason_counts=dict(self._reason_counts),
            cumulative_timings_s=dict(self._timings),
            route_cache_entries=len(self.cache), route_cache_limit=MAX_CACHE_ENTRIES,
            geometry_timestamp=self.geometry_timestamp.isoformat(), eta_beta_time="ACTUAL_DECISION_TIMESTAMP",
            traffic_policy="EXPLICIT_U_FOR_NEW_ROUTE_WITHOUT_FROZEN_M3_FEATURE_ROWS",
            traffic_unknown_share=1.0, routing_budget_s=self.routing_budget_s,
            routing_budget_is_separate_from_solver=True, last_epoch=dict(self.last_diagnostics),
            seam_evidence=self.seam_evidence.diagnostics() if hasattr(self.seam_evidence, "diagnostics") else None)

    def __call__(self, c, arcs, now):
        started = perf_counter()
        counts, reasons = Counter(), Counter()
        valid, indices = [], []
        route_time = seam_time = 0.0
        self.pickup_paths = {}
        c.city_pickup_paths = {}
        idle_routing_before = float(getattr(self.seam_evidence, "departure_context_routing_time_s", 0.))
        actual_stamp = c._timestamp(float(now))
        if c.eta_adapter is not self.empty_router.eta_adapter:
            raise ValueError("city pickup checks must share the existing ETA adapter and Actor")
        bin_index, beta = c.eta_adapter.beta_for(actual_stamp)
        beta = float(beta)
        if not isfinite(beta) or beta <= 0:
            raise ValueError("invalid frozen decision-time pickup multiplier")
        staged = []
        for index, arc in enumerate(arcs):
            counts["input_arcs"] += 1
            runtime, request, estimate = arc.payload
            if float(request.sim_time_s) > float(now):
                raise ValueError("city pickup evidence queried an unrevealed request")
            meta = c.request_meta[int(arc.request_id)]
            remaining = float(meta["pickup_deadline_s"]) - float(now)
            if not isfinite(remaining) or remaining < 0:
                reasons["PICKUP_DEADLINE_PASSED"] += 1
                continue
            if arc.vehicle_type != "AV":
                if (not isfinite(float(arc.pickup_eta_s)) or arc.pickup_eta_s < 0
                        or arc.pickup_eta_s > remaining):
                    reasons["EXISTING_HV_PICKUP_ETA_INVALID_OR_LATE"] += 1
                    continue
                valid.append(arc)
                indices.append(index)
                counts["hv_existing_auto_arcs_retained"] += 1
                continue
            counts["av_pickup_arcs_examined"] += 1
            if not meta.get("research_route_compatible", False):
                reasons["CUSTOMER_BODY_OUTSIDE_C_PROFILE"] += 1
                continue
            if self.routing_budget_s <= 0:
                reasons["PICKUP_SUPPORT_ROUTING_BUDGET_EXHAUSTED"] += 1
                continue
            origin = tuple(map(float, c.routing_engine.return_position_coordinates(runtime.native_vehicle.pos)))
            target = (float(request.pickup_lon_wgs84), float(request.pickup_lat_wgs84))
            key = (*origin, *target)
            cached = key in self.cache
            if cached:
                routed = self.cache[key]
                self.cache.move_to_end(key)
                counts["pickup_route_cache_hits"] += 1
            else:
                if route_time >= self.routing_budget_s:
                    reasons["PICKUP_SUPPORT_ROUTING_BUDGET_EXHAUSTED"] += 1
                    continue
                route_started = perf_counter()
                routed = self.empty_router.route(*origin, *target, self.geometry_timestamp)
                route_time += perf_counter() - route_started
                counts["pickup_scalar_route_evaluations"] += 1
                self.cache[key] = routed
                while len(self.cache) > MAX_CACHE_ENTRIES:
                    self.cache.popitem(last=False)
            if not routed["common_supported"]:
                reasons.update(routed["reason_codes"] or ("PICKUP_COMMON_SUPPORT_UNAVAILABLE",))
                continue
            if "C" not in routed["compatible_profiles"]:
                reasons.update(routed.get("profile_reason_codes", {}).get("C", ()) or ("EMPTY_PICKUP_OUTSIDE_C_PROFILE",))
                continue
            tolerance = float(getattr(self.seam_evidence, "tolerance", 80.0))
            if any(not isfinite(float(routed[name])) or float(routed[name]) > tolerance
                    for name in ("snap_gap_origin_m", "snap_gap_target_m")):
                reasons["SNAP_GAP_EXCEEDS_COMMON_INSTANCE_TOLERANCE"] += 1
                continue
            raw, distance = float(routed["raw_time_s"]), float(routed["distance_m"])
            duration = raw * beta
            coincident = bool(routed.get("coincident_od", False))
            if (not all(isfinite(value) and value >= 0 for value in (raw, distance, duration))
                    or (not coincident and min(raw, distance, duration) <= 0)):
                reasons["PICKUP_SCALAR_TIME_OR_DISTANCE_INVALID"] += 1
                continue
            if duration > remaining:
                reasons["SCALAR_PICKUP_EXCEEDS_REMAINING_PATIENCE"] += 1
                continue
            staged.append((index, arc, runtime, request, routed, raw, distance, duration, cached))
        if staged:
            evidence_started = perf_counter()
            self.seam_evidence.prepare(c, [item[1] for item in staged], now)
            prepare_routes = getattr(self.seam_evidence, "prepare_routes", None)
            if prepare_routes is not None:
                prepare_routes([item[4] for item in staged])
            seam_time += perf_counter() - evidence_started
        for index, arc, runtime, request, routed, raw, distance, duration, cached in staged:
            seam_started = perf_counter()
            joined = self.seam_evidence.evaluate_pickup(c, runtime, request, routed, now)
            seam_time += perf_counter() - seam_started
            if not joined["supported"]:
                reasons.update(joined.get("reason_codes") or ("PICKUP_CUSTOMER_JOIN_UNSUPPORTED",))
                continue
            if "C" not in joined.get("compatible_profiles", ()):
                reasons.update(joined.get("C_reason_codes") or ("PICKUP_CUSTOMER_JOIN_OUTSIDE_C_PROFILE",))
                continue
            corrected = PickupEstimate(valhalla_time_s=raw, corrected_pickup_eta_s=duration,
                route_distance_m=distance, beta=beta, time_bin_index=int(bin_index), cache_hit=cached)
            updated = replace(arc, pickup_eta_s=duration, payload=(runtime, request, corrected))
            valid.append(updated)
            indices.append(index)
            counts["av_pickup_arcs_retained"] += 1
            path = dict(routed)
            path.update(duration_s=duration, beta=beta, time_bin_index=int(bin_index), seam=joined,
                geometry_timestamp=self.geometry_timestamp.isoformat(), decision_timestamp=pd.Timestamp(actual_stamp).isoformat())
            self.pickup_paths[(int(arc.vehicle_id), int(arc.request_id))] = path
            c.city_pickup_paths[(int(arc.vehicle_id), int(arc.request_id))] = path["points"]
        ordered = sorted(zip(indices, valid), key=lambda item: item[0])
        indices, valid = [item[0] for item in ordered], [item[1] for item in ordered]
        elapsed = perf_counter() - started
        idle_routing = float(getattr(self.seam_evidence, "departure_context_routing_time_s", 0.)) - idle_routing_before
        self.last_diagnostics = dict(input_arc_count=len(arcs), validated_arc_count=len(valid),
            original_indices=indices, counts=dict(counts), rejection_reason_counts=dict(reasons),
            pickup_support_time_s=elapsed, pickup_routing_time_s=route_time,
            idle_prefix_routing_time_s=idle_routing,
            pickup_seam_evidence_time_s=seam_time, routing_budget_s=self.routing_budget_s,
            routing_budget_is_separate_from_solver=True, geometry_timestamp=self.geometry_timestamp.isoformat(),
            decision_timestamp=pd.Timestamp(actual_stamp).isoformat(), eta_beta=beta,
            route_cache_entries=len(self.cache), retained_pickup_path_count=len(self.pickup_paths))
        self._counts.update(counts)
        self._reason_counts.update(reasons)
        self._timings.update(total_pickup_support=elapsed, pickup_scalar_route=route_time,
            idle_prefix_routing=idle_routing, seam_nonrouting_evidence=max(0., seam_time - idle_routing))
        return valid
