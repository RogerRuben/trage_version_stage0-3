"""Historical typed targets; common physical limits and shared move capacity."""
from collections import Counter
from math import cos, radians
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from .historical_deficit_movement import HistoricalDeficitMovementManager
from .repositioning_policy import largest_remainder_quotas, time_bin_index


class TypeConditionedHistoricalDeficitMovementManager(HistoricalDeficitMovementManager):
    def __init__(self, *args, typed_reference, typed_manifest=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.typed_targets = {str(profile): {int(index): group.sort_values("node_id", kind="stable").reset_index(drop=True)
            for index, group in frame.groupby("time_bin_index", sort=True)}
            for profile, frame in typed_reference.groupby("profile_id", sort=True)}
        self.typed_manifest = typed_manifest or {}
        self.targets_by_profile = {"HV": self.targets, **self.typed_targets}

    def _typed_surplus(self, now):
        scale = np.array([111320 * cos(radians(34.25)), 110540])
        index = time_bin_index(self.control._timestamp(now))
        occupancy = self.targets.get(index)
        if occupancy is None or occupancy.empty:
            raise ValueError("existing physical occupancy-node support is missing")
        occupancy_tree = cKDTree(occupancy[["lon_wgs84", "lat_wgs84"]].to_numpy(float) * scale)
        by_profile = {}
        for vid in self.control.event_calendar.active_ids(now):
            runtime = self.control.runtime_by_vid[int(vid)]
            profile = "HV" if runtime.fixture.vehicle_type == "HV" else self.control.config["profile_id"]
            by_profile.setdefault(profile, []).append((int(vid), runtime))
        surplus, contexts, idle_total = [], {}, 0
        for profile, vehicles in sorted(by_profile.items()):
            target = self.targets_by_profile.get(profile, {}).get(index)
            if target is None or target.empty:
                self.counts[f"NO_HISTORICAL_TYPED_TARGET:{profile}"] += 1
                idle_total += sum(vid not in self.active and self.control._available(runtime, now)
                                  for vid, runtime in vehicles)
                continue  # Absence of observed demand is not a route/capability ban.
            xy = target[["lon_wgs84", "lat_wgs84"]].to_numpy(float) * scale
            tree = cKDTree(xy)
            idle, inflight = [], Counter()
            for vid, runtime in vehicles:
                if vid in self.active:
                    inflight[self.rows[self.active[vid]]["target_node_id"]] += 1
                elif self.control._available(runtime, now):
                    lon, lat = self.network.return_position_coordinates(runtime.native_vehicle.pos)
                    # Occupancy uses the ORIGINAL full physical node support.
                    # Using the filtered C targets here would count a remote
                    # vehicle as already at C's nearest desired destination.
                    node = int(occupancy.iloc[int(occupancy_tree.query(np.array([lon, lat]) * scale)[1])].node_id)
                    idle.append((runtime, node, float(lon), float(lat)))
            idle_total += len(idle)
            # All inflight resources retain their already-issued routes even when
            # the new bin has different targets. Do not silently discard supply.
            quotas = largest_remainder_quotas(target.set_index("node_id").demand_share, len(idle) + sum(inflight.values()))
            current = Counter(row[1] for row in idle) + inflight
            deficits = {int(node): max(0, int(quota) - current[int(node)]) for node, quota in quotas.items()}
            groups = {}
            for row in idle:
                groups.setdefault(row[1], []).append(row)
            for node, rows in groups.items():
                rows.sort(key=lambda row: row[0].fixture.vehicle_id)
                keep = max(0, int(quotas.get(node, 0)) - inflight[node])
                surplus.extend((profile, *row) for row in rows[keep:])
            contexts[profile] = target, xy, tree, deficits, scale
        surplus.sort(key=lambda row: row[1].fixture.vehicle_id)  # No artificial AV-first priority.
        return surplus, contexts, idle_total

    def after_normal_dispatch(self, control, simulation_time):
        if control is not self.control:
            raise ValueError("typed movement attached to another controller")
        if simulation_time % 900 or not 0 <= simulation_time < self.day_end_s:
            return
        started_at = perf_counter()
        self._deadline = started_at + self.wall_limit_s
        self._budget_reported = False
        started = attempts = 0
        try:
            surplus, contexts, idle_total = self._typed_surplus(simulation_time)
            for profile, runtime, node, lon, lat in surplus:
                if started >= self.max_moves or self._movement_budget_exhausted():
                    break
                target, xy, tree, deficits, scale = contexts[profile]
                origin_xy = np.array([lon, lat]) * scale
                nearby = tree.query_ball_point(origin_xy, self.radius)
                nearby = [i for i in nearby if deficits[int(target.iloc[i].node_id)] > 0 and int(target.iloc[i].node_id) != node]
                nearby.sort(key=lambda i: (float(np.sum((xy[i] - origin_xy) ** 2)), str(int(target.iloc[i].node_id))))
                for i in nearby[:self.top_k]:
                    if self._movement_budget_exhausted():
                        break
                    destination = target.iloc[i]
                    attempts += 1
                    routed_at = perf_counter()
                    result = self._route_candidate(runtime, destination, simulation_time, lon, lat)
                    self.routing_wall_time_s += perf_counter() - routed_at
                    if self._movement_budget_exhausted():
                        break
                    if not result["common_supported"]:
                        self.counts["COMMON_ROUTE_UNSUPPORTED"] += 1
                        continue
                    if profile not in result["compatible_profiles"]:
                        self.counts["PROFILE_REJECTED"] += 1
                        continue
                    duration, distance = float(result["duration_s"]), float(result["distance_m"])
                    if not np.isfinite(duration) or not np.isfinite(distance) or min(duration, distance) <= 0 or duration > self.max_eta:
                        self.counts["ETA_OR_ZERO_MOVE_REJECTED"] += 1
                        continue
                    vid = int(runtime.fixture.native_id)
                    if simulation_time + duration > control.fixture_windows_s[vid][1]:
                        self.counts["OPTIONAL_MOVE_CROSSES_SHIFT"] += 1
                        continue
                    origin = runtime.native_vehicle.pos
                    position = self.network.registry.position_for(float(destination.lon_wgs84), float(destination.lat_wgs84))
                    self.network.register_vehicle_leg((0, vid), origin, position, duration, distance)
                    self.geometry.register(vid, origin, position, result["points"])
                    row = dict(native_vehicle_id=vid, vehicle_id=runtime.fixture.vehicle_id,
                        vehicle_type=runtime.fixture.vehicle_type, start_time_s=int(simulation_time), target_node_id=int(destination.node_id),
                        destination_position=position, planned_duration_s=duration, planned_distance_m=distance, status="IN_PROGRESS",
                        actual_duration_s=None, actual_distance_m=None, traffic_unknown_share=float(result["traffic_unknown_share"]),
                        start_lon_wgs84=lon, start_lat_wgs84=lat, shape_polyline6=result.get("shape_polyline6"),
                        geometry_point_count=len(result["points"]), decision_source="TYPE_CONDITIONED_HISTORICAL_DEFICIT",
                        target_profile_id=profile, connection_evidence_state=result.get("connection_evidence_state"),
                        geometry_timestamp=result.get("geometry_timestamp"),
                        snap_gap_origin_m=float(result.get("snap_gap_origin_m", 0.)), snap_gap_target_m=float(result.get("snap_gap_target_m", 0.)))
                    self.active[vid] = len(self.rows)
                    self.rows.append(row)
                    self._remember_issued_route(runtime, row, result)
                    leg = self.bindings.vehicle_route_leg(self.bindings.states.REPOSITION, position, {})
                    runtime.native_vehicle.assign_vehicle_plan([leg], int(simulation_time))
                    runtime.state = "NATIVE_REPOSITIONING"
                    deficits[int(destination.node_id)] -= 1
                    started += 1
                    self.counts["STARTED"] += 1
                    break
            self.epochs.append(dict(simulation_time_s=simulation_time, idle_after_dispatch=idle_total,
                routing_candidates=attempts, started=started, target_mode="TYPE_CONDITIONED_HISTORICAL_DEFICIT"))
        finally:
            self._deadline = None
            self.total_decision_time_s += perf_counter() - started_at

    def summary(self):
        return dict(super().summary(), control_mode="TYPE_CONDITIONED_HISTORICAL_DEMAND_DEFICIT",
            target_scope="EXISTING_NODE_SUPPORT_WITH_PROFILE_CONDITIONAL_HISTORICAL_SHARES",
            typed_source_dates=self.typed_manifest.get("source_partition_dates"),
            HV_target_reference_unchanged=True, shared_current_move_cap=self.max_moves,
            empty_typed_bin_is_not_capability_failure=True)
