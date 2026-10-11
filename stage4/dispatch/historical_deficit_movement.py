"""Historical deficit base layer, independent of speculative service value."""
from __future__ import annotations

from collections import OrderedDict
from time import perf_counter

from .controlled_movement import CommonIdleMovementManager
from .scheme_a_city_graph import RouteState


class HistoricalDeficitMovementManager(CommonIdleMovementManager):
    def __init__(self, *args, wall_limit_s=10., **kwargs):
        super().__init__(*args, **kwargs)
        self.connectors = None
        self.wall_limit_s = float(wall_limit_s)
        if not 0 < self.wall_limit_s <= 10.:
            raise ValueError("optional common movement retains a bounded ten-second budget")
        self._deadline = None
        self._budget_reported = False
        self._issued_routes = OrderedDict()
        self._latest_move = {}

    def _remember_issued_route(self, runtime, row, result):
        vid = int(runtime.fixture.native_id)
        if runtime.fixture.vehicle_type != "AV":
            return
        self._latest_move[vid] = len(self.rows) - 1
        # One compact, issued path per recent C vehicle; no Valhalla JSON or
        # route copies per epoch. Eviction falls back to the ORIGINAL snapshot.
        edges = [dict(stage3_edge_uid=edge["stage3_edge_uid"],
            begin_shape_index=int(edge["begin_shape_index"])) for edge in result.get("edges", ())]
        if edges:
            edges[0]["trace_shape_polyline6"] = result["edges"][0].get("trace_shape_polyline6")
        route = dict(common_supported=True, shape_polyline6=result.get("shape_polyline6"),
            points=tuple(tuple(point) for point in result["points"]), edges=edges)
        self._issued_routes[vid] = (int(row["start_time_s"]), route)
        self._issued_routes.move_to_end(vid)
        while len(self._issued_routes) > 256:
            self._issued_routes.popitem(last=False)

    def issued_direction_route(self, vid, row):
        saved = self._issued_routes.get(int(vid))
        if (saved is not None and saved[0] == int(row["start_time_s"])
                and saved[1]["shape_polyline6"] == row.get("shape_polyline6")):
            self._issued_routes.move_to_end(int(vid))
            return saved[1]
        return None

    def has_movement_since(self, vid, previous_finished_s):
        index = self._latest_move.get(int(vid))
        if index is None:
            return False
        row = self.rows[index]
        if row["start_time_s"] < previous_finished_s:
            return False
        if row["status"] == "IN_PROGRESS":
            pos = self.control.runtime_by_vid[int(vid)].native_vehicle.pos
            return len(pos) > 2 and pos[2] is not None and float(pos[2]) > 0
        return float(row["actual_distance_m"] or 0.) > 0

    def _forget_direction_evidence(self, vid):
        self._issued_routes.pop(int(vid), None)
        self._latest_move.pop(int(vid), None)

    def queue_actions(self, moves, now):
        if list(moves):
            raise ValueError("common deficit movements must not also execute value-selected MOVEs")

    def after_normal_dispatch(self, control, simulation_time):
        if control is not self.control:
            raise ValueError("common movement attached to a different controller")
        if simulation_time % 900 or not 0 <= simulation_time < self.day_end_s:
            return
        self._deadline = perf_counter() + self.wall_limit_s
        self._budget_reported = False
        try:
            super().after_normal_dispatch(control, simulation_time)
        finally:
            self._deadline = None

    def _movement_budget_exhausted(self):
        expired = self._deadline is not None and perf_counter() >= self._deadline
        if expired and not self._budget_reported:
            self.counts["OPTIONAL_COMMON_MOVE_WALL_CUTOFF"] += 1
            self._budget_reported = True
        return expired

    def _route_candidate(self, runtime, destination, simulation_time, lon, lat):
        if self.connectors is None:
            raise RuntimeError("common movement needs the existing directed current-route validator")
        vid = int(runtime.fixture.native_id)
        profile = "HV" if runtime.fixture.vehicle_type == "HV" else self.control.config["profile_id"]
        state = RouteState((lon, lat), float(simulation_time),
            float(self.control.fixture_windows_s[vid][1]), profile,
            dict(kind="NATIVE_VEHICLE", native_vehicle_id=vid, timestamp_s=float(simulation_time)))
        target = dict(site_id=f"HISTORICAL_NODE:{int(destination.node_id)}",
            node_id=int(destination.node_id),
            position=(float(destination.lon_wgs84), float(destination.lat_wgs84)), require_geometry=True)
        result = self.connectors(state, target, simulation_time, profile)
        if not result["supported"]:
            for reason in set(result.get("reason_codes", ())):
                self.counts[f"COMMON_MOVE_REJECTION:{reason}"] += 1
            return dict(common_supported=False)
        routed = result.get("routed")
        if routed is None:
            raise ValueError("common movement has no executable route geometry")
        if profile == "C" and result.get("connection_evidence_state") != "C_DIRECTED_SEAM_CERTIFIED":
            raise ValueError("common C movement must retain its departure direction/seam check")
        return dict(routed, common_supported=True, compatible_profiles=(profile,))

    def summary(self):
        return dict(super().summary(), control_mode="COMMON_HISTORICAL_DEMAND_DEFICIT",
            automatic_historical_deficit_rule_active=True,
            independent_of_future_service_value=True, value_selected_moves_enabled=False,
            target_scope="ALL_HISTORICAL_NODES_IN_CURRENT_15_MINUTE_BIN",
            max_optional_movement_wall_s=self.wall_limit_s,
            wall_cutoff_may_reduce_optional_moves=True,
            C_departure_seam_certificate_required=True)
