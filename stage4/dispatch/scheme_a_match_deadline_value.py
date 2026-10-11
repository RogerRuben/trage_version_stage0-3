"""Opt-in forecast dispatch-deadline / independent pickup-limit time chain.

Every historical task has one 300-second dispatch slot, not a fixed boarding
time. A site/predecessor connection supplies pickup ETA at that dispatch time;
completion is dispatch + ETA + frozen M3 service time. A seven-bin readiness DP
prices at most three services over sparse C Top3 / nested HV Top6 successors.
This is the same planning approximation and shared-customer LP as before, not
a Valhalla route certificate, new ETA calibration, or executable future order.
"""
from __future__ import annotations

from math import isfinite

import numpy as np
from scipy.spatial import cKDTree

from .scheme_a_city_graph import ChainTask, _xy
from .scheme_a_two_scale_value import _SceneDAG


GRID_S = 300
PICKUP_ETA_LIMIT_S = 600
READY_BINS = 7
MAX_SERVICES = 3


class MatchDeadlineSceneDAG(_SceneDAG):
    """Reuse competitive LP and common prices; replace only its time oracle.

    ``pickup_slot`` is retained for the existing diagnostic interface but is
    an alias of ``dispatch_slot``. ``finish`` is a ZERO-PICKUP lower bound used
    only to screen impossible successors; it is not a predicted completion.
    Actual predecessor-specific completions are ``successor_finish`` and
    ``root_finish``. LP columns still contain physical task indices exactly once.
    """
    def __init__(self, tasks, sites, now, horizon, pace, cfg, check):
        if cfg.get("max_pickup_eta_s", PICKUP_ETA_LIMIT_S) != PICKUP_ETA_LIMIT_S:
            raise ValueError("the frozen dispatch time chain requires the authorized 600-s pickup cap")
        self.tasks = tuple(tasks)
        for task in self.tasks:
            if (not isinstance(task, ChainTask) or task.source_kind != "FORECAST"
                    or task.request_id >= 0 or not task.source_date or task.source_date > "20161024"):
                raise ValueError("coarse values accept earlier-history prediction tasks only")
        if len({task.request_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("duplicate synthetic task in one coarse scenario")
        if (not isfinite(float(now)) or not isfinite(float(horizon))
                or not 0 < horizon - now <= 1800 or now % GRID_S):
            raise ValueError("dispatch time chain requires a 300-s origin and at most 1800-s horizon")
        self.now, self.horizon, self.pace, self.cfg = float(now), float(horizon), pace, dict(cfg)
        self.site_points = np.asarray([_xy(site["position"]) for site in sites], float).reshape(-1, 2)
        self.pickup = np.asarray([_xy(task.pickup) for task in self.tasks], float).reshape(-1, 2)
        self.dropoff = np.asarray([_xy(task.dropoff) for task in self.tasks], float).reshape(-1, 2)
        self.release = np.asarray([task.release_s for task in self.tasks], float)
        self.ids = np.asarray([task.request_id for task in self.tasks], np.int64)
        self.deadline = np.asarray([task.deadline_s for task in self.tasks], float)
        self.service_time = np.asarray([task.service_time_s for task in self.tasks], float)
        self.dispatch_slot = np.ceil(self.release / GRID_S) * GRID_S
        self.pickup_slot = self.dispatch_slot  # Compatibility alias: dispatch, NOT boarding.
        self.finish = self.dispatch_slot + self.service_time  # Zero-pickup lower bound only.
        if not np.all(np.isfinite(self.finish)):
            raise ValueError("forecast completion lower bounds must be finite")
        self.dispatch_bin = self._ready_bins(self.dispatch_slot)
        self.minimum_ready_bin = self._ready_bins(self.finish)
        self.dispatch_valid = ((self.release <= self.dispatch_slot)
            & (self.dispatch_slot < self.deadline)
            & (self.dispatch_slot >= self.now) & (self.dispatch_slot < self.horizon))
        n = len(self.tasks)
        self.allowed = {
            "HV": self.dispatch_valid & np.asarray(["HV" in t.compatible_profiles for t in self.tasks], bool),
            "C": self.dispatch_valid & np.asarray(["C" in t.compatible_profiles
                and t.passenger_accepts_av for t in self.tasks], bool),
        }
        if np.any(self.allowed["C"] & ~self.allowed["HV"]):
            raise ValueError("nested coarse profiles require C customer support to be a subset of HV")
        self.tree = cKDTree(self.pickup) if n else None
        self.site_neighbors, self.root_eta, self.root_finish, self.root_ready_bin = [], [], [], []
        for zone, point in enumerate(self.site_points):
            if zone % 8 == 0:
                check()
            indices = np.asarray(self.tree.query_ball_point(point, 2000.) if self.tree else (), np.int32)
            distance = np.linalg.norm(self.pickup[indices] - point, axis=1)
            eta = self._etas(distance, indices)
            completion = self._completions(indices, eta)
            self.site_neighbors.append((indices, distance))
            self.root_eta.append(eta)
            self.root_finish.append(completion)
            self.root_ready_bin.append(self._ready_bins(completion))
        self.successors, self.departures = {}, {}
        self.successor_eta, self.successor_finish, self.successor_ready_bin = {}, {}, {}
        for profile, width in (("C", 3), ("HV", 6)):
            self.successors[profile] = np.full((n, width), -1, np.int32)
            self.departures[profile] = np.full((n, width), np.inf)
            self.successor_eta[profile] = np.full((n, width), np.inf)
            self.successor_finish[profile] = np.full((n, width), np.inf)
            self.successor_ready_bin[profile] = np.full((n, width), 6, np.int8)
        for i in range(n):
            if i % 32 == 0:
                check()
            if not self.allowed["HV"][i]:
                continue
            indices = np.asarray(self.tree.query_ball_point(self.dropoff[i], 2000.), np.int32)
            distance = np.linalg.norm(self.pickup[indices] - self.dropoff[i], axis=1)
            eta = self._etas(distance, indices)
            completion = self._completions(indices, eta)
            keep = ((indices != i) & self.dispatch_valid[indices] & (eta <= PICKUP_ETA_LIMIT_S)
                & (self.dispatch_bin[indices] >= self.minimum_ready_bin[i]))
            choices = {}
            for profile in ("HV", "C"):
                eligible = np.flatnonzero(keep & self.allowed[profile][indices]) if self.allowed[profile][i] else np.array([], np.int32)
                ordered = np.lexsort((self.ids[indices[eligible]], self.release[indices[eligible]],
                    self.dispatch_slot[indices[eligible]], distance[eligible]))[:3]
                choices[profile] = eligible[ordered].tolist()
            # Preserve both original Top3 choices, with at most six HV arcs.
            choices["HV"] = list(dict.fromkeys(choices["HV"] + choices["C"]))
            for profile in ("C", "HV"):
                offsets = np.asarray(choices[profile], np.int32)
                count = len(offsets)
                self.successors[profile][i, :count] = indices[offsets]
                self.departures[profile][i, :count] = self.dispatch_slot[indices[offsets]]
                self.successor_eta[profile][i, :count] = eta[offsets]
                self.successor_finish[profile][i, :count] = completion[offsets]
                self.successor_ready_bin[profile][i, :count] = self._ready_bins(completion[offsets])
        check()
        self.edge_count = sum(int(np.count_nonzero(values >= 0)) for values in self.successors.values())
        self.alpha, self.beta = np.zeros(n), {}
        self.dp_cache, self.entry_cache, self.entry_ready_cache = {}, {}, {}
        self.info = {}

    def _ready_bins(self, completion):
        # Bin six is terminal. A service already dispatched may complete later
        # than horizon/availability, but cannot start another future service.
        return np.clip(np.ceil((completion - self.now) / GRID_S), 0, 6).astype(np.int8)

    def _etas(self, distance, indices):
        eta = distance * self._paces(self.dispatch_slot[indices])
        if not np.all(np.isfinite(eta)) or np.any(eta < 0):
            raise ValueError("forecast pickup ETA must be finite and nonnegative")
        return eta

    def _completions(self, indices, eta):
        completion = self.dispatch_slot[indices] + eta + self.service_time[indices]
        if not np.all(np.isfinite(completion)):
            raise ValueError("forecast completion including pickup ETA must be finite")
        return completion

    def entries(self, key):
        cached = self.entry_cache.get(key)
        if cached is not None:
            return cached
        profile, zone, ready_bin, end_bin = key
        if profile not in self.allowed or not 0 <= zone < len(self.site_neighbors):
            raise ValueError("unknown forecast profile or site state")
        indices, distance = self.site_neighbors[zone]
        keep = (self.allowed[profile][indices] & (self.root_eta[zone] <= PICKUP_ETA_LIMIT_S)
            & (self.dispatch_bin[indices] >= ready_bin) & (self.dispatch_bin[indices] < min(6, end_bin)))
        offsets = np.flatnonzero(keep) if ready_bin < min(6, end_bin) else np.array([], np.int32)
        order = np.lexsort((self.ids[indices[offsets]], self.release[indices[offsets]],
            self.dispatch_slot[indices[offsets]], distance[offsets]))
        offsets = offsets[order]
        result = indices[offsets]
        self.entry_cache[key] = result
        self.entry_ready_cache[key] = self.root_ready_bin[zone][offsets]
        return result

    def _dp(self, profile, end_bin):
        key = profile, end_bin
        cached = self.dp_cache.get(key)
        if cached is not None:
            return cached
        n = len(self.tasks)
        # Dropoff-origin value at a ready bin, with optional idle termination.
        values = np.zeros((MAX_SERVICES + 1, READY_BINS, n))
        following = np.full(values.shape, -1, np.int8)  # Sparse successor COLUMN, not task ID.
        successors = self.successors[profile]
        safe = np.maximum(successors, 0)
        dispatch = self.dispatch_bin[safe]
        valid = (successors >= 0) & (dispatch < min(6, end_bin))
        reward = 1. - self.alpha
        for depth in range(1, MAX_SERVICES + 1):
            if not n:
                break
            continuation = values[depth - 1, self.successor_ready_bin[profile], safe]
            candidate = reward[safe] + continuation
            for ready_bin in range(6):
                offered = np.where(valid & (dispatch >= ready_bin), candidate, -np.inf)
                best_column = np.argmax(offered, axis=1)
                best = offered[np.arange(n), best_column]
                use = (best > 0) & self.allowed[profile]
                values[depth, ready_bin, use] = best[use]
                following[depth, ready_bin, use] = best_column[use]
        result = values, following
        self.dp_cache[key] = result
        return result

    def price(self, key):
        profile, _, _, end_bin = key
        entries = self.entries(key)
        if not len(entries):
            return 0., ()
        values, following = self._dp(profile, end_bin)
        ready = self.entry_ready_cache[key]
        offered = np.stack([1. - self.alpha[entries] + values[depth - 1, ready, entries]
            for depth in range(1, MAX_SERVICES + 1)])
        score = float(np.max(offered))
        if not isfinite(score):
            raise ValueError("forecast price must be finite")
        if score <= 1e-9:
            return 0., ()
        ties = np.argwhere(offered == score)
        depth, offset = min(ties, key=lambda item: (int(item[0]), int(self.ids[entries[int(item[1])]])))
        node, ready_bin, remaining = int(entries[int(offset)]), int(ready[int(offset)]), int(depth)
        path = [node]
        while remaining > 0:
            column = int(following[remaining, ready_bin, node])
            if column < 0:
                break
            next_node = int(self.successors[profile][node, column])
            ready_bin = int(self.successor_ready_bin[profile][node, column])
            path.append(next_node)
            node, remaining = next_node, remaining - 1
        if len(path) != len(set(path)):
            raise RuntimeError("dispatch time chain repeated a physical forecast customer")
        return min(3., max(0., score)), tuple(path)

    def solve_competition(self, supply, check, remaining, cfg):
        super().solve_competition(supply, check, remaining, cfg)
        self.info.update(forecast_time_contract="MATCH_BY_DEADLINE_WITH_INDEPENDENT_PICKUP_LIMIT",
            fixed_dispatch_grid_s=GRID_S, independent_pickup_eta_limit_s=PICKUP_ETA_LIMIT_S,
            pickup_slot_alias_semantics="DISPATCH_SLOT_NOT_BOARDING_TIME",
            finish_alias_semantics="ZERO_PICKUP_LOWER_BOUND_NOT_ACTUAL_COMPLETION",
            completion_depends_on_predecessor_pickup_eta=True, ready_bin_count=READY_BINS,
            dispatch_eligible_forecast_nodes=int(np.count_nonzero(self.dispatch_valid)),
            profile_successor_domain_nested=True, maximum_services=MAX_SERVICES)

    def resident_extra_array_bytes(self):
        """Additional ndarray bytes beyond the old provider's array/DP list."""
        counted = {id(array) for array in (self.pickup, self.dropoff, self.release, self.ids,
            self.pickup_slot, self.finish, self.deadline, self.alpha,
            *self.allowed.values(), *self.successors.values(), *self.departures.values())}
        counted.update(id(array) for pair in self.dp_cache.values() for array in pair)
        extra = [self.site_points, self.service_time, self.dispatch_bin, self.minimum_ready_bin,
            self.dispatch_valid, *self.root_eta, *self.root_finish, *self.root_ready_bin,
            *self.successor_eta.values(), *self.successor_finish.values(), *self.successor_ready_bin.values(),
            *self.entry_cache.values(), *self.entry_ready_cache.values()]
        extra.extend(array for pair in self.site_neighbors for array in pair)
        size = 0
        for array in extra:
            if id(array) not in counted:
                size += array.nbytes
                counted.add(id(array))
        return size
