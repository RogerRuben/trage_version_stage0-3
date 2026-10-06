"""Bounded static raw-OD reuse; beta and admission are always time-specific."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from math import atan2, cos, isfinite, radians, sin, sqrt
from pathlib import Path

import pandas as pd

from .deterministic_routing import SINGLE_SOURCE_MATRIX
from .routing_v3 import DemandRoutingAdapter

STATIC_TAG = "STATIC_AUTO_OD_3_8_2_V1"


@lru_cache(maxsize=50_000)
def _well_separated(hex_coordinates):
    """Conservative guard, NOT a snapped-road travel-time bound.

    Loki clears location times when integer metre distance exceeds zero.
    A 6,000-km spherical radius and 10-m guard stay well below its Earth
    radius/distance and away from the sub-metre integer boundary. Coordinates
    themselves remain exact; near-zero pairs keep their original minute key.
    """
    lon1, lat1, lon2, lat2 = (float.fromhex(x) for x in hex_coordinates)
    if not all(isfinite(x) for x in (lon1, lat1, lon2, lat2)):
        return False
    if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180 and -90 <= lat1 <= 90 and -90 <= lat2 <= 90):
        return False
    a, b = radians(lat1), radians(lat2)
    h = sin((b - a) / 2.) ** 2 + cos(a) * cos(b) * sin(radians(lon2 - lon1) / 2.) ** 2
    h = min(1., max(0., h))
    return 12_000_000. * atan2(sqrt(h), sqrt(1. - h)) > 10.


def static_matrix_certificate(root):
    """Bind the narrow, inspected engine branch; refuse mutable traffic."""
    from valhalla._valhalla import VALHALLA_PRINT_VERSION
    config_path = Path(json.loads((Path(root) / "stage3/config/stage3_finalization.json").read_text())["valhalla_config"])
    encoded = config_path.read_bytes()
    cfg = json.loads(encoded)
    graph = cfg.get("mjolnir", {})
    if (VALHALLA_PRINT_VERSION != "3.8.2"
            or cfg.get("service_limits", {}).get("max_timedep_distance_matrix") != 0
            or cfg.get("thor", {}).get("source_to_target_algorithm") != "select_optimal"
            or any(graph.get(name) for name in ("traffic_extract", "incident_log", "incident_dir", "tile_url"))):
        raise ValueError("static raw OD reuse requires the inspected frozen Valhalla 3.8.2 matrix context")
    return dict(schema=STATIC_TAG, valhalla_version=VALHALLA_PRINT_VERSION,
        config_sha256=hashlib.sha256(encoded).hexdigest(), timedep_matrix_distance=0,
        conservative_static_guard_m=10., mutable_traffic_or_remote_tiles=False)


@dataclass(frozen=True)
class PickupEtaBudget:
    remaining_patience_s: float
    simulation_time_s: float
    predicted_service_with_overhead_s: float
    empirical_session_end_s: float | None = None

    def rejection(self, eta):
        # Same comparison/order of additions as rolling_or_control. Do not
        # rearrange into an ETA cap: that can move floating boundary cases.
        if not isfinite(eta) or eta > self.remaining_patience_s:
            return "PATIENCE"
        if self.empirical_session_end_s is not None:
            predicted = self.predicted_service_with_overhead_s
            if not isfinite(predicted):
                return "HV_SESSION_EVIDENCE"
            if self.simulation_time_s + eta + predicted > self.empirical_session_end_s:
                return "HV_SESSION_END"
        return None


class StaticRawRoutingAdapter(DemandRoutingAdapter):
    """v3 cache precedence plus a separate bounded raw-time/distance LRU."""
    def __init__(self, *args, static_raw_od=True, raw_od_cache_size=50_000,
                 certified_eta_pruning=True, **kwargs):
        if not isinstance(raw_od_cache_size, int) or not 0 <= raw_od_cache_size <= 50_000:
            raise ValueError("raw OD memory cache exceeds 50000 entries")
        certificate = None
        if static_raw_od:
            if kwargs.get("routing_mode") != SINGLE_SOURCE_MATRIX:
                raise ValueError("static raw OD reuse is matrix-only")
            certificate = static_matrix_certificate(args[0] if args else kwargs["root"])
        super().__init__(*args, **kwargs)
        self.static_raw_od = bool(static_raw_od)
        self.static_certificate = certificate
        self.raw_od_cache_size = raw_od_cache_size
        self.certified_eta_pruning = bool(certified_eta_pruning)
        self._raw_od_cache = OrderedDict()
        self._memory_answers = set()
        self.raw_od_memory_queries_avoided = 0
        self.raw_od_cross_minute_reuses = 0
        self.raw_od_beta_bin_reuses = 0
        self.certified_eta_prunes = 0
        self.known_eta_prunes_before_backend = 0
        self.last_certified_prunes = []
        self.last_pruned_estimates = []

    def _normal_key(self, key):
        if self.static_raw_od and key[0] == SINGLE_SOURCE_MATRIX and _well_separated(tuple(key[1:5])):
            return (*key[:5], STATIC_TAG)
        return tuple(key)

    def _raw_key(self, vehicle, lon, lat, local):
        return self._normal_key(super()._raw_key(vehicle, lon, lat, local))

    def _remember_raw(self, key, raw, distance, minute, bin_index):
        if not self.raw_od_cache_size:
            return
        self._raw_od_cache[key] = (float(raw), float(distance), minute, int(bin_index))
        self._raw_od_cache.move_to_end(key)
        while len(self._raw_od_cache) > self.raw_od_cache_size:
            self._raw_od_cache.popitem(last=False)

    def _remember(self, key, estimate):
        super()._remember(key, estimate)
        self._remember_raw(self._normal_key(key[:6]), estimate.valhalla_time_s,
                           estimate.route_distance_m, key[5], key[6])

    def _answer_origin(self, key):
        return "RAW_OD_MEMORY" if key in self._memory_answers else "COMPUTED"

    def _groupable(self, key, query):
        # A far target clears the shared source time, which could otherwise
        # change a near-zero/time-aware cell in the same matrix request.
        return key[-1] == STATIC_TAG

    def _raw_answers_before_disk(self, needed):
        found = {}
        for key, q in needed.items():
            cached = self._raw_od_cache.get(key)
            if cached is None:
                continue
            raw, distance, minute, bin_index = cached
            self._raw_od_cache.move_to_end(key)
            self._memory_answers.add(key)
            self.raw_od_memory_queries_avoided += 1
            self.raw_od_cross_minute_reuses += int(minute != q[5])
            current_bin = (int(q[5][11:13]) * 60 + int(q[5][14:16])) // 15
            self.raw_od_beta_bin_reuses += int(current_bin != bin_index)
            found[key] = (raw, distance)
        return found

    def _evaluate(self, payloads):
        values, missing = {}, []
        for key, q in payloads:
            cached = self._raw_od_cache.get(key)
            if cached is None:
                self._memory_answers.discard(key)
                missing.append((key, q))
                continue
            self._raw_od_cache.move_to_end(key)
            raw, distance, minute, bin_index = cached
            self._memory_answers.add(key)
            self.raw_od_memory_queries_avoided += 1
            self.raw_od_cross_minute_reuses += int(minute != q[5])
            current_bin = (int(q[5][11:13]) * 60 + int(q[5][14:16])) // 15
            self.raw_od_beta_bin_reuses += int(current_bin != bin_index)
            values[key] = (raw, distance, None, 0.)
        for key, value in super()._evaluate(missing):
            values[key] = value
        # Return in original raw-request order, not grouped/search finish order.
        return [(key, values[key]) for key, _ in payloads]

    def _known_eta(self, vehicle, lon, lat, local, bin_index, beta):
        value = self.cache.get(self._matrix_key(vehicle, lon, lat, local, bin_index))
        if value is None:
            value = self._persistent_cache.get(self._persistent_key(vehicle, lon, lat, local, bin_index, beta))
        if value is not None:
            return value.corrected_pickup_eta_s
        raw = self._raw_od_cache.get(self._raw_key(vehicle, lon, lat, local))
        return None if raw is None else raw[0] * beta

    def estimate_epoch(self, batches, *, eta_budgets=None):
        if eta_budgets is not None and len(eta_budgets) != len(batches):
            raise ValueError("ETA budgets must match original sparse route batches")
        self._memory_answers.clear()
        self.last_certified_prunes = [{} for _ in batches]
        self.last_pruned_estimates = [{} for _ in batches]
        if sum(len(b[0]) for b in batches) > 50_000:
            # Preserve bounded chunks AND the budget/diagnostic alignment.
            found, pruned, pruned_values, left, size = [], [], [], 0, 0
            for i, b in enumerate(batches):
                if len(b[0]) > 50_000:
                    raise ValueError("single sparse route batch exceeds 50000 arcs")
                if size and size + len(b[0]) > 50_000:
                    found.extend(self.estimate_epoch(batches[left:i], eta_budgets=None if eta_budgets is None else eta_budgets[left:i]))
                    pruned.extend(self.last_certified_prunes)
                    pruned_values.extend(self.last_pruned_estimates)
                    left, size = i, 0
                size += len(b[0])
            found.extend(self.estimate_epoch(batches[left:], eta_budgets=None if eta_budgets is None else eta_budgets[left:]))
            pruned.extend(self.last_certified_prunes)
            pruned_values.extend(self.last_pruned_estimates)
            self.last_certified_prunes = pruned
            self.last_pruned_estimates = pruned_values
            return found
        known = []
        if eta_budgets is not None and self.certified_eta_pruning:
            for vehicles, lon, lat, stamp in batches:
                local = pd.Timestamp(stamp).tz_convert("Asia/Shanghai")
                bin_index, beta = self.beta_for(local)
                known.append({v.native_vehicle_id: self._known_eta(v, lon, lat, local, bin_index, beta) for v in vehicles})
        found = super().estimate_epoch(batches)
        if eta_budgets is not None and self.certified_eta_pruning:
            for i, (estimates, budgets) in enumerate(zip(found, eta_budgets)):
                for vid, estimate in tuple(estimates.items()):
                    budget = budgets.get(vid)
                    reason = None if budget is None else budget.rejection(estimate.corrected_pickup_eta_s)
                    if reason is None or known[i].get(vid) != estimate.corrected_pickup_eta_s:
                        continue
                    # v3 already filled rounded aliases and both LRUs. Pruning
                    # must not change which precise OD supplies a later alias.
                    self.last_certified_prunes[i][vid] = reason
                    self.last_pruned_estimates[i][vid] = estimate
                    self.certified_eta_prunes += 1
                    self.known_eta_prunes_before_backend += 1
                    del estimates[vid]
        return found

    def diagnostics(self):
        return dict(static_matrix_certificate=self.static_certificate,
            raw_od_cache_entries=len(self._raw_od_cache),
            raw_od_memory_queries_avoided=self.raw_od_memory_queries_avoided,
            raw_od_cross_minute_reuses=self.raw_od_cross_minute_reuses,
            raw_od_beta_bin_reuses=self.raw_od_beta_bin_reuses,
            certified_eta_prunes=self.certified_eta_prunes,
            known_eta_prunes_before_backend=self.known_eta_prunes_before_backend,
            grouped_source_queries=self.source_group_queries,
            grouped_cell_fallback_queries=self.group_cell_fallback_queries)
