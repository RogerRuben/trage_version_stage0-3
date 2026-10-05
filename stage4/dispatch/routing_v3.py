"""Cache-aware sparse planning before work submission, ordered v1 replay."""
from __future__ import annotations

from collections import OrderedDict, defaultdict
import time

import pandas as pd

from .deterministic_routing import ArcDeterministicValhallaAdapter, SINGLE_SOURCE_MATRIX
from .routing_v3_workers import query_group
from stage4.fleetpy_adapter.valhalla_time_adapter import PickupEstimate


class _ShadowLRU:
    """Lazy LRU simulation: no 50k-entry copy on every epoch."""
    def __init__(self, data, capacity):
        self.data, self.capacity = data, capacity
        self.tail, self.removed = OrderedDict(), set()
        self.oldest = iter(data)
        self.count = len(data)

    def get(self, key):
        if key in self.tail:
            self.tail.move_to_end(key)
            return self.tail[key]
        if key not in self.removed and key in self.data:
            self.tail[key] = self.data[key]
            return self.tail[key]
        return None

    def put(self, key, value):
        if not self.capacity:
            return
        exists = key in self.tail or (key not in self.removed and key in self.data)
        self.count += int(not exists)
        self.tail[key] = value
        self.tail.move_to_end(key)
        self.removed.discard(key)
        while self.count > self.capacity:
            victim = None
            for candidate in self.oldest:
                if candidate not in self.removed and candidate not in self.tail:
                    victim = candidate
                    break
            if victim is None:
                victim, _ = self.tail.popitem(last=False)
            self.removed.add(victim)
            self.count -= 1


class DemandRoutingAdapter(ArcDeterministicValhallaAdapter):
    def __init__(self, *args, grouped_sources=False, **kwargs):
        bank = kwargs.pop("disk_cache_path", None)
        context = kwargs.pop("routing_context", None)
        limit = kwargs.pop("disk_cache_limit_mib", 512)
        entries = kwargs.pop("disk_cache_max_entries", 2_000_000)
        super().__init__(*args, **kwargs)
        self.grouped_sources = bool(grouped_sources)
        if self.grouped_sources and self.routing_mode != SINGLE_SOURCE_MATRIX:
            raise ValueError("grouped sources require the forward matrix costing mode")
        if bank is not None:
            if not context:
                raise ValueError("buffered raw bank requires engine/tile context")
            from .route_bank_v3 import BufferedRouteBank
            self._disk_cache = BufferedRouteBank(bank, context, limit_mib=limit, max_entries=entries)
        self.raw_prefetch_avoided = 0
        self.source_group_queries = 0
        self.group_cell_fallback_queries = 0

    def _evaluate(self, payloads):
        if self.grouped_sources:
            by_source = defaultdict(list)
            for key, q in payloads:
                by_source[q[0], q[1], q[2], q[5]].append((key, q))
            groups = [group[left:left + 32] for group in by_source.values() for left in range(0, len(group), 32)]
        else:
            groups = [payloads[left:left + 8] for left in range(0, len(payloads), 8)]
        results = []
        for left in range(0, len(groups), 32):
            block = groups[left:left + 32]
            inputs = [([q for _, q in group], self.grouped_sources) for group in block]
            start = time.perf_counter()
            if self.route_workers > 1:
                answers = list(self._worker_pool().map(query_group, inputs, chunksize=1))
            else:
                # Injected actor remains usable for small joint tests.
                from . import routing_workers
                previous = routing_workers._ACTOR
                routing_workers._ACTOR = self.actor
                try:
                    answers = [query_group(item) for item in inputs]
                finally:
                    routing_workers._ACTOR = previous
            self.routing_time_s += time.perf_counter() - start
            for group, answer in zip(block, answers):
                self.source_group_queries += int(self.grouped_sources)
                self.routing_queries += answer["queries"]
                self.group_cell_fallback_queries += answer["fallbacks"]
                self.routing_arc_evaluations += len(group)
                results.extend((key, value) for (key, _), value in zip(group, answer["results"]))
        return results

    def estimate_epoch(self, batches):
        if not batches:
            return []
        if sum(len(x[0]) for x in batches) > 50_000:
            found, block, size = [], [], 0
            for batch in batches:
                if len(batch[0]) > 50_000:
                    raise ValueError("single sparse route batch exceeds 50000 arcs")
                if block and size + len(batch[0]) > 50_000:
                    found.extend(self.estimate_epoch(block)); block, size = [], 0
                block.append(batch); size += len(batch[0])
            return found + self.estimate_epoch(block)
        self.epoch_route_batches += 1
        started = time.perf_counter()
        virtual_legacy = dict(self.cache)
        lru = _ShadowLRU(self._persistent_cache, self.persistent_cache_size)
        plans, needed, all_missing = [], OrderedDict(), set()
        # Simulate the ORIGINAL two-phase fill order before asking the backend.
        for candidates, lon, lat, timestamp in batches:
            local = pd.Timestamp(timestamp).tz_convert("Asia/Shanghai")
            bin_index, beta = self.beta_for(local)
            minute = local.strftime("%Y-%m-%dT%H:%M")
            rows, misses = [], []
            for vehicle in candidates:
                key = self._matrix_key(vehicle, lon, lat, local, bin_index)
                raw_key = self._raw_key(vehicle, lon, lat, local)
                exact = self._persistent_key(vehicle, lon, lat, local, bin_index, beta)
                row = (vehicle, key, raw_key, exact)
                rows.append(row)
                if key not in self.cache and exact not in self._persistent_cache:
                    all_missing.add(raw_key)
                if key not in virtual_legacy:
                    misses.append(row)
            uncached = []
            for row in misses:
                _, key, raw_key, exact = row
                value = lru.get(exact)
                if value is not None:
                    virtual_legacy[key] = value
                else:
                    uncached.append(row)
            for vehicle, key, raw_key, exact in uncached:
                self._validate_wgs84(vehicle.lon_wgs84, vehicle.lat_wgs84)
                self._validate_wgs84(lon, lat)
                needed.setdefault(raw_key, (self.routing_mode, float(vehicle.lon_wgs84), float(vehicle.lat_wgs84), float(lon), float(lat), minute))
                reference = ("PENDING_RAW", raw_key)
                lru.put(exact, reference)
                virtual_legacy[key] = reference
            plans.append((rows, local, bin_index, beta, lon, lat))
        self.raw_prefetch_avoided += len(all_missing) - len(needed)
        answers, origins = {}, {}
        if self._disk_cache is not None and needed:
            t = time.perf_counter()
            digests = {self._disk_cache.key(key): key for key in needed}
            for digest, value in self._disk_cache.get_many(digests).items():
                key = digests[digest]
                answers[key], origins[key] = (*value, None), "DISK"
            self.disk_cache_lookup_time_s += time.perf_counter() - t
        pending = [(key, q) for key, q in needed.items() if key not in answers]
        for key, (raw, distance, error, work) in self._evaluate(pending):
            self.routing_backend_work_time_s += work
            answers[key], origins[key] = (raw, distance, error), "COMPUTED"
            if error is None and self._disk_cache is not None:
                self._disk_cache.remember(self._disk_cache.key(key), raw, distance)
        results, consumed = [], set()
        for rows, local, bin_index, beta, lon, lat in plans:
            found, uncached = {}, []
            hits = {v.native_vehicle_id: self.cache[key] for v, key, _, _ in rows if key in self.cache}
            for vehicle, key, raw_key, exact in rows:
                self.arc_lookup_count += 1
                hit = hits.get(vehicle.native_vehicle_id)
                if hit is None:
                    hit = self._persistent_cache.get(exact)
                    if hit is not None:
                        self._persistent_cache.move_to_end(exact)
                        self.cache[key] = hit
                        self.raw_lru_hits += 1
                        self.persistent_cache_hits += 1
                if hit is not None:
                    self.cache_hit_count += 1
                    found[vehicle.native_vehicle_id] = PickupEstimate(**{**hit.__dict__, "cache_hit": True})
                else:
                    uncached.append((vehicle, key, raw_key, exact))
            for vehicle, key, raw_key, exact in uncached:
                answer = answers.get(raw_key)
                if answer is None or (answer[2] is not None and raw_key in consumed):
                    # A planned hit may have failed. Resolve it, never omit it.
                    q = (self.routing_mode, float(vehicle.lon_wgs84), float(vehicle.lat_wgs84), float(lon), float(lat), local.strftime("%Y-%m-%dT%H:%M"))
                    _, value = self._evaluate([(raw_key, q)])[0]
                    raw, distance, error, work = value
                    self.routing_backend_work_time_s += work
                    answers[raw_key], origins[raw_key] = (raw, distance, error), "COMPUTED"
                    answer = answers[raw_key]
                raw, distance, error = answer
                duplicate = raw_key in consumed
                consumed.add(raw_key)
                if error is not None:
                    self.routing_failures += 1
                    self.matrix_failed_arcs += int(self.routing_mode == SINGLE_SOURCE_MATRIX)
                    self._record_failed_arc(vehicle, lon, lat, local, error)
                    continue
                disk_hit = origins[raw_key] == "DISK"
                self.cache_hit_count += int(disk_hit)
                self.disk_cache_hits += int(disk_hit)
                self.raw_query_deduplications += int(duplicate and not disk_hit)
                stored = PickupEstimate(raw, raw * beta, distance, beta, bin_index, False)
                self.cache[key] = stored
                self._remember(exact, stored)
                found[vehicle.native_vehicle_id] = PickupEstimate(**{**stored.__dict__, "cache_hit": disk_hit})
            results.append(found)
        self.prefetched_unused_raw_arcs += sum(origins[k] == "COMPUTED" and k not in consumed for k in answers)
        if self._disk_cache is not None:
            self._disk_cache.flush()  # Bounded background queue, not an fsync here.
        self.routing_queue_pipeline_time_s += time.perf_counter() - started
        return results

    def estimate_many(self, candidates, pickup_lon, pickup_lat, timestamp):
        return self.estimate_epoch([(candidates, pickup_lon, pickup_lat, timestamp)])[0]
