"""Bounded per-request counters, not an all-arc trace or causal attribution."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, fields
import json
from math import isfinite


@dataclass(slots=True)
class RequestRecord:
    native_request_id: int
    order_id: str
    release_s: float
    deadline_s: float
    activation_lag_s: float
    status: str = "PENDING"
    end_s: float | None = None
    attempt_rounds: int = 0
    no_global_available_rounds: int = 0
    first_attempt_s: float | None = None
    last_attempt_s: float | None = None
    spatial_hv_opportunities: int = 0
    spatial_av_opportunities: int = 0
    eligible_spatial_opportunities: int = 0
    topk_arc_attempts: int = 0
    returned_route_attempts: int = 0
    patience_feasible_arc_attempts: int = 0
    validated_arc_attempts: int = 0
    withheld_idle_opportunity_rounds: int = 0
    minimum_returned_pickup_eta_s: float | None = None
    rejection_counts: Counter = field(default_factory=Counter)


class RequestDecisionDiagnostics:
    def __init__(self):
        self.records: dict[int, RequestRecord] = {}

    def activate(self, rid, order_id, release_s, deadline_s, now):
        if int(rid) in self.records:
            raise ValueError("request activated twice")
        self.records[int(rid)] = RequestRecord(int(rid), str(order_id), float(release_s),
            float(deadline_s), float(now) - float(release_s))

    def round(self, rid, now, available, raw_hv, raw_av):
        row = self.records[int(rid)]
        if row.status != "PENDING":
            raise ValueError("only pending requests may enter a dispatch round")
        row.attempt_rounds += 1
        row.no_global_available_rounds += int(available == 0)
        if row.first_attempt_s is None:
            row.first_attempt_s = float(now)
        row.last_attempt_s = float(now)
        row.spatial_hv_opportunities += int(raw_hv)
        row.spatial_av_opportunities += int(raw_av)

    def candidates(self, rid, eligible, topk):
        row = self.records[int(rid)]
        row.eligible_spatial_opportunities += int(eligible)
        row.topk_arc_attempts += int(topk)

    def route(self, rid, eta):
        row = self.records[int(rid)]
        row.returned_route_attempts += 1
        if isfinite(float(eta)) and eta >= 0:
            row.minimum_returned_pickup_eta_s = (float(eta) if row.minimum_returned_pickup_eta_s is None
                else min(row.minimum_returned_pickup_eta_s, float(eta)))

    def feasible(self, rid):
        self.records[int(rid)].patience_feasible_arc_attempts += 1

    def validated(self, rid):
        self.records[int(rid)].validated_arc_attempts += 1

    def reject(self, rid, *reasons, count=1):
        counts = self.records[int(rid)].rejection_counts
        for reason in set(reasons):
            # Preserve bounded reason codes, never vehicle/edge identities.
            key = str(reason)
            if key not in counts and len(counts) >= 64:
                key = "OTHER_REASON"
            counts[key] += int(count)

    def withheld(self, rid):
        self.records[int(rid)].withheld_idle_opportunity_rounds += 1

    def finish(self, rid, now, status):
        if status not in ("ASSIGNED", "EXPIRED"):
            raise ValueError("invalid request decision terminal state")
        row = self.records[int(rid)]
        if row.status == status == "EXPIRED":
            return  # Expiry may be called through the bridge and controller.
        if row.status != "PENDING":
            raise ValueError("an assigned request must never become patience-expired")
        if status == "EXPIRED" and float(now) < row.deadline_s:
            raise ValueError("request expired before its release-based deadline")
        row.status, row.end_s = status, float(now)

    @staticmethod
    def expiry_stage(row):
        if row.status != "EXPIRED":
            return None
        if not row.attempt_rounds:
            return "NO_DISPATCH_ROUND"
        if row.spatial_hv_opportunities + row.spatial_av_opportunities == 0:
            return "NO_NEARBY_AVAILABLE_VEHICLE"
        if not row.eligible_spatial_opportunities:
            return "NO_ELIGIBLE_SPATIAL_CANDIDATE"
        if not row.returned_route_attempts and not row.patience_feasible_arc_attempts:
            if row.rejection_counts.get("CACHED_ETA_EXCEEDS_REMAINING_PATIENCE", 0):
                return "NO_PICKUP_WITHIN_REMAINING_PATIENCE"
            return "NO_RETURNED_PICKUP_ROUTE"
        if not row.patience_feasible_arc_attempts:
            return "NO_PICKUP_WITHIN_REMAINING_PATIENCE"
        if not row.validated_arc_attempts:
            return "NO_FINAL_VALIDATED_ARC"
        if row.withheld_idle_opportunity_rounds:
            return "VALID_ARC_WITH_IDLE_OPPORTUNITY_WITHHELD"
        return "VALID_ARCS_NOT_SELECTED"

    def rows(self):
        for rid in sorted(self.records):
            record = self.records[rid]
            row = {item.name: getattr(record, item.name) for item in fields(record)}
            row["rejection_counts_json"] = json.dumps(dict(sorted(row.pop("rejection_counts").items())))
            row["expiry_last_reached_stage"] = self.expiry_stage(record)
            row["evidence_scope"] = "OBSERVED_CANDIDATE_ATTEMPTS_NOT_CAUSAL_DECOMPOSITION"
            yield row

    def summary(self):
        return dict(request_count=len(self.records),
            terminal_states=dict(Counter(row.status for row in self.records.values())),
            expiry_stage_counts=dict(Counter(self.expiry_stage(row) for row in self.records.values()
                if row.status == "EXPIRED")),
            stores_all_arcs=False, resets_patience_on_retry=False,
            arc_counters_count_repeated_attempts_not_unique_vehicles=True,
            categories_are_diagnostic_not_exclusive_physical_causes=True)
