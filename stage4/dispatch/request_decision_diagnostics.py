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
    typed_final_AV_arc_attempts: int = 0
    typed_final_HV_arc_attempts: int = 0
    typed_rounds_with_final_AV: int = 0
    typed_rounds_with_final_HV: int = 0
    typed_last_final_AV_count: int = 0
    typed_last_final_HV_count: int = 0
    typed_last_min_AV_eta_s: float | None = None
    typed_last_min_HV_eta_s: float | None = None
    typed_HV_assigned_with_final_AV_same_epoch: bool | None = None
    typed_AV_alternatives_idle_after_choice: int | None = None
    typed_AV_alternatives_serving_other_request: int | None = None
    typed_AV_alternatives_moving_after_choice: int | None = None
    typed_rejections: Counter | None = None


class RequestDecisionDiagnostics:
    def __init__(self, *, deadline_semantics="REQUEST_TO_PICKUP", typed_candidates=False):
        self.records: dict[int, RequestRecord] = {}
        self.deadline_semantics = deadline_semantics
        if type(typed_candidates) is not bool:
            raise ValueError("typed diagnostic switch must be boolean")
        self.typed_candidates = typed_candidates

    def activate(self, rid, order_id, release_s, deadline_s, now):
        if int(rid) in self.records:
            raise ValueError("request activated twice")
        self.records[int(rid)] = RequestRecord(int(rid), str(order_id), float(release_s),
            float(deadline_s), float(now) - float(release_s))
        if self.typed_candidates:
            self.records[int(rid)].typed_rejections = Counter()

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
        if self.typed_candidates:
            row.typed_last_final_AV_count = row.typed_last_final_HV_count = 0
            row.typed_last_min_AV_eta_s = row.typed_last_min_HV_eta_s = None

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

    def validated(self, rid, vehicle_type=None, pickup_eta_s=None):
        row = self.records[int(rid)]
        row.validated_arc_attempts += 1
        if self.typed_candidates:
            if vehicle_type not in ("AV", "HV"):
                raise ValueError("final typed candidate requires its vehicle type")
            name = f"typed_last_final_{vehicle_type}_count"
            if not getattr(row, name):
                rounds = f"typed_rounds_with_final_{vehicle_type}"
                setattr(row, rounds, getattr(row, rounds) + 1)
            setattr(row, name, getattr(row, name) + 1)
            attempts = f"typed_final_{vehicle_type}_arc_attempts"
            setattr(row, attempts, getattr(row, attempts) + 1)
            if pickup_eta_s is not None:
                if not isfinite(float(pickup_eta_s)) or pickup_eta_s < 0:
                    raise ValueError("invalid final typed pickup ETA")
                minimum = f"typed_last_min_{vehicle_type}_eta_s"
                old = getattr(row, minimum)
                setattr(row, minimum, float(pickup_eta_s) if old is None else min(old, float(pickup_eta_s)))

    def reject(self, rid, *reasons, count=1, stage="CANDIDATE_GATE", vehicle_type=None):
        counts = self.records[int(rid)].rejection_counts
        for reason in set(reasons):
            # Preserve bounded reason codes, never vehicle/edge identities.
            key = str(reason)
            if key not in counts and len(counts) >= 64:
                key = "OTHER_REASON"
            counts[key] += int(count)
        if stage is not None:
            self.reject_stage(rid, stage, vehicle_type, reasons, count=count)

    def reject_stage(self, rid, stage, vehicle_type, reasons, *, count=1):
        if not self.typed_candidates:
            return
        if vehicle_type not in ("AV", "HV"):
            vehicle_type = "UNATTRIBUTED"
        counts = self.records[int(rid)].typed_rejections
        for reason in set(reasons):
            key = f"{vehicle_type}:{stage}:{reason}"
            if key not in counts and len(counts) >= 64:
                key = "OTHER_TYPED_REASON"
            counts[key] += int(count)

    def withheld(self, rid):
        self.records[int(rid)].withheld_idle_opportunity_rounds += 1

    def HV_alternatives(self, rid, *, idle, serving_other, moving):
        if not self.typed_candidates:
            return
        if any(type(v) is not int or v < 0 for v in (idle, serving_other, moving)):
            raise ValueError("invalid final AV alternative state counts")
        row = self.records[int(rid)]
        if idle + serving_other + moving != row.typed_last_final_AV_count:
            raise ValueError("selected-graph AV alternatives do not reconcile")
        row.typed_AV_alternatives_idle_after_choice = idle
        row.typed_AV_alternatives_serving_other_request = serving_other
        row.typed_AV_alternatives_moving_after_choice = moving

    def finish(self, rid, now, status, *, vehicle_type=None):
        if status not in ("ASSIGNED", "EXPIRED"):
            raise ValueError("invalid request decision terminal state")
        row = self.records[int(rid)]
        if row.status == status == "EXPIRED":
            return  # Expiry may be called through the bridge and controller.
        if row.status != "PENDING":
            raise ValueError("an assigned request must never become patience-expired")
        if status == "EXPIRED" and float(now) < row.deadline_s:
            raise ValueError("request expired before its release-based deadline")
        if self.typed_candidates and status == "ASSIGNED" and vehicle_type not in ("AV", "HV"):
            raise ValueError("typed assignment requires selected vehicle type")
        row.status, row.end_s = status, float(now)
        if self.typed_candidates and status == "ASSIGNED":
            if vehicle_type == "HV":
                row.typed_HV_assigned_with_final_AV_same_epoch = row.typed_last_final_AV_count > 0

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
            if row.rejection_counts.get("CACHED_PICKUP_ETA_EXCEEDS_INDEPENDENT_CAP", 0):
                return "NO_PICKUP_WITHIN_INDEPENDENT_CAP"
            if row.rejection_counts.get("CACHED_ETA_EXCEEDS_REMAINING_PATIENCE", 0):
                return "NO_PICKUP_WITHIN_REMAINING_PATIENCE"
            return "NO_RETURNED_PICKUP_ROUTE"
        if not row.patience_feasible_arc_attempts:
            if row.rejection_counts.get("PICKUP_ETA_EXCEEDS_INDEPENDENT_CAP", 0):
                return "NO_PICKUP_WITHIN_INDEPENDENT_CAP"
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
            if self.typed_candidates:
                row["typed_rejection_counts_json"] = json.dumps(dict(sorted(row.pop("typed_rejections").items())))
                row["typed_final_candidate_scope"] = "PRE_JOINT_ASSIGNMENT_GRAPH_NOT_FREE_AFTER_OTHER_ASSIGNMENTS"
            else:
                row = {key: value for key, value in row.items() if not key.startswith("typed_")}
            row["rejection_counts_json"] = json.dumps(dict(sorted(row.pop("rejection_counts").items())))
            row["expiry_last_reached_stage"] = self.expiry_stage(record)
            row["evidence_scope"] = "OBSERVED_CANDIDATE_ATTEMPTS_NOT_CAUSAL_DECOMPOSITION"
            row["deadline_semantics"] = self.deadline_semantics
            yield row

    def summary(self):
        result = dict(request_count=len(self.records),
            terminal_states=dict(Counter(row.status for row in self.records.values())),
            expiry_stage_counts=dict(Counter(self.expiry_stage(row) for row in self.records.values()
                if row.status == "EXPIRED")),
            stores_all_arcs=False, resets_patience_on_retry=False,
            deadline_semantics=self.deadline_semantics,
            arc_counters_count_repeated_attempts_not_unique_vehicles=True,
            categories_are_diagnostic_not_exclusive_physical_causes=True)
        if self.typed_candidates:
            result.update(typed_final_candidate_counts_enabled=True,
                typed_final_AV_arc_attempts=sum(r.typed_final_AV_arc_attempts for r in self.records.values()),
                typed_final_HV_arc_attempts=sum(r.typed_final_HV_arc_attempts for r in self.records.values()),
                HV_assigned_with_final_AV_same_epoch=sum(r.typed_HV_assigned_with_final_AV_same_epoch is True for r in self.records.values()),
                HV_assigned_without_final_AV_same_epoch=sum(r.typed_HV_assigned_with_final_AV_same_epoch is False for r in self.records.values()),
                HV_assigned_with_unoccupied_valid_AV=sum(r.typed_HV_assigned_with_final_AV_same_epoch is True
                    and (r.typed_AV_alternatives_idle_after_choice or 0) > 0 for r in self.records.values()),
                AV_alternatives_selected_for_other_requests=sum(r.typed_AV_alternatives_serving_other_request or 0 for r in self.records.values()),
                typed_final_scope="PRE_JOINT_ASSIGNMENT_GRAPH",
                typed_rejection_reason_limit_per_request=64)
        return result
