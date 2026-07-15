"""Audit Stage4 Simulator v3 outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage4.simulator_v3.economy_ledger import audit_ledger


LEGAL_TRANSITIONS = {
    "UNREVEALED": {"PENDING", "CANCELLED"},
    "PENDING": {"CANDIDATE_SELECTED", "OFFERED", "RESERVED", "ASSIGNED", "CANCELLED", "PICKUP_STARTED"},
    "CANDIDATE_SELECTED": {"OFFERED", "ASSIGNED", "PENDING", "CANCELLED"},
    "OFFERED": {"ASSIGNED", "RESERVED", "PENDING", "CANCELLED"},
    "RESERVED": {"ASSIGNED", "PENDING", "CANCELLED"},
    "ASSIGNED": {"PICKUP_PENDING", "PICKUP_STARTED", "CANCELLED"},
    "PICKUP_PENDING": {"PICKUP_STARTED", "CANCELLED"},
    "PICKUP_STARTED": {"BOARDED", "CANCELLED"},
    "BOARDED": {"IN_SERVICE"},
    "IN_SERVICE": {"COMPLETED"},
    "COMPLETED": set(),
    "CANCELLED": set(),
}

EVENT_PRIORITY = {
    "LEG_COMPLETED": 1,
    "SERVICE_COMPLETED": 2,
    "HV_SESSION_END": 3,
    "HV_SESSION_START": 4,
    "REQUEST_REVEALED": 5,
    "DRIVER_RESPONSE": 6,
    "RESERVATION_EXPIRED": 7,
    "PLAN_INVALIDATED": 8,
    "DECISION_EPOCH": 9,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("stage4/output/simulator_v3/replication=1/strategy=Safe_GlobalMatch-MinPickup/operation=O0/RT-Base"))
    parser.add_argument("--results-dir", type=Path, default=Path("stage4/docs/results/simulator_v3"))
    return parser.parse_args()


def status(ok: bool) -> str:
    return "PASS" if bool(ok) else "FAIL"


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    req = pd.read_parquet(args.run_dir / "request_log.parquet")
    legs = pd.read_parquet(args.run_dir / "vehicle_leg_log.parquet")
    plans = pd.read_parquet(args.run_dir / "plan_revision_log.parquet")
    offers = pd.read_parquet(args.run_dir / "offer_log.parquet")
    epochs = pd.read_parquet(args.run_dir / "system_epoch_log.parquet")
    transitions = pd.read_parquet(args.run_dir / "request_transition_log.parquet")
    events = pd.read_parquet(args.run_dir / "event_execution_log.parquet")
    vehicle_state_path = args.run_dir / "vehicle_state_log.parquet"
    vehicle_states = pd.read_parquet(vehicle_state_path) if vehicle_state_path.exists() else pd.DataFrame()
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))

    formal_full_day = int(summary.get("max_orders", -1)) == 0
    condition_known_count = int(req.get("condition_available", pd.Series(False, index=req.index)).fillna(False).astype(bool).sum())
    condition_unknown_count = int(len(req) - condition_known_count)
    request_audit = {
        "orders": int(len(req)),
        "formal_full_day_run": formal_full_day,
        "condition_known_orders": condition_known_count,
        "condition_unknown_orders": condition_unknown_count,
        "demand_universe_count_pass": status(
            (len(req) == 114_356 and condition_known_count == 112_165 and condition_unknown_count == 2_191)
            if formal_full_day else len(req) == int(summary.get("orders", -1))
        ),
        "completed_orders": int(req["final_status"].eq("COMPLETED").sum()),
        "cancelled_orders": int(req["final_status"].eq("CANCELLED").sum()),
        "completed_plus_cancelled_pass": status(int(req["final_status"].isin(["COMPLETED", "CANCELLED"]).sum()) == len(req)),
        "duplicate_request_rows": int(req["order_id"].duplicated().sum()),
        "duplicate_request_rows_pass": status(req["order_id"].is_unique),
        "completed_without_assignment": int((req["final_status"].eq("COMPLETED") & req["assigned_vehicle_id"].fillna("").eq("")).sum()),
        "completed_without_assignment_pass": status((~(req["final_status"].eq("COMPLETED") & req["assigned_vehicle_id"].fillna("").eq(""))).all()),
    }

    illegal = transitions[~transitions.apply(lambda r: str(r["new_status"]) in LEGAL_TRANSITIONS.get(str(r["old_status"]), set()), axis=1)]
    # Vectorised lifecycle checks: an O(number_of_orders * log_rows) loop makes
    # a full-day audit take tens of minutes and unnecessarily retains large
    # temporary frames in RAM.
    transition_status = transitions["new_status"].astype(str)
    row_number = transitions.groupby("order_id", sort=False).cumcount()
    last_row = transitions.groupby("order_id", sort=False)["order_id"].transform("size") - 1
    terminal_reentry = int((transition_status.isin(["COMPLETED", "CANCELLED"]) & row_number.ne(last_row)).sum())
    status_counts = transitions.assign(_status=transition_status).groupby(
        ["order_id", "_status"], sort=False
    ).size()
    completed_counts = status_counts[status_counts.index.get_level_values(1) == "COMPLETED"]
    cancelled_counts = status_counts[status_counts.index.get_level_values(1) == "CANCELLED"]
    duplicate_completed = int((completed_counts - 1).clip(lower=0).sum())
    duplicate_cancelled = int((cancelled_counts - 1).clip(lower=0).sum())
    pickup_seen = transition_status.eq("PICKUP_STARTED").groupby(transitions["order_id"], sort=False).cumsum()
    boarded_seen = transition_status.eq("BOARDED").groupby(transitions["order_id"], sort=False).cumsum()
    missing_prereq = int(
        (transition_status.eq("BOARDED") & pickup_seen.eq(0)).sum()
        + (transition_status.eq("IN_SERVICE") & boarded_seen.eq(0)).sum()
    )
    transition_audit = {
        "transition_rows": int(len(transitions)),
        "illegal_transition_count": int(len(illegal)),
        "terminal_reentry_count": int(terminal_reentry),
        "duplicate_completed_transition_count": int(duplicate_completed),
        "duplicate_cancelled_transition_count": int(duplicate_cancelled),
        "missing_prerequisite_transition_count": int(missing_prereq),
    }
    transition_audit["request_transition_pass"] = status(all(v == 0 for k, v in transition_audit.items() if k.endswith("_count")))

    service_like = legs[legs["leg_type"].isin(["PICKUP", "SERVICE"])] if "leg_type" in legs.columns else legs
    leg_columns = {"leg_id", "vehicle_id", "distance_m", "realized_time_sec", "start_time", "end_time"}
    leg_evidence_present = len(legs) > 0 and leg_columns.issubset(legs.columns)
    leg_audit = {
        "vehicle_leg_count": int(len(legs)),
        "negative_distance_count": int((pd.to_numeric(legs["distance_m"], errors="coerce") < 0).sum()) if len(legs) else 0,
        "negative_time_count": int((pd.to_numeric(legs["realized_time_sec"], errors="coerce") < 0).sum()) if len(legs) else 0,
        "leg_evidence_present_pass": status(leg_evidence_present),
        "nonnegative_leg_metrics_pass": status(
            leg_evidence_present
            and pd.to_numeric(legs["distance_m"], errors="coerce").notna().all()
            and pd.to_numeric(legs["realized_time_sec"], errors="coerce").notna().all()
            and (pd.to_numeric(legs["distance_m"], errors="coerce") >= 0).all()
            and (pd.to_numeric(legs["realized_time_sec"], errors="coerce") >= 0).all()
        ),
        "completed_order_leg_pair_pass": status(len(service_like) == 2 * request_audit["completed_orders"]),
    }
    if len(legs):
        work = legs.copy()
        work["start_time_dt"] = pd.to_datetime(work["start_time"], utc=True, errors="coerce")
        work["end_time_dt"] = pd.to_datetime(work["end_time"], utc=True, errors="coerce")
        overlap = 0
        spatial_break = 0
        duplicate_leg = int(work["leg_id"].duplicated().sum())
        for _, group in work.sort_values(["vehicle_id", "start_time_dt"]).groupby("vehicle_id", sort=False):
            prev = None
            for row in group.itertuples(index=False):
                if prev is not None:
                    if row.start_time_dt < prev.end_time_dt - pd.Timedelta(seconds=1):
                        overlap += 1
                    if abs(float(row.start_lon) - float(prev.end_lon)) > 1e-6 or abs(float(row.start_lat) - float(prev.end_lat)) > 1e-6:
                        spatial_break += 1
                prev = row
        leg_audit.update({
            "leg_overlap_count": int(overlap),
            "leg_spatial_break_count": int(spatial_break),
            "duplicate_leg_completion_count": duplicate_leg,
            "vehicle_leg_continuity_pass": status(overlap == 0 and spatial_break == 0 and duplicate_leg == 0),
        })

    completed_requests = req[req["final_status"].astype(str).eq("COMPLETED")]
    completed_index = completed_requests[["order_id", "assigned_vehicle_id"]].astype(str).set_index("order_id")
    if len(service_like):
        leg_work = service_like.assign(
            request_id=service_like["request_id"].astype(str),
            vehicle_id=service_like["vehicle_id"].astype(str),
            _pickup=service_like["leg_type"].astype(str).eq("PICKUP").astype(int),
            _service=service_like["leg_type"].astype(str).eq("SERVICE").astype(int),
        )
        leg_summary = leg_work.groupby("request_id", sort=False).agg(
            leg_count=("request_id", "size"), pickup_count=("_pickup", "sum"),
            service_count=("_service", "sum"), vehicle_count=("vehicle_id", "nunique"),
            vehicle_id=("vehicle_id", "first"),
        )
        joined_leg = completed_index.join(leg_summary, how="left")
        completed_leg_pair_mismatch = int((
            joined_leg["leg_count"].fillna(0).ne(2)
            | joined_leg["pickup_count"].fillna(0).ne(1)
            | joined_leg["service_count"].fillna(0).ne(1)
            | joined_leg["vehicle_count"].fillna(0).ne(1)
            | joined_leg["vehicle_id"].fillna("").ne(joined_leg["assigned_vehicle_id"])
        ).sum())
    else:
        completed_leg_pair_mismatch = int(len(completed_requests))
    if len(plans):
        plan_pairs = pd.MultiIndex.from_frame(pd.DataFrame({
            "vehicle_id": plans["vehicle_id"].astype(str),
            "order_id": plans["added_request_id"].astype(str),
        }).drop_duplicates())
        completed_pairs = pd.MultiIndex.from_frame(pd.DataFrame({
            "vehicle_id": completed_requests["assigned_vehicle_id"].astype(str),
            "order_id": completed_requests["order_id"].astype(str),
        }))
        plan_request_evidence_missing = int((~completed_pairs.isin(plan_pairs)).sum())
    else:
        plan_request_evidence_missing = int(len(completed_requests))
    plan_audit = {
        "plan_revision_count": int(len(plans)),
        "completed_request_leg_pair_mismatch_count": completed_leg_pair_mismatch,
        "completed_request_plan_evidence_missing_count": plan_request_evidence_missing,
        "one_plan_per_completed_order_pass": status(plan_request_evidence_missing == 0),
        "completed_legs_recorded": int(len(legs)),
        "plan_and_leg_evidence_pass": status(
            completed_leg_pair_mismatch == 0
            and plan_request_evidence_missing == 0
        ),
    }

    pickup_sources = legs.loc[legs["leg_type"].eq("PICKUP"), "route_source"].astype(str) if len(legs) else pd.Series(dtype=str)
    service_sources = legs.loc[legs["leg_type"].eq("SERVICE"), "route_source"].astype(str) if len(legs) else pd.Series(dtype=str)
    truncation = pd.to_numeric(epochs.get("candidate_truncation_rate", pd.Series(dtype=float)), errors="coerce")
    routing_audit = {
        "routing_query_count": int(summary.get("routing_query_count", 0)),
        "routing_cache_hit_count": int(summary.get("routing_cache_hit_count", 0)),
        "routing_cache_hit_rate": float(summary.get("routing_cache_hit_rate", 0.0)),
        "pickup_leg_count": int(len(pickup_sources)),
        "service_leg_count": int(len(service_sources)),
        "fixed_haversine_8mps_source_count": int(pickup_sources.str.contains("fixed_haversine_8mps", case=False).sum()),
        "service_using_pickup_backend_count": int(service_sources.str.contains("empty_speed|pickup", case=False, regex=True).sum()),
        "mean_candidate_truncation_rate": float(truncation.mean()) if len(truncation) else 0.0,
        "p95_candidate_truncation_rate": float(truncation.quantile(0.95)) if len(truncation) else 0.0,
        "orders_hitting_candidate_cap": int(pd.to_numeric(epochs.get("orders_hitting_candidate_cap", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "peak_candidate_edge_count": int(pd.to_numeric(epochs.get("candidate_plans", pd.Series(dtype=float)), errors="coerce").fillna(0).max()) if len(epochs) else 0,
        "maximum_matching_runtime_sec": float(pd.to_numeric(epochs.get("matching_runtime", pd.Series(dtype=float)), errors="coerce").fillna(0).max()) if len(epochs) else 0.0,
    }
    sensitivity_path = Path("stage4/docs/results/simulator_v3/candidate_sensitivity_summary.csv")
    sensitivity_pass = False
    sensitivity_max_difference = None
    if sensitivity_path.exists():
        sensitivity = pd.read_csv(sensitivity_path)
        required_caps = {40, 80, 120}
        if required_caps.issubset(set(pd.to_numeric(sensitivity["candidate_maximum"], errors="coerce").dropna().astype(int))):
            metric_differences = []
            for metric in ["match_rate", "av_assignment_share"]:
                values = pd.to_numeric(sensitivity[metric], errors="coerce").dropna()
                denominator = max(float(values.abs().max()), 1e-12)
                metric_differences.append(float((values.max() - values.min()) / denominator))
            sensitivity_max_difference = max(metric_differences)
            sensitivity_pass = sensitivity_max_difference <= 0.02
    routing_audit.update({
        "candidate_sensitivity_evidence_path": str(sensitivity_path),
        "candidate_sensitivity_max_relative_difference": sensitivity_max_difference,
        "candidate_sensitivity_pass": status(sensitivity_pass),
    })
    routing_audit["pickup_service_backend_separation_pass"] = status(
        routing_audit["pickup_leg_count"] == request_audit["completed_orders"]
        and routing_audit["service_leg_count"] == request_audit["completed_orders"]
        and routing_audit["fixed_haversine_8mps_source_count"] == 0
        and routing_audit["service_using_pickup_backend_count"] == 0
    )
    routing_audit["candidate_truncation_pass"] = status(
        (
            routing_audit["mean_candidate_truncation_rate"] <= 0.20
            and routing_audit["p95_candidate_truncation_rate"] <= 0.40
        )
        or sensitivity_pass
    )

    event_order = events.copy()
    event_order["event_time_dt"] = pd.to_datetime(event_order["event_time"], utc=True, errors="coerce")
    monotonic = event_order["event_time_dt"].is_monotonic_increasing
    priority_bad = int((
        event_order.groupby("event_time", sort=False)["event_priority"].diff().fillna(0) < 0
    ).groupby(event_order["event_time"], sort=False).any().sum())
    payload_evidence = "event_payload" in events.columns
    completed_leg_ids: list[str] = []
    if payload_evidence:
        for payload in events.loc[events["event_type"].astype(str).eq("LEG_COMPLETED"), "event_payload"]:
            try:
                completed_leg_ids.append(str(json.loads(str(payload)).get("leg_id", "")))
            except (TypeError, ValueError, json.JSONDecodeError):
                completed_leg_ids.append("")
    duplicate_leg_event_count = (
        int(pd.Series(completed_leg_ids).duplicated().sum()) if completed_leg_ids else 0
    )
    session_events = events[events["event_type"].astype(str).isin(["HV_SESSION_START", "HV_SESSION_END"])]
    duplicate_session_event_count = int(
        session_events.duplicated(subset=["event_type", "entity_id"]).sum()
    )
    event_audit = {
        "event_rows": int(len(events)),
        "event_time_monotonic_pass": status(monotonic),
        "same_time_priority_violation_count": int(priority_bad),
        "unhandled_event_count": int((~events["handled"].astype(bool)).sum()),
        "duplicate_request_reveal_count": int(events[events["event_type"].eq("REQUEST_REVEALED")]["entity_id"].duplicated().sum()),
        "event_payload_evidence_pass": status(payload_evidence),
        "duplicate_leg_completed_event_count": duplicate_leg_event_count,
        "duplicate_session_event_count": duplicate_session_event_count,
        "event_audit_pass": status(
            monotonic
            and priority_bad == 0
            and int((~events["handled"].astype(bool)).sum()) == 0
            and payload_evidence
            and duplicate_leg_event_count == 0
            and duplicate_session_event_count == 0
        ),
    }

    session_required = {"vehicle_id", "online_start", "online_end"}
    session_evidence = len(vehicle_states) > 0 and session_required.issubset(vehicle_states.columns)
    if session_evidence:
        sessions = vehicle_states[["vehicle_id", "online_start", "online_end"]].copy()
        sessions["online_start_dt"] = pd.to_datetime(sessions["online_start"], utc=True, errors="coerce")
        sessions["online_end_dt"] = pd.to_datetime(sessions["online_end"], utc=True, errors="coerce")
        leg_session = work.merge(sessions, on="vehicle_id", how="left") if len(legs) else pd.DataFrame()
        unknown_leg_vehicle = int(leg_session["online_start_dt"].isna().sum()) if len(leg_session) else 0
        before_online = int((leg_session["start_time_dt"] < leg_session["online_start_dt"]).sum()) if len(leg_session) else 0
        after_online = int((leg_session["start_time_dt"] > leg_session["online_end_dt"]).sum()) if len(leg_session) else 0
        invalid_window = int((sessions["online_start_dt"] >= sessions["online_end_dt"]).sum())
    else:
        unknown_leg_vehicle = before_online = after_online = invalid_window = -1
    session_audit = {
        "vehicle_state_log_present_pass": status(session_evidence),
        "invalid_online_window_count": invalid_window,
        "unknown_leg_vehicle_count": unknown_leg_vehicle,
        "leg_started_before_online_count": before_online,
        "leg_started_after_online_end_count": after_online,
        "session_end_behavior_pass": status(
            session_evidence
            and invalid_window == 0
            and unknown_leg_vehicle == 0
            and before_online == 0
            and after_online == 0
        ),
    }

    index_columns = {"full_vehicle_scan_count", "decision_epoch_full_scan_count"}
    index_evidence_present = len(epochs) > 0 and index_columns.issubset(epochs.columns)
    full_scan = pd.to_numeric(epochs["full_vehicle_scan_count"], errors="coerce") if index_evidence_present else pd.Series(dtype=float)
    decision_scan = pd.to_numeric(epochs["decision_epoch_full_scan_count"], errors="coerce") if index_evidence_present else pd.Series(dtype=float)
    state_index_audit = {
        "state_index_evidence_present_pass": status(
            index_evidence_present and full_scan.notna().all() and decision_scan.notna().all()
        ),
        "full_vehicle_scan_count": int(full_scan.max()) if len(full_scan) else -1,
        "decision_epoch_full_scan_count": int(decision_scan.max()) if len(decision_scan) else -1,
    }
    state_index_audit["incremental_state_index_pass"] = status(
        index_evidence_present
        and full_scan.notna().all()
        and decision_scan.notna().all()
        and state_index_audit["decision_epoch_full_scan_count"] == 0
    )

    odd_required_columns = {
        "condition_available",
        "assigned_vehicle_id",
        "assigned_vehicle_type",
        "pickup_odd_feasible",
        "service_odd_feasible",
        "combined_odd_feasible",
        "capability_profile",
        "capability_mapping_version",
    }
    odd_missing_columns = sorted(odd_required_columns.difference(req.columns))
    if odd_missing_columns:
        unknown_av_assignments = pickup_violations = service_violations = combined_violations = 0
        assigned_av_count = 0
        missing_profile_count = missing_mapping_version_count = 0
    else:
        assigned = req["assigned_vehicle_id"].fillna("").astype(str).ne("")
        assigned_av = assigned & (
            req["assigned_vehicle_type"].fillna("").astype(str).eq("AV")
            | req["assigned_vehicle_id"].fillna("").astype(str).str.startswith("AV_")
        )
        assigned_av_count = int(assigned_av.sum())
        unknown_av_assignments = int((assigned_av & ~req["condition_available"].fillna(False).astype(bool)).sum())
        pickup_violations = int((assigned_av & ~req["pickup_odd_feasible"].fillna(False).astype(bool)).sum())
        service_violations = int((assigned_av & ~req["service_odd_feasible"].fillna(False).astype(bool)).sum())
        combined_violations = int((assigned_av & ~req["combined_odd_feasible"].fillna(False).astype(bool)).sum())
        missing_profile_count = int((assigned_av & req["capability_profile"].fillna("").astype(str).eq("")).sum())
        missing_mapping_version_count = int((assigned_av & req["capability_mapping_version"].fillna("").astype(str).eq("")).sum())
    odd_fields_present = not odd_missing_columns
    odd_audit = {
        "odd_log_required_columns": sorted(odd_required_columns),
        "odd_log_missing_columns": odd_missing_columns,
        "odd_log_fields_present_pass": status(odd_fields_present),
        "assigned_av_count": assigned_av_count,
        "unknown_condition_av_assignment_count": unknown_av_assignments,
        "pickup_odd_violation_count": pickup_violations,
        "service_odd_violation_count": service_violations,
        "combined_odd_violation_count": combined_violations,
        "assigned_av_missing_capability_profile_count": missing_profile_count,
        "assigned_av_missing_mapping_version_count": missing_mapping_version_count,
        "unknown_condition_av_assignment_pass": status(odd_fields_present and unknown_av_assignments == 0),
        "pickup_odd_violation_pass": status(odd_fields_present and pickup_violations == 0),
        "service_odd_violation_pass": status(odd_fields_present and service_violations == 0),
        "combined_odd_violation_pass": status(odd_fields_present and combined_violations == 0),
        "odd_provenance_complete_pass": status(
            odd_fields_present and missing_profile_count == 0 and missing_mapping_version_count == 0
        ),
    }
    odd_audit["odd_audit_pass"] = status(
        odd_fields_present
        and unknown_av_assignments == 0
        and pickup_violations == 0
        and service_violations == 0
        and combined_violations == 0
        and missing_profile_count == 0
        and missing_mapping_version_count == 0
    )

    reserved_count = int(transitions["new_status"].astype(str).eq("RESERVED").sum())
    preassignment_enabled = str(summary.get("operation", "")).upper() in {"O2", "O3"} or reserved_count > 0
    reserved_return_pending = int(((transitions["old_status"].astype(str) == "RESERVED") & (transitions["new_status"].astype(str) == "PENDING")).sum())
    reservation_path = args.run_dir / "reservation_log.parquet"
    failure_path = args.run_dir / "preassignment_failure_log.parquet"
    reservation_log = pd.read_parquet(reservation_path) if reservation_path.exists() else pd.DataFrame()
    failure_log = pd.read_parquet(failure_path) if failure_path.exists() else pd.DataFrame()
    if preassignment_enabled and len(reservation_log):
        buffer_ok = (
            reservation_log["buffer_source"].fillna("").astype(str).str.startswith("validation_q0.90:").all()
            and pd.to_numeric(reservation_log["buffer_sample_count"], errors="coerce").gt(0).all()
            and pd.to_numeric(reservation_log["buffer_quantile"], errors="coerce").sub(0.9).abs().le(1e-9).all()
        )
    else:
        buffer_ok = not preassignment_enabled
    terminal_statuses = {"COMPLETED", "EXPIRED", "INVALIDATED", "RELEASED"}
    terminal_record_count = int(
        reservation_log.get("status", pd.Series(dtype=str)).astype(str).isin(terminal_statuses).sum()
    )
    active_record_count = int(
        reservation_log.get("status", pd.Series(dtype=str)).astype(str).eq("ACTIVE").sum()
    )
    reservation_cycle_ok = (
        (not preassignment_enabled)
        or (
            len(reservation_log) == reserved_count
            and terminal_record_count == reserved_count
            and active_record_count == 0
        )
    )
    def nearest_transition_matches(
        left: pd.DataFrame, left_time: str, right: pd.DataFrame, right_time: str
    ) -> int:
        if left.empty or right.empty:
            return 0
        lhs = pd.DataFrame({
            "request_id": left["request_id"].astype(str),
            "event_time": pd.to_datetime(left[left_time], utc=True, errors="coerce"),
        }).dropna().sort_values(["event_time", "request_id"])
        rhs = pd.DataFrame({
            "request_id": right["order_id"].astype(str),
            "transition_time_match": pd.to_datetime(right[right_time], utc=True, errors="coerce"),
        }).dropna().sort_values(["transition_time_match", "request_id"])
        if lhs.empty or rhs.empty:
            return 0
        matched = pd.merge_asof(
            lhs, rhs, by="request_id", left_on="event_time", right_on="transition_time_match",
            direction="nearest", tolerance=pd.Timedelta(seconds=1),
        )
        return int(matched["transition_time_match"].notna().sum())

    failure_transitions = transitions[
        transitions["old_status"].astype(str).eq("RESERVED")
        & transitions["new_status"].astype(str).isin(["PENDING", "CANCELLED"])
    ]
    failure_release_matches = nearest_transition_matches(
        failure_log, "failure_time", failure_transitions, "transition_time"
    )
    completed_records = reservation_log[
        reservation_log.get("status", pd.Series(dtype=str)).astype(str).eq("COMPLETED")
    ] if len(reservation_log) else reservation_log
    promotion_transitions = transitions[
        transitions["old_status"].astype(str).eq("RESERVED")
        & transitions["new_status"].astype(str).eq("ASSIGNED")
    ]
    completed_promotion_matches = nearest_transition_matches(
        completed_records, "close_time", promotion_transitions, "transition_time"
    )
    failure_release_ok = (
        (not preassignment_enabled)
        or failure_release_matches == len(failure_log)
    )
    completed_promotion_ok = (
        (not preassignment_enabled)
        or completed_promotion_matches == len(completed_records)
    )
    # Synthetic decoupled HV identifiers use the ``DHV_`` prefix; vehicle
    # type is therefore determined by excluding the canonical AV prefix.
    hv_offer = offers[~offers["vehicle_id"].astype(str).str.startswith("AV_")] if len(offers) else offers
    assigned_hv_count = int(
        (req["final_status"].astype(str).eq("COMPLETED")
        & req["assigned_vehicle_type"].fillna("").astype(str).eq("HV")).sum()
    ) if "assigned_vehicle_type" in req.columns else 0
    driver_response_event_count = int(events["event_type"].astype(str).eq("DRIVER_RESPONSE").sum())
    hv_delay_ok = (
        (not len(hv_offer) and assigned_hv_count == 0)
        or (
            len(hv_offer) > 0
            and driver_response_event_count >= len(hv_offer)
            and (pd.to_datetime(hv_offer["response_time"], utc=True, errors="coerce")
                 > pd.to_datetime(hv_offer["offer_time"], utc=True, errors="coerce")).all()
        )
    )
    preassignment_audit = {
        "preassignment_enabled": bool(preassignment_enabled),
        "reserved_transition_count": reserved_count,
        "reserved_return_pending_count": reserved_return_pending,
        "reservation_record_count": int(len(reservation_log)),
        "terminal_reservation_record_count": terminal_record_count,
        "active_reservation_record_count": active_record_count,
        "completed_reservation_promotion_matches": completed_promotion_matches,
        "current_reserved_dual_layer_pass": status(reservation_cycle_ok and completed_promotion_ok),
        "reservation_failure_rows": int(len(failure_log)),
        "reservation_failure_transition_matches": failure_release_matches,
        "reservation_invalidation_pass": status(failure_release_ok),
        "q09_buffer_source_recorded_pass": status(buffer_ok),
        "assigned_hv_count": assigned_hv_count,
        "hv_offer_count": int(len(hv_offer)),
        "driver_response_event_count": driver_response_event_count,
        "hv_response_event_delay_pass": status(hv_delay_ok),
        "preassignment_audit_pass": status(
            reservation_cycle_ok and completed_promotion_ok
            and failure_release_ok and buffer_ok and hv_delay_ok
        ),
    }

    ledger_path = args.run_dir / "economy_ledger.csv"
    ledger = pd.read_csv(ledger_path) if ledger_path.exists() else pd.DataFrame()
    economy_audit = audit_ledger(ledger)
    economy_audit["ledger_file_present"] = bool(ledger_path.exists())
    economy_audit["fixed_cost_in_served_edge_rows"] = float(
        ledger.loc[ledger.get("ledger_type", pd.Series(dtype=str)).eq("served_order"), [
            "av_fixed_cost", "depot_cost", "lost_demand_cost", "preassignment_failure_cost"
        ]].sum().sum()
    ) if len(ledger) else 0.0
    economy_audit["fixed_cost_excluded_from_edge_objective_pass"] = status(
        ledger_path.exists() and economy_audit["fixed_cost_in_served_edge_rows"] == 0.0
    )
    economy_audit["economy_ledger_pass"] = status(
        ledger_path.exists() and economy_audit.get("economy_audit_status") == "PASS"
    )

    operation = str(summary.get("operation", "")).upper()
    idle_enabled = operation in {"O1", "O3"}
    idle_path = args.run_dir / "idle_movement_log.parquet"
    idle_log = pd.read_parquet(idle_path) if idle_path.exists() else pd.DataFrame()
    movement_legs = legs[
        legs.get("leg_type", pd.Series(dtype=str)).astype(str).isin(["HV_REPOSITION", "AV_REBALANCE"])
    ] if len(legs) else legs
    planned_idle = idle_log[
        idle_log.get("plan_created", pd.Series(dtype=bool)).fillna(False).astype(bool)
    ] if len(idle_log) else idle_log
    hv_idle = idle_log[idle_log.get("vehicle_type", pd.Series(dtype=str)).astype(str).eq("HV")] if len(idle_log) else idle_log
    av_idle = idle_log[idle_log.get("vehicle_type", pd.Series(dtype=str)).astype(str).eq("AV")] if len(idle_log) else idle_log
    idle_required_columns = {
        "vehicle_id", "vehicle_type", "movement_time", "origin_zone", "target_zone",
        "movement_reason", "plan_created", "policy_source",
    }
    idle_schema_ok = idle_required_columns.issubset(idle_log.columns)
    prior_path = REPO_ROOT / "stage4" / "data" / "decoupled_abm" / "idle_management_prior_audit.json"
    prior_doc = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else {}
    prior_no_future = (
        prior_doc.get("hv_transition", {}).get("uses_test_day_future_demand") is False
        and prior_doc.get("av_demand_prior", {}).get("uses_test_day_future_demand") is False
    )
    hv_source_ok = (
        not len(hv_idle)
        or (
            hv_idle["movement_reason"].astype(str).str.startswith("empirical_idle_repositioning").all()
            and hv_idle["policy_source"].fillna("").astype(str).ne("").all()
            and pd.to_numeric(hv_idle.get("training_sample_count"), errors="coerce").fillna(0).gt(0).all()
        )
    ) if idle_schema_ok else False
    av_source_ok = (
        not len(av_idle)
        or (
            av_idle["movement_reason"].astype(str).eq("platform_training_demand_rebalancing").all()
            and av_idle["policy_source"].fillna("").astype(str).isin(["time_bin", "global_time_fallback"]).all()
            and av_idle.get("solver", pd.Series("", index=av_idle.index)).fillna("").astype(str).ne("").all()
        )
    ) if idle_schema_ok else False
    scenario_row = ledger[ledger.get("ledger_type", pd.Series(dtype=str)).astype(str).eq("scenario_costs")] if len(ledger) else ledger
    hv_move_cost = float(pd.to_numeric(scenario_row.get("hv_repositioning_cost", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    av_move_cost = float(pd.to_numeric(scenario_row.get("av_rebalancing_cost", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    idle_evidence_ok = (
        idle_schema_ok
        and len(idle_log) > 0
        and len(planned_idle) == len(movement_legs)
        and prior_no_future
        and hv_source_ok
        and av_source_ok
        and (not (movement_legs["leg_type"].astype(str).eq("HV_REPOSITION").any()) or hv_move_cost > 0)
        and (not (movement_legs["leg_type"].astype(str).eq("AV_REBALANCE").any()) or av_move_cost > 0)
    ) if idle_enabled else (
        len(movement_legs) == 0 and not idle_path.exists()
    )
    idle_audit = {
        "idle_management_enabled": idle_enabled,
        "idle_log_rows": int(len(idle_log)),
        "planned_idle_movement_count": int(len(planned_idle)),
        "executed_idle_movement_leg_count": int(len(movement_legs)),
        "hv_idle_record_count": int(len(hv_idle)),
        "av_idle_record_count": int(len(av_idle)),
        "hv_repositioning_cost": hv_move_cost,
        "av_rebalancing_cost": av_move_cost,
        "training_prior_no_test_day_future_pass": status(prior_no_future if idle_enabled else True),
        "hv_training_transition_source_pass": status(hv_source_ok if idle_enabled else len(hv_idle) == 0),
        "av_shortage_optimization_source_pass": status(av_source_ok if idle_enabled else len(av_idle) == 0),
        "idle_plan_leg_cost_evidence_pass": status(idle_evidence_ok),
        "idle_management_audit_pass": status(idle_evidence_ok),
    }

    balanced_enabled = str(summary.get("strategy", "")) == "Three-Stakeholder Balanced"
    balanced_required = {
        "stress_constraint_active", "zone_service_constraint_active",
        "balanced_constraint_table_hash", "remaining_stress_budget_total",
        "minimum_zone_service_target_total", "balanced_constraint_source",
        "balanced_price_aware_equivalent", "balanced_edge_objective",
        "balanced_constraint_model",
    }
    balanced_schema_ok = balanced_required.issubset(epochs.columns)
    if balanced_schema_ok:
        active_epochs = epochs[
            epochs["stress_constraint_active"].fillna(False).astype(bool)
            | epochs["zone_service_constraint_active"].fillna(False).astype(bool)
        ]
        binding_count = int((
            epochs.get("stress_constraint_binding", pd.Series(False, index=epochs.index)).fillna(False).astype(bool)
            | epochs.get("zone_service_constraint_binding", pd.Series(False, index=epochs.index)).fillna(False).astype(bool)
        ).sum())
        table_evidence_ok = (
            len(active_epochs) > 0
            and active_epochs["balanced_constraint_table_hash"].fillna("").astype(str).ne("").all()
            and active_epochs["balanced_constraint_source"].fillna("").astype(str).ne("").all()
            and pd.to_numeric(active_epochs["remaining_stress_budget_total"], errors="coerce").ge(0).all()
            and pd.to_numeric(active_epochs["minimum_zone_service_target_total"], errors="coerce").ge(0).all()
        )
        differentiated = (
            active_epochs["balanced_price_aware_equivalent"].fillna(True).eq(False).all()
            and active_epochs["balanced_constraint_model"].fillna("").astype(str).str.contains("zone_time_hv_stress").all()
            and active_epochs["balanced_edge_objective"].fillna("").astype(str).ne("").all()
        )
    else:
        active_epochs = pd.DataFrame()
        binding_count = 0
        table_evidence_ok = False
        differentiated = False
    balanced_pass = (
        balanced_schema_ok and table_evidence_ok and differentiated and binding_count > 0
    ) if balanced_enabled else (
        not epochs.get("stress_constraint_active", pd.Series(False, index=epochs.index)).fillna(False).astype(bool).any()
        and not epochs.get("zone_service_constraint_active", pd.Series(False, index=epochs.index)).fillna(False).astype(bool).any()
    )
    balanced_audit = {
        "balanced_enabled": balanced_enabled,
        "balanced_constraint_schema_pass": status(balanced_schema_ok if balanced_enabled else True),
        "balanced_active_epoch_count": int(len(active_epochs)),
        "balanced_binding_epoch_count": binding_count,
        "balanced_constraint_table_evidence_pass": status(table_evidence_ok if balanced_enabled else True),
        "balanced_price_aware_differentiation_pass": status(differentiated if balanced_enabled else True),
        "balanced_audit_pass": status(balanced_pass),
    }

    overall = {
        "request_lifecycle": request_audit,
        "request_transitions": transition_audit,
        "vehicle_leg": leg_audit,
        "plan_execution_separation": plan_audit,
        "event": event_audit,
        "session": session_audit,
        "state_index": state_index_audit,
        "routing": routing_audit,
        "odd": odd_audit,
        "preassignment": preassignment_audit,
        "idle_management": idle_audit,
        "balanced": balanced_audit,
        "economy": economy_audit,
        "summary": summary,
    }
    all_pass = []
    for section in [request_audit, transition_audit, leg_audit, plan_audit, event_audit, session_audit, state_index_audit, routing_audit, odd_audit, preassignment_audit, idle_audit, balanced_audit, economy_audit]:
        all_pass.extend(v for k, v in section.items() if k.endswith("_pass"))
    overall["overall_phase1_status"] = status(all(v == "PASS" for v in all_pass))

    (args.results_dir / "simulator_v3_request_lifecycle_audit.json").write_text(json.dumps(request_audit, indent=2), encoding="utf-8")
    (args.results_dir / "simulator_v3_request_transition_audit.json").write_text(json.dumps(transition_audit, indent=2), encoding="utf-8")
    (args.results_dir / "simulator_v3_vehicle_leg_continuity_audit.json").write_text(json.dumps(leg_audit, indent=2), encoding="utf-8")
    (args.results_dir / "simulator_v3_event_audit.json").write_text(json.dumps(event_audit, indent=2), encoding="utf-8")
    (args.results_dir / "simulator_v3_session_audit.json").write_text(json.dumps(session_audit, indent=2), encoding="utf-8")
    (args.results_dir / "simulator_v3_state_index_audit.json").write_text(json.dumps(state_index_audit, indent=2), encoding="utf-8")
    (args.results_dir / "simulator_v3_state_audit.json").write_text(json.dumps(plan_audit | leg_audit, indent=2), encoding="utf-8")
    (args.results_dir / "simulator_v3_preassignment_audit.json").write_text(json.dumps(preassignment_audit, indent=2), encoding="utf-8")
    (args.results_dir / "simulator_v3_idle_management_audit.json").write_text(json.dumps(idle_audit, indent=2), encoding="utf-8")
    (args.results_dir / "simulator_v3_balanced_audit.json").write_text(json.dumps(balanced_audit, indent=2), encoding="utf-8")
    (args.results_dir / "simulator_v3_routing_audit.json").write_text(json.dumps(routing_audit, indent=2), encoding="utf-8")
    (args.results_dir / "simulator_v3_odd_audit.json").write_text(json.dumps(odd_audit, indent=2), encoding="utf-8")
    (args.results_dir / "simulator_v3_economy_audit.json").write_text(json.dumps(economy_audit, indent=2), encoding="utf-8")
    (args.results_dir / "simulator_v3_audit_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    (args.run_dir / "audit_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    (args.run_dir / "full_day_audit.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    (args.run_dir / "odd_audit.json").write_text(json.dumps(odd_audit, indent=2), encoding="utf-8")
    (args.run_dir / "economy_audit.json").write_text(json.dumps(economy_audit, indent=2), encoding="utf-8")
    (args.run_dir / "state_event_leg_audit.json").write_text(json.dumps({
        "state_index": state_index_audit,
        "event": event_audit,
        "session": session_audit,
        "vehicle_leg": leg_audit,
    }, indent=2), encoding="utf-8")
    backend_usage = (
        legs.groupby(["leg_type", "route_source"], dropna=False).size()
        .rename("leg_count").reset_index()
    )
    backend_usage.to_csv(args.run_dir / "routing_backend_usage_summary.csv", index=False)
    backend_usage.to_csv(args.results_dir / "routing_backend_usage_summary.csv", index=False)
    pd.DataFrame([summary]).to_csv(args.results_dir / "simulator_v3_operation_summary.csv", index=False)
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
