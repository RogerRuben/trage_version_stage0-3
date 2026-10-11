"""Read completed v4 products; no simulation, inference, routing or tuning.

Calendar-day admission time is partitioned by recorded physical events. The
residual is simulated unoccupied admission time, not observed commercial idle.
Only aggregates are exported; identifiers and coordinates remain in memory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from math import floor, isfinite
from pathlib import Path
import re
from time import perf_counter

import pandas as pd
import psutil

from stage4.analysis.capability_chain_instances import atomic_json
from stage4.dispatch.controlled_fleet import build_controlled_fleet, FLEET_REL

OUT = Path("stage4/output/capability_chain_planning/two_scale_city_v4")
DOC = Path("stage4/docs/capability_chain_planning/unified_operations_v1")
ORDER_BASE = Path("stage4/input/replay_foundation/stage4_order_replay_base.parquet")
GROUPS = ("HOTSPOT_BASELINE", "CHAIN_LAYOUT_BASELINE", "CHAIN_LAYOUT_JOINT")
START = pd.Timestamp("2016-10-31T00:00:00+08:00")
DAY_S = 86400.0
TOL = 1e-6
KINDS = ("PICKUP", "LOADED", "MOVE")
GRID_DEGREES = 0.02
MIN_ZONE_REQUESTS = 30
MAX_RSS_MIB = 512


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def overlap(a, b, left, right):
    return max(0.0, min(float(b), float(right)) - max(float(a), float(left)))


def clipped_ledger(activities, admission_start, admission_end, left=0., right=DAY_S):
    """Disjoint recorded activities plus their admission-window complement."""
    if not all(isfinite(float(v)) for v in (admission_start, admission_end, left, right)):
        raise ValueError("nonfinite ledger boundary")
    if admission_end <= admission_start or right < left:
        raise ValueError("invalid admission or reporting window")
    lo, hi = max(float(left), float(admission_start)), min(float(right), float(admission_end))
    totals = dict.fromkeys(KINDS, 0.0)
    if hi <= lo:
        return dict(admission_s=0., **totals, idle_s=0., longest_idle_s=0., idle_intervals=0)
    clipped = []
    for begin, end, kind in activities:
        if kind not in KINDS or not all(isfinite(float(x)) for x in (begin, end)) or end < begin:
            raise ValueError("invalid physical activity")
        a, b = max(float(begin), lo), min(float(end), hi)
        if b > a:
            clipped.append((a, b, kind))
    cursor, longest, gaps = lo, 0., 0
    idle = 0.
    for a, b, kind in sorted(clipped):
        if a < cursor - TOL:
            raise ValueError("overlapping customer/movement activity in the same slot")
        if a > cursor:
            gap = a - cursor
            idle += gap
            longest = max(longest, gap)
            gaps += int(gap > TOL)
        # Treat sub-microsecond callback boundary differences as roundoff only.
        a = max(a, cursor)
        totals[kind] += max(0., b - a)
        cursor = max(cursor, b)
    if hi > cursor:
        idle += hi - cursor
        longest = max(longest, hi - cursor)
        gaps += int(hi - cursor > TOL)
    error = abs(hi - lo - idle - sum(totals.values()))
    if error > TOL:
        raise ValueError("admission-time partition does not conserve time")
    return dict(admission_s=hi - lo, **totals, idle_s=idle,
                longest_idle_s=longest, idle_intervals=gaps)


def request_partition(frame):
    """Observed nearby AV opportunities do not certify a legal AV candidate."""
    eligible = frame.C_compatible & frame.passenger_accepts_av
    hv = frame.loc[eligible & frame.completed & frame.vehicle_type.eq("HV")]
    opportunities = hv.spatial_av_opportunities
    if opportunities.isna().any() or not opportunities.ge(0).all():
        raise ValueError("missing/invalid nearby-C diagnostics")
    attempts, requests = Counter(), Counter()
    for value in hv.rejection_counts_json:
        reasons = json.loads(value)
        record_reasons = Counter()
        for code, count in reasons.items():
            if not isinstance(count, int) or count < 0:
                raise ValueError("invalid diagnostic rejection counter")
            public_code = code if re.fullmatch(r"[A-Z][A-Z0-9_:-]{0,100}", code) else "OTHER_TEXT_REASON"
            record_reasons[public_code] += count
        attempts.update(record_reasons)
        requests.update(code for code, count in record_reasons.items() if count > 0)
    nearby = int(opportunities.gt(0).sum())
    counts = dict(eligible_orders=int(eligible.sum()),
        eligible_served_C=int((eligible & frame.completed & frame.vehicle_type.eq("AV")).sum()),
        eligible_served_HV=len(hv), eligible_expired=int((eligible & frame.expired).sum()),
        HV_served_with_no_observed_nearby_C=len(hv) - nearby,
        HV_served_with_observed_nearby_C=nearby,
        nearby_counts_are_repeated_attempts_not_unique_vehicles=True,
        nearby_does_not_prove_final_C_arc_feasibility=True,
        final_validated_C_arc_count_available=False,
        rejection_reason_request_counts=dict(sorted(requests.items())),
        rejection_reason_attempt_counts=dict(sorted(attempts.items())))
    if sum(counts[k] for k in ("eligible_served_C", "eligible_served_HV", "eligible_expired")) != counts["eligible_orders"]:
        raise ValueError("eligible-request categories do not reconcile")
    return counts


def _guard():
    memory = psutil.Process().memory_info()
    peak = max(memory.rss, getattr(memory, "peak_wset", memory.rss)) / 2**20
    if peak > MAX_RSS_MIB:
        raise MemoryError(f"read-only diagnostic RSS {peak:.1f} MiB exceeds {MAX_RSS_MIB}")
    return peak


def _hours(totals):
    keys = ("admission_s", *KINDS, "idle_s")
    result = {k.replace("_s", "") + "_hours": float(totals[k] / 3600.) for k in keys}
    denominator = totals["admission_s"]
    result["idle_share"] = float(totals["idle_s"] / denominator) if denominator else None
    result["customer_busy_share"] = float((totals["PICKUP"] + totals["LOADED"]) / denominator) if denominator else None
    return result


def _zone_id(lon, lat):
    if not all(isfinite(float(v)) for v in (lon, lat)):
        raise ValueError("missing diagnostic location")
    key = f"{floor(float(lon) / GRID_DEGREES)}|{floor(float(lat) / GRID_DEGREES)}"
    return "Z_" + hashlib.sha256(key.encode()).hexdigest()[:10]


def _summarize_orders(frame):
    eligible = frame.C_compatible & frame.passenger_accepts_av
    return dict(requests=len(frame), C_body_compatible=int(frame.C_compatible.sum()),
        C_and_passenger_eligible=int(eligible.sum()), served_C=int(frame.vehicle_type.eq("AV").sum()),
        C_eligible_served_HV=int((eligible & frame.completed & frame.vehicle_type.eq("HV")).sum()),
        C_eligible_expired=int((eligible & frame.expired).sum()),
        completed=int(frame.completed.sum()), expired=int(frame.expired.sum()))


def analyze(root):
    started = perf_counter()
    root = Path(root).resolve()
    first = json.loads((root / OUT / GROUPS[0] / "summary.json").read_text(encoding="utf-8"))
    cfg = first["frozen_protocol"]
    if (cfg["test_date"] != "20161031" or cfg["requested_q_a"] != .1
            or cfg["profile_id"] != "C" or cfg["passenger_acceptance_rate"] != .7):
        raise ValueError("only the completed C/q10/P70 v4 comparison is in scope")
    inputs = {}
    for path in (FLEET_REL, ORDER_BASE):
        key = path.as_posix()
        digest = sha(root / path)
        if first["inputs_sha256"][key] != digest:
            raise ValueError(f"frozen source changed: {key}")
        inputs[key] = digest
    fleet = build_controlled_fleet(root, benchmark_start=START,
        simulation_end=START + pd.Timedelta(seconds=cfg["physical_drain_limit_s"]),
        requested_q_a=cfg["requested_q_a"], seed=cfg["supply_seed"])
    expected = first["all_day_supply"]["accounting"]
    if (fleet.accounting["type_label_sha256"] != expected["type_label_sha256"]
            or len(fleet.scenario_fleet) != first["all_day_supply"]["slot_count"]):
        raise ValueError("reconstructed admission/type ledger differs from completed supply")
    plans = {int(r.native_id): (str(r.vehicle_type),
        float((r.availability_start_time - START).total_seconds()),
        float((r.availability_end_time - START).total_seconds()))
        for r in fleet.scenario_fleet.itertuples(index=False)}
    del fleet
    # A small, projected in-memory join. No exact coordinate or identifier export.
    coordinates = pd.read_parquet(root / ORDER_BASE,
        columns=["order_id", "pickup_lon_wgs84", "pickup_lat_wgs84"], filters=[("profile_id", "==", "M")])
    if not coordinates.order_id.is_unique:
        raise ValueError("source M cohort coordinates are not unique")
    coordinates["origin_zone"] = [_zone_id(x, y) for x, y in
        zip(coordinates.pickup_lon_wgs84, coordinates.pickup_lat_wgs84)]
    coordinates = coordinates[["order_id", "origin_zone"]]
    result = dict(version="unified_operations_diagnostic_v1", status="ANALYZED_EXISTING_PRODUCTS",
        experiment_producer_sha=first["code_sha"], test_date="20161031",
        calendar_window_s=[0., DAY_S], demand_tail_retained=True,
        supply_unit="EFFECTIVE_COMMON_SERVICE_SESSION_SLOT_NOT_PHYSICAL_FLEET_OR_TRUE_ONLINE",
        type_label_sha256=expected["type_label_sha256"], source_inputs=inputs,
        new_simulation=False, new_routing=False, new_inference=False, model_or_threshold_change=False,
        future_Test31_demand_used_for_decisions=False, per_order_or_coordinates_exported=False,
        grid_degrees=GRID_DEGREES, minimum_public_zone_requests=MIN_ZONE_REQUESTS, groups={})
    result["zone_privacy_scope"] = "NO_EXPLICIT_COORDINATES_OR_ORDER_IDS_NOT_IRREVERSIBLE_ANONYMIZATION"
    peak = _guard()
    for group in GROUPS:
        directory = root / OUT / group
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if (summary["status"] != "TWO_SCALE_CITY_GROUP_COMPLETE"
                or summary["version"] != "scheme_a_two_scale_city_v4"
                or summary["inputs_sha256"] != first["inputs_sha256"]
                or summary["all_day_supply"]["accounting"]["type_label_sha256"] != expected["type_label_sha256"]):
            raise ValueError("requires unchanged complete v4 groups")
        assignment = pd.read_parquet(directory / "assignments.parquet", columns=[
            "native_vehicle_id", "vehicle_type", "simulation_time_s", "pickup_time_s", "service_end_time_s", "completed"])
        moves = pd.read_parquet(directory / "empty_movements.parquet", columns=[
            "native_vehicle_id", "vehicle_type", "start_time_s", "end_time_s", "actual_duration_s", "actual_distance_m", "status"])
        outcome = pd.read_parquet(directory / "cohort_outcomes.parquet", columns=[
            "order_id", "release_s", "C_compatible", "passenger_accepts_av", "completed", "expired", "vehicle_type",
            "spatial_av_opportunities", "rejection_counts_json"])
        if (not assignment.completed.all() or not outcome.order_id.is_unique
                or len(outcome) != 28367 or not (outcome.completed ^ outcome.expired).all()):
            raise ValueError("incomplete or duplicate final products")
        activities = defaultdict(list)
        full_customer = Counter()
        calendar_customer = Counter()
        full_moves = Counter()
        move_distance = Counter()
        serving = set()
        for r in assignment.itertuples(index=False):
            vid = int(r.native_vehicle_id)
            kind, a, b = plans[vid]
            if kind != r.vehicle_type or not (a - TOL <= r.simulation_time_s < b):
                raise ValueError("customer commitment outside source admission window")
            if not r.simulation_time_s <= r.pickup_time_s <= r.service_end_time_s:
                raise ValueError("invalid customer physical time chain")
            activities[vid].extend(((r.simulation_time_s, r.pickup_time_s, "PICKUP"),
                                    (r.pickup_time_s, r.service_end_time_s, "LOADED")))
            full_customer[kind] += r.service_end_time_s - r.simulation_time_s
            calendar_customer[kind] += overlap(r.simulation_time_s, r.service_end_time_s, 0, DAY_S)
            serving.add(vid)
        for r in moves.itertuples(index=False):
            vid = int(r.native_vehicle_id)
            kind = plans[vid][0]
            if (kind != r.vehicle_type or r.status not in ("COMPLETED", "INTERRUPTED_BY_CUSTOMER")
                    or abs(r.end_time_s - r.start_time_s - r.actual_duration_s) > TOL):
                raise ValueError("incomplete/invalid movement event")
            activities[vid].append((r.start_time_s, r.end_time_s, "MOVE"))
            full_moves[kind] += r.actual_duration_s
            move_distance[kind] += r.actual_distance_m
        totals = {k: Counter() for k in ("AV", "HV")}
        hourly = {k: [Counter() for _ in range(24)] for k in ("AV", "HV")}
        slot_counts = {k: Counter() for k in ("AV", "HV")}
        for vid, (kind, a, b) in plans.items():
            ledger = clipped_ledger(activities[vid], a, b)
            totals[kind].update({key: ledger[key] for key in ("admission_s", *KINDS, "idle_s")})
            if ledger["admission_s"]:
                slot_counts[kind]["day_slots"] += 1
                slot_counts[kind]["never_served_slots"] += int(vid not in serving)
                slot_counts[kind]["fully_idle_slots"] += int(ledger["idle_s"] >= ledger["admission_s"] - TOL)
                slot_counts[kind]["slots_with_idle_gap_over_300s"] += int(ledger["longest_idle_s"] > 300)
            for hour in range(max(0, int(max(a, 0) // 3600)), min(24, int(min(b, DAY_S) // 3600) + 1)):
                part = clipped_ledger(activities[vid], a, b, hour * 3600., (hour + 1) * 3600.)
                hourly[kind][hour].update({key: part[key] for key in ("admission_s", *KINDS, "idle_s")})
        state = {}
        for kind in ("AV", "HV"):
            total = totals[kind]
            committed_inside = total["PICKUP"] + total["LOADED"]
            state[kind] = dict(**_hours(total), **slot_counts[kind],
                full_committed_customer_hours=full_customer[kind] / 3600.,
                calendar_customer_after_admission_hours=max(0., calendar_customer[kind] - committed_inside) / 3600.,
                customer_outside_calendar_hours=(full_customer[kind] - calendar_customer[kind]) / 3600.,
                full_movement_hours=full_moves[kind] / 3600., full_movement_km=move_distance[kind] / 1000.,
                partition_error_s=abs(total["admission_s"] - committed_inside - total["MOVE"] - total["idle_s"]))
            planned_key = "achieved_av_vehicle_hours" if kind == "AV" else "achieved_hv_vehicle_hours"
            if abs(state[kind]["admission_hours"] - expected[planned_key]) > TOL:
                raise ValueError("calendar admission denominator changed")
        joined = outcome.merge(coordinates, on="order_id", validate="one_to_one", how="left")
        if joined.origin_zone.isna().any():
            raise ValueError("missing common-order diagnostic zone")
        joined["hour"] = (joined.release_s // 3600).astype(int)
        hourly_orders = {int(h): frame for h, frame in joined.groupby("hour", sort=True)}
        hours = [dict(hour=h, requests=_summarize_orders(hourly_orders[h]) if h in hourly_orders else {},
            C_state=_hours(hourly["AV"][h]), HV_state=_hours(hourly["HV"][h])) for h in range(24)]
        tail = _summarize_orders(joined.loc[joined.hour.ge(24)])
        zones = [dict(zone_id=str(z), **_summarize_orders(frame)) for z, frame in joined.groupby("origin_zone", sort=True)
                 if len(frame) >= MIN_ZONE_REQUESTS]
        zones.sort(key=lambda r: (-r["requests"], r["zone_id"]))
        result["groups"][group] = dict(orders=_summarize_orders(outcome), admission_state=state,
            C_eligible_HV_diagnostics=request_partition(outcome), hourly=hours, post_midnight_requests=tail,
            public_origin_zones=zones, suppressed_zone_requests=len(outcome) - sum(z["requests"] for z in zones),
            source_products={name: sha(directory / name) for name in
                ("summary.json", "assignments.parquet", "empty_movements.parquet", "cohort_outcomes.parquet")})
        peak = max(peak, _guard())
        del assignment, moves, outcome, joined, activities, hourly_orders
        gc.collect()
    if any(sha(root / path) != digest for path, digest in inputs.items()):
        raise ValueError("source input changed during read-only analysis")
    result.update(runtime_s=perf_counter() - started, peak_rss_mib=peak,
        limitations=["One date and seed; descriptive, not statistical superiority",
            "No type-specific final AV arc trace; cannot identify every HV substitution cause",
            "Residual idle is simulated unoccupied admission time, not real online standby",
            "Test31 zone summaries are diagnostics, never training/deployment inputs",
            "No new C-compatible routes or dynamic predictions for empty routes"])
    atomic_json(root / DOC / "diagnostic_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = analyze(args.root)
    print(json.dumps(dict(status=result["status"], runtime_s=result["runtime_s"],
        peak_rss_mib=result["peak_rss_mib"], groups={name:dict(C_state=row["admission_state"]["AV"],
            eligible_HV=row["C_eligible_HV_diagnostics"]) for name,row in result["groups"].items()}), ensure_ascii=False))


if __name__ == "__main__":
    main()
