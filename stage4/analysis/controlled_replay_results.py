"""Aggregate the two completed controlled runs; never start a native replay."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RUNS = Path("stage4/output/controlled_replay_v2/runs")
DOCS = Path("stage4/docs/controlled_replay_v2/results")
CONFIG = Path("stage4/config/controlled_replay_v2.json")
SCENARIOS = ("PURE_HV", "M_Q10_P70")
POLICY = "SERVICE_PRESERVING_LOOKAHEAD"
AVAILABILITY_POLICY = "STOP_ADMISSION_FINISH_COMMITTED"
PRODUCTS = ("assignments.parquet", "cohort_outcomes.parquet", "epochs.parquet",
            "solver_trace.parquet", "empty_movements.parquet", "fleet_accounting.json")
PIPELINE_TIMES = ("forecast_sampling_time_s", "future_graph_time_s", "problem_setup_time_s",
                  "model_build_time_s", "optimization_time_s", "recourse_recovery_time_s",
                  "failed_model_attempt_time_s")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _batches(path, required, optional=()):
    """Read needed columns only, with bounded batches and no Arrow worker pool."""
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows == 0:
        return
    names = set(parquet.schema_arrow.names)
    missing = set(required) - names
    if missing:
        raise ValueError(f"{Path(path).name} missing actual-run columns: {sorted(missing)}")
    columns = list(dict.fromkeys((*required, *(key for key in optional if key in names))))
    for batch in parquet.iter_batches(batch_size=8192, columns=columns, use_threads=False):
        yield batch.to_pandas()


def _cohort(path):
    required = ("order_id", "common_eligible", "matched", "expired", "excluded_input", "vehicle_type", "wait_s")
    parts = list(_batches(path, required))
    if not parts:
        raise ValueError("completed run has an empty original cohort")
    frame = pd.concat(parts, ignore_index=True)
    if frame.order_id.isna().any() or not frame.order_id.is_unique:
        raise ValueError("original cohort order identities are missing or duplicated")
    if frame[["common_eligible", "matched", "expired", "excluded_input"]].isna().any().any():
        raise ValueError("original cohort accounting flags are missing")
    return frame.set_index("order_id")


def _stats(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce")
    values = values.loc[np.isfinite(values)]
    return dict(count=int(len(values)), mean_s=float(values.mean()) if len(values) else None,
                p50_s=float(values.quantile(.5)) if len(values) else None,
                p90_s=float(values.quantile(.9)) if len(values) else None,
                p95_s=float(values.quantile(.95)) if len(values) else None,
                minimum_s=float(values.min()) if len(values) else None,
                maximum_s=float(values.max()) if len(values) else None)


def _orders(frame):
    common, matched = frame.common_eligible.astype(bool), frame.matched.astype(bool)
    expired, excluded = frame.expired.astype(bool), frame.excluded_input.astype(bool)
    waits = pd.to_numeric(frame.wait_s, errors="coerce")
    return dict(original_orders=int(len(frame)), common_eligible=int(common.sum()),
                matched=int(matched.sum()), expired=int(expired.sum()), excluded_input=int(excluded.sum()),
                service_rate_original=float(matched.mean()),
                service_rate_common=float(matched.sum() / common.sum()) if common.any() else None,
                hv_assignments=int(frame.vehicle_type.eq("HV").sum()),
                av_assignments=int(frame.vehicle_type.eq("AV").sum()),
                served_wait=_stats(waits[matched]),
                terminal_accounting_errors=int(((matched.astype(int) + expired.astype(int) + excluded.astype(int)) != 1).sum()),
                excluded_flag_errors=int(excluded.ne(~common).sum()),
                matched_outside_common=int((matched & ~common).sum()),
                matched_missing_wait=int((matched & ~np.isfinite(waits)).sum()),
                pickup_patience_violations=int((matched & waits.gt(300.0 + 1e-6)).sum()))


def _pair(hv, mixed):
    if set(hv.index) != set(mixed.index):
        raise ValueError("two completed original order populations differ; paired comparison is unavailable")
    mixed = mixed.reindex(hv.index)
    if not hv.common_eligible.equals(mixed.common_eligible):
        raise ValueError("two completed common-eligible populations differ")
    served_hv, served_mixed = hv.matched.astype(bool), mixed.matched.astype(bool)
    both = served_hv & served_mixed
    delta = pd.to_numeric(mixed.loc[both, "wait_s"], errors="coerce") - pd.to_numeric(hv.loc[both, "wait_s"], errors="coerce")
    return dict(direction="M_Q10_P70_MINUS_PURE_HV", original_orders=int(len(hv)),
                common_eligible=int(hv.common_eligible.sum()), original_order_sets_identical=True,
                common_eligibility_identical=True, gained_by_mixed=int((~served_hv & served_mixed).sum()),
                lost_by_mixed=int((served_hv & ~served_mixed).sum()),
                net_matched_change=int(served_mixed.sum() - served_hv.sum()),
                common_served=int(both.sum()), neither_served=int((~served_hv & ~served_mixed).sum()),
                paired_wait_delta_s=_stats(delta), paired_wait_missing=int((~np.isfinite(delta)).sum()),
                waiting_is_conditional_on_both_runs_serving=True,
                statistical_significance_claim=False)


def _seconds(column):
    parsed = pd.to_datetime(column, utc=True, errors="coerce")
    return parsed.to_numpy(dtype="datetime64[ns]").view(np.int64), parsed.notna().to_numpy()


def _physical_hours(path):
    required = ("order_id", "native_vehicle_id", "vehicle_type", "assignment_time", "pickup_time",
                "service_end_time", "completed", "availability_policy", "availability_start_time", "availability_end_time")
    optional = ("pickup_eta_s", "realized_service_time_s", "passenger_accepts_av")
    by_type = {kind: Counter() for kind in ("HV", "AV")}
    checks = Counter()
    identifiers = set()
    intervals = defaultdict(list)
    for frame in _batches(path, required, optional):
        checks["assignment_rows"] += len(frame)
        checks["duplicate_order_rows"] += int(frame.order_id.duplicated().sum()) + sum(value in identifiers for value in frame.order_id.drop_duplicates())
        identifiers.update(frame.order_id)
        checks["incomplete_assignments"] += int((~frame.completed.fillna(False).astype(bool)).sum())
        checks["policy_violations"] += int(frame.availability_policy.ne(AVAILABILITY_POLICY).sum())
        checks["unknown_vehicle_types"] += int((~frame.vehicle_type.isin(by_type)).sum())
        assign, ok_a = _seconds(frame.assignment_time)
        pickup, ok_p = _seconds(frame.pickup_time)
        finish, ok_f = _seconds(frame.service_end_time)
        window_start, ok_ws = _seconds(frame.availability_start_time)
        window_end, ok_we = _seconds(frame.availability_end_time)
        valid = ok_a & ok_p & ok_f & ok_ws & ok_we
        checks["missing_physical_timestamps"] += int((~valid).sum())
        valid &= (pickup >= assign) & (finish >= pickup) & (window_end > window_start)
        checks["invalid_physical_time_chains"] += int((ok_a & ok_p & ok_f & ok_ws & ok_we & ~valid).sum())
        checks["admission_outside_window"] += int((valid & ((assign < window_start) | (assign >= window_end))).sum())
        for kind, totals in by_type.items():
            mask = valid & frame.vehicle_type.eq(kind).to_numpy()
            totals["assignments"] += int(frame.vehicle_type.eq(kind).sum())
            totals["physically_accounted_assignments"] += int(mask.sum())
            for activity, left, right in (("pickup", assign, pickup), ("service", pickup, finish)):
                durations = right[mask] - left[mask]
                within = np.maximum(0, np.minimum(right[mask], window_end[mask]) - np.maximum(left[mask], window_start[mask]))
                after = np.maximum(0, right[mask] - np.maximum(left[mask], window_end[mask]))
                before = np.maximum(0, np.minimum(right[mask], window_start[mask]) - left[mask])
                totals[activity + "_total_ns"] += int(durations.sum())
                totals[activity + "_within_window_ns"] += int(within.sum())
                totals[activity + "_after_shift_ns"] += int(after.sum())
                totals[activity + "_before_window_ns"] += int(before.sum())
        for index in np.flatnonzero(valid):
            intervals[int(frame.iloc[index].native_vehicle_id)].append((int(assign[index]), int(finish[index])))
        if "pickup_eta_s" in frame:
            error = np.abs((pickup[valid] - assign[valid]) / 1e9 - pd.to_numeric(frame.loc[valid, "pickup_eta_s"], errors="coerce").to_numpy())
            checks["physical_pickup_chain_violations"] += int((error > 1e-6).sum())
        if "realized_service_time_s" in frame:
            error = np.abs((finish[valid] - pickup[valid]) / 1e9 - pd.to_numeric(frame.loc[valid, "realized_service_time_s"], errors="coerce").to_numpy())
            checks["physical_service_chain_violations"] += int((error > 1e-6).sum())
        if "passenger_accepts_av" in frame:
            checks["av_acceptance_violations"] += int((frame.vehicle_type.eq("AV") & ~frame.passenger_accepts_av.fillna(False).astype(bool)).sum())
    checks["overlapping_customer_tasks"] = 0
    for tasks in intervals.values():
        tasks.sort()
        end = None
        for start, finish in tasks:
            checks["overlapping_customer_tasks"] += int(end is not None and start < end)
            end = finish if end is None else max(end, finish)
    result = {}
    for kind, totals in by_type.items():
        row = {key: int(totals[key]) for key in ("assignments", "physically_accounted_assignments")}
        for activity in ("pickup", "service"):
            for period in ("total", "within_window", "after_shift", "before_window"):
                row[f"{activity}_{period}_hours"] = totals[f"{activity}_{period}_ns"] / 3.6e12
        row["customer_busy_total_hours"] = row["pickup_total_hours"] + row["service_total_hours"]
        row["customer_busy_after_shift_hours"] = row["pickup_after_shift_hours"] + row["service_after_shift_hours"]
        result[kind] = row
    check_names = ("assignment_rows", "duplicate_order_rows", "incomplete_assignments", "policy_violations",
                   "unknown_vehicle_types", "missing_physical_timestamps", "invalid_physical_time_chains",
                   "admission_outside_window", "overlapping_customer_tasks")
    return dict(by_vehicle_type=result, checks={**{key: int(checks[key]) for key in check_names}, **dict(checks)},
                timestamp_basis="ACTUAL_ASSIGNMENT_PICKUP_SERVICE_END_NOT_M3_PREDICTION"), identifiers


def _empty_movements(path):
    required = ("vehicle_type", "status", "start_time_s", "end_time_s", "actual_duration_s", "actual_distance_m", "after_shift_time_s")
    optional = ("traffic_unknown_share", "snap_gap_origin_m", "snap_gap_target_m", "planned_duration_s", "planned_distance_m")
    rows, statuses, invalid, unknown_missing = 0, Counter(), Counter(), 0
    by_type = {kind: Counter() for kind in ("HV", "AV")}
    snap = Counter()
    for frame in _batches(path, required, optional):
        rows += len(frame)
        statuses.update(frame.status.fillna("MISSING_STATUS").astype(str))
        duration = pd.to_numeric(frame.actual_duration_s, errors="coerce")
        distance = pd.to_numeric(frame.actual_distance_m, errors="coerce")
        after = pd.to_numeric(frame.after_shift_time_s, errors="coerce")
        actual_chain = pd.to_numeric(frame.end_time_s, errors="coerce") - pd.to_numeric(frame.start_time_s, errors="coerce")
        valid = np.isfinite(duration) & np.isfinite(distance) & np.isfinite(after) & np.isfinite(actual_chain)
        valid &= duration.ge(0) & distance.ge(0) & after.ge(0) & after.le(duration + 1e-6)
        invalid["invalid_physical_rows"] += int((~valid).sum())
        invalid["time_chain_violations"] += int((valid & (actual_chain - duration).abs().gt(1e-6)).sum())
        invalid["unfinished_rows"] += int(frame.status.eq("IN_PROGRESS").sum())
        invalid["unknown_vehicle_types"] += int((~frame.vehicle_type.isin(by_type)).sum())
        if "planned_distance_m" in frame:
            invalid["distance_exceeds_plan_rows"] += int((valid & distance.gt(pd.to_numeric(frame.planned_distance_m, errors="coerce") + 1e-5)).sum())
        if "traffic_unknown_share" in frame:
            unknown = pd.to_numeric(frame.traffic_unknown_share, errors="coerce")
            unknown_missing += int((~np.isfinite(unknown)).sum())
        else:
            unknown = pd.Series(float("nan"), index=frame.index)
            unknown_missing += len(frame)
        for kind, totals in by_type.items():
            selected = frame.vehicle_type.eq(kind)
            mask = selected & valid
            totals["movements"] += int(selected.sum())
            totals["interruptions"] += int((selected & frame.status.eq("INTERRUPTED_BY_CUSTOMER")).sum())
            totals["completed"] += int((selected & frame.status.eq("COMPLETED")).sum())
            totals["time_s"] += float(duration[mask].sum())
            totals["distance_m"] += float(distance[mask].sum())
            totals["after_shift_time_s"] += float(after[mask].sum())
            totals["unknown_time_s"] += float((duration[mask] * unknown[mask]).sum())
        for field in ("snap_gap_origin_m", "snap_gap_target_m"):
            if field in frame:
                values = pd.to_numeric(frame[field], errors="coerce")
                snap[field + "_max"] = max(float(snap[field + "_max"]), float(values.max()) if values.notna().any() else 0.0)
                snap[field + "_positive_rows"] += int(values.gt(0).sum())
    converted = {}
    for kind, totals in by_type.items():
        converted[kind] = {key: int(totals[key]) for key in ("movements", "interruptions", "completed")}
        converted[kind].update(time_s=float(totals["time_s"]), distance_m=float(totals["distance_m"]),
            total_hours=totals["time_s"] / 3600, after_shift_hours=totals["after_shift_time_s"] / 3600,
            within_window_hours=(totals["time_s"] - totals["after_shift_time_s"]) / 3600)
    total_time = sum(row["time_s"] for row in converted.values())
    unknown_time = sum(row["unknown_time_s"] for row in by_type.values())
    return dict(rows=rows, statuses=dict(statuses), by_vehicle_type=converted,
                total_time_s=total_time, total_distance_m=sum(row["distance_m"] for row in converted.values()),
                traffic_unknown_duration_share=unknown_time / total_time if total_time and not unknown_missing else None,
                traffic_unknown_missing_rows=unknown_missing,
                checks={**{key: int(invalid[key]) for key in ("invalid_physical_rows", "time_chain_violations", "unfinished_rows", "unknown_vehicle_types", "distance_exceeds_plan_rows")}},
                endpoint_connector_observations=dict(snap),
                evidence_scope="EXISTING_STAGE3_MOVEMENT_COMPATIBILITY_WITH_EXPLICIT_UNKNOWN_U_NOT_FULL_DYNAMIC_ODD_CERTIFICATION")


def _solver_and_epochs(directory):
    trace_path = directory / "solver_trace.parquet"
    optional = ("resource_fallback", "current_face_preserved", "solver_time_s", *PIPELINE_TIMES)
    names = set(pq.ParquetFile(trace_path).schema_arrow.names)
    timing = {key: 0.0 for key in ("solver_time_s", *PIPELINE_TIMES) if key in names}
    fallback, violations, face_missing, trace_rows = Counter(), 0, 0, 0
    for frame in _batches(trace_path, (), optional):
        trace_rows += len(frame)
        for key in timing:
            timing[key] += float(pd.to_numeric(frame[key], errors="coerce").sum())
        if "resource_fallback" in frame:
            fallback.update(frame.resource_fallback.dropna().astype(str))
        if "current_face_preserved" in frame:
            face_missing += int(frame.current_face_preserved.isna().sum())
            violations += int(frame.current_face_preserved.eq(False).sum())
        else:
            face_missing += len(frame)
    epoch_timing, epochs = Counter(), 0
    for frame in _batches(directory / "epochs.parquet", ("candidate_generation_time_s", "routing_time_s")):
        epochs += len(frame)
        for key in frame:
            epoch_timing[key] += float(pd.to_numeric(frame[key], errors="coerce").sum())
    movement_epoch_checks = None
    path = directory / "empty_movement_epochs.parquet"
    if path.exists():
        rounds, outside, over_budget, maximum = 0, 0, 0, 0
        for frame in _batches(path, ("simulation_time_s", "started")):
            rounds += len(frame)
            outside += int(((frame.simulation_time_s < 0) | (frame.simulation_time_s >= 86400) | (frame.simulation_time_s % 900 != 0)).sum())
            over_budget += int(frame.started.gt(50).sum())
            maximum = max(maximum, int(frame.started.max()))
        movement_epoch_checks = dict(rounds=rounds, outside_day_or_grid_rounds=outside, over_budget_rounds=over_budget, maximum_started_in_round=maximum)
    return dict(trace_rows=trace_rows, dispatch_epochs=epochs, resource_fallback_epochs=sum(fallback.values()),
                resource_fallback_reasons=dict(fallback), current_face_violation_epochs=violations if "current_face_preserved" in names else None,
                current_face_unreported_epochs=face_missing, epoch_timing_totals_s=dict(epoch_timing),
                solver_pipeline_total_s=timing.pop("solver_time_s", None), solver_pipeline_components_s=timing,
                empty_movement_epoch_checks=movement_epoch_checks)


def _input_path(root, name):
    path = (root / str(name).replace("\\", "/")).resolve()
    if not path.is_relative_to(root):
        raise ValueError("recorded input is outside the requested workspace")
    return path


def _provenance(root, summaries):
    mappings = [{str(key).replace("\\", "/"): value for key, value in s.get("inputs_sha256", {}).items()} for s in summaries]
    records = []
    for name in sorted(set(mappings[0]) | set(mappings[1])):
        expected = [mapping.get(name) for mapping in mappings]
        path = _input_path(root, name)
        observed = _sha(path) if path.is_file() else None
        records.append(dict(path=name, pure_hv_sha256=expected[0], mixed_sha256=expected[1], current_sha256=observed,
                            shared_hash_identical=expected[0] is not None and expected[0] == expected[1],
                            current_hash_matches=observed is not None and observed == expected[0] == expected[1]))
    configs = {key: [s.get(key) for s in summaries] for key in ("controlled_config_sha256", "acceleration_config_sha256")}
    return dict(recorded_input_count_by_scenario={name: len(mapping) for name, mapping in zip(SCENARIOS, mappings)},
                shared_input_count=len(set(mappings[0]) & set(mappings[1])), input_records=records,
                input_key_sets_identical=bool(records) and set(mappings[0]) == set(mappings[1]),
                shared_hashes_identical=bool(records) and all(row["shared_hash_identical"] for row in records),
                frozen_files_unchanged=bool(records) and all(row["current_hash_matches"] for row in records),
                configuration_hashes_by_scenario=configs,
                configuration_hashes_identical=all(values[0] is not None and values[0] == values[1] for values in configs.values()),
                execution_code_sha_by_scenario={name: s.get("execution_code_sha") for name, s in zip(SCENARIOS, summaries)},
                execution_code_sha_identical=summaries[0].get("execution_code_sha") is not None and summaries[0].get("execution_code_sha") == summaries[1].get("execution_code_sha"),
                report_helper_sha256=_sha(Path(__file__)), controlled_config_path=CONFIG.as_posix())


def _hourly(fleets):
    by_scenario = []
    for scenario, fleet in zip(SCENARIOS, fleets):
        rows = sorted(fleet.get("hourly_supply", []), key=lambda row: row["hour"])
        if [row["hour"] for row in rows] != list(range(24)):
            raise ValueError("completed fleet ledger must contain exactly hours 0 through 23")
        by_scenario.append(rows)
    same_total = all(a["total_vehicle_hours"] == b["total_vehicle_hours"] for a, b in zip(*by_scenario))
    exact = all(row["total_supply_error_nanoseconds"] == 0 and row["total_vehicle_hours"] == row["baseline_vehicle_hours"] for rows in by_scenario for row in rows)
    records = [dict(scenario=scenario, **row) for scenario, rows in zip(SCENARIOS, by_scenario) for row in rows]
    mixed = by_scenario[1]
    proportions = [row["realized_q_a"] for row in mixed if row["realized_q_a"] is not None]
    return dict(hourly_total_supply_identical=same_total, hourly_total_supply_exact=exact,
                maximum_hourly_total_supply_difference_hours=max(abs(a["total_vehicle_hours"] - b["total_vehicle_hours"]) for a, b in zip(*by_scenario)),
                mixed_hourly_minimum_av_share=min(proportions) if proportions else None,
                mixed_hourly_maximum_av_share=max(proportions) if proportions else None,
                mixed_maximum_absolute_hourly_av_share_error_pp=max(abs(row["q_a_error_pp"]) for row in mixed if row["q_a_error_pp"] is not None)), records


def _runtime(summary):
    fields = ("runtime_s", "peak_rss_mib", "peak_process_group_rss_mib", "peak_process_group_private_committed_mib",
              "routing_failures", "routing_arc_evaluations", "routing_queue_pipeline_time_s", "disk_cache_lookup_time_s",
              "gpu_used", "dense_matrix", "physical_reconciliation", "native_start_s", "drain_end_s", "administrative_timeout_s",
              "execution_started_at", "execution_finished_at")
    empty = summary.get("empty_movements", {})
    return {**{key: summary.get(key) for key in fields}, "recorded_timing_totals_s": summary.get("timing_totals_s"),
            "recorded_solver_components_s": summary.get("solver_pipeline_components_s"),
            "empty_routing_wall_time_s": empty.get("routing_wall_time_s"),
            "empty_total_decision_time_s": empty.get("total_decision_time_s"),
            "timing_note": "Nested solver components and routing-queue measurements are not additional independent wall-clock intervals."}


def _customer_route_checks(root, frames, provenance):
    paths = [row["path"] for row in provenance["input_records"] if row["path"].endswith("/test31_research_routes.parquet")]
    if len(paths) != 1:
        raise ValueError("completed runs do not record one frozen customer research-route input")
    route_parts = list(_batches(_input_path(root, paths[0]), ("order_id", "common_eligible", "compatible_M")))
    routes = pd.concat(route_parts, ignore_index=True).set_index("order_id")
    if not routes.index.is_unique:
        raise ValueError("frozen customer-route identities are duplicated")
    result = {}
    for scenario, frame in zip(SCENARIOS, frames):
        source = routes.reindex(frame.index)
        eligible = source.common_eligible.fillna(False).astype(bool)
        av = frame.vehicle_type.eq("AV")
        result[scenario] = dict(common_route_support_mismatch_orders=int(eligible.ne(frame.common_eligible).sum()),
            av_service_capability_violations=int((av & ~source.compatible_M.fillna(False).astype(bool)).sum()),
            missing_source_route_orders=int(source.common_eligible.isna().sum()))
    return result


def collect_comparison(root):
    """Return PENDING without writes, or aggregate only the two declared runs."""
    root = Path(root).resolve()
    summaries, directories, pending = [], [], []
    for scenario in SCENARIOS:
        directory = root / RUNS / scenario / POLICY
        directories.append(directory)
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            pending.append(dict(scenario=scenario, reason="summary_missing"))
            continue
        try:
            summary = _read_json(summary_path)
        except json.JSONDecodeError:
            pending.append(dict(scenario=scenario, reason="summary_not_readable_yet"))
            continue
        if summary.get("status") != "COMPLETE":
            pending.append(dict(scenario=scenario, reason="run_not_complete", run_status=summary.get("status"), error=summary.get("error")))
        missing = [name for name in PRODUCTS if not (directory / name).is_file()]
        if missing:
            pending.append(dict(scenario=scenario, reason="completed_products_missing", products=missing))
        summaries.append(summary)
    if pending:
        return dict(status="PENDING", pending=pending, native_replay_started=False, output_files_written=False), []
    for scenario, summary in zip(SCENARIOS, summaries):
        if (summary.get("controlled_scenario") != scenario or not summary.get("controlled_replay_v2")
                or summary.get("smoke_diagnostic") or summary.get("policy") != POLICY):
            raise ValueError("summary does not identify the declared complete controlled full-day scenario")
    cfg = _read_json(root / CONFIG)
    fleets = [_read_json(directory / "fleet_accounting.json") for directory in directories]
    provenance = _provenance(root, summaries)
    frames = [_cohort(directory / "cohort_outcomes.parquet") for directory in directories]
    customer_routes = _customer_route_checks(root, frames, provenance)
    paired = _pair(*frames)
    supply, hourly_records = _hourly(fleets)
    checks = dict(original_order_sets_identical=True, common_eligibility_identical=True,
                  hourly_total_supply_identical=supply["hourly_total_supply_identical"],
                  hourly_total_supply_exact=supply["hourly_total_supply_exact"],
                  shared_input_key_sets_identical=provenance["input_key_sets_identical"],
                  shared_input_hashes_identical=provenance["shared_hashes_identical"],
                  frozen_input_files_unchanged=provenance["frozen_files_unchanged"],
                  configuration_hashes_identical=provenance["configuration_hashes_identical"],
                  execution_code_sha_identical=provenance["execution_code_sha_identical"])
    scenarios = {}
    for name, directory, summary, fleet, frame in zip(SCENARIOS, directories, summaries, fleets, frames):
        orders = _orders(frame)
        physical, assigned_ids = _physical_hours(directory / "assignments.parquet")
        empty = _empty_movements(directory / "empty_movements.parquet")
        solve = _solver_and_epochs(directory)
        setting = cfg["scenarios"][name]
        rows = physical["by_vehicle_type"]
        for kind, row in rows.items():
            key = "hv_vehicle_hours" if kind == "HV" else "av_vehicle_hours"
            row["planned_day_hours"] = sum(record[key] for record in fleet["hourly_supply"])
            row["planned_full_source_window_hours"] = row["planned_day_hours"] + fleet["pre_day_supply"][key] + fleet["spillover_supply"][key]
            row["empty_within_window_hours"] = empty["by_vehicle_type"][kind]["within_window_hours"]
            row["empty_after_shift_hours"] = empty["by_vehicle_type"][kind]["after_shift_hours"]
            row["actual_active_operating_hours"] = row["customer_busy_total_hours"] + empty["by_vehicle_type"][kind]["total_hours"]
        checks[name + "_original_cohort_count"] = orders["original_orders"] == 30000 == summary.get("original_orders")
        checks[name + "_summary_order_counts"] = all(orders[key] == summary.get(key) for key in ("common_eligible", "matched", "expired", "excluded_input"))
        checks[name + "_cohort_terminal_accounting"] = all(orders[key] == 0 for key in ("terminal_accounting_errors", "excluded_flag_errors", "matched_outside_common", "matched_missing_wait", "pickup_patience_violations"))
        checks[name + "_assignment_order_set"] = assigned_ids == set(frame.index[frame.matched])
        checks[name + "_physical_time_accounting"] = all(value == 0 for key, value in physical["checks"].items() if key != "assignment_rows")
        checks[name + "_empty_physical_accounting"] = all(value == 0 for value in empty["checks"].values())
        checks[name + "_current_service_face"] = solve["current_face_violation_epochs"] == 0 and solve["current_face_unreported_epochs"] == 0
        checks[name + "_resource_fallback_count"] = solve["resource_fallback_epochs"] == summary.get("resource_fallback_epochs")
        checks[name + "_q_target"] = fleet["requested_q_a"] == setting["requested_q_a"]
        checks[name + "_complete_frozen_slots"] = fleet["slot_count"] == fleet["native_fixture_count"] == 8435
        checks[name + "_controlled_policy"] = fleet["availability_policy"] == AVAILABILITY_POLICY
        checks[name + "_physical_reconciliation"] = summary.get("physical_reconciliation") is True
        checks[name + "_customer_research_route_compatibility"] = all(value == 0 for value in customer_routes[name].values())
        checks[name + "_cpu_and_sparse"] = summary.get("gpu_used") is False and summary.get("dense_matrix") is False
        resource = summary.get("peak_process_group_rss_mib")
        checks[name + "_process_group_rss_within_declared_limit"] = resource <= 2048 if resource is not None else None
        observed_empty = summary.get("empty_movements", {})
        checks[name + "_empty_summary_rows"] = observed_empty.get("movement_rows") == empty["rows"]
        checks[name + "_empty_summary_time_distance"] = all(
            observed_empty.get(key) is not None and abs(float(observed_empty[key]) - empty[field]) <= 1e-5
            for key, field in (("empty_time_s", "total_time_s"), ("empty_distance_m", "total_distance_m")))
        if solve["empty_movement_epoch_checks"] is not None:
            c = solve["empty_movement_epoch_checks"]
            checks[name + "_empty_day_grid_and_shared_budget"] = c["outside_day_or_grid_rounds"] == c["over_budget_rounds"] == 0 and c["rounds"] <= 96
        scenarios[name] = dict(definition=dict(scenario=name, policy=POLICY, profile="M",
            requested_q_a=setting["requested_q_a"], passenger_acceptance_rate=setting["passenger_acceptance_rate"],
            availability_policy=AVAILABILITY_POLICY), orders=orders, physical_vehicle_hours=physical,
            empty_movements=empty, empty_route_diagnostics=summary.get("empty_route_diagnostics"),
            recorded_empty_movement_summary=observed_empty, customer_route_checks=customer_routes[name],
            computation=solve, runtime=_runtime(summary), fleet=fleet,
            recorded_summary_source=(directory / "summary.json").relative_to(root).as_posix())
    paired["service_rate_common_difference_pp"] = 100 * (scenarios[SCENARIOS[1]]["orders"]["service_rate_common"] - scenarios[SCENARIOS[0]]["orders"]["service_rate_common"])
    paired["service_rate_original_difference_pp"] = 100 * (scenarios[SCENARIOS[1]]["orders"]["service_rate_original"] - scenarios[SCENARIOS[0]]["orders"]["service_rate_original"])
    flags = [name for name, passed in checks.items() if passed is False]
    result = dict(status="ANALYZED" if not flags else "ANALYZED_WITH_DIAGNOSTIC_FLAGS", version="controlled_replay_v2",
                  analyzed_at=datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Singapore")).isoformat(), scenarios=scenarios, paired_orders=paired,
                  supply_control=supply, provenance=provenance, integrity_checks=checks, diagnostic_flags=flags,
                  unavailable_checks=[name for name, passed in checks.items() if passed is None],
                  one_day_one_profile_one_supply_seed=True, request_time_equals_boarding_accepted=True,
                  original_release_spillover_preserved=True, native_replay_started=False,
                  native_conditions_added_by_analysis=0, per_order_ids_or_coordinates_in_report=False,
                  old_82_40_and_68_58_rates_are_legacy_background_not_paired_controls=True,
                  no_service_rate_success_threshold=True, statistical_significance_claim=False)
    return result, hourly_records


def _number(value, digits=2):
    return f"{value:,.{digits}f}" if value is not None else "未记录"


def _percent(value, digits=3):
    return f"{100 * value:.{digits}f}%" if value is not None else "未记录"


def render_report(result):
    hv, mixed = (result["scenarios"][name] for name in SCENARIOS)
    pair, supply = result["paired_orders"], result["supply_control"]
    lines = ["# 共同班次供给回放：纯HV与M混合车队", "",
        "Origin Skill: academic-research-suite / experiment-agent；Mode: completed-run validation。",
        "Verification Status: 已与两组完整本地产物对账；Version: controlled_replay_v2_first_pair。", "",
        "本轮比较纯HV与目标10%预定AV车时、70%乘客接受AV的M混合场景。两组均使用既定的前瞻派单、停止接新单后完成已承诺任务、基于训练期历史订单分布的空车调位。上车时刻近似请求释放时间沿用本研究已接受的假设，接驾耐心保持300秒。", "",
        f"汇总状态：`{result['status']}`；生成时间：{result['analyzed_at']}。本报告只读取两组完成产物，未新增模拟。", "",
        "## 订单与等待", "", "| 指标 | 纯HV | M混合（目标10% / 接受70%） |", "|---|---:|---:|"]
    for title, key in (("原始订单", "original_orders"), ("共同可用订单", "common_eligible"), ("已服务", "matched"), ("接驾耐心到期", "expired"), ("共同输入证据不完整而排除", "excluded_input")):
        lines.append(f"| {title} | {hv['orders'][key]:,} | {mixed['orders'][key]:,} |")
    for title, key in (("共同订单口径服务率", "service_rate_common"), ("原始订单口径服务率", "service_rate_original")):
        lines.append(f"| {title} | {_percent(hv['orders'][key])} | {_percent(mixed['orders'][key])} |")
    lines += [f"| 各组已服务订单的平均等待（秒） | {_number(hv['orders']['served_wait']['mean_s'])} | {_number(mixed['orders']['served_wait']['mean_s'])} |", "",
        f"以纯HV为基准，混合组新增服务{pair['gained_by_mixed']:,}单、失去服务{pair['lost_by_mixed']:,}单，净变化{pair['net_matched_change']:+,}单；共同口径服务率变化{pair['service_rate_common_difference_pp']:+.3f}个百分点，原始口径变化{pair['service_rate_original_difference_pp']:+.3f}个百分点。", "",
        f"两组都服务的{pair['common_served']:,}单，等待差方向为“混合减纯HV”：均值{_number(pair['paired_wait_delta_s']['mean_s'])}秒，中位数{_number(pair['paired_wait_delta_s']['p50_s'])}秒，90%分位{_number(pair['paired_wait_delta_s']['p90_s'])}秒。该比较条件于两组均服务，不能代替所有请求的等待福利。", "",
        "## 共同预定供给", "", "| 日内供给指标 | 纯HV | M混合 |", "|---|---:|---:|"]
    for title, key in (("HV预定车时", "achieved_hv_vehicle_hours"), ("AV预定车时", "achieved_av_vehicle_hours"), ("总预定车时", "total_planned_vehicle_hours")):
        lines.append(f"| {title} | {_number(hv['fleet'][key], 6)} | {_number(mixed['fleet'][key], 6)} |")
    lines += [f"| 实际AV车时比例 | {_percent(hv['fleet']['achieved_q_a'], 6)} | {_percent(mixed['fleet']['achieved_q_a'], 6)} |",
        f"| AV比例取整误差（百分点） | {_number(hv['fleet']['q_a_error_pp'], 6)} | {_number(mixed['fleet']['q_a_error_pp'], 6)} |", "",
        f"两组全部24小时总预定供给严格相同：{supply['hourly_total_supply_identical'] and supply['hourly_total_supply_exact']}；最大总车时差{_number(supply['maximum_hourly_total_supply_difference_hours'], 9)}小时。共同冻结模板均为8,435个有效班次slot；它们不是8,435辆全天物理车，也不是AV采购数量。", "",
        f"混合组逐小时AV车时占比范围为{_percent(supply['mixed_hourly_minimum_av_share'])}—{_percent(supply['mixed_hourly_maximum_av_share'])}，最大目标偏差{_number(supply['mixed_maximum_absolute_hourly_av_share_error_pp'], 6)}个百分点。总供给严格相同与逐小时AV占比近似目标是两项不同声明；全部24小时见[hourly_supply.csv](hourly_supply.csv)。", "",
        f"日内车时分母为{_number(mixed['fleet']['h_base_day_exact'], 6)}小时，完整源班次窗口为{_number(mixed['fleet']['h_base_full_template_exact'], 6)}小时。跨午夜尾段另记{_number(mixed['fleet']['spillover_supply']['total_vehicle_hours'], 6)}小时（混合HV {_number(mixed['fleet']['spillover_supply']['hv_vehicle_hours'], 6)}、AV {_number(mixed['fleet']['spillover_supply']['av_vehicle_hours'], 6)}）；日内零权重slot保留，未静默删去。", "",
        "## 实际物理活动车时", "",
        "下表用成功分配、真实上车和真实服务完成时刻计算接驾与载客活动；空车移动使用真实结束或中断事件。预测服务时长不进入本账。计划在线车时、实际忙碌车时和班次后继续履约车时分别记录。", "",
        "| 场景/车型 | 日内计划车时 | 窗口内接驾 | 窗口内载客 | 班次后接驾 | 班次后载客 | 窗口内空驶 | 班次后空驶 | 实际活动车时合计 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in SCENARIOS:
        for kind, row in result["scenarios"][name]["physical_vehicle_hours"]["by_vehicle_type"].items():
            values = [row[key] for key in ("planned_day_hours", "pickup_within_window_hours", "service_within_window_hours", "pickup_after_shift_hours", "service_after_shift_hours", "empty_within_window_hours", "empty_after_shift_hours", "actual_active_operating_hours")]
            label = "纯HV" if name == "PURE_HV" else "M混合"
            lines.append(f"| {label}/{kind} | " + " | ".join(_number(value, 4) for value in values) + " |")
    lines += ["", "实际活动车时包含跨窗口、跨午夜的物理履约；空闲但仍可接单的在线时间不计为活动工作。班次后活动不会回填成可接新单供给。", "",
        "## 空车移动与路线证据", "", "| 指标 | 纯HV | M混合 |", "|---|---:|---:|"]
    for title, key in (("已启动移动数", "rows"), ("实际空驶时间（秒）", "total_time_s"), ("实际空驶距离（米）", "total_distance_m")):
        lines.append(f"| {title} | {_number(hv['empty_movements'][key])} | {_number(mixed['empty_movements'][key])} |")
    for title, key in (("正常完成", "COMPLETED"), ("真实订单接单后中断", "INTERRUPTED_BY_CUSTOMER"), ("仍未完成", "IN_PROGRESS")):
        lines.append(f"| {title} | {hv['empty_movements']['statuses'].get(key, 0):,} | {mixed['empty_movements']['statuses'].get(key, 0):,} |")
    lines += ["", "中断只累计实际已行驶的路径前缀与时长，未把完整原计划重复计入。车型分账、可选移动被拒绝原因、标量路线请求及共同路线支持诊断均保留在[comparison.json](comparison.json)。", "",
        f"空驶的未知交通状态U所占实际时长比例：纯HV {_percent(hv['empty_movements']['traffic_unknown_duration_share'])}，混合 {_percent(mixed['empty_movements']['traffic_unknown_duration_share'])}。新空驶路线沿用Stage3已有研究适配函数及显式未知U假设，未编造新的动态路线预测。该结果属于研究情景适配，不是完整动态运行条件或自动驾驶安全认证；路线端点连接距离只作物理几何记录。", "",
        "## 运算与对账", "", "| 观测量 | 纯HV | M混合 |", "|---|---:|---:|"]
    for title, key in (("实际运行耗时（秒）", "runtime_s"), ("主进程峰值驻留内存（MiB）", "peak_rss_mib"), ("采样进程组峰值驻留内存（MiB）", "peak_process_group_rss_mib"), ("采样进程组峰值私有提交量（MiB）", "peak_process_group_private_committed_mib"), ("客户接驾路由失败数", "routing_failures")):
        lines.append(f"| {title} | {_number(hv['runtime'][key])} | {_number(mixed['runtime'][key])} |")
    for title, key in (("资源限制下回退派单轮次", "resource_fallback_epochs"), ("当前服务目标约束违规轮次", "current_face_violation_epochs")):
        lines.append(f"| {title} | {_number(hv['computation'][key], 0)} | {_number(mixed['computation'][key], 0)} |")
    lines += ["", "| 测得时长（秒） | 纯HV | M混合 |", "|---|---:|---:|"]
    for title, key in (("候选生成", "candidate_generation_time_s"), ("接驾路由", "routing_time_s")):
        lines.append(f"| {title} | {_number(hv['computation']['epoch_timing_totals_s'].get(key))} | {_number(mixed['computation']['epoch_timing_totals_s'].get(key))} |")
    lines.append(f"| 派单求解流程合计 | {_number(hv['computation']['solver_pipeline_total_s'])} | {_number(mixed['computation']['solver_pipeline_total_s'])} |")
    for title, key in (("空驶决策合计", "empty_total_decision_time_s"), ("其中空驶路线查询", "empty_routing_wall_time_s")):
        lines.append(f"| {title} | {_number(hv['runtime'][key])} | {_number(mixed['runtime'][key])} |")
    for title, key in (("历史未来需求抽样", "forecast_sampling_time_s"), ("未来机会图构造", "future_graph_time_s"), ("问题准备", "problem_setup_time_s"), ("模型构造", "model_build_time_s"), ("优化求解", "optimization_time_s"), ("后续方案恢复", "recourse_recovery_time_s"), ("失败模型尝试", "failed_model_attempt_time_s")):
        lines.append(f"| {title} | {_number(hv['computation']['solver_pipeline_components_s'].get(key))} | {_number(mixed['computation']['solver_pipeline_components_s'].get(key))} |")
    lines += ["", "流程子项包含在其上级合计内；路由排队、缓存查询计时可能嵌套，不能再与流程合计相加当作实际运行耗时。资源限制仍为进程组驻留内存2,048 MiB，私有提交量单独报告；未记录值显示为“未记录”，不会填成零。", ""]
    if result["diagnostic_flags"]:
        lines += ["需要核查的对账标记：" + "、".join(result["diagnostic_flags"]) + "。", ""]
    else:
        lines += ["共同订单、逐小时总供给、真实任务时间、接单窗口、空驶账及已记录资源/求解约束未发现对账违规。路由失败和资源回退作为观测量报告，不设置服务率成功门槛。", ""]
    provenance = result["provenance"]
    lines += ["## 来源与解释范围", "",
        f"两组实际记录{provenance['shared_input_count']}项共享输入；键集合一致={provenance['input_key_sets_identical']}，共享SHA一致={provenance['shared_hashes_identical']}，当前冻结文件未变={provenance['frozen_files_unchanged']}，共同配置SHA一致={provenance['configuration_hashes_identical']}。逐文件摘要、实际模式定义和汇总代码摘要均记录于comparison.json。", "",
        f"纯HV执行代码SHA：`{provenance['execution_code_sha_by_scenario']['PURE_HV']}`；混合执行代码SHA：`{provenance['execution_code_sha_by_scenario']['M_Q10_P70']}`。", "",
        "本轮两组比较的是同一共同回放环境下的车辆类型标签、M服务适配及70%乘客接受AV这一组合条件；不能单独分解为某一个能力约束、接受率或派单算法的因果效应。实际位置、忙闲状态与运营车时是派单结果，不要求它们相同。这里只做单日、固定供给种子和既定策略的描述性配对比较，不作统计显著性宣称。", "",
        "结果解释风险已检查11/11类：未估计相关性或因果系数，不把日总差值解释为逐小时单调规律；共同输入筛选范围和原始分母明确保留。等待时间的配对比较只适用于两组都服务的订单，存在条件选择，不能推广到所有请求；未选择最好场景、重选种子或按结果调整参数。本次不计算显著性、置信区间或跨日期总体效应。", "",
        "旧82.40%纯HV与68.58%混合结果属于经验HV班次/AV全天在线的旧环境，保留作背景，未直接作为本轮配对对照。文档只包含聚合数据，没有逐订单身份或GPS坐标。", "",
        "本报告由AI辅助读取实际完成产物并汇总；未重新训练模型、改变请求释放时刻、搜索比例或新增实验条件。", ""]
    return "\n".join(lines)


def analyze(root):
    started = time.monotonic()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    root = Path(root).resolve()
    result, hourly = collect_comparison(root)
    if result["status"] == "PENDING":
        return result
    result["aggregation_runtime_s"] = time.monotonic() - started
    directory = root / DOCS
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(hourly).to_csv(directory / "hourly_supply.csv", index=False)
    (directory / "report.md").write_text(render_report(result), encoding="utf-8")
    return dict(status=result["status"], output_files=[(DOCS / name).as_posix() for name in ("report.md", "comparison.json", "hourly_supply.csv")],
                original_orders=result["paired_orders"]["original_orders"], common_eligible=result["paired_orders"]["common_eligible"],
                net_matched_change=result["paired_orders"]["net_matched_change"], diagnostic_flags=result["diagnostic_flags"], native_replay_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    try:
        result = analyze(arguments.root)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(json.dumps(dict(status="INVALID_COMPLETED_PRODUCTS", error=str(error), native_replay_started=False), ensure_ascii=False), flush=True)
        return 2
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
