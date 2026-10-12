"""Explain three completed layouts from projected products, never rerun them.

All-slot first-customer clocks retain unsuccessful admission windows. Spatial
attribution is to birth locations, not trip geography or local causal effects.
Historical body-demand and parked-HV inventories are descriptive opportunity
proxies, not legal pickup arcs, matching capacity, or new policy features.
"""
from __future__ import annotations

import argparse
import gc
import json
from math import cos, floor, isfinite, pi
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import psutil
from scipy.spatial import cKDTree

from .capability_chain_instances import atomic_json, atomic_parquet
from .c_operations_diagnostic import START, DAY_S, TOL, clipped_ledger, sha
from .scheme_a_two_scale_experiment import LAYOUT_RULES
from stage4.dispatch.controlled_fleet import build_controlled_fleet

SOURCE = Path("stage4/output/capability_chain_planning/layout_comparison_300_600_v1")
COMPARISON = Path("stage4/docs/capability_chain_planning/layout_comparison_300_600_v1/comparison.json")
DOC = Path("stage4/docs/capability_chain_planning/layout_mechanism_v1")
OUT = Path("stage4/output/capability_chain_planning/layout_mechanism_v1")
MODES = dict(zip(LAYOUT_RULES, ("all_hot", "compatible_hot", "opportunity_joint")))
GRID = .01
RADIUS_M = 2000.
DELAYS_S = (30, 60, 120, 300, 600, 900, 1800, 3600)
ASSIGN_COLUMNS = ["native_vehicle_id", "vehicle_type", "simulation_time_s", "pickup_time_s",
    "service_end_time_s", "completion_lon_wgs84", "completion_lat_wgs84", "completed"]
MOVE_COLUMNS = ["native_vehicle_id", "vehicle_type", "start_time_s", "end_time_s",
    "end_lon_wgs84", "end_lat_wgs84", "status"]


def xy(point):
    return float(point[0]) * 111320 * cos(34.25 * pi / 180), float(point[1]) * 110540


def cell(point):
    return f"{floor(float(point[0]) / GRID)}_{floor(float(point[1]) / GRID)}"


def cell_center(key):
    x, y = map(int, key.split("_"))
    return (x + .5) * GRID, (y + .5) * GRID


def slot_accounting(start, end, jobs, moves):
    """First commitment is not first boarding; no-customer time is censored.

    The all-slot restricted span includes repositioning. Pure unoccupied time
    subtracts recorded movement through the same disjoint activity ledger.
    A committed trip may board/finish after admission ends; loaded time clips.
    """
    jobs = sorted(jobs, key=lambda r: r[0])  # commitment, pickup, completion
    if any(not start - TOL <= a < end or not a <= p <= f for a, p, f in jobs):
        raise ValueError("customer event outside its original admission window")
    activities = [(a, p, "PICKUP") for a, p, f in jobs] + [(p, f, "LOADED") for a, p, f in jobs]
    activities += [(a, b, "MOVE") for a, b in moves]
    ledger = clipped_ledger(activities, start, end)
    lo, hi = max(0., start), min(DAY_S, end)
    first = jobs[0][0] if jobs else None
    pickup = jobs[0][1] if jobs else None
    cut = max(lo, min(hi, first)) if first is not None else hi
    before = clipped_ledger(activities, start, end, lo, max(lo, cut))
    result = dict(orders=len(jobs), admission_s=ledger["admission_s"], loaded_s=ledger["LOADED"],
        pickup_s=ledger["PICKUP"], move_s=ledger["MOVE"], idle_s=ledger["idle_s"],
        served=bool(jobs), first_assignment_delay_s=None if first is None else first - lo,
        first_boarding_delay_s=None if pickup is None else pickup - lo,
        first_assignment_in_calendar_window=first is not None and lo - TOL <= first < hi,
        first_boarding_in_calendar_window=pickup is not None and lo - TOL <= pickup < hi,
        restricted_first_customer_span_s=max(0., cut - lo), pre_first_idle_s=before["idle_s"],
        pre_first_move_s=before["MOVE"], post_first_idle_s=ledger["idle_s"] - before["idle_s"])
    if result["post_first_idle_s"] < -TOL:
        raise ValueError("first-customer partition lost unoccupied time")
    result["post_first_idle_s"] = max(0., result["post_first_idle_s"])
    return result


def activation_decomposition(before, after):
    """Exact algebra of number working × loaded hours per working slot.

    Symmetric allocation of the product interaction; not a causal component
    experiment and not conditioned-slot treatment-effect estimation.
    """
    n0, n1 = int(before.served.sum()), int(after.served.sum())
    l0, l1 = float(before.loaded_s.sum() / 3600), float(after.loaded_s.sum() / 3600)
    if not n0 or not n1:
        return dict(available=False, total_loaded_delta_h=l1 - l0, reason="ZERO_WORKING_SLOTS")
    m0, m1 = l0 / n0, l1 / n1
    extensive = (n1 - n0) * (m0 + m1) / 2
    intensive = (m1 - m0) * (n0 + n1) / 2
    if abs(extensive + intensive - (l1 - l0)) > TOL:
        raise ValueError("activation product decomposition does not reconcile")
    return dict(available=True, working_slots_before=n0, working_slots_after=n1,
        conditional_loaded_h_before=m0, conditional_loaded_h_after=m1,
        working_count_component_h=extensive, conditional_intensity_component_h=intensive,
        total_loaded_delta_h=l1 - l0, causal_component_effects_identified=False)


class ParkedInventory:
    """Observed event replay of admitted, unoccupied, non-moving HV only.

    Coordinates update at completed trips/moves. Busy or moving resources are
    excluded, not placed at guessed future destinations. Counts are not arcs.
    """
    def __init__(self, plans, jobs, moves, *, template_only=False):
        self.ids = [int(r.native_id) for r in plans.itertuples(index=False)]
        self.index = {vid: i for i, vid in enumerate(self.ids)}
        self.points = plans[["initial_lon_wgs84", "initial_lat_wgs84"]].to_numpy(float, copy=True)
        self.admitted = np.zeros(len(plans), bool)
        self.busy = np.zeros(len(plans), int)
        self.known = np.all(np.isfinite(self.points), axis=1)
        events = []
        for r in plans.itertuples(index=False):
            events.extend(((float(r.activation_s), 1, int(r.native_id), 0, None),
                           (float(r.admission_end_s), 0, int(r.native_id), 0, None)))
        if not template_only:
            for r in jobs.itertuples(index=False):
                events.extend(((float(r.simulation_time_s), 3, int(r.native_vehicle_id), 1, None),
                    (float(r.service_end_time_s), 2, int(r.native_vehicle_id), -1,
                     (float(r.completion_lon_wgs84), float(r.completion_lat_wgs84)))))
            for r in moves.itertuples(index=False):
                events.extend(((float(r.start_time_s), 3, int(r.native_vehicle_id), 1, None),
                    (float(r.end_time_s), 2, int(r.native_vehicle_id), -1,
                     (float(r.end_lon_wgs84), float(r.end_lat_wgs84)))))
        self.events = sorted(events, key=lambda e: (e[0], e[1], e[2]))
        self.cursor, self.now, self.tree, self.unknown_free = 0, -float("inf"), None, 0

    def advance(self, now):
        if now < self.now:
            raise ValueError("inventory snapshots must be chronological")
        self.now = now
        while self.cursor < len(self.events) and self.events[self.cursor][0] <= now + TOL:
            _, kind, vid, delta, point = self.events[self.cursor]
            i = self.index[vid]
            if kind < 2:
                self.admitted[i] = bool(kind)
            else:
                self.busy[i] += delta
                if point is not None:
                    self.known[i] = all(isfinite(v) for v in point)
                    self.points[i] = point
            self.cursor += 1
        if np.any(self.busy < 0):
            raise ValueError("physical inventory activity counter became negative")
        free = self.admitted & (self.busy == 0)
        self.unknown_free = int(np.count_nonzero(free & ~self.known))
        points = self.points[free & self.known]
        self.tree = cKDTree(np.asarray([xy(p) for p in points])) if len(points) else None

    def nearby(self, point):
        return int(self.tree.query_ball_point(xy(point), RADIUS_M, return_length=True)) if self.tree is not None else 0


def summarize(frame):
    working = frame.loc[frame.served]
    quantiles = lambda s: dict(mean=float(s.mean()), p50=float(s.median()), p90=float(s.quantile(.9))) if len(s) else None
    out = dict(slots=len(frame), served_slots=len(working), unserved_slots=len(frame) - len(working),
        served_slot_share=len(working) / len(frame), orders=int(frame.orders.sum()),
        orders_per_served_slot=float(working.orders.mean()), loaded_h=float(frame.loaded_s.sum() / 3600),
        pre_first_idle_h=float(frame.pre_first_idle_s.sum() / 3600),
        post_first_idle_h=float(frame.post_first_idle_s.sum() / 3600),
        pre_first_move_h=float(frame.pre_first_move_s.sum() / 3600),
        restricted_first_customer_span_h=float(frame.restricted_first_customer_span_s.sum() / 3600),
        first_assignment_delay_served_only_s=quantiles(working.first_assignment_delay_s),
        first_boarding_delay_served_only_s=quantiles(working.first_boarding_delay_s),
        first_assignment_in_calendar_slots=int(frame.first_assignment_in_calendar_window.sum()),
        first_boarding_in_calendar_slots=int(frame.first_boarding_in_calendar_window.sum()),
        all_slots_retained=True, first_customer_span_includes_repositioning=True,
        failed_slots_are_not_zero_waits=True, survival_distribution_estimated=False)
    out["first_customer_cumulative_all_slot_rates"] = [dict(delay_s=t,
        assignment_share=float((frame.first_assignment_in_calendar_window & frame.first_assignment_delay_s.le(t)).mean()),
        boarding_within_admission_share=float((frame.first_boarding_in_calendar_window & frame.first_boarding_delay_s.le(t)).mean())) for t in DELAYS_S]
    for col in ("historical_body_C_count_3days", "historical_all_count_3days", "historical_body_C_predicted_h_3days",
                "template_HV_nearby", "baseline_parked_HV_nearby", "own_parked_HV_nearby",
                "unknown_parked_HV_position_count"):
        out[col] = quantiles(frame[col])
    return out


def paired(before, after):
    if not before.index.equals(after.index) or not before[["activation_s", "admission_end_s"]].equals(after[["activation_s", "admission_end_s"]]):
        raise ValueError("C slot identities or original windows differ")
    work = after.copy()
    work["transition"] = np.select([~before.served & after.served, before.served & ~after.served,
        before.served & after.served], ["NEWLY_SERVED", "LOST_SERVICE", "BOTH_SERVED"], default="NEITHER_SERVED")
    work["orders_delta"] = after.orders - before.orders
    work["loaded_delta_h"] = (after.loaded_s - before.loaded_s) / 3600
    work["pre_first_idle_delta_h"] = (after.pre_first_idle_s - before.pre_first_idle_s) / 3600
    work["post_first_idle_delta_h"] = (after.post_first_idle_s - before.post_first_idle_s) / 3600
    transitions = []
    for label, f in work.groupby("transition", sort=True):
        transitions.append(dict(transition=label, slots=len(f), orders_delta=int(f.orders_delta.sum()),
            loaded_delta_h=float(f.loaded_delta_h.sum()), pre_first_idle_delta_h=float(f.pre_first_idle_delta_h.sum()),
            post_first_idle_delta_h=float(f.post_first_idle_delta_h.sum())))
    cells = []
    for key, f in work.groupby("birth_cell", sort=True):
        cx, cy = cell_center(key)
        cells.append(dict(birth_cell=key, center_lon_coarse=cx, center_lat_coarse=cy, slots=len(f),
            newly_served=int(f.transition.eq("NEWLY_SERVED").sum()), lost_service=int(f.transition.eq("LOST_SERVICE").sum()),
            loaded_delta_h=float(f.loaded_delta_h.sum()), orders_delta=int(f.orders_delta.sum()),
            supported_for_rank=len(f) >= 10))
    hours = [dict(activation_hour=int(hour), slots=len(f), newly_served=int(f.transition.eq("NEWLY_SERVED").sum()),
        lost_service=int(f.transition.eq("LOST_SERVICE").sum()), loaded_delta_h=float(f.loaded_delta_h.sum()),
        orders_delta=int(f.orders_delta.sum())) for hour, f in work.groupby("activation_hour", sort=True)]
    subsets = {}
    for label in ("NEWLY_SERVED", "LOST_SERVICE", "BOTH_SERVED", "NEITHER_SERVED"):
        mask = work.transition.eq(label)
        rows = {}
        for col in ("historical_body_C_count_3days", "historical_body_C_predicted_h_3days",
                    "template_HV_nearby", "baseline_parked_HV_nearby"):
            rows[col] = dict(before_mean=float(before.loc[mask, col].mean()), after_mean=float(after.loc[mask, col].mean())) if mask.any() else None
        subsets[label] = dict(slots=int(mask.sum()), opportunity_proxies=rows)
    return dict(transitions=transitions, activation_product_decomposition=activation_decomposition(before, after),
        loaded_delta_h=float(work.loaded_delta_h.sum()), orders_delta=int(work.orders_delta.sum()),
        birth_cells_after_layout=cells, activation_hours=hours, transition_proxies=subsets,
        spatial_grouping_by_after_birth_location_not_service_location=True,
        local_treatment_effect_or_HV_competition_causal_component_identified=False)


def analyze(root):
    started = perf_counter()
    root = Path(root).resolve()
    comparison = json.loads((root / COMPARISON).read_text(encoding="utf-8"))
    if comparison["status"] != "PREPARED_LAYOUT_COMPARISON_ANALYZED":
        raise ValueError("requires the completed unchanged three-layout comparison")
    cfg = json.loads((root / SOURCE / next(iter(MODES)) / "summary.json").read_text())["frozen_protocol"]
    inputs = dict(comparison["execution_binding"]["inputs_sha256"])
    inputs[COMPARISON.as_posix()] = sha(root / COMPARISON)
    for group in MODES:
        for name in ("assignments.parquet", "empty_movements.parquet", "summary.json"):
            key = (SOURCE / group / name).as_posix()
            digest = comparison["groups"][group]["source_products_sha256"][name]
            if sha(root / key) != digest:
                raise ValueError("a completed native product changed")
            inputs[key] = digest
    if any(sha(root / path) != digest for path, digest in inputs.items()):
        raise ValueError("a frozen analysis input changed")

    def check():
        memory = psutil.Process().memory_info()
        peak = max(memory.rss, getattr(memory, "peak_wset", memory.rss)) / 2**20
        if peak > 512:
            raise MemoryError("projected mechanism analysis exceeds 512 MiB")
        if perf_counter() - started > 180:
            raise TimeoutError("offline mechanism analysis exceeds 180 seconds")
        return peak

    fleet = build_controlled_fleet(root, benchmark_start=START,
        simulation_end=START + pd.Timedelta(seconds=cfg["physical_drain_limit_s"]),
        requested_q_a=cfg["requested_q_a"], seed=cfg["supply_seed"])
    if fleet.accounting["type_label_sha256"] != comparison["type_label_sha256"]:
        raise ValueError("reconstructed source supply differs")
    plans = fleet.scenario_fleet.copy()
    plans["activation_s"] = (plans.availability_start_time - START).dt.total_seconds()
    plans["admission_end_s"] = (plans.availability_end_time - START).dt.total_seconds()
    del fleet
    av, hv = plans.loc[plans.vehicle_type.eq("AV")].copy(), plans.loc[plans.vehicle_type.eq("HV")].copy()
    av.sort_values(["activation_s", "native_id"], inplace=True, kind="stable")
    layout = json.loads((root / cfg["source_layout"]).read_text(encoding="utf-8"))["placements"]
    snapshots, slots = {}, {}
    for group, mode in MODES.items():
        jobs = pd.read_parquet(root / SOURCE / group / "assignments.parquet", columns=ASSIGN_COLUMNS)
        moves = pd.read_parquet(root / SOURCE / group / "empty_movements.parquet", columns=MOVE_COLUMNS)
        if not jobs.completed.all() or not moves.status.isin(["COMPLETED", "INTERRUPTED_BY_CUSTOMER"]).all():
            raise ValueError("unfinished activity cannot be analyzed as a complete event ledger")
        snapshots[group] = ParkedInventory(hv, jobs.loc[jobs.vehicle_type.eq("HV")], moves.loc[moves.vehicle_type.eq("HV")])
        cjobs = {int(vid): [(float(r.simulation_time_s), float(r.pickup_time_s), float(r.service_end_time_s))
            for r in f.itertuples(index=False)] for vid, f in jobs.loc[jobs.vehicle_type.eq("AV")].groupby("native_vehicle_id")}
        cmoves = {int(vid): [(float(r.start_time_s), float(r.end_time_s)) for r in f.itertuples(index=False)]
            for vid, f in moves.loc[moves.vehicle_type.eq("AV")].groupby("native_vehicle_id")}
        rows = []
        for r in av.itertuples(index=False):
            vid = int(r.native_id)
            point = layout[mode][str(r.vehicle_id)]["position"]
            row = slot_accounting(float(r.activation_s), float(r.admission_end_s), cjobs.get(vid, []), cmoves.get(vid, []))
            rows.append(dict(row, native_id=vid, activation_s=float(r.activation_s), admission_end_s=float(r.admission_end_s),
                activation_hour=int(max(0, r.activation_s) // 3600), birth_lon=float(point[0]), birth_lat=float(point[1]),
                birth_cell=cell(point)))
        slots[group] = pd.DataFrame(rows).set_index("native_id").sort_index()
        expected = comparison["groups"][group]
        if (int(slots[group].orders.sum()) != expected["served_C"]
                or int(slots[group].served.sum()) != 851 - expected["admission_state"]["AV"]["never_served_slots"]
                or abs(slots[group].loaded_s.sum() / 3600 - expected["admission_state"]["AV"]["LOADED_hours"]) > TOL
                or abs(slots[group].idle_s.sum() / 3600 - expected["admission_state"]["AV"]["idle_hours"]) > TOL):
            raise ValueError("C-slot events do not reconcile with the existing full-day result")
        del jobs, moves, cjobs, cmoves
        gc.collect()
        check()
        print(json.dumps(dict(projected_group_loaded=group)), flush=True)
    template = ParkedInventory(hv, None, None, template_only=True)
    hcols = ["release_second", "start_lon_wgs84", "start_lat_wgs84", "predicted_route_time_p50_s",
        "research_data_ready", "common_eligible", "compatible_C", "date"]
    history = pd.read_parquet(root / cfg["source_history_templates"], columns=hcols)
    if set(history.date.astype(str)) != set(cfg["forecast_train_dates"]):
        raise ValueError("historical opportunity proxy has unexpected dates")
    history = history.loc[history.research_data_ready & history.common_eligible].sort_values("release_second", kind="stable")
    times = history.release_second.to_numpy(float)
    hx = np.asarray([xy(p) for p in history[["start_lon_wgs84", "start_lat_wgs84"]].to_numpy(float)])
    hc = history.compatible_C.to_numpy(bool)
    ht = history.predicted_route_time_p50_s.to_numpy(float)
    if not np.all(np.isfinite(ht)) or np.any(ht <= 0):
        raise ValueError("historical M3 service-time proxy is invalid")
    base = next(iter(MODES))
    for ordinal, (now, born) in enumerate(av.groupby("activation_s", sort=True)):
        check()
        for inventory in (*snapshots.values(), template):
            inventory.advance(float(now))
        trees = {}
        for r in born.itertuples(index=False):
            stop = min(float(r.admission_end_s), float(now) + 1800, DAY_S)
            lo = int(np.searchsorted(times, max(0., now - 300), side="left"))
            hi = int(np.searchsorted(times, stop, side="left"))
            if hi > lo and (lo, hi) not in trees:
                trees[lo, hi] = cKDTree(hx[lo:hi])
            for group in MODES:
                row = slots[group].loc[int(r.native_id)]
                point = (row.birth_lon, row.birth_lat)
                indices = np.asarray(trees[lo, hi].query_ball_point(xy(point), RADIUS_M), int) + lo if hi > lo else np.asarray([], int)
                compatible = indices[hc[indices]]
                values = dict(historical_all_count_3days=len(indices), historical_body_C_count_3days=len(compatible),
                    historical_body_C_predicted_h_3days=float(ht[compatible].sum() / 3600),
                    template_HV_nearby=template.nearby(point),
                    baseline_parked_HV_nearby=snapshots[base].nearby(point),
                    own_parked_HV_nearby=snapshots[group].nearby(point),
                    unknown_parked_HV_position_count=snapshots[group].unknown_free)
                for col, value in values.items():
                    slots[group].loc[int(r.native_id), col] = value
        if ordinal % 200 == 0:
            print(json.dumps(dict(birth_time_groups_processed=ordinal + 1, runtime_s=perf_counter() - started)), flush=True)
    public = {group: summarize(frame) for group, frame in slots.items()}
    group_names = tuple(MODES)
    pairs = {f"{a}_to_{b}": paired(slots[a], slots[b]) for a, b in
        ((group_names[0], group_names[1]), (group_names[1], group_names[2]), (group_names[0], group_names[2]))}
    if any(sha(root / path) != digest for path, digest in inputs.items()):
        raise ValueError("a source changed during read-only mechanism analysis")
    payload = dict(status="LAYOUT_MECHANISM_ANALYZED_EXISTING_PRODUCTS", verification_status="ANALYZED_NOT_NEW_CAUSAL_EXPERIMENT",
        analysis_source_sha256=sha(Path(__file__)), native_producer_sha=comparison["execution_binding"]["code_sha"],
        source_input_sha256=inputs, groups=public, paired=pairs, projected_calendar_window_s=[0, DAY_S],
        historical_proxy_dates=cfg["forecast_train_dates"], radius_m=RADIUS_M, birth_cell_grid_degrees=GRID,
        historical_window="RELEASE_FROM_ACTIVATION_MINUS_300_TO_MIN_ADMISSION_END_ACTIVATION_PLUS_1800",
        historical_proxy_is_body_compatibility_not_a_legal_pickup_arc=True,
        parked_HV_proxy_excludes_moving_and_busy_and_unknown_positions=True,
        baseline_HV_reference_is_all_hot_replay_not_exogenous_supply=True,
        own_replay_HV_states_are_endogenous_descriptive_mediators=True,
        per_vehicle_or_exact_coordinates_in_git=False, new_simulation=False, new_routing=False,
        new_inference=False, model_or_threshold_change=False, dense_matrix=False, gpu_used=False,
        all_slots_retained=True, independent_component_effects_identified=False,
        runtime_s=perf_counter() - started, peak_rss_mib=check())
    for group, frame in slots.items():
        atomic_parquet(root / OUT / f"{group.lower()}_slots.parquet", frame.reset_index())
    payload.update(runtime_s=perf_counter() - started, peak_rss_mib=check())
    atomic_json(root / DOC / "mechanism.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = analyze(args.root)
    print(json.dumps(dict(status=result["status"], runtime_s=result["runtime_s"], peak_rss_mib=result["peak_rss_mib"])), flush=True)


if __name__ == "__main__":
    main()
