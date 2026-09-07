"""Bounded MVC analysis on the ten existing frozen canonical decision states.

No vehicle progression or full-day simulation. Production time_trigger is
intercepted at its solver call to inspect its actual candidate construction.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from stage4.analysis import frozen_state_prediction_ablation as frozen
from stage4.dispatch import rolling_or_control as production
from stage4.dispatch.deterministic_routing import ArcDeterministicValhallaAdapter, SINGLE_SOURCE_MATRIX
from stage4.dispatch.exposure import CumulativeExposureState, ExposureExcess
from stage4.dispatch.solver import solve_lexicographic

OUT = Path('stage4/output/paper_enhancement/mechanism_validity')


def graph_metrics(pairs):
    """Compute E/U/nu from exactly the same sparse bipartite edge set."""
    pairs = sorted(set(pairs))
    if not pairs:
        return {'E': 0, 'U': 0, 'M': 0}
    orders = {v: i for i, v in enumerate(sorted({o for o, _ in pairs}))}
    vehicles = {v: i for i, v in enumerate(sorted({v for _, v in pairs}))}
    graph = csr_matrix((np.ones(len(pairs), dtype=np.int8),
                        ([orders[o] for o, v in pairs], [vehicles[v] for o, v in pairs])),
                       shape=(len(orders), len(vehicles)))
    matching = maximum_bipartite_matching(graph, perm_type='column')
    return {'E': len(pairs), 'U': len(orders), 'M': int((matching >= 0).sum())}


class _Captured(Exception):
    pass


def production_neutral_arcs(vehicles, fixtures, waiting, timestamp, start, config, adapter):
    """Exercise the production branch, disabling only explicit AV restrictions.

    Positions, IDs, windows, predicted durations and request state are shared.
    An implicit vehicle-label-dependent window branch is intentionally NOT
    patched out: finding such a branch is the purpose of this diagnostic.
    """
    core = production._RollingORFleetControlCore.__new__(production._RollingORFleetControlCore)
    sim_s = int((timestamp - start).total_seconds())
    core.repositioning_manager = None
    core.run_started_perf = time.perf_counter()
    core.runtime_guard_s = 3600
    core.matching_end_s = 90000
    core.config = config
    core.dispatch_interval_s = 30
    core.rid_to_assigned_vid = {}
    core.demand = SimpleNamespace(waiting_rq={r.native_id: None for r, *_ in waiting})
    core.runtime_by_vid = {v.native_vehicle_id: SimpleNamespace(
        fixture=replace(fixtures[v.native_vehicle_id], vehicle_type=v.vehicle_type), spatial=v)
        for v in vehicles}
    core._available = lambda runtime, t: True  # already frozen as available
    core._spatial_vehicle = lambda runtime: runtime.spatial
    core._timestamp = lambda t: start + pd.Timedelta(seconds=t)
    core._fixture_seconds = lambda ts: (pd.Timestamp(ts) - start).total_seconds()
    core.request_by_rid = {}
    core.request_meta = {}
    for r, failed, carry, critical in waiting:
        neutral = SimpleNamespace(**r.__dict__)
        neutral.av_smoke_eligible = True
        core.request_by_rid[r.native_id] = neutral
        core.request_meta[r.native_id] = {
            'first_attempt_time': None, 'attempt_count': 0,
            'pickup_deadline_s': r.sim_time_s + 300,
            'entered_critical': critical, 'failed_round_count': failed,
            'carry_over_flag': carry, 'passenger_accepts_av': True,
            'exposure': ExposureExcess(0., 0., 0.)}
    core.prospective_gate_logging = False
    core.eta_adapter = adapter
    core.cost_level_enabled = False
    core.eta_cost_av_to_hv = 1.
    core.hv_session_end_exclusions = set()
    core.candidate_generation_time_s = 0.
    core.av_candidates_pruned_by_acceptance = 0
    core.av_candidates_pruned_by_missing_exposure = 0
    core.exposure_state = CumulativeExposureState()
    core.gammas = {f: None for f in ('static', 'dynamic', 'speed')}
    core.pickup_cost_epsilon = 0.
    core.solver_tolerance = 1e-7
    captured = []
    def capture(arcs, **kwargs):
        captured.extend(arcs)
        raise _Captured
    with patch.object(production, 'solve_lexicographic', capture):
        try:
            core.time_trigger(sim_s)
        except _Captured:
            pass
    return captured


def _arc_summary(arcs):
    pairs = sorted((a.request_id, a.vehicle_id) for a in arcs)
    solved = solve_lexicographic(arcs, cost_level_enabled=False)
    selected = sorted((arcs[i].request_id, arcs[i].vehicle_id) for i in solved.selected_indices)
    return {**graph_metrics(pairs), 'candidate_identity': json.dumps(pairs),
            'candidate_sha256': frozen._sha(pairs),
            'selected_count': len(selected), 'selected_identity': json.dumps(selected),
            'selected_sha256': frozen._sha(selected),
            'pickup_objective_s': solved.pickup_eta_optimum_s}


def load_states(root):
    """Use exactly the prior frozen-state reconstruction and verify its hashes."""
    scenario = root / frozen.SCENARIO_REL
    config = json.loads((scenario / 'scenario_config.json').read_text())['runtime_configuration']
    start = pd.Timestamp('2016-10-31T00:00:00+08:00')
    requests = frozen.load_all_test31_requests(root, start=start, end=start + pd.Timedelta(days=1), profile_id='M')
    assignments = pd.read_parquet(scenario / 'assignment_log.parquet')
    for c in ('assignment_time', 'service_end_time'):
        assignments[c] = pd.to_datetime(assignments[c], utc=True).dt.tz_convert('Asia/Shanghai')
    exposure = pd.read_parquet(scenario / 'exposure_state.parquet')
    # Restore the pre-existing session selection verbatim; no new sampling rule.
    fleet = frozen.build_fleet_scenario(root, benchmark_start=start,
        simulation_end=start + pd.Timedelta(days=1, seconds=max(r.realized_service_time_s for r in requests) + 60),
        requested_q_a=.5, seed=20260824, max_hv_hour_error_pct=2.)
    registry = pd.read_csv(root / frozen.OUTPUT_REL / 'frozen_epoch_registry.csv', dtype={'epoch_id': str})
    for row in registry.itertuples(index=False):
        ts = pd.Timestamp(row.timestamp)
        sim_s = int((ts - start).total_seconds())
        vehicles = frozen._vehicle_state(fleet.native_fixtures, assignments, ts)
        waiting = frozen._waiting_requests(requests, assignments, sim_s)
        state = frozen._exposure_before(exposure, sim_s)
        payload = {'timestamp': ts.isoformat(), 'waiting': [r.order_id for r, *_ in waiting],
            'vehicles': [(v.vehicle_id, v.vehicle_type, round(v.lon_wgs84, 7), round(v.lat_wgs84, 7)) for v in vehicles],
            'exposure': state.__dict__,
            'candidate_rules': {k: config[k] for k in ('search_radius_initial_m','search_radius_step_m','search_radius_cap_m','candidate_top_k','max_pickup_wait_s')}}
        if frozen._sha(payload) != row.state_sha256:
            raise RuntimeError(f'Frozen state mismatch: {row.epoch_id}')
        yield row, vehicles, waiting, fleet.native_fixtures, config, start


def run(root):
    root = Path(root).resolve()
    output = root / OUT
    output.mkdir(parents=True, exist_ok=True)
    adapter = ArcDeterministicValhallaAdapter(root, routing_mode=SINGLE_SOURCE_MATRIX)
    neutral_rows = []
    started = time.perf_counter()
    for registry, vehicles, waiting, fixtures, config, start in load_states(root):
        ts = pd.Timestamp(registry.timestamp)
        originals = production_neutral_arcs(vehicles, fixtures, waiting, ts, start, config, adapter)
        relabeled = [replace(v, vehicle_type='AV') for v in vehicles]
        alternate = production_neutral_arcs(relabeled, fixtures, waiting, ts, start, config, adapter)
        a, b = _arc_summary(originals), _arc_summary(alternate)
        same_graph = a['candidate_sha256'] == b['candidate_sha256']
        same_assignment = a['selected_sha256'] == b['selected_sha256']
        for name, values in [('HV_ORIGINAL', a), ('AV_NEUTRAL', b)]:
            neutral_rows.append({'epoch_id': registry.epoch_id, 'variant': name,
                'physical_state_sha256': registry.state_sha256, **values,
                'candidate_graph_identical': same_graph, 'assignment_identical': same_assignment})
        frozen._atomic_csv(pd.DataFrame(neutral_rows), output / 'neutral_av_identity.csv')
        print(json.dumps({'epoch': registry.epoch_id, 'neutral_graph_equal': same_graph,
                          'neutral_assignment_equal': same_assignment, 'E_original': a['E'], 'E_neutral': b['E']}), flush=True)
        if not same_graph or not same_assignment:
            # Required B5 stop: report actual branch difference before any
            # mechanism interpretation or additional state analysis.
            summary = {'classification': 'IMPLEMENTATION_DEFECT',
                'neutral_classification': 'HIDDEN_ASYMMETRY',
                'first_differing_epoch': registry.epoch_id,
                'snapshots_tested': len(neutral_rows) // 2,
                'cause': 'production HV-only predicted-service-end / finite-service gate is removed by relabeling identical session to AV',
                'runtime_s': time.perf_counter() - started,
                'routing_arcs': adapter.routing_arc_evaluations,
                'routing_failures': adapter.routing_failures,
                'state_identity_verified': True, 'full_day_simulation': False}
            (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            return summary
    summary = {'classification': 'PASS_IDENTITY', 'snapshots_tested': len(neutral_rows)//2}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    print(json.dumps(run(parser.parse_args().root), indent=2), flush=True)
