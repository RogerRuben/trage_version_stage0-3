"""Bounded frozen-state arc reconstruction and offline assignment-gap diagnosis."""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import psutil
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, vstack

from stage4.analysis import mechanism_validity as mv
from stage4.analysis import frozen_state_prediction_ablation as frozen
from stage4.dispatch.deterministic_routing import ArcDeterministicValhallaAdapter, SINGLE_SOURCE_MATRIX
from stage4.dispatch.exposure import CumulativeExposureState

OUT = Path('stage4/output/paper_enhancement/control_freedom_structure')
TOL = 1e-7
ALPHAS = (0, .005, .01, .02, .05)
START = time.perf_counter()
PEAK = 0


def guard():
    global PEAK
    rss = psutil.Process().memory_info().rss
    PEAK = max(PEAK, rss)
    if rss > 2 * 1024**3 or time.perf_counter()-START > 600:
        raise RuntimeError('RESOURCE_LIMIT')


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


class LimitedAdapter(ArcDeterministicValhallaAdapter):
    def estimate_many(self, candidates, *args):
        guard()
        self.state_lookups += len(candidates)
        if self.state_lookups > 5000 or self.routing_arc_evaluations + len(candidates) > 20000:
            raise RuntimeError('ROUTING_ARC_BUDGET')
        return super().estimate_many(candidates, *args)


class Problem:
    def __init__(self, arcs, state, gammas):
        self.arcs = arcs
        self.n = len(arcs)
        self.orders = sorted({a.request_id for a in arcs})
        self.rows, self.low, self.high = [], [], []
        for attr in ('vehicle_id', 'request_id'):
            groups = {}
            for i, a in enumerate(arcs):
                groups.setdefault(getattr(a, attr), {})[i] = 1.
            for coeff in groups.values():
                self.add(coeff, -np.inf, 1)
        for family, gamma in gammas.items():
            if gamma is not None:
                self.add({i: getattr(a, 'exposure_'+family)-gamma for i, a in enumerate(arcs) if a.vehicle_type == 'AV'},
                         -np.inf, gamma*state.av_assignments-getattr(state, family))
        self.eta = np.array([a.pickup_eta_s for a in arcs])
        self.levels = [np.array([a.critical for a in arcs], float), np.ones(self.n),
                       np.array([a.carry_over for a in arcs], float)]

    def add(self, coeff, low, high):
        if isinstance(coeff, dict):
            ids = list(coeff)
            row = csr_matrix(([coeff[i] for i in ids], ([0]*len(ids), ids)), shape=(1, self.n))
        else:
            row = csr_matrix(np.asarray(coeff).reshape(1, -1))
        self.rows.append(row)
        self.low.append(low)
        self.high.append(high)

    def solve(self, objective):
        guard()
        matrix = vstack(self.rows, format='csr')
        began = time.perf_counter()
        result = milp(objective, integrality=np.ones(self.n), bounds=Bounds(0, 1),
                      constraints=LinearConstraint(matrix, self.low, self.high),
                      options={'time_limit': 10., 'mip_rel_gap': 0., 'presolve': True})
        result.elapsed = time.perf_counter()-began
        result.binary = None
        if result.x is not None:
            binary = (result.x > .5).astype(float)
            values = matrix @ binary
            if (np.max(np.abs(result.x-binary)) > TOL or
                    np.any(values < np.asarray(self.low)-TOL) or np.any(values > np.asarray(self.high)+TOL)):
                raise RuntimeError('INVALID_INTEGER_WITNESS')
            result.binary = binary
        guard()
        return result


def require_optimal(result):
    if result.status != 0 or result.binary is None:
        raise RuntimeError('BASELINE_NOT_CERTIFIED: '+str(result.message))


def diagnose(arcs, state, gammas, historical):
    if not arcs:
        raise RuntimeError('EMPTY_GRAPH_REQUIRES_EXPLICIT_REFERENCE_CHECK')
    problem = Problem(arcs, state, gammas)
    optima = []
    for level in problem.levels:
        result = problem.solve(-level)
        require_optimal(result)
        value = int(round(level @ result.binary))
        optima.append(value)
        problem.add(level, value, value)
    result = problem.solve(problem.eta)
    require_optimal(result)
    pstar = float(problem.eta @ result.binary)
    historical_pairs = {(str(r.order_id), str(r.vehicle_id)) for r in historical.itertuples()}
    x0 = np.array([tuple(a.payload) in historical_pairs for a in arcs], float)
    if int(x0.sum()) != len(historical_pairs) or any(int(level @ x0) != value for level, value in zip(problem.levels, optima)):
        raise RuntimeError('HISTORICAL_REFERENCE_IDENTITY_OR_COUNTS_MISMATCH')
    if abs(float(problem.eta @ x0)-pstar) > TOL:
        raise RuntimeError('HISTORICAL_PICKUP_OBJECTIVE_MISMATCH')
    hist_eta = {(str(r.order_id), str(r.vehicle_id)): float(r.pickup_eta_s) for r in historical.itertuples()}
    if any(abs(a.pickup_eta_s-hist_eta[tuple(a.payload)]) > TOL for i, a in enumerate(arcs) if x0[i]):
        raise RuntimeError('HISTORICAL_SELECTED_ARC_ETA_MISMATCH')
    matrix = vstack(problem.rows, format='csr')
    if np.any(matrix @ x0 < np.asarray(problem.low)-TOL) or np.any(matrix @ x0 > np.asarray(problem.high)+TOL):
        raise RuntimeError('HISTORICAL_GAMMA_INFEASIBLE')
    base_len = len(problem.rows)
    base_selected = {i for i, selected in enumerate(x0) if selected}
    y0 = {arcs[i].request_id for i in base_selected}
    b0 = {arcs[i].request_id for i in base_selected if arcs[i].vehicle_type == 'AV'}
    results = []
    for mode in ('A1', 'A2'):
        for alternative in ('ANY_ARC_CHANGE', 'AV_SERVICE_INDICATOR_CHANGE'):
            problem.rows = problem.rows[:base_len]
            problem.low = problem.low[:base_len]
            problem.high = problem.high[:base_len]
            if mode == 'A1':
                for order in problem.orders:
                    problem.add({i: 1. for i, a in enumerate(arcs) if a.request_id == order}, int(order in y0), int(order in y0))
            if alternative == 'ANY_ARC_CHANGE':
                problem.add({i: 1. for i in base_selected}, -np.inf, len(base_selected)-1)
            else:
                problem.add({i: (-1. if a.request_id in b0 else 1.) for i, a in enumerate(arcs) if a.vehicle_type == 'AV'}, 1-len(b0), np.inf)
            r = problem.solve(problem.eta)
            incumbent = float(problem.eta @ r.binary) if r.binary is not None else None
            bound = getattr(r, 'mip_dual_bound', None)
            bound = float(bound) if bound is not None and np.isfinite(bound) else None
            if r.status == 0:
                require_optimal(r)
                bound = incumbent
            if incumbent is not None and incumbent < pstar-TOL:
                raise RuntimeError('NEGATIVE_ALTERNATIVE_GAP')
            status = 'OPTIMAL' if r.status == 0 else ('INFEASIBLE' if r.status == 2 else 'UNRESOLVED')
            row = {'mode': mode, 'alternative': alternative, 'status': status, 'solver_message': str(r.message),
                   'pstar_s': pstar, 'alternative_pickup_s': incumbent,
                   'gap_upper_s': None if incumbent is None else max(0., incumbent-pstar),
                   'gap_lower_s': None if bound is None else max(0., bound-pstar), 'runtime_s': r.elapsed,
                   'gap_relative': None if incumbent is None or pstar == 0 else max(0., incumbent-pstar)/pstar,
                   'budget_exists': {}, 'witness': []}
            for alpha in ALPHAS:
                budget = pstar*(1+alpha)
                row['budget_exists'][str(alpha)] = ('YES_WITHIN_NUMERICAL_TOLERANCE' if incumbent is not None and incumbent <= budget+TOL else
                    'NO' if status == 'INFEASIBLE' or (bound is not None and bound > budget+TOL) else 'UNRESOLVED')
            if r.binary is not None:
                selected = {i for i, chosen in enumerate(r.binary) if chosen}
                row['served_order_set_changed'] = {arcs[i].request_id for i in selected} != y0
                row['vehicle_identity_set_changed'] = {arcs[i].vehicle_id for i in selected} != {arcs[i].vehicle_id for i in base_selected}
                base_types = {arcs[i].request_id: arcs[i].vehicle_type for i in base_selected}
                new_types = {arcs[i].request_id: arcs[i].vehicle_type for i in selected}
                row['same_order_hv_av_swap_count'] = sum(base_types[o] != new_types[o] for o in base_types.keys() & new_types.keys())
                row['witness'] = [{'order_id': arcs[i].payload[0], 'vehicle_id': arcs[i].payload[1],
                                   'vehicle_type': arcs[i].vehicle_type, 'pickup_eta_s': arcs[i].pickup_eta_s,
                                   'selected_before': i in base_selected, 'selected_after': i in selected}
                                  for i in sorted(selected ^ base_selected)]
            results.append(row)
    return {'counts': optima, 'pstar_s': pstar, 'reference': 'HISTORICAL_P_SELECTED_SET_VERIFIED_OPTIMAL', 'results': results}


def run(root):
    output = root / OUT
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / 'summary.json'
    if summary_path.exists():
        raise RuntimeError('OUTPUT_EXISTS: preserve prior diagnostic; do not silently overwrite')
    historical = pd.read_csv(root / frozen.OUTPUT_REL / 'selected_assignments_diagnostic.csv', dtype={'epoch_id': str})
    old_decisions = pd.read_csv(root / frozen.OUTPUT_REL / 'decision_variant_metrics.csv', dtype={'epoch_id': str})
    config_path = root / frozen.SCENARIO_REL / 'scenario_config.json'
    stage3_config = json.loads((root/'stage3/config/stage3_finalization.json').read_text())
    routing_config = Path(stage3_config['valhalla_config'])
    tile_dir = Path(json.loads(routing_config.read_text())['mjolnir']['tile_dir'])
    tile_files = sorted(tile_dir.rglob('*.gph'))
    tile_inventory = [(str(p.relative_to(tile_dir)), p.stat().st_size, p.stat().st_mtime_ns) for p in tile_files]
    provenance = {'source_config_sha256': sha(config_path), 'registry_sha256': sha(root/frozen.OUTPUT_REL/'frozen_epoch_registry.csv'),
                  'routing_config_path': str(routing_config), 'routing_config_sha256': sha(routing_config),
                  'routing_mode': SINGLE_SOURCE_MATRIX, 'costing': 'auto', 'clock': 'Asia/Shanghai minute precision',
                  'beta_sha256': sha(root/'stage4/input/replay_foundation/pickup_eta_calibration_15min.parquet'),
                  'tiles_path': str(tile_dir), 'tile_count': len(tile_inventory), 'tile_inventory_sha256': frozen._sha(tile_inventory),
                  'tile_inventory_is_content_hash': False, 'historical_full_tiles_sha': json.loads((root/'stage0/docs/stage0_v6_freeze_manifest.json').read_text())['valhalla_tiles_sha'],
                  'historical_full_tiles_sha_recomputed': False, 'input_code_sha256': sha(Path(__file__)),
                  'limits': {'rss_bytes': 2*1024**3, 'wall_s': 600, 'milp_s': 10, 'arcs_per_state': 5000, 'routing_arc_total': 20000}}
    write_json(output/'provenance.json', provenance)
    adapter = LimitedAdapter(root, routing_mode=SINGLE_SOURCE_MATRIX)
    completed = []
    try:
        for registry, vehicles, waiting, fixtures, config, start in mv.load_states(root):
            guard()
            if len(completed) >= 10:
                raise RuntimeError('STATE_BUDGET')
            print('START_STATE', registry.epoch_id, flush=True)
            began = time.perf_counter()
            adapter.state_lookups = 0
            before_failures = adapter.routing_failures
            before_arcs = adapter.routing_arc_evaluations
            captured = mv.production_neutral_arcs(vehicles, fixtures, waiting, pd.Timestamp(registry.timestamp), start, config, adapter, neutral=False)
            if len(captured) > 5000 or adapter.routing_failures != before_failures:
                raise RuntimeError('ARC_LIMIT_OR_ROUTING_FAILURE')
            request_ids = {r.native_id: r.order_id for r, *_ in waiting}
            fixture_ids = {f.native_id: str(f.vehicle_id) for f in fixtures}
            # Persist only scalar solver input, never production runtime objects.
            from dataclasses import replace, asdict
            arcs = [replace(a, payload=(request_ids[a.request_id], fixture_ids[a.vehicle_id])) for a in captured]
            del captured
            if len({(a.request_id, a.vehicle_id) for a in arcs}) != len(arcs):
                raise RuntimeError('DUPLICATE_ARCS')
            if any(not np.isfinite(a.pickup_eta_s) or a.pickup_eta_s < 0 for a in arcs):
                raise RuntimeError('INVALID_ETA')
            for rid in {a.request_id for a in arcs}:
                if len({(a.critical, a.carry_over) for a in arcs if a.request_id == rid}) != 1:
                    raise RuntimeError('INCONSISTENT_ORDER_FLAGS')
            state = CumulativeExposureState(int(registry.cumulative_av_assignments), float(registry.cumulative_static_excess), float(registry.cumulative_dynamic_excess), float(registry.cumulative_speed_excess))
            gammas = {f: config['gamma_'+f] for f in ('static', 'dynamic', 'speed')}
            if config['cost_level_enabled']:
                raise RuntimeError('UNAUTHORIZED_COST_SCENARIO')
            archive = {'epoch_id': registry.epoch_id, 'timestamp': str(registry.timestamp), 'state_sha256': registry.state_sha256,
                       'exposure_state': asdict(state), 'gammas': gammas, 'arcs': [asdict(a) for a in arcs]}
            write_json(output/f'arcs_{registry.epoch_id}.json', archive)
            old = historical.loc[historical.epoch_id.eq(registry.epoch_id) & historical.variant.eq('P')]
            decision = diagnose(arcs, state, gammas, old)
            old_row = old_decisions.loc[old_decisions.epoch_id.eq(registry.epoch_id) & old_decisions.variant.eq('P')].iloc[0]
            if abs(decision['pstar_s']-float(old_row.pickup_objective_s)) > TOL or decision['counts'][1] != int(old_row.matched_order_count):
                raise RuntimeError('HISTORICAL_SUMMARY_MISMATCH')
            record = {'epoch_id': registry.epoch_id, 'timestamp': str(registry.timestamp), 'state_sha256': registry.state_sha256,
                      'arc_count': len(arcs), 'av_arc_count': sum(a.vehicle_type == 'AV' for a in arcs),
                      'routed_arcs': adapter.routing_arc_evaluations-before_arcs, 'runtime_s': time.perf_counter()-began, **decision}
            write_json(output/f'diagnosis_{registry.epoch_id}.json', record)
            completed.append(record)
            print('DONE_STATE', registry.epoch_id, len(arcs), decision['pstar_s'], flush=True)
            adapter.cache.clear()
            del arcs, archive
            gc.collect()
        status = 'COMPLETE' if len(completed) == 10 else 'INCOMPLETE'
    except Exception as exc:
        status = 'STOPPED: '+str(exc)
        print(status, flush=True)
    summary = {'status': status, 'completed_states': len(completed), 'registered_states': 10,
               'runtime_s': time.perf_counter()-START, 'sampled_peak_rss_mb': PEAK/1024**2,
               'routing_arc_evaluations': adapter.routing_arc_evaluations, 'routing_failures': adapter.routing_failures,
               'numerical_tolerance': TOL, 'alphas': ALPHAS, 'production_modified': False, 'dynamic_simulation': False,
               'budget_counts': []}
    for mode in ('A1', 'A2'):
        for alternative in ('ANY_ARC_CHANGE', 'AV_SERVICE_INDICATOR_CHANGE'):
            relevant = [r for c in completed for r in c['results'] if r['mode'] == mode and r['alternative'] == alternative]
            for alpha in ALPHAS:
                labels = [r['budget_exists'][str(alpha)] for r in relevant]
                summary['budget_counts'].append({'mode': mode, 'alternative': alternative, 'alpha': alpha,
                    'yes': labels.count('YES_WITHIN_NUMERICAL_TOLERANCE'), 'no': labels.count('NO'),
                    'unresolved': labels.count('UNRESOLVED'), 'denominator': len(labels)})
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if status == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(run(Path.cwd()))
