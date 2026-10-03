"""Frozen-state attribution, cached pickup only, Gamma/cost disabled."""
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from stage4.analysis import mechanism_validity as mv
from stage4.dispatch.acceptance import passenger_acceptance
from stage4.dispatch.candidate_graph import SparseCandidateIndex, search_radius_m
from stage4.dispatch.solver import AssignmentArc
from stage4.tools import control_freedom_diagnostic as cf
from stage4.tools.complete_attribution_pickup import OUT, PHASE0, key

CONDITIONS = ('D0', 'D1', 'D2', 'D3', 'D4')
LAYERS = ('SPATIAL', 'PASSENGER', 'PHYSICAL_DIRECTION_HARD', 'CAPABILITY_HARD',
          'STRUCTURAL_ROUTE', 'PRE_TOPK_EVIDENCE', 'TOPK', 'VALID_ETA', 'PATIENCE', 'SESSION')
REVERSE = 'KNOWN_REVERSE_DIRECTION_AV_UNROUTABLE'
UTURN = 'UTURN_PROFILE_INCOMPATIBLE'


def codes(value):
    return set(json.loads(value)) if isinstance(value, str) else set(value)


def solve_condition(arcs):
    """Condition-local count optima, then pickup minimum, then minimum AV flip."""
    if not arcs:
        return {'counts': [0, 0, 0], 'pstar_s': 0., 'reference': [], 'alternatives': [
            {'mode': mode, 'status': 'INFEASIBLE', 'gap_s': None, 'gap_relative': None,
             'budget_exists': {str(a): False for a in cf.ALPHAS}, 'same_order_swap_count': 0,
             'served_set_changed': False, 'witness': []} for mode in ('A1', 'A2')]}
    problem = cf.Problem(arcs, None, {})
    values = []
    for level in problem.levels:
        result = problem.solve(-level)
        cf.require_optimal(result)
        value = int(round(level @ result.binary))
        values.append(value)
        problem.add(level, value, value)
    result = problem.solve(problem.eta)
    cf.require_optimal(result)
    x0 = result.binary
    pstar = float(problem.eta @ x0)
    reference = [i for i in range(len(arcs)) if x0[i]]
    base_types = {arcs[i].request_id: arcs[i].vehicle_type for i in reference}
    b0 = {o for o, t in base_types.items() if t == 'AV'}
    y0 = set(base_types)
    base_len = len(problem.rows)
    alternatives = []
    for mode in ('A1', 'A2'):
        problem.rows = problem.rows[:base_len]
        problem.low = problem.low[:base_len]
        problem.high = problem.high[:base_len]
        if mode == 'A1':
            for order in problem.orders:
                val = int(order in y0)
                problem.add({i: 1. for i, a in enumerate(arcs) if a.request_id == order}, val, val)
        problem.add({i: (-1. if a.request_id in b0 else 1.) for i, a in enumerate(arcs)
                     if a.vehicle_type == 'AV'}, 1-len(b0), np.inf)
        result = problem.solve(problem.eta)
        if result.status not in (0, 2):
            raise RuntimeError('UNRESOLVED_ALTERNATIVE: '+str(result.message))
        row = {'mode': mode, 'status': 'OPTIMAL' if result.status == 0 else 'INFEASIBLE',
               'gap_s': None, 'gap_relative': None, 'witness': [],
               'same_order_swap_count': 0, 'served_set_changed': False,
               'budget_exists': {str(a): False for a in cf.ALPHAS}}
        if result.status == 0:
            cf.require_optimal(result)
            selected = [i for i in range(len(arcs)) if result.binary[i]]
            new_types = {arcs[i].request_id: arcs[i].vehicle_type for i in selected}
            assert len(new_types) == len(selected)
            assert len({arcs[i].vehicle_id for i in selected}) == len(selected)
            assert all(int(round(level @ result.binary)) == value for level, value in zip(problem.levels, values))
            assert {o for o, t in new_types.items() if t == 'AV'} != b0
            swaps = sum(base_types[o] != new_types[o] for o in y0 & set(new_types))
            if mode == 'A1':
                assert set(new_types) == y0 and swaps > 0
            pickup = float(problem.eta @ result.binary)
            assert pickup >= pstar-cf.TOL
            gap = max(0., pickup-pstar)
            row.update(gap_s=gap, gap_relative=gap/pstar if pstar else None,
                       witness=selected, same_order_swap_count=swaps,
                       served_set_changed=set(new_types) != y0,
                       budget_exists={str(a): pickup <= pstar*(1+a)+cf.TOL for a in cf.ALPHAS})
        alternatives.append(row)
    return {'counts': values, 'pstar_s': pstar, 'reference': reference, 'alternatives': alternatives}


def metrics(pairs, vehicle_types):
    av = {(o, v) for o, v in pairs if vehicle_types[v] == 'AV'}
    hv = {(o, v) for o, v in pairs if vehicle_types[v] != 'AV'}
    m = mv.graph_metrics(av)
    return {'E_AV': m['E'], 'U_AV': m['U'], 'M_AV': m['M'],
            'O_HA': len({o for o, _ in av} & {o for o, _ in hv}), 'E_HV': len(hv)}


def main():
    cf.START = time.perf_counter()
    root = Path.cwd()
    out = root/OUT
    if (out/'experiment_summary.json').exists():
        raise RuntimeError('Existing experiment: do not overwrite')
    qa = json.loads((out/'routing_qa.json').read_text())
    assert qa['status'] == 'PASS' and qa['valid_eta']+qa['certified_routing_failure'] == 16386 and qa['unresolved'] == 0
    assert cf.sha(out/'pickup_receipts.jsonl') == qa['journal_sha256']
    cache = json.loads((out/'pickup_cache.json').read_text())
    assert len(cache) == 16386 and all(v['status'] in ('VALID_ETA', 'CERTIFIED_ROUTING_FAILURE') for v in cache.values())
    union = json.loads((root/PHASE0/'required_pickup_union.json').read_text())
    assert cf.sha(root/PHASE0/'required_pickup_union.json') == qa['union_sha256']
    lookup = {(r['epoch_id'], r['native_request_id'], r['native_vehicle_id']): cache[key(r)] for r in union}
    variants = {(r['order_id'], r['condition']): r for r in json.loads((root/PHASE0/'route_counterfactuals.json').read_text())}
    expected = {(r['epoch_id'], r['condition']): r['required'] for r in json.loads((root/PHASE0/'condition_input_counts.json').read_text())}
    ids = sorted({o for o, _ in variants})
    filters = [('order_id', 'in', ids), ('profile_id', '==', 'M')]
    original = pd.read_parquet(root/'stage3/output/odd_tod/s4/test31_av_operational_suitability.parquet', filters=filters).set_index('order_id')
    final = pd.read_parquet(root/'stage3/output/odd_tod/final/test31_stage3_to_stage4_interface.parquet', filters=filters).set_index('order_id')
    layer_rows, endpoint_rows, base_rows, provenance_rows = [], [], [], []
    total_union_seen = set()
    for registry, vehicles, waiting, fixtures, config, start in mv.load_states(root):
        cf.guard()
        epoch = registry.epoch_id
        index = SparseCandidateIndex(vehicles)
        types = {v.native_vehicle_id: v.vehicle_type for v in vehicles}
        sim_s = (pd.Timestamp(registry.timestamp)-start).total_seconds()
        category = {}
        for r, *_ in waiting:
            orig, fin = original.loc[r.order_id], final.loc[r.order_id]
            category[r.native_id] = {name for name, applies in (
                ('ORIGINAL_REVERSE', REVERSE in codes(orig.hard_reason_codes)),
                ('M_CAPABILITY_HARD', UTURN in codes(orig.hard_reason_codes)),
                ('FALLBACK_EVIDENCE', fin.selected_route_type == 'FALLBACK' and fin.hard_state == 'FEASIBLE' and not fin.evidence_complete)) if applies}
        d0_served, d0_hv_top = None, None
        for condition in CONDITIONS:
            graphs = {name: set() for name in LAYERS}
            arcs = []
            for r, failed, carry, _ in waiting:
                variant = variants[r.order_id, condition]
                fin = final.loc[r.order_id]
                accepts = passenger_acceptance(r.order_id, config['passenger_acceptance_rate'], config['passenger_acceptance_seed']).passenger_accepts_av
                radius = search_radius_m(failed, config['search_radius_initial_m'], config['search_radius_step_m'], config['search_radius_cap_m'])
                nearby, _ = index.query(r.pickup_lon_wgs84, r.pickup_lat_wgs84, radius, len(vehicles), True)
                hard = codes(fin.hard_reason_codes)
                structural = fin.hard_state == 'FEASIBLE' and fin.selected_route_type not in ('NONE', '')
                if condition in ('D1', 'D2'):
                    hard = set(variant.get('remaining_hard', []))
                    structural = variant['hard_state'] == 'FEASIBLE' and variant['selected_route_type'] not in ('NONE', '')
                if condition == 'D4':
                    hard, structural = set(), True
                gates = [True, accepts, accepts and not (hard-{UTURN}),
                         accepts and not hard, accepts and structural,
                         accepts and variant['av_eligible']]
                # Unknown/NONE is removed at STRUCTURAL_ROUTE, never called certified physical prohibition.
                for v, _ in nearby:
                    pair = (r.native_id, v.native_vehicle_id)
                    for name, gate in zip(LAYERS[:6], gates):
                        if v.vehicle_type != 'AV' or gate:
                            graphs[name].add(pair)
                selected, _ = index.query(r.pickup_lon_wgs84, r.pickup_lat_wgs84, radius, int(config['candidate_top_k']), bool(gates[-1]))
                remaining = r.sim_time_s+config['max_pickup_wait_s']-sim_s
                for v, _ in selected:
                    pair = (r.native_id, v.native_vehicle_id)
                    graphs['TOPK'].add(pair)
                    identity = (epoch, *pair)
                    assert identity in lookup
                    total_union_seen.add(identity)
                    receipt = lookup[identity]
                    if receipt['status'] != 'VALID_ETA':
                        continue
                    graphs['VALID_ETA'].add(pair)
                    eta = receipt['eta_s']
                    if eta > remaining:
                        continue
                    graphs['PATIENCE'].add(pair)
                    fixture = fixtures[v.native_vehicle_id]
                    if fixture.availability_policy == 'EMPIRICAL_SESSION':
                        predicted = float(r.predicted_service_time_s)
                        if not math.isfinite(predicted) or sim_s+eta+predicted > (pd.Timestamp(fixture.availability_end_time)-start).total_seconds():
                            continue
                    graphs['SESSION'].add(pair)
                    arcs.append(AssignmentArc(v.native_vehicle_id, r.native_id, eta, 0 < remaining <= 30, bool(carry),
                                              payload=(r.order_id, v.vehicle_id), vehicle_type=v.vehicle_type))
            assert len(graphs['TOPK']) == expected[epoch, condition]
            for before, after in zip(LAYERS, LAYERS[1:]):
                assert graphs[after] <= graphs[before], (epoch, condition, before, after)
            for layer, pairs in graphs.items():
                layer_rows.append({'epoch_id': epoch, 'condition': condition, 'layer': layer, **metrics(pairs, types)})
                for name in ('ORIGINAL_REVERSE', 'M_CAPABILITY_HARD', 'FALLBACK_EVIDENCE'):
                    subset = {p for p in pairs if name in category[p[0]]}
                    provenance_rows.append({'epoch_id': epoch, 'condition': condition, 'layer': layer, 'category': name,
                                            'waiting_order_count': sum(name in c for c in category.values()), **metrics(subset, types)})
            arcs.sort(key=lambda a: (a.request_id, a.vehicle_id))
            result = solve_condition(arcs)
            served = {arcs[i].request_id for i in result['reference']}
            hv_top = {p for p in graphs['TOPK'] if types[p[1]] != 'AV'}
            if condition == 'D0':
                d0_served, d0_hv_top = served, hv_top
            base_rows.append({'epoch_id': epoch, 'condition': condition, 'critical': result['counts'][0],
                              'total': result['counts'][1], 'carryover': result['counts'][2], 'pstar_s': result['pstar_s'],
                              'reference_orders_added_vs_D0': len(served-d0_served), 'reference_orders_removed_vs_D0': len(d0_served-served),
                              'HV_topk_displaced_vs_D0': len(d0_hv_top-hv_top)})
            for alt in result['alternatives']:
                endpoint_rows.append({'epoch_id': epoch, 'condition': condition, 'pstar_s': result['pstar_s'],
                                      **{k: v for k, v in alt.items() if k != 'witness'}})
            private = {'result': result, 'arcs': [a.__dict__ for a in arcs]}
            cf.write_json(out/f'private_{epoch}_{condition}.json', private)
            print(epoch, condition, 'arcs', len(arcs), 'counts', result['counts'], flush=True)
        cf.guard()
    assert total_union_seen == set(lookup)
    for name, rows in (('graph_layers', layer_rows), ('endpoints', endpoint_rows), ('baselines', base_rows), ('provenance_funnels', provenance_rows)):
        cf.write_json(out/(name+'.json'), rows)
    cf.write_json(out/'experiment_summary.json', {'status': 'PASS', 'condition_states': len(base_rows),
                  'endpoints': len(endpoint_rows), 'routing_keys': 16386, 'routing_calls': 0,
                  'Gamma': 'OFF', 'cost': 'OFF', 'runtime_s': time.perf_counter()-cf.START,
                  'internally_sampled_peak_rss_mib': cf.PEAK/1024**2, 'union_reconciled': True,
                  'all_MILPs_certified': True, 'all_witnesses_verified': True})


if __name__ == '__main__':
    main()
