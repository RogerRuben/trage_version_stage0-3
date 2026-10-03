"""Q0 and Phase0 only. No routing actor, new scientific MILP, or simulation."""
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from stage4.tools.control_freedom_diagnostic import diagnose, write_json, sha, guard, TOL
from stage4.dispatch.solver import AssignmentArc
from stage4.dispatch.exposure import CumulativeExposureState
from stage4.dispatch.acceptance import passenger_acceptance
from stage4.dispatch.candidate_graph import SparseCandidateIndex, search_radius_m
from stage4.analysis import mechanism_validity as mv
from stage4.analysis import frozen_state_prediction_ablation as frozen
from stage3.odd_tod import finalization as final

OUT = Path('stage4/output/paper_enhancement/stage3_attribution_phase0')
OLD = Path('stage4/output/paper_enhancement/control_freedom_structure')
REVERSE = 'KNOWN_REVERSE_DIRECTION_AV_UNROUTABLE'
UTURN = 'UTURN_PROFILE_INCOMPATIBLE'


def reasons(value):
    return set(final._json_list(value))


def hard_state(hard, unknown):
    return 'INFEASIBLE' if hard else ('UNKNOWN' if unknown else 'FEASIBLE')


def q0(root, out):
    historical = pd.read_csv(root/frozen.OUTPUT_REL/'selected_assignments_diagnostic.csv', dtype={'epoch_id': str})
    records = []
    for path in sorted((root/OLD).glob('arcs_*.json')):
        guard()
        arcset = json.loads(path.read_text())
        epoch = arcset['epoch_id']
        arcs = [AssignmentArc(**{**a, 'payload': tuple(a['payload'])}) for a in arcset['arcs']]
        h = historical.loc[historical.epoch_id.eq(epoch) & historical.variant.eq('P')]
        actual = diagnose(arcs, CumulativeExposureState(**arcset['exposure_state']), arcset['gammas'], h)
        expected = json.loads((root/OLD/f'diagnosis_{epoch}.json').read_text())
        assert actual['counts'] == expected['counts'] and abs(actual['pstar_s']-expected['pstar_s']) <= TOL
        for a, e in zip(actual['results'], expected['results']):
            assert (a['mode'], a['alternative'], a['status'], a['budget_exists']) == (e['mode'], e['alternative'], e['status'], e['budget_exists'])
            for field in ('gap_upper_s', 'gap_lower_s'):
                assert (a[field] is None and e[field] is None) or (a[field] is not None and e[field] is not None and abs(a[field]-e[field]) <= TOL)
        records.append({'epoch_id': epoch, 'status': 'PASS', 'pstar_s': actual['pstar_s'], 'counts': actual['counts']})
    assert len(records) == 10
    write_json(out/'q0.json', {'status': 'PASS', 'scientific_condition': False, 'states': records})
    print('Q0_PASS_10', flush=True)


def route_variant(original, descriptor, fallback, code):
    hard = reasons(original.hard_reason_codes)-{code}
    unknown = reasons(original.unknown_reason_codes)
    if code == REVERSE and int(descriptor.unresolved_token_count) > 0:
        unknown.add('UNRESOLVED_ROUTE_IDENTITY')
    state = hard_state(hard, unknown)
    if state != 'INFEASIBLE':
        complete = state == 'FEASIBLE' and all(np.isfinite(float(getattr(original, 'rho_'+f))) for f in ('static','dynamic','speed'))
        return {'selected_route_type': 'ORIGINAL', 'hard_state': state, 'evidence_complete': bool(complete),
                'av_eligible': bool(complete), 'remaining_hard': sorted(hard), 'remaining_unknown': sorted(unknown)}
    if fallback is None:
        return {'selected_route_type': 'NONE', 'hard_state': 'UNKNOWN', 'evidence_complete': False, 'av_eligible': False,
                'remaining_hard': sorted(hard), 'remaining_unknown': ['NOT_ESTABLISHED_UNDER_LIMITED_K1_SEARCH']}
    fh = reasons(fallback.hard_reason_codes)-{code}
    fu = reasons(fallback.structural_unknown_reason_codes)
    fs = hard_state(fh, fu)
    return {'selected_route_type': 'FALLBACK' if fs == 'FEASIBLE' else 'NONE',
            'hard_state': 'FEASIBLE' if fs == 'FEASIBLE' else 'UNKNOWN', 'evidence_complete': False,
            'av_eligible': False, 'remaining_hard': sorted(fh), 'remaining_unknown': sorted(fu),
            'fallback_dynamic_contract': 'MISSING_NOT_IMPUTED'}


def phase0(root, out):
    states = list(mv.load_states(root))  # Original physical hashes are checked by the loader.
    ids = sorted({r.order_id for _, _, waiting, *_ in states for r, *_ in waiting})
    filters = [('order_id', 'in', ids)]
    original = pd.read_parquet(root/final.ORIGINAL_SUITABILITY_REL, filters=filters)
    original = original.loc[original.profile_id.eq('M')].set_index('order_id')
    descriptor = pd.read_parquet(root/final.ORIGINAL_DESCRIPTOR_REL, filters=filters).set_index('order_id')
    interface = pd.read_parquet(root/final.INTERFACE_REL, filters=filters)
    interface = interface.loc[interface.profile_id.eq('M')].set_index('order_id')
    statuses = pd.read_parquet(root/final.FALLBACK_STATUS_REL, filters=filters)
    routes = pd.read_parquet(root/final.FALLBACK_ROUTE_REL, filters=filters)
    guard()
    # Static evaluation only of existing edges. No generate_candidates/finalize call.
    candidate = final._candidate_profile_table(root, statuses, routes)
    candidate = candidate.loc[candidate.profile_id.eq('M')].set_index('order_id')
    guard()
    # Reconcile unmodified static fallback outcome to the frozen final interface first.
    for oid in ids:
        row = interface.loc[oid]
        if original.loc[oid].hard_state == 'INFEASIBLE':
            found = oid in candidate.index and candidate.loc[oid].hard_state == 'FEASIBLE'
            assert found == (row.selected_route_type == 'FALLBACK'), 'FALLBACK_EVALUATOR_RECONCILIATION'
    route_rows = []
    eligibility = {}
    for oid in ids:
        row, orig, desc = interface.loc[oid], original.loc[oid], descriptor.loc[oid]
        fallback = candidate.loc[oid] if oid in candidate.index else None
        variants = {
            'D0': {'av_eligible': bool(row.hard_state == 'FEASIBLE' and row.evidence_complete and all(np.isfinite(float(getattr(row,'rho_'+f))) for f in ('static','dynamic','speed')))},
            'D1': route_variant(orig, desc, fallback, REVERSE),
            'D2': route_variant(orig, desc, fallback, UTURN),
            'D3': {'av_eligible': bool(row.hard_state == 'FEASIBLE' and row.selected_route_type not in ('NONE',''))},
            'D4': {'av_eligible': True}}
        for d, variant in variants.items():
            eligibility[oid, d] = variant['av_eligible']
            route_rows.append({'order_id': oid, 'condition': d, **variant})
    write_json(out/'route_counterfactuals.json', route_rows)
    rows, unions, fallback_rows = [], [], []
    for registry, vehicles, waiting, fixtures, config, start in states:
        guard()
        index = SparseCandidateIndex(vehicles)
        arcset = json.loads((root/OLD/f'arcs_{registry.epoch_id}.json').read_text())
        archived = {(a['request_id'], a['vehicle_id']): a['pickup_eta_s'] for a in arcset['arcs']}
        union = {}
        by_condition = {d: set() for d in ('D0','D1','D2','D3','D4')}
        for r, failed, *_ in waiting:
            accepts = passenger_acceptance(r.order_id, config['passenger_acceptance_rate'], config['passenger_acceptance_seed']).passenger_accepts_av
            radius = search_radius_m(failed, config['search_radius_initial_m'], config['search_radius_step_m'], config['search_radius_cap_m'])
            expected = bool(r.av_smoke_eligible and all(np.isfinite(float(getattr(r,'rho_'+f))) for f in ('static','dynamic','speed')))
            assert eligibility[r.order_id, 'D0'] == expected
            for d in by_condition:
                selected, _ = index.query(r.pickup_lon_wgs84, r.pickup_lat_wgs84, radius, int(config['candidate_top_k']), bool(accepts and eligibility[r.order_id, d]))
                for v, _ in selected:
                    key = (r.native_id, v.native_vehicle_id)
                    by_condition[d].add(key)
                    union[key] = {'epoch_id': registry.epoch_id, 'timestamp': str(registry.timestamp),
                        'order_id': r.order_id, 'vehicle_id': v.vehicle_id, 'native_request_id': r.native_id,
                        'native_vehicle_id': v.native_vehicle_id, 'vehicle_type': v.vehicle_type,
                        'source_lon': v.lon_wgs84, 'source_lat': v.lat_wgs84,
                        'pickup_lon': r.pickup_lon_wgs84, 'pickup_lat': r.pickup_lat_wgs84,
                        'eta_archived': key in archived, 'archived_eta_s': archived.get(key)}
            f = interface.loc[r.order_id]
            if f.selected_route_type == 'FALLBACK' and f.hard_state == 'FEASIBLE' and not f.evidence_complete:
                av_space = index.count_vehicle_type_within(r.pickup_lon_wgs84, r.pickup_lat_wgs84, radius, 'AV')
                av_top = sum(k[0] == r.native_id and union[k]['vehicle_type'] == 'AV' for k in by_condition['D3'])
                fallback_rows.append({'epoch_id': registry.epoch_id, 'order_id': r.order_id, 'av_spatial_count': av_space,
                                      'D3_topk_av_count': av_top, 'patience_pass': 'NOT_RUN'})
        for d, keys in by_condition.items():
            rows.append({'epoch_id': registry.epoch_id, 'condition': d, 'required': len(keys),
                         'archived': sum(k in archived for k in keys), 'missing': sum(k not in archived for k in keys)})
        unions.extend(union.values())
        print('PHASE0_STATE', registry.epoch_id, len(union), flush=True)
    assert len(rows) == 50
    write_json(out/'required_pickup_union.json', unions)
    write_json(out/'condition_input_counts.json', rows)
    write_json(out/'fallback_phase0_provenance.json', fallback_rows)
    missing = sum(not r['eta_archived'] for r in unions)
    result = {'status': 'READY_AFTER_BOUNDED_PICKUP_ETA_COMPLETION' if missing else 'READY',
              'physical_states': 10, 'unique_waiting_orders': len(ids), 'required_state_arcs': len(unions),
              'archived_state_arcs': len(unions)-missing, 'missing_state_arcs': missing,
              'route_counterfactual_unresolved': 0, 'existing_fallback_routes_evaluated': len(candidate),
              'fallback_hardF_evidenceIncomplete_state_orders': len(fallback_rows),
              'fallback_state_orders_with_AV_spatial': sum(r['av_spatial_count'] > 0 for r in fallback_rows),
              'fallback_state_orders_with_D3_AV_topk': sum(r['D3_topk_av_count'] > 0 for r in fallback_rows),
              'routing_calls': 0, 'scientific_condition_MILPs': 0, 'next_step_requires_user_authorization': True,
              'archive_scope': 'Certified old same-state solver arcs only; other routing samples not certified for exact keys'}
    write_json(out/'phase0_summary.json', result)
    print(json.dumps(result), flush=True)


def main():
    root = Path.cwd()
    out = root/OUT
    out.mkdir(parents=True, exist_ok=True)
    if (out/'phase0_summary.json').exists() or (out/'failure.json').exists():
        raise RuntimeError('EXISTING_OUTPUT_DO_NOT_OVERWRITE')
    began = time.perf_counter()
    write_json(out/'execution_source.json', {'code_sha256': sha(Path(__file__)),
        'protocol_sha256': sha(root/'stage4/docs/stage3_action_space_attribution/stage3_action_space_attribution_protocol_v1.md'),
        'authorized_scope': 'Q0_AND_PHASE0_ONLY'})
    try:
        q0(root, out)
        phase0(root, out)
    except Exception as exc:
        write_json(out/'failure.json', {'status': 'NOT_IDENTIFIABLE', 'error': repr(exc)})
        raise
    print('ELAPSED_S', time.perf_counter()-began, flush=True)


if __name__ == '__main__':
    main()
