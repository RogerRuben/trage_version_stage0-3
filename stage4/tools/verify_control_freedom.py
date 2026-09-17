"""Three small formulation fixtures and verification of archived witnesses; no routing."""
import json
from pathlib import Path

import pandas as pd

from stage4.dispatch.exposure import CumulativeExposureState
from stage4.dispatch.solver import AssignmentArc
from stage4.tools.control_freedom_diagnostic import diagnose, sha, write_json, OUT, TOL


def main():
    empty = CumulativeExposureState()
    none = dict(static=None, dynamic=None, speed=None)
    hist = pd.DataFrame([dict(order_id='o', vehicle_id='h', pickup_eta_s=10.)])
    h = AssignmentArc(1, 1, 10., False, False, payload=('o', 'h'))
    av = AssignmentArc(2, 1, 11., False, False, payload=('o', 'a'), vehicle_type='AV')
    first = diagnose([h, av], empty, none, hist)
    assert all(r['gap_upper_s'] == 1. for r in first['results'])
    second = diagnose([h, AssignmentArc(1, 2, 10., False, False, payload=('other', 'h'))], empty, none, hist)
    assert second['results'][0]['status'] == 'INFEASIBLE'
    assert second['results'][2]['gap_upper_s'] == 0.
    assert second['results'][2]['served_order_set_changed']
    blocked_av = AssignmentArc(2, 1, 0., False, False, payload=('o', 'a'), vehicle_type='AV', exposure_static=1.)
    third = diagnose([h, blocked_av], empty, dict(static=0., dynamic=None, speed=None), hist)
    assert third['pstar_s'] == 10.
    assert all(r['status'] == 'INFEASIBLE' for r in third['results'])

    root = Path.cwd()
    out = root / OUT
    historical = pd.read_csv(root/'stage4/output/paper_enhancement/frozen_state_prediction_ablation/selected_assignments_diagnostic.csv', dtype={'epoch_id': str})
    registry = pd.read_csv(root/'stage4/output/paper_enhancement/frozen_state_prediction_ablation/frozen_epoch_registry.csv', dtype={'epoch_id': str})
    feasible_witnesses = 0
    dispositions = 0
    seen = []
    for path in sorted(out.glob('diagnosis_*.json')):
        result = json.loads(path.read_text())
        epoch = result['epoch_id']
        seen.append(epoch)
        archive = json.loads((out/f'arcs_{epoch}.json').read_text())
        source_hash = registry.set_index('epoch_id').loc[epoch, 'state_sha256']
        assert archive['state_sha256'] == result['state_sha256'] == source_hash
        arcs = {tuple(a['payload']): a for a in archive['arcs']}
        assert len(arcs) == len(archive['arcs']) == result['arc_count']
        rows = historical.loc[historical.epoch_id.eq(epoch) & historical.variant.eq('P')]
        baseline = set(zip(rows.order_id.astype(str), rows.vehicle_id.astype(str)))
        for check in [None, *result['results']]:
            selected = set(baseline)
            if check is not None:
                dispositions += 1
                if check['status'] == 'INFEASIBLE':
                    assert not check['witness'] and all(v == 'NO' for v in check['budget_exists'].values())
                    continue
                assert check['status'] == 'OPTIMAL'
                for edge in check['witness']:
                    pair = (edge['order_id'], edge['vehicle_id'])
                    assert pair in arcs and (pair in baseline) == edge['selected_before']
                    if edge['selected_after']:
                        selected.add(pair)
                    else:
                        selected.remove(pair)
                assert selected != baseline
                feasible_witnesses += 1
            ordered = [arcs[key] for key in selected]
            assert len({a['vehicle_id'] for a in ordered}) == len(ordered)
            assert len({a['request_id'] for a in ordered}) == len(ordered)
            assert [sum(a['critical'] for a in ordered), len(ordered), sum(a['carry_over'] for a in ordered)] == result['counts']
            eta = sum(a['pickup_eta_s'] for a in ordered)
            target = result['pstar_s'] if check is None else check['alternative_pickup_s']
            assert abs(eta-target) <= TOL
            state = archive['exposure_state']
            av_arcs = [a for a in ordered if a['vehicle_type'] == 'AV']
            for family, gamma in archive['gammas'].items():
                if gamma is not None:
                    assert state[family]+sum(a['exposure_'+family] for a in av_arcs) <= gamma*(state['av_assignments']+len(av_arcs))+TOL
            if check:
                if check['mode'] == 'A1':
                    assert {o for o, _ in selected} == {o for o, _ in baseline}
                if check['alternative'] == 'AV_SERVICE_INDICATOR_CHANGE':
                    assert {o for o, v in selected if arcs[o, v]['vehicle_type'] == 'AV'} != {o for o, v in baseline if arcs[o, v]['vehicle_type'] == 'AV'}
    assert sorted(seen) == sorted(registry.epoch_id.tolist()) and len(seen) == 10
    provenance = json.loads((out/'provenance.json').read_text())
    assert sha(Path(provenance['routing_config_path'])) == provenance['routing_config_sha256']
    assert sha(root/'stage4/input/replay_foundation/pickup_eta_calibration_15min.parquet') == provenance['beta_sha256']
    assert sha(root/'stage4/output/final_experiments/ODD_Q50_M_P70_REFERENCE/scenario_config.json') == provenance['source_config_sha256']
    assert sha(root/'stage4/tools/control_freedom_diagnostic.py') == provenance['input_code_sha256']
    report = {'status': 'PASS', 'formulation_fixtures': 3, 'states_verified': 10,
              'alternative_dispositions': dispositions, 'feasible_witnesses_verified': feasible_witnesses,
              'infeasible_dispositions_from_solver': dispositions-feasible_witnesses,
              'infeasibility_independently_reproved': False, 'routing_calls': 0,
              'checks': ['counts', 'pickup', 'one_vehicle_one_order', 'Gamma', 'A1_order_set', 'AV_indicator_change', 'source_hashes']}
    write_json(out/'verification.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
