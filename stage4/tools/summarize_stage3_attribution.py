"""Aggregate existing results only; no routing, inference, or optimization."""
import csv
import json
from pathlib import Path

from stage4.tools.complete_attribution_pickup import OUT
from stage4.tools.control_freedom_diagnostic import write_json


def main():
    root = Path.cwd()
    out = root/OUT
    public = root/'stage4/docs/stage3_action_space_attribution/results_v1'
    public.mkdir(parents=True, exist_ok=True)
    def read(name):
        return json.loads((out/(name+'.json')).read_text())
    assert read('independent_qa')['status'] == 'PASS'
    endpoints, layers = read('endpoints'), read('graph_layers')
    lookup = {(r['epoch_id'], r['condition'], r['mode']): r for r in endpoints}
    graph = {(r['epoch_id'], r['condition'], r['layer']): r for r in layers}
    epochs = sorted({r['epoch_id'] for r in endpoints})
    transitions, comparison = [], []
    for condition in ('D0', 'D1', 'D2', 'D3', 'D4'):
        row = {'condition': condition}
        for field in ('E_AV', 'U_AV', 'M_AV', 'O_HA'):
            row[field+'_state_sum'] = sum(graph[e, condition, 'SESSION'][field] for e in epochs)
        for mode in ('A1', 'A2'):
            for alpha in ('0', '0.005', '0.01', '0.02', '0.05'):
                row[mode+'_states_at_'+alpha] = sum(lookup[e, condition, mode]['budget_exists'][alpha] for e in epochs)
                gained = [e for e in epochs if lookup[e, condition, mode]['budget_exists'][alpha] and not lookup[e, 'D0', mode]['budget_exists'][alpha]]
                lost = [e for e in epochs if not lookup[e, condition, mode]['budget_exists'][alpha] and lookup[e, 'D0', mode]['budget_exists'][alpha]]
                overlap = [e for e in gained if graph[e, condition, 'SESSION']['O_HA'] > graph[e, 'D0', 'SESSION']['O_HA']]
                transitions.append({'condition': condition, 'mode': mode, 'tolerance': float(alpha),
                                    'gained_states': gained, 'lost_states': lost, 'gained_with_O_HA_increase': overlap})
        comparison.append(row)
    factor = [r['condition'] for r in transitions if r['condition'] in ('D1','D2','D3') and r['mode']=='A1'
              and r['tolerance']==.05 and len(r['gained_with_O_HA_increase']) >= 3]
    d4 = {r['mode']: r for r in transitions if r['condition']=='D4' and r['tolerance']==.05}
    labels = []
    if factor:
        labels.append('STAGE3_SUBSTITUTION_BOTTLENECK')
    capacity_only = [r['condition'] for r in comparison[1:4] if r['M_AV_state_sum'] > comparison[0]['M_AV_state_sum'] and not any(
        t['gained_states'] for t in transitions if t['condition']==r['condition'] and t['mode']=='A1' and t['tolerance']==.05)]
    if capacity_only:
        labels.append('STAGE3_CAPACITY_ONLY')
    if not factor and d4['A1']['gained_states']:
        labels.append('STAGE3_INTERACTION')
    if not factor and len(d4['A1']['gained_states']) <= 1 and len(d4['A2']['gained_states']) <= 1:
        labels.extend(['STOP_STAGE3_INVESTMENT_ON_THIS_SAMPLE'])
    classification = {'labels': labels, 'investigate_factors': factor, 'capacity_only_factors': capacity_only,
                      'downstream_spatial_patience_bottleneck': 'REVIEW_REQUIRED: report pre/post capacity; no frozen dominance threshold',
                      'scope': 'Ten registered M physical states; D4 is not an assignment upper bound',
                      'Stage3_modification_authorized': False, 'Stage5_authorized': False}
    write_json(out/'comparison.json', comparison)
    write_json(out/'transitions.json', transitions)
    write_json(out/'classification.json', classification)
    names = ('routing_qa','experiment_summary','independent_qa','graph_layers','endpoints','baselines','provenance_funnels','comparison','transitions','classification')
    for name in names:
        value = read(name)
        write_json(public/(name+'.json'), value)
        if isinstance(value, list):
            flat = [{k: json.dumps(v, separators=(',', ':')) if isinstance(v, (dict,list)) else v for k,v in r.items()} for r in value]
            with (public/(name+'.csv')).open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
                writer.writeheader()
                writer.writerows(flat)
    print(json.dumps({'comparison': comparison, 'classification': classification}, indent=2))


if __name__ == '__main__':
    main()
