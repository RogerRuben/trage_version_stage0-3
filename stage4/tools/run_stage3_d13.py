"""One cached D13 condition and read-only D4 witness attribution."""
import csv
import json
import math
from collections import Counter
from pathlib import Path
import time

import pandas as pd

from stage4.analysis import mechanism_validity as mv
from stage4.dispatch.acceptance import passenger_acceptance
from stage4.dispatch.candidate_graph import SparseCandidateIndex, search_radius_m
from stage4.dispatch.solver import AssignmentArc
from stage4.tools import control_freedom_diagnostic as cf
from stage4.tools.complete_attribution_pickup import OUT, PHASE0, key
from stage4.tools.run_stage3_attribution import solve_condition, metrics

NEW = Path('stage4/output/paper_enhancement/stage3_attribution_d13')
PUBLIC = Path('stage4/docs/stage3_action_space_attribution/results_d13')
TARGETS = {'0730', '0830', '1300', '1800', '1830'}


def eligible(variant):
    return variant['hard_state'] == 'FEASIBLE' and variant['selected_route_type'] not in ('NONE', '')


def main():
    root = Path.cwd()
    out, old = root/NEW, root/OUT
    out.mkdir(parents=True, exist_ok=True)
    if (out/'summary.json').exists():
        raise RuntimeError('Existing output; do not overwrite')
    cf.START = time.perf_counter()
    qa = json.loads((old/'routing_qa.json').read_text())
    assert qa['status'] == 'PASS' and qa['unresolved'] == 0
    assert cf.sha(old/'pickup_receipts.jsonl') == qa['journal_sha256']
    cache = json.loads((old/'pickup_cache.json').read_text())
    variants = {r['order_id']: r for r in json.loads((root/PHASE0/'route_counterfactuals.json').read_text()) if r['condition']=='D1'}
    # Minimal definition fixtures: evidence bypass never bypasses hard/unknown/NONE.
    for state, route, expected in [('FEASIBLE','ORIGINAL',True),('FEASIBLE','FALLBACK',True),('UNKNOWN','ORIGINAL',False),('INFEASIBLE','ORIGINAL',False),('FEASIBLE','NONE',False)]:
        assert eligible(dict(hard_state=state, selected_route_type=route)) == expected
    states = []
    required = set()
    # Bound to ten modest physical states, not a full-day order/vehicle matrix.
    for registry, vehicles, waiting, fixtures, config, start in mv.load_states(root):
        cf.guard()
        index = SparseCandidateIndex(vehicles)
        graphs = {n: set() for n in ('SPATIAL','PASSENGER','STRUCTURAL_D13','TOPK','VALID_ETA','PATIENCE','SESSION')}
        arcs, reasons = [], {}
        sim_s = (pd.Timestamp(registry.timestamp)-start).total_seconds()
        for r, failed, carry, _ in waiting:
            variant = variants[r.order_id]
            ready = eligible(variant)
            reasons[r.native_id] = variant
            accepts = passenger_acceptance(r.order_id,config['passenger_acceptance_rate'],config['passenger_acceptance_seed']).passenger_accepts_av
            radius = search_radius_m(failed,config['search_radius_initial_m'],config['search_radius_step_m'],config['search_radius_cap_m'])
            near,_ = index.query(r.pickup_lon_wgs84,r.pickup_lat_wgs84,radius,len(vehicles),True)
            for v,_ in near:
                pair=(r.native_id,v.native_vehicle_id)
                graphs['SPATIAL'].add(pair)
                if v.vehicle_type!='AV' or accepts:
                    graphs['PASSENGER'].add(pair)
                    if v.vehicle_type!='AV' or ready:
                        graphs['STRUCTURAL_D13'].add(pair)
            selected,_ = index.query(r.pickup_lon_wgs84,r.pickup_lat_wgs84,radius,int(config['candidate_top_k']),bool(accepts and ready))
            remaining = r.sim_time_s+config['max_pickup_wait_s']-sim_s
            for v,_ in selected:
                pair=(r.native_id,v.native_vehicle_id)
                graphs['TOPK'].add(pair)
                k=key(dict(timestamp=str(registry.timestamp),source_lon=v.lon_wgs84,source_lat=v.lat_wgs84,pickup_lon=r.pickup_lon_wgs84,pickup_lat=r.pickup_lat_wgs84))
                required.add(k)
                assert k in cache and cache[k]['status'] in ('VALID_ETA','CERTIFIED_ROUTING_FAILURE'), 'STOP_MISSING_KEY'
                if cache[k]['status']!='VALID_ETA':
                    continue
                graphs['VALID_ETA'].add(pair)
                eta=cache[k]['eta_s']
                if eta>remaining:
                    continue
                graphs['PATIENCE'].add(pair)
                f=fixtures[v.native_vehicle_id]
                if f.availability_policy=='EMPIRICAL_SESSION':
                    p=float(r.predicted_service_time_s)
                    if not math.isfinite(p) or sim_s+eta+p>(pd.Timestamp(f.availability_end_time)-start).total_seconds():
                        continue
                graphs['SESSION'].add(pair)
                arcs.append(AssignmentArc(v.native_vehicle_id,r.native_id,eta,0<remaining<=30,bool(carry),payload=(r.order_id,v.vehicle_id),vehicle_type=v.vehicle_type))
        layers=list(graphs)
        assert all(graphs[b]<=graphs[a] for a,b in zip(layers,layers[1:]))
        states.append((registry.epoch_id,arcs,graphs,{v.native_vehicle_id:v.vehicle_type for v in vehicles},reasons))
    assert len(states)==10 and len(required)==12352
    cf.write_json(out/'preflight.json',dict(status='PASS',unique_required_keys=len(required),missing_keys=0,routing_calls=0))
    outputs={name:[] for name in ('endpoints','graph_layers','state_comparison','blocking_summary','blocking_reasons')}
    for epoch,arcs,graphs,types,reasons in states:
        cf.guard()
        arcs.sort(key=lambda a:(a.request_id,a.vehicle_id))
        result=solve_condition(arcs)
        cf.write_json(out/f'private_{epoch}_D13.json',dict(arcs=[a.__dict__ for a in arcs],result=result))
        served={arcs[i].request_id for i in result['reference']}
        for layer,pairs in graphs.items():
            outputs['graph_layers'].append(dict(epoch_id=epoch,condition='D13',layer=layer,**metrics(pairs,types)))
        for a in result['alternatives']:
            outputs['endpoints'].append(dict(epoch_id=epoch,condition='D13',pstar_s=result['pstar_s'],**{k:v for k,v in a.items() if k!='witness'}))
        for d in ('D0','D1','D3','D4'):
            previous=json.loads((old/f'private_{epoch}_{d}.json').read_text())
            pr=previous['result']; pa=previous['arcs']
            ps={pa[i]['request_id'] for i in pr['reference']}
            outputs['state_comparison'].append(dict(epoch_id=epoch,comparison=d,counts_D13=result['counts'],counts_previous=pr['counts'],
                pstar_D13=result['pstar_s'],pstar_previous=pr['pstar_s'],served_added=len(served-ps),served_removed=len(ps-served),
                A1_D13=result['alternatives'][0]['budget_exists'],A1_previous=pr['alternatives'][0]['budget_exists'],
                A1_gap_s_D13=result['alternatives'][0]['gap_s'],A1_gap_s_previous=pr['alternatives'][0]['gap_s']))
            if d!='D4' or epoch not in TARGETS:
                continue
            for role,indices in [('BASELINE',pr['reference']),('A1_ALTERNATIVE',pr['alternatives'][0]['witness'])]:
                counts=Counter(); hard=Counter(); unknown=Counter()
                for i in indices:
                    a=pa[i]; pair=(a['request_id'],a['vehicle_id']); variant=reasons[a['request_id']]
                    if pair in graphs['SESSION']:
                        reason='PRESENT'
                    elif a['vehicle_type']=='AV' and not eligible(variant):
                        reason='STAGE3_NOT_ESTABLISHED_OR_NOT_FEASIBLE'
                        hard.update(variant.get('remaining_hard',[])); unknown.update(variant.get('remaining_unknown',[]))
                    elif pair not in graphs['TOPK']:
                        reason='SHARED_TOPK_DISPLACEMENT'
                    elif pair not in graphs['VALID_ETA']:
                        reason='CERTIFIED_ROUTING_FAILURE'
                    elif pair not in graphs['PATIENCE']:
                        reason='PATIENCE'
                    else:
                        reason='SESSION'
                    counts[reason]+=1
                outputs['blocking_summary'].append(dict(epoch_id=epoch,witness_role=role,witness_arcs=len(indices),counts=dict(counts),
                    all_arcs_present=counts['PRESENT']==len(indices),D13_A1_at_5pct=result['alternatives'][0]['budget_exists']['0.05']))
                for kind,rs in [('remaining_hard',hard),('remaining_unknown',unknown)]:
                    for reason,n in sorted(rs.items()):
                        outputs['blocking_reasons'].append(dict(epoch_id=epoch,witness_role=role,reason_kind=kind,reason=reason,arc_occurrences=n))
        print(epoch,result['counts'],result['alternatives'][0]['gap_relative'],flush=True)
    summary=dict(status='PASS',states=10,routing_calls=0,unique_cached_keys=len(required),
        Gamma='OFF',cost='OFF',runtime_s=time.perf_counter()-cf.START,internally_sampled_peak_rss_mib=cf.PEAK/1024**2,
        protocol_sha256=cf.sha(root/'stage4/docs/stage3_action_space_attribution/d13_protocol.md'),
        code_sha256=cf.sha(Path(__file__)),all_MILPs_certified=True,
        capacity={k:sum(r[k] for r in outputs['graph_layers'] if r['layer']=='SESSION') for k in ('E_AV','U_AV','M_AV','O_HA')},
        budget_state_counts={m:{str(a):sum(r['budget_exists'][str(a)] for r in outputs['endpoints'] if r['mode']==m) for a in cf.ALPHAS} for m in ('A1','A2')})
    public=root/PUBLIC; public.mkdir(parents=True,exist_ok=True)
    for name,rows in outputs.items():
        cf.write_json(out/(name+'.json'),rows); cf.write_json(public/(name+'.json'),rows)
        if rows:
            with (public/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader()
                writer.writerows({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)
    cf.write_json(out/'summary.json',summary); cf.write_json(public/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
