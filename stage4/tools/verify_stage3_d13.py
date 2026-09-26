"""Read-only solution checks and provenance summaries, without new solves."""
import json
from collections import Counter
from pathlib import Path

from stage4.tools.complete_attribution_pickup import OUT, PHASE0
from stage4.tools.run_stage3_d13 import NEW, PUBLIC, TARGETS, eligible
from stage4.tools.control_freedom_diagnostic import sha, write_json, ALPHAS, TOL


def main():
    root=Path.cwd(); out=root/NEW; public=root/PUBLIC
    summary=json.loads((out/'summary.json').read_text())
    assert summary['code_sha256']==sha(root/'stage4/tools/run_stage3_d13.py')
    variants={(r['order_id'],r['condition']):r for r in json.loads((root/PHASE0/'route_counterfactuals.json').read_text())}
    checks=[]; swaps=[]
    for path in sorted(out.glob('private_*_D13.json')):
        epoch=path.stem.split('_')[1]
        doc=json.loads(path.read_text()); arcs=doc['arcs']; r=doc['result']
        def selection(indices):
            selected=[arcs[i] for i in indices]
            assert len({a['vehicle_id'] for a in selected})==len(selected)
            assert len({a['request_id'] for a in selected})==len(selected)
            assert [sum(a['critical'] for a in selected),len(selected),sum(a['carry_over'] for a in selected)]==r['counts']
            for a in selected:
                if a['vehicle_type']=='AV':
                    assert eligible(variants[a['payload'][0],'D1'])
            return {a['request_id']:a['vehicle_type'] for a in selected},sum(a['pickup_eta_s'] for a in selected)
        base,p=selection(r['reference']); assert abs(p-r['pstar_s'])<=TOL
        for a in r['alternatives']:
            assert a['status'] in ('OPTIMAL','INFEASIBLE')
            if a['status']=='INFEASIBLE':
                assert not any(a['budget_exists'].values())
                continue
            types,q=selection(a['witness']); gap=max(0,q-p)
            assert abs(gap-a['gap_s'])<=TOL
            assert {o for o in types if types[o]=='AV'}!={o for o in base if base[o]=='AV'}
            if a['mode']=='A1':
                assert types.keys()==base.keys() and any(types[o]!=base[o] for o in base)
            assert all(a['budget_exists'][str(t)]==(q<=p*(1+t)+TOL) for t in ALPHAS)
        checks.append(dict(epoch_id=epoch,status='PASS',arcs=len(arcs)))
        if epoch not in TARGETS:
            continue
        d4=json.loads((root/OUT/f'private_{epoch}_D4.json').read_text()); aa=d4['arcs']; rr=d4['result']
        before={aa[i]['request_id']:aa[i] for i in rr['reference']}
        after={aa[i]['request_id']:aa[i] for i in rr['alternatives'][0]['witness']}
        present={(a['request_id'],a['vehicle_id']) for a in arcs}
        for oid in sorted(before):
            if before[oid]['vehicle_type']==after[oid]['vehicle_type']:
                continue
            pair=[before[oid],after[oid]]
            v=variants[pair[0]['payload'][0],'D1']
            swaps.append(dict(epoch_id=epoch,D4_swap_direction=before[oid]['vehicle_type']+'->'+after[oid]['vehicle_type'],
                both_arcs_present_in_D13=all((a['request_id'],a['vehicle_id']) in present for a in pair),
                D13_route_type=v['selected_route_type'],D13_hard_state=v['hard_state'],
                remaining_hard=v.get('remaining_hard',[]),remaining_unknown=v.get('remaining_unknown',[])))
    divergence=Counter()
    for (oid,d),v in variants.items():
        if d=='D3' and v['av_eligible'] and not eligible(variants[oid,'D1']):
            x=variants[oid,'D1']
            divergence[(x['selected_route_type'],x['hard_state'],tuple(x.get('remaining_unknown',[])))]+=1
    rows=[dict(route_type=k[0],hard_state=k[1],remaining_unknown=list(k[2]),unique_order_count=n) for k,n in sorted(divergence.items())]
    qa=dict(status='PASS',condition_states=len(checks),witness_checks=checks,additional_MILPs=0,additional_routing_calls=0,
            maximum_final_arcs=max(r['arcs'] for r in checks),D3_eligible_not_D13_unique_orders=sum(divergence.values()),
            old_outputs_unchanged_scope='New output directory only; old files opened read-only')
    for name,data in [('qa',qa),('swap_blocking_detail',swaps),('route_selection_divergence',rows)]:
        write_json(out/(name+'.json'),data);write_json(public/(name+'.json'),data)
    print(json.dumps(dict(qa=qa,swaps=swaps,divergence=rows),indent=2))


if __name__=='__main__':
    main()
