"""Summarize all prespecified pairs; never choose or rerun conditions."""
from pathlib import Path
import json
import pandas as pd
from stage3.scripts.traffic_state_batch1 import write_json
from stage4.analysis.traffic_mechanism import OUT, DOC


def compare(a,b):
    assert not a.order_id.duplicated().any() and not b.order_id.duplicated().any()
    a=a.set_index('order_id').sort_index(); b=b.set_index('order_id').sort_index()
    assert a.index.equals(b.index)
    both=a.matched & b.matched
    types=pd.crosstab(a.vehicle_type.fillna('UNSERVED'),b.vehicle_type.fillna('UNSERVED'))
    return dict(cohort=len(a),both_matched=int(both.sum()),
        gained=int((~a.matched & b.matched).sum()),lost=int((a.matched & ~b.matched).sum()),
        both_unserved=int((~a.matched & ~b.matched).sum()),
        common_matched_wait_difference_s=float((b.loc[both,'wait_s']-a.loc[both,'wait_s']).mean()) if both.any() else None,
        transitions=[dict(before=str(i),after=str(j),count=int(types.loc[i,j])) for i in types.index for j in types.columns])


def run():
    root=Path.cwd(); out=root/OUT/'dynamic'
    summary=json.loads((out/'summary.json').read_text()); assert summary['status']=='COMPLETE'
    cfg=json.loads((root/'stage4/config/traffic_mechanism_v1.json').read_text())
    pairs=[]; details=[]
    for profile in cfg['dynamic_profiles']:
        for cut in cfg['cuts_s']:
            frames={v:pd.read_parquet(out/f'{profile}_{cut}_{v}/cohort_outcomes.parquet') for v in cfg['variants']}
            for left,right in [('F','V'),('V','VT'),('F','VT')]:
                pairs.append(dict(profile=profile,cut=cut,contrast=left+'->'+right,**compare(frames[left],frames[right])))
            for v in cfg['variants']:
                d=out/f'{profile}_{cut}_{v}'
                a=pd.read_parquet(d/'assignments.parquet'); e=pd.read_parquet(d/'epochs.parquet')
                x=pd.read_parquet(d/'exposure.parquet'); av=a[a.vehicle_type.eq('AV')]
                counts={c:int(e[c].sum()) for c in e if c.startswith('gate_av_') or c=='traffic_policy_pruned_av_opportunities'}
                fields=['traffic_outside_share','traffic_unknown_share','traffic_low_support_share']
                traffic={c:dict(mean=float(av[c].mean()),maximum=float(av[c].max())) for c in fields if c in av and len(av)}
                details.append(dict(profile=profile,cut=cut,variant=v,new_assignments=len(a),new_av_assignments=len(av),
                    epochs=len(e),gate_counts=counts,selected_av_traffic=traffic,
                    final_exposure=x.tail(1).to_dict('records')[0]))
    result=dict(status='COMPLETE',pairs=pairs,details=details,
        interpretation='Paired descriptive conditional continuations, not independent replications or full-day effects')
    write_json(root/DOC/'mechanism_comparison.json',result)
    print(json.dumps(pairs),flush=True)


if __name__=='__main__': run()
