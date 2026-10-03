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
    policy=pd.read_parquet(root/'stage4/output/traffic_research/policy_date=20161031.parquet',
        columns=['order_id','profile_id','outside_share','unknown_share','low_support_share','predicted_time_s'])
    for profile in cfg['dynamic_profiles']:
        for cut in cfg['cuts_s']:
            frames={v:pd.read_parquet(out/f'{profile}_{cut}_{v}/cohort_outcomes.parquet') for v in cfg['variants']}
            for left,right in [('F','V'),('V','VT'),('F','VT')]:
                keys=['native_request_id','native_vehicle_id','simulation_time_s','pickup_eta_s','pickup_time','service_end_time']
                aa=pd.read_parquet(out/f'{profile}_{cut}_{left}/assignments.parquet',columns=keys).sort_values(keys[:3]).reset_index(drop=True)
                bb=pd.read_parquet(out/f'{profile}_{cut}_{right}/assignments.parquet',columns=keys).sort_values(keys[:3]).reset_index(drop=True)
                pairs.append(dict(profile=profile,cut=cut,contrast=left+'->'+right,
                    physical_assignments_identical=aa.equals(bb),**compare(frames[left],frames[right])))
            for v in cfg['variants']:
                d=out/f'{profile}_{cut}_{v}'
                a=pd.read_parquet(d/'assignments.parquet'); e=pd.read_parquet(d/'epochs.parquet')
                x=pd.read_parquet(d/'exposure.parquet'); av=a[a.vehicle_type.eq('AV')]
                counts={c:int(e[c].sum()) for c in e if c.startswith('gate_av_') or c=='traffic_policy_pruned_av_opportunities'}
                # Annotate F too, using exact ORIGINAL evidence already preflighted.
                desc=av[['order_id']].merge(policy[policy.profile_id.eq(profile)],on='order_id',validate='one_to_one')
                assert len(desc)==len(av)
                traffic={c:dict(mean=float(desc[c].mean()),maximum=float(desc[c].max()),
                    predicted_time_weighted_mean=float((desc[c]*desc.predicted_time_s).sum()/desc.predicted_time_s.sum()))
                    for c in ['outside_share','unknown_share','low_support_share'] if len(desc)}
                cohort_av=frames[v].loc[frames[v].vehicle_type.eq('AV')]
                cohort_traffic={c:dict(mean=float(cohort_av[c].mean()),maximum=float(cohort_av[c].max()))
                    for c in ['research_outside_share','research_unknown_share'] if len(cohort_av)}
                details.append(dict(profile=profile,cut=cut,variant=v,new_assignments=len(a),new_av_assignments=len(av),
                    epochs=len(e),gate_counts=counts,selected_av_traffic=traffic,
                    cohort_av_count=len(cohort_av),cohort_av_traffic=cohort_traffic,
                    final_exposure=x.tail(1).to_dict('records')[0]))
    result=dict(status='COMPLETE',pairs=pairs,details=details,
        interpretation='Paired descriptive conditional continuations, not independent replications or full-day effects')
    write_json(root/DOC/'mechanism_comparison.json',result)
    print(json.dumps(pairs),flush=True)


if __name__=='__main__': run()
