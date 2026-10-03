"""Offline semantic attribution only. Does not import routing or simulation."""
from pathlib import Path
import gc
import json
import os
import threading
import time

import numpy as np
import pandas as pd
import psutil
from stage3.scripts.traffic_state_batch1 import EDGE, sha, write_json, write_parquet
from stage3.scripts.traffic_state_support import decompose
from stage4.analysis.traffic_research_prepare import aggregate

OUT = Path('stage4/output/traffic_mixed_attribution_v1')
DOC = Path('stage4/docs/traffic_research')
REL = 'RELATIVE_SLOWDOWN_WITHOUT_HIGH_LOW_SPEED_SHARE'
LIMIT = .05 + 1e-6


def attribution(relative, congestion):
    """Exhaustive accounting, with exactly the live policy's tolerance."""
    valid = np.isfinite(relative) & np.isfinite(congestion)
    r, c = relative.gt(LIMIT), congestion.gt(LIMIT)
    return pd.Series(np.select(
        [~valid, r & c, c, r, (relative + congestion).gt(LIMIT)],
        ['MISSING', 'BOTH_INDIVIDUALLY', 'CONGESTION_ONLY', 'RELATIVE_ONLY', 'COMBINED_ONLY'],
        default='PASS'), index=relative.index)


def weighted_stats(values, weights):
    v, w = np.asarray(values, float), np.asarray(weights, float)
    valid = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[valid], w[valid]
    if not len(v):
        return dict(valid_tokens=0, valid_time_s=0., mean=None, p10=None, p50=None, p90=None)
    ix = np.argsort(v, kind='stable'); v, w = v[ix], w[ix]
    cum = np.cumsum(w)
    q = v[np.minimum(np.searchsorted(cum, cum[-1]*np.array([.1,.5,.9]), side='left'), len(v)-1)]
    return dict(valid_tokens=len(v), valid_time_s=float(w.sum()), mean=float(np.average(v, weights=w)),
                **{k:float(x) for k,x in zip(['p10','p50','p90'],q)})


def token_summary(g):
    w = g.travel_time_p50_s.astype(float)
    metrics = ['predicted_speed_kmh','reference_speed_kmh','pace_ratio','pred_crawl','pred_stop','low_speed_share']
    return dict(tokens=len(g), orders=int(g.order_id.nunique()), edges=int(g[EDGE].nunique()),
                predicted_time_s=float(w.sum()), metrics={m:weighted_stats(g[m],w) for m in metrics})


def paired(a, b):
    assert not a.order_id.duplicated().any() and not b.order_id.duplicated().any()
    assert set(a.order_id) == set(b.order_id)
    x = a[['order_id','matched','vehicle_type']].merge(
        b[['order_id','matched','vehicle_type']], on='order_id', suffixes=('_F','_VT'), validate='one_to_one')
    for v in ['F','VT']:
        assert x.loc[x[f'matched_{v}'],f'vehicle_type_{v}'].isin(['AV','HV']).all()
        x[f'type_{v}'] = x[f'vehicle_type_{v}'].where(x[f'matched_{v}'], 'UNSERVED')
    x['outcome'] = np.select([x.matched_F & ~x.matched_VT, ~x.matched_F & x.matched_VT,
                            x.matched_F & x.matched_VT], ['LOST','GAINED','RETAINED'], default='BOTH_UNSERVED')
    return x


def counts(d, keys):
    return d.groupby(keys, dropna=False).size().rename('orders').reset_index().to_dict('records')


def run():
    started=time.monotonic(); OUT.mkdir(parents=True,exist_ok=False)
    sources={}; done=threading.Event(); resource={'peak_rss_mib':0.}
    def monitor():
        alerted=False
        while not done.wait(.2):
            rss=psutil.Process().memory_info().rss/2**20
            resource['peak_rss_mib']=max(resource['peak_rss_mib'],rss)
            if rss>2048 and not alerted:
                print('RSS_ADVISORY >2048MiB',flush=True); alerted=True
            if time.monotonic()-started>1800:
                print('HARD_TIMEOUT stopping offline analysis',flush=True); os._exit(124)
    threading.Thread(target=monitor,daemon=True).start()
    def source(p):
        p=Path(p)
        if str(p) not in sources: sources[str(p)]=sha(p)
        return p
    def read(p, columns=None):
        return pd.read_parquet(source(p), columns=columns)
    protected=[Path(p) for p in ['stage3/config/stage3_av_capability_profiles.json',
        'stage3/config/traffic_state_batch1.json','stage4/config/traffic_mechanism_v1.json',
        'stage2/output_v5_2/development/M3/epoch_004.pt']]
    before={str(p):sha(p) for p in protected}
    cfg=json.loads(protected[1].read_text())
    ref=read('stage3/output/traffic_state_batch1/train_reference.parquet')
    semantic=[]; route_counts=[]; support_counts=[]; comparisons=[]
    for date in ['20161025','20161026','20161027','20161031']:
        policy=read(f'stage4/output/traffic_research/policy_date={date}.parquet')
        policy=policy.loc[policy.profile_id.eq('C')].copy()
        route=read(f'stage2/output_v4/route_conditioned_dataset/revealed_route_proxy/day={date}.parquet',
                   ['order_id','traversal_id',EDGE,'canonical_highway'])
        if date=='20161031':
            p=read('stage3/output/odd_tod/s4/test31_m3_predictions.parquet',
                   ['order_id','traversal_id','pred_pace_p50','pred_crawl','pred_stop','travel_time_p50_s'])
            p=p.loc[p.order_id.isin(policy.order_id)].merge(route,on=['order_id','traversal_id'],validate='one_to_one')
            p=decompose(p.merge(ref,on=EDGE,how='left',validate='many_to_one'),cfg)
        else:
            p=read(f'stage3/output/traffic_state_support/tokens_date={date}.parquet')
            p=p.merge(route.drop(columns=EDGE),on=['order_id','traversal_id'],validate='one_to_one')
        del route; gc.collect()
        assert not p.duplicated(['order_id','traversal_id']).any()
        a=aggregate(p).set_index('order_id').sort_index()
        old=policy.set_index('order_id').sort_index()
        assert a.index.equals(old.index)
        assert np.allclose(a,old[a.columns],rtol=1e-6,atol=1e-7)
        p['state_detail']=p.research_state.where(p.mixed_subtype.eq('NOT_MIXED'),p.mixed_subtype)
        p['predicted_speed_kmh']=3.6/p.pred_pace_p50
        p['reference_speed_kmh']=3.6/p.reference_pace_q20
        p['canonical_highway']=p.canonical_highway.fillna('MISSING').astype(str)
        scopes={'ALL':np.ones(len(p),dtype=bool)}
        if date=='20161031':
            joint=read('stage4/output/traffic_mechanism_v1/offline/test31_joint.parquet')
            joint=joint.loc[joint.profile_id.eq('C')]
            eligible=set(joint.loc[joint.base_eligible,'order_id'])
            rejected=set(joint.loc[joint.base_eligible & joint.outside_share.gt(LIMIT),'order_id'])
            scopes.update(C_BASE_ELIGIBLE=p.order_id.isin(eligible),C_INDEPENDENT_REJECTED=p.order_id.isin(rejected))
        for scope,mask in scopes.items():
            for state,g in p.loc[mask].groupby('state_detail'):
                semantic.append(dict(date=date,scope=scope,grouping='state',state=str(state),**token_summary(g)))
            relative=p.loc[mask & p.mixed_subtype.eq(REL)]
            for (road,support),g in relative.groupby(['canonical_highway','reference_support'],dropna=False):
                semantic.append(dict(date=date,scope=scope,grouping='relative_road_support',road_class=str(road),
                                     support=str(support),**token_summary(g)))
            for support,g in relative.groupby('reference_support'):
                semantic.append(dict(date=date,scope=scope,grouping='relative_support',support=str(support),**token_summary(g)))
        r=policy.copy()
        r['congestion_share']=r.congested_share+r.severe_share
        r['attribution']=attribution(r.relative_mixed_share,r.congestion_share)
        assert r.attribution.ne('PASS').equals(r.outside_share.gt(LIMIT))
        masks={'relative':p.mixed_subtype.eq(REL),'congestion':p.research_state.isin(['CONGESTED_PROXY','SEVERE_CONGESTION_PROXY']),
               'severe':p.research_state.eq('SEVERE_CONGESTION_PROXY')}
        w=p.travel_time_p50_s.astype(float); total=w.groupby(p.order_id).sum()
        for component,mask in masks.items():
            amount=w.where(mask & p.reference_support.eq('LOW_REFERENCE_SUPPORT'),0).groupby(p.order_id).sum()/total
            r[f'{component}_low_share']=r.order_id.map(amount)
        r['outside_low_share']=r.relative_low_share+r.congestion_low_share
        r['outside_supported_share']=r.outside_share-r.outside_low_share
        assert r.outside_supported_share.ge(-1e-6).all()
        r['supported_alone_exceeds']=r.outside_supported_share.gt(LIMIT)
        r['low_alone_exceeds']=r.outside_low_share.gt(LIMIT)
        route_counts.extend(dict(date=date,scope='ALL',**row) for row in counts(r,['attribution']))
        if date=='20161031':
            r['base_eligible']=r.order_id.isin(eligible)
            route_counts.extend(dict(date=date,scope='C_BASE_ELIGIBLE',**row) for row in counts(r.loc[r.base_eligible],['attribution']))
            rej=r.loc[r.base_eligible & r.outside_share.gt(LIMIT)]
            support_counts=counts(rej,['attribution','supported_alone_exceeds','low_alone_exceeds'])
            assert len(rej)==len(rejected)
            all_routes=r.copy()
        write_parquet(OUT/f'routes_date={date}.parquet',r)
        print(f'{date}: {len(p)} tokens; {len(r)} routes; policy reconciliation PASS',flush=True)
        del p,r,a,old,relative,policy; gc.collect()
    # Only existing cohort outcomes; no simulation imports or calls.
    for profile in ['C','M']:
        for cut in [37800,63000]:
            frames=[read(f'stage4/output/traffic_mechanism_v1/dynamic/{profile}_{cut}_{v}/cohort_outcomes.parquet') for v in ['F','VT']]
            x=paired(*frames).merge(all_routes.drop(columns=['profile_id']),on='order_id',how='left',validate='one_to_one')
            x['attribution']=x.attribution.fillna('MISSING')
            # C taxonomy is explicitly diagnostic-only for M; M's real gate is severe.
            x['actual_gate_share']=x.outside_share if profile=='C' else x.severe_share
            x['actual_gate']=np.select([x.actual_gate_share.isna(),x.actual_gate_share.gt(LIMIT)],['MISSING','EXCEEDS'],default='PASS')
            x['profile_id']=profile; x['cut_s']=cut
            write_parquet(OUT/f'paired_{profile}_{cut}.parquet',x)
            comparisons.append(dict(profile=profile,cut=cut,cohort=len(x),
                outcomes=counts(x,['outcome']),
                transitions=counts(x,['type_F','type_VT','attribution','actual_gate']),
                outcome_attribution=counts(x,['outcome','attribution','actual_gate']),
                missing_descriptors=int(x.attribution.eq('MISSING').sum())))
    original=json.loads(source(DOC/'mechanism_comparison.json').read_text())
    for row in comparisons:
        prior=next(v for v in original['pairs'] if v['profile']==row['profile'] and v['cut']==row['cut'] and v['contrast']=='F->VT')
        c={r['outcome']:r['orders'] for r in row['outcomes']}
        assert c.get('LOST',0)==prior['lost'] and c.get('GAINED',0)==prior['gained']
        assert c.get('RETAINED',0)==prior['both_matched'] and c.get('BOTH_UNSERVED',0)==prior['both_unserved']
    assert before=={str(p):sha(p) for p in protected}
    result=dict(status='COMPLETE',semantic=semantic,route_attribution=route_counts,
        test31_C_rejected_support=support_counts,dynamic=comparisons,sources=sources,protected_sha256=before,
        runtime_s=time.monotonic()-started,**resource,gpu_used=False,new_dispatch=False,new_inference=False)
    write_json(OUT/'summary.json',result); write_json(DOC/'mixed_attribution.json',result)
    done.set(); print({k:result[k] for k in ['status','runtime_s','peak_rss_mib']},flush=True)


if __name__=='__main__':
    run()
