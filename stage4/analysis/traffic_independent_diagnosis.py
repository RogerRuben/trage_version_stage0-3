"""Read-only, bounded-memory descriptor/eligibility overlap; no dispatch labels."""
from pathlib import Path
import gc
import time
import numpy as np
import pandas as pd
import psutil
from stage3.scripts.traffic_state_batch1 import sha, write_json, write_parquet

DOC = Path('stage4/docs/traffic_research')
OUT = Path('stage4/output/traffic_mechanism_v1/offline')
COMPONENTS = ['relative_mixed_share','congested_share','severe_share','background_mixed_share']


def describe(g):
    outside = g.outside_share.gt(.05+1e-6)
    upper = (g.outside_share+g.unknown_share).clip(upper=1).gt(.05+1e-6)
    return dict(routes=len(g), outside_count=int(outside.sum()),
        unknown_upper_exceeded_count=int(upper.sum()),
        unknown_sensitive_count=int((~outside & upper).sum()),
        unknown_mean=float(g.unknown_share.mean()), low_support_mean=float(g.low_support_share.mean()),
        outside_mean=float(g.outside_share.mean()),
        components={c:dict(mean=float(g[c].mean()), alone_exceeds_budget=int(g[c].gt(.05+1e-6).sum())) for c in COMPONENTS})


def joint(g):
    present=g.outside_share.notna()
    eligible=g.base_eligible
    outside=g.outside_share.gt(.05+1e-6)
    return dict(orders=len(g), descriptor_present=int(present.sum()),
        base_eligible=int(eligible.sum()),
        eligible_descriptor_missing=int((eligible & ~present).sum()),
        independent_traffic_rejection=int((eligible & present & outside).sum()),
        already_excluded_with_outside=int((~eligible & present & outside).sum()),
        eligible_traffic_pass=int((eligible & present & ~outside).sum()),
        unknown_mean=float(g.loc[present,'unknown_share'].mean()) if present.any() else None,
        low_support_mean=float(g.loc[present,'low_support_share'].mean()) if present.any() else None)


def run():
    started=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=False)
    rows=[]; sources=[]
    for date in ['20161025','20161026','20161027','20161031']:
        p=Path(f'stage4/output/traffic_research/policy_date={date}.parquet')
        d=pd.read_parquet(p)
        sources.append(dict(path=str(p),sha256=sha(p)))
        for profile,g in d.groupby('profile_id'):
            rows.append(dict(date=date,profile=profile,**describe(g)))
        del d; gc.collect()
    p=Path('stage4/input/replay_foundation/stage4_order_replay_base.parquet')
    cols=['order_id','profile_id','request_time','pickup_lon_wgs84','pickup_lat_wgs84',
          'selected_route_type','hard_state','evidence_complete','rho_static','rho_dynamic','rho_speed','predicted_service_time_s']
    base=pd.read_parquet(p,columns=cols); sources.append(dict(path=str(p),sha256=sha(p)))
    assert not base.duplicated(['order_id','profile_id']).any()
    base['base_eligible']=base.hard_state.eq('FEASIBLE') & base.evidence_complete & np.isfinite(base[['rho_static','rho_dynamic','rho_speed','predicted_service_time_s']]).all(axis=1) & base.predicted_service_time_s.gt(0)
    base['selected_route_reference']=base.selected_route_type+':'+base.order_id.astype(str)
    policy=pd.read_parquet('stage4/output/traffic_research/policy_date=20161031.parquet')
    d=base.merge(policy,on=['order_id','profile_id','selected_route_reference'],how='left',validate='one_to_one')
    t=pd.to_datetime(d.request_time,utc=True).dt.tz_convert('Asia/Shanghai')
    d['release_date']=t.dt.strftime('%Y%m%d'); d['release_hour']=t.dt.hour
    d['pickup_grid_x']=np.floor(d.pickup_lon_wgs84/.02).astype(int)
    d['pickup_grid_y']=np.floor(d.pickup_lat_wgs84/.02).astype(int)
    grouped={}
    for name,keys in [('profile',['profile_id']),('hour',['profile_id','release_date','release_hour']),
                      ('space',['profile_id','pickup_grid_x','pickup_grid_y'])]:
        result=[]
        for key,g in d.groupby(keys):
            key=key if isinstance(key,tuple) else (key,)
            result.append({**{k:(int(v) if isinstance(v,(int,np.integer)) else str(v)) for k,v in zip(keys,key)},**joint(g)})
        grouped[name]=result
    write_parquet(OUT/'test31_joint.parquet',d)
    summary=dict(status='COMPLETE',descriptor_summary=rows,test31_joint=grouped,sources=sources,
        validation_joint_eligibility='UNAVAILABLE_NOT_IMPUTED',
        interpretation='Descriptive counts only; no cumulative Gamma or passenger acceptance applied; unknown upper is diagnostic only',
        runtime_s=time.perf_counter()-started,peak_rss_mib=psutil.Process().memory_info().peak_wset/2**20)
    write_json(OUT/'summary.json',summary); write_json(DOC/'independent_diagnosis.json',summary)
    print({k:summary[k] for k in ['status','runtime_s','peak_rss_mib']},flush=True)
    print(grouped['profile'],flush=True)


if __name__=='__main__':
    run()
