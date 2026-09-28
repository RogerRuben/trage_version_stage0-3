"""Twelve prespecified continuations from shared M physical checkpoints."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import psutil
from stage3.scripts.traffic_state_batch1 import sha, write_json
from stage4.analysis import traffic_research_window as w
from stage4.analysis.native_checkpoint_closure import signature
from stage4.analysis.traffic_mechanism_compare import compare
from stage4.dispatch.exposure import exposure_excess
from stage4.dispatch.traffic_research_policy import MODE, VARIABILITY_MODE
from stage4.fleetpy_adapter.test31_demand_adapter import load_all_test31_requests
from stage4.fleetpy_adapter.upstream import load_fleetpy_bindings

CONFIG=Path('stage4/config/traffic_common_state_v1.json')
OUT=Path('stage4/output/traffic_common_state_v1')
DOC=Path('stage4/docs/traffic_research')
PHYSICAL=['native_id','order_id','request_time','sim_time_s','pickup_lon_wgs84','pickup_lat_wgs84',
          'dropoff_lon_wgs84','dropoff_lat_wgs84','realized_service_time_s','predicted_service_time_s']
DECISION=['profile_id','hard_state','evidence_complete','rho_static','rho_dynamic','rho_speed','selected_route_type']


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def physical_signature(sim):
    c=sim.operators[0]; s=signature(sim)
    s.pop('metadata'); s.pop('exposure')
    s['waiting_history']={rid:{k:v for k,v in m.items() if k not in ['exposure'] and not k.startswith('traffic_')}
                          for rid,m in c.request_meta.items()}
    s['fixtures']=[vars(r.fixture) for _,r in sorted(c.runtime_by_vid.items())]
    s['demand']=[[getattr(r,k) for k in PHYSICAL] for _,r in sorted(c.request_by_rid.items())]
    s['future']={t:sorted(v) for t,v in sim.demand.future_requests.items()}
    return digest(s)


def retarget(sim, requests, profile):
    """Keep native references/physical history; update only decision descriptors."""
    c=sim.operators[0]; before=physical_signature(sim)
    target={r.order_id:r for r in requests}
    assert len(target)==len(c.request_by_rid)
    for rid,r in c.request_by_rid.items():
        other=target[r.order_id]
        for field in PHYSICAL:
            a,b=getattr(r,field),getattr(other,field)
            assert a==b or (isinstance(a,float) and isinstance(b,float) and np.isnan(a) and np.isnan(b)), field
        for field in DECISION: setattr(r,field,getattr(other,field))
        if rid in c.request_meta:
            c.request_meta[rid]['exposure']=exposure_excess(r.rho_static,r.rho_dynamic,r.rho_speed)
            assert not any(k.startswith('traffic_') for k in c.request_meta[rid])
    c.config={**c.config,'profile_id':profile,'gamma_static':None,'gamma_dynamic':None,'gamma_speed':None}
    c.gammas={'static':None,'dynamic':None,'speed':None}
    assert not c.config.get('repositioning_enabled',False)
    assert before==physical_signature(sim)
    # Record all control settings; only profile_id is allowed to differ across profiles.
    controls={k:v for k,v in c.config.items() if k!='profile_id'}
    return dict(physical_sha256=before,control_sha256=digest(controls),profile=profile,
                acceptance_rate=c.acceptance_rate,acceptance_seed=c.acceptance_seed,
                inherited_ledger=dict(c.exposure_state.__dict__),gammas=c.gammas)


def summarize(root, cfg):
    rows=[]; pairs=[]
    policy=pd.read_parquet(root/'stage4/output/traffic_research/policy_date=20161031.parquet')
    for cut in cfg['cuts_s']:
        for profile in cfg['profiles']:
            frames={v:pd.read_parquet(root/OUT/f'{profile}_{cut}_{v}/cohort_outcomes.parquet') for v in cfg['variants']}
            pair=dict(profile=profile,cut=cut,**compare(frames['OFF'],frames['ON']))
            keys=['native_request_id','native_vehicle_id','simulation_time_s','pickup_eta_s','pickup_time','service_end_time']
            a,b=[pd.read_parquet(root/OUT/f'{profile}_{cut}_{v}/assignments.parquet',columns=keys).sort_values(keys[:3]).reset_index(drop=True) for v in cfg['variants']]
            pair['physical_assignments_identical']=a.equals(b)
            if profile=='A': assert pair['physical_assignments_identical']
            pairs.append(pair)
            for variant,x in frames.items():
                av=x.loc[x.vehicle_type.eq('AV'),['order_id']].merge(policy.loc[policy.profile_id.eq(profile)],on='order_id',validate='one_to_one')
                assert len(av)==int(x.vehicle_type.eq('AV').sum())
                if variant=='ON': assert av.outside_share.le(cfg['budget']+1e-6).all()
                rows.append(dict(profile=profile,cut=cut,variant=variant,cohort=len(x),matched=int(x.matched.sum()),
                    av=len(av),hv=int(x.vehicle_type.eq('HV').sum()),mean_wait_s=float(x.wait_s.mean()),
                    traffic={k:dict(mean=float(av[k].mean()),p50=float(av[k].median()),maximum=float(av[k].max()),
                        time_weighted_mean=float(np.average(av[k],weights=av.predicted_time_s)))
                             for k in ['outside_share','unknown_share','low_support_share']} if len(av) else {}))
    result=dict(status='COMPLETE',rows=rows,pairs=pairs)
    write_json(root/DOC/'common_state_comparison.json',result)
    return result


def run(root, fleetpy_root):
    cfg=json.loads((root/CONFIG).read_text()); load_fleetpy_bindings(fleetpy_root)
    cps=json.loads((root/'stage4/output/traffic_mechanism_v1/states/checkpoints.json').read_text())['rows']
    cps={r['cut']:r for r in cps if r['profile']=='M'}
    protected={str(p):sha(root/p) for p in [CONFIG,Path('stage3/config/stage3_av_capability_profiles.json'),
        Path('stage2/output_v5_2/development/M3/epoch_004.pt'),Path('stage4/input/replay_foundation/stage4_order_replay_base.parquet')]}
    out=root/OUT; out.mkdir(parents=True,exist_ok=False)
    summary=dict(status='RUNNING',rows=[],protected_sha256=protected)
    start=time.perf_counter(); write_json(out/'summary.json',summary)
    baseline={}
    try:
        for cut in cfg['cuts_s']:
            cp=cps[cut]; assert sha(root/cp['path'])==cp['sha256']
            for profile in cfg['profiles']:
                day=pd.Timestamp('2016-10-31T00:00:00+08:00')
                requests=load_all_test31_requests(root,start=day,end=day+pd.Timedelta(days=1,seconds=60),profile_id=profile)
                for variant in cfg['variants']:
                    name=f'{profile}_{cut}_{variant}'; dest=out/name; dest.mkdir()
                    summary['active']=name; write_json(out/'summary.json',summary)
                    captured={}
                    def prepare(sim):
                        captured.update(retarget(sim,requests,profile))
                        sig=(captured['physical_sha256'],captured['control_sha256'])
                        if cut not in baseline: baseline[cut]=sig
                        assert sig==baseline[cut], 'common start/control mismatch'
                    wc=dict(checkpoint=cp['path'],profile_id=profile,main_budget=cfg['budget'],checkpoint_s=cut,
                        measurement_start_s=cut,measurement_end_s=cut+cfg['window_duration_s'],
                        last_dispatch_s=cut+cfg['window_duration_s']+cfg['patience_s'],routing_mode=cfg['routing_mode'],
                        scenario_timeout_s=cfg['scenario_timeout_s'],rss_warning_mib=cfg['rss_warning_mib'],
                        prospective_gate_logging=True,baseline_reference=None,preserve_inherited_exposure=True)
                    result=w.condition(root,wc,MODE if variant=='ON' else VARIABILITY_MODE,dest,on_loaded=prepare)
                    assert result['routing_failures']==0
                    result.update(profile=profile,cut=cut,variant=variant,common_start=captured)
                    summary['rows'].append(result); write_json(out/'summary.json',summary)
                    print(json.dumps(dict(completed=name,matched=result['matched'],av=result['matched_AV'],rss=result['peak_rss_mib'])),flush=True)
                    gc.collect()
                del requests; gc.collect()
        summarize(root,cfg)
        assert protected=={p:sha(root/p) for p in protected}
        summary.update(status='COMPLETE',active=None,runtime_s=time.perf_counter()-start,
                       peak_rss_mib=psutil.Process().memory_info().peak_wset/2**20)
    except Exception as e:
        summary.update(status='STOPPED',error=repr(e)); write_json(out/'summary.json',summary)
        raise
    write_json(out/'summary.json',summary); write_json(root/DOC/'common_state_summary.json',summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--fleetpy-root',type=Path,required=True)
    args=parser.parse_args(); run(Path.cwd(),args.fleetpy_root)
