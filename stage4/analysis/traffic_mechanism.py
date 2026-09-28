"""Bounded F/V/VT extension, profile-own history, sparse captured production arcs."""
import argparse
from dataclasses import replace
import gc
import json
import math
from pathlib import Path
from types import MethodType
from unittest.mock import patch
import time

import pandas as pd
import psutil
from joblib.externals import cloudpickle
from stage3.scripts.traffic_state_batch1 import sha, write_json, write_parquet
from stage4.analysis import native_checkpoint_closure as n
from stage4.analysis import traffic_research_window as w
from stage4.analysis.mechanism_validity import graph_metrics
from stage4.dispatch import rolling_or_control as production
from stage4.dispatch.solver import solve_lexicographic
from stage4.dispatch.traffic_research_policy import MODE, VARIABILITY_MODE

CONFIG=Path('stage4/config/traffic_mechanism_v1.json')
OUT=Path('stage4/output/traffic_mechanism_v1')
DOC=Path('stage4/docs/traffic_research')
MODES={'F':'FROZEN','V':VARIABILITY_MODE,'VT':MODE}


def checkpoint(root, cfg, profile, cut):
    """Forced-action native replay, no prehistory routing/optimization."""
    if profile=='M' and cut==37800:
        p=root/'stage4/output/paper_enhancement/native_checkpoint_closure/retry1/q50/native_checkpoint.pkl'
        return p, dict(reused=True, path=str(p.relative_to(root)),sha256=sha(p))
    dest=root/OUT/'checkpoints'/f'{profile}_{cut}'
    dest.mkdir(parents=True,exist_ok=False)
    source=root/'stage4/output/final_experiments'/cfg['scenarios'][profile]
    config=json.loads((source/'scenario_config.json').read_text())['runtime_configuration']
    sources={str(p.relative_to(root)):sha(p) for p in [source/'scenario_config.json',source/'assignment_log.parquet',source/'exposure_state.parquet']}
    actions=pd.read_parquet(source/'assignment_log.parquet')
    start=pd.Timestamp('2016-10-31T00:00:00+08:00')
    requests=n.f.load_all_test31_requests(root,start=start,end=start+pd.Timedelta(days=1,seconds=60),profile_id=profile)
    step=int(config['dispatch_interval_s'])
    end_s=int(config['matching_end_s'])+math.ceil(max(r.realized_service_time_s for r in requests)/step)*step+step
    end=start+pd.Timedelta(seconds=end_s)
    fleet=n.f.build_fleet_scenario(root,benchmark_start=start,simulation_end=end,requested_q_a=.5,
        seed=config['fleet_sampling_seed'],max_hv_hour_error_pct=config['max_hv_vehicle_hour_error_pct'])
    bindings=n.load_fleetpy_bindings(cfg['fleetpy_root'])
    registry=n.CoordinateRegistry()
    n.attach_fleetpy_requests(requests,bindings,registry)
    network=n.create_native_network(bindings,registry)
    demand=n.create_native_demand(bindings,requests,registry,network,dest)
    vehicles,native_output=n.create_native_vehicles(fleet.native_fixtures,bindings,registry,demand.rq_db,
        dest/'runtime',native_movement=True,routing_engine=network)
    c=n.create_rolling_or_fleet_control(bindings,vehicles,requests,demand,network,None,start,end,config)
    c.recorded_actions={int(t):g.to_dict('records') for t,g in actions[actions.simulation_time_s<cut].groupby('simulation_time_s')}
    c.time_trigger=MethodType(n.forced_actions,c)
    sim=n.create_native_simulation(bindings,simulation_end_s=end_s,time_step_s=step,demand=demand,
        vehicles=[v.native_vehicle for v in vehicles],fleet_control=c,network=network,native_output=native_output)
    started=time.perf_counter()
    with patch.object(n,'CUT',cut):
        try:
            for tick in range(0,cut+step,step):
                if time.perf_counter()-started>cfg['scenario_timeout_s']:
                    raise TimeoutError('checkpoint reconstruction timeout')
                sim.step(tick)
                if tick%7200==0:
                    print(json.dumps(dict(stage='checkpoint',profile=profile,cut=cut,tick=tick,rss_mib=psutil.Process().memory_info().rss/2**20)),flush=True)
        except n.CheckpointReached:
            pass
        else:
            raise AssertionError('checkpoint not reached')
    del c.time_trigger; del c.recorded_actions
    strict=n.f.pre_decision_vehicle_state(fleet.native_fixtures,actions,start+pd.Timedelta(seconds=cut))
    actual=[c._spatial_vehicle(v) for v in vehicles if c._available(v,cut)]
    assert n.f._sha([v.__dict__ for v in sorted(actual,key=lambda v:v.vehicle_id)])==n.f._sha([v.__dict__ for v in sorted(strict,key=lambda v:v.vehicle_id)])
    prior=actions[actions.simulation_time_s<cut]
    busy=prior[pd.to_datetime(prior.service_end_time)>start+pd.Timedelta(seconds=cut)]
    assert set(c.rid_to_assigned_vid)-c.completed_rids==set(busy.native_request_id.astype(int))
    expected=n.f._exposure_before(pd.read_parquet(source/'exposure_state.parquet'),cut)
    assert all(abs(v-expected.__dict__[k])<1e-7 for k,v in c.exposure_state.__dict__.items())
    c.exposure_state.validate(c.gammas,c.solver_tolerance)
    p=dest/'native_checkpoint.pkl'
    with p.with_suffix('.tmp').open('wb') as f: cloudpickle.dump(sim,f)
    p.with_suffix('.tmp').replace(p)
    with p.open('rb') as f: restored=cloudpickle.load(f)
    assert n.f._sha(n.signature(sim))==n.f._sha(n.signature(restored))
    observed=pd.DataFrame(c.assignment_rows).set_index('native_request_id')
    errors={}
    for col in ['pickup_time','service_end_time']:
        # Busy completion fields can be pending; compare completed history only.
        ids=sorted(c.completed_rids & set(prior.native_request_id.astype(int)))
        errors[col]=float((pd.to_datetime(observed.loc[ids,col])-pd.to_datetime(prior.set_index('native_request_id').loc[ids,col])).dt.total_seconds().abs().max())
        assert errors[col]<1e-6
    assert sources=={s:sha(root/s) for s in sources}
    report=dict(reused=False,path=str(p.relative_to(root)),sha256=sha(p),sources=sources,
        strict_available_equal=True,busy_equal=True,roundtrip_equal=True,history_errors=errors,
        prior_assignments=len(prior),busy=len(busy),available=len(actual),gammas=c.gammas)
    write_json(dest/'summary.json',report)
    return p,report


def window_config(root,cfg,profile,cut,path):
    return dict(checkpoint=str(path.relative_to(root)),profile_id=profile,main_budget=cfg['budget'],
        checkpoint_s=cut,measurement_start_s=cut,measurement_end_s=cut+cfg['window_duration_s'],
        last_dispatch_s=cut+cfg['window_duration_s']+cfg['patience_s'],routing_mode=cfg['routing_mode'],
        scenario_timeout_s=cfg['scenario_timeout_s'],rss_warning_mib=cfg['rss_warning_mib'],
        prospective_gate_logging=True,
        baseline_reference='stage4/output/traffic_research/window/retry1/FROZEN/assignments.parquet' if profile=='M' and cut==37800 else None)


class Captured(Exception):
    pass


def solve_metrics(arcs,kwargs):
    mixed=graph_metrics([(a.request_id,a.vehicle_id) for a in arcs])
    av=graph_metrics([(a.request_id,a.vehicle_id) for a in arcs if a.vehicle_type=='AV'])
    chosen=solve_lexicographic(arcs,**kwargs)
    # Maximum cardinality subject to actual Gamma, distinct from critical-first control.
    cardinal=solve_lexicographic([replace(a,critical=False,carry_over=False) for a in arcs],**kwargs)
    return dict(mixed=mixed,av=av,constrained_maximum=cardinal.total_matched,
        selected=chosen.total_matched,selected_av=sum(arcs[i].vehicle_type=='AV' for i in chosen.selected_indices),
        selected_pairs=sorted((arcs[i].request_id,arcs[i].vehicle_id) for i in chosen.selected_indices),
        selected_pickup_sum_s=chosen.pickup_eta_optimum_s)


def fixed_state(root,cfg,profile,cut,path):
    rows=[]; v_pairs=None; v_arcs=None; v_kwargs=None; allowed=None
    wc=window_config(root,cfg,profile,cut,path)
    for variant in cfg['variants']:
        captured={}
        def ready(c):
            captured['allowed']={rid:meta.get('traffic_allowed',True) for rid,meta in c.request_meta.items()}
            captured['before']=dict(c.exposure_state.__dict__)
        def intercept(arcs,**kwargs):
            captured['arcs']=arcs; captured['kwargs']=kwargs
            raise Captured()
        with patch.object(production,'solve_lexicographic',intercept):
            try: w.condition(root,wc,MODES[variant],None,on_ready=ready)
            except Captured: pass
            else: raise AssertionError('solver not intercepted')
        arcs=captured['arcs']; kwargs=captured['kwargs']
        pairs={(a.request_id,a.vehicle_id) for a in arcs}
        if variant=='F': f_pairs=pairs
        if variant=='V':
            assert pairs==f_pairs
            # Payload includes native objects: drop it for bounded diagnostic retention.
            v_arcs=[replace(a,payload=None) for a in arcs]; v_kwargs=kwargs
            v_pairs=pairs
        if variant=='VT':
            allowed=captured['allowed']
        result=dict(profile=profile,cut=cut,variant=variant,source_scenario=cfg['scenarios'][profile],
            gammas=kwargs['gammas'],history_exposure=captured['before'],**solve_metrics(arcs,kwargs))
        if variant=='VT':
            result['topk_refill_added_pairs']=len(pairs-v_pairs)
            result['removed_pairs']=len(v_pairs-pairs)
            deleted=[a for a in v_arcs if a.vehicle_type!='AV' or allowed.get(a.request_id,True)]
            result['fixed_V_arc_deletion']=solve_metrics(deleted,v_kwargs)
        rows.append(result)
        print(json.dumps({k:result[k] for k in ['profile','cut','variant','mixed','av','selected','constrained_maximum']}),flush=True)
        del captured,arcs; gc.collect()
    return rows


def run(root,fleetpy_root,stage):
    cfg=json.loads((root/CONFIG).read_text()); cfg['fleetpy_root']=fleetpy_root
    n.load_fleetpy_bindings(fleetpy_root)
    assert json.loads((root/DOC/'independent_diagnosis.json').read_text())['status']=='COMPLETE'
    protected={str(p):sha(root/p) for p in [CONFIG,Path('stage3/config/stage3_av_capability_profiles.json'),Path('stage2/output_v5_2/development/M3/epoch_004.pt')]}
    out=root/OUT/stage; out.mkdir(parents=True,exist_ok=False)
    summary=dict(status='RUNNING',stage=stage,protocol_commit='6fcc31a',protected_sha256=protected,rows=[])
    write_json(out/'summary.json',summary)
    started=time.perf_counter()
    try:
        if stage=='states':
            cps=[]
            for profile in cfg['state_profiles']:
                for cut in cfg['cuts_s']:
                    summary['active']=f'{profile}_{cut}'; write_json(out/'summary.json',summary)
                    path,record=checkpoint(root,cfg,profile,cut)
                    cps.append(dict(profile=profile,cut=cut,**record))
                    write_json(out/'checkpoints.json',dict(rows=cps))
                    summary['rows'].extend(fixed_state(root,cfg,profile,cut,path))
                    write_json(out/'summary.json',summary); gc.collect()
        else:
            assert json.loads((root/OUT/'states/summary.json').read_text())['status']=='COMPLETE'
            cps=json.loads((root/OUT/'states/checkpoints.json').read_text())['rows']
            for cp in cps:
                profile,cut=cp['profile'],cp['cut']
                if profile not in cfg['dynamic_profiles']: continue
                path=root/cp['path']; assert sha(path)==cp['sha256']
                wc=window_config(root,cfg,profile,cut,path)
                for variant in cfg['variants']:
                    name=f'{profile}_{cut}_{variant}'; dest=out/name; dest.mkdir()
                    summary['active']=name; write_json(out/'summary.json',summary)
                    result=w.condition(root,wc,MODES[variant],dest)
                    result.update(profile=profile,cut=cut,variant=variant)
                    summary['rows'].append(result); write_json(out/'summary.json',summary)
                    print(json.dumps(result),flush=True); gc.collect()
        assert protected=={p:sha(root/p) for p in protected}
        summary.update(status='COMPLETE',active=None,runtime_s=time.perf_counter()-started,
            peak_rss_mib=psutil.Process().memory_info().peak_wset/2**20)
    except Exception as e:
        summary.update(status='STOPPED',error=repr(e)); write_json(out/'summary.json',summary)
        raise
    write_json(out/'summary.json',summary); write_json(root/DOC/f'mechanism_{stage}.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['states','dynamic'])
    p.add_argument('--fleetpy-root',type=Path,required=True)
    args=p.parse_args(); run(Path.cwd(),args.fleetpy_root,args.stage)
