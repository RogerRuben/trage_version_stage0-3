import gc
import json
from pathlib import Path

import pandas as pd
import pytest
from joblib.externals import cloudpickle
from stage4.analysis.traffic_common_state import retarget
from stage4.analysis.traffic_research_window import preflight_requests
from stage4.dispatch.traffic_research_policy import TrafficResearchPolicy, VARIABILITY_MODE
from stage4.fleetpy_adapter.test31_demand_adapter import load_all_test31_requests
from stage4.fleetpy_adapter.upstream import load_fleetpy_bindings


def test_prespecified_design():
    cfg=json.loads(Path('stage4/config/traffic_common_state_v1.json').read_text())
    assert len(cfg['profiles'])*len(cfg['cuts_s'])*len(cfg['variants'])==12
    assert all(cfg['gamma_'+k] is None for k in ['static','dynamic','speed'])
    assert cfg['budget']==.05 and cfg['source_profile']=='M'


def test_local_shared_checkpoint_retarget_preflight():
    import os
    fleetpy=os.environ.get('FLEETPY_ROOT')
    path=Path('stage4/output/traffic_mechanism_v1/states/checkpoints.json')
    if not fleetpy or not path.exists():
        pytest.skip('Requires local frozen checkpoints and FleetPy')
    load_fleetpy_bindings(Path(fleetpy))
    cps=[r for r in json.loads(path.read_text())['rows'] if r['profile']=='M']
    day=pd.Timestamp('2016-10-31T00:00:00+08:00')
    policy=TrafficResearchPolicy(pd.read_parquet('stage4/output/traffic_research/policy_date=20161031.parquet'),mode=VARIABILITY_MODE)
    for cp in cps:
        baseline=None
        for profile in ['C','M','A']:
            with Path(cp['path']).open('rb') as f: sim=cloudpickle.load(f)
            requests=load_all_test31_requests(Path.cwd(),start=day,end=day+pd.Timedelta(days=1,seconds=60),profile_id=profile)
            r=retarget(sim,requests,profile)
            sig=(r['physical_sha256'],r['control_sha256'])
            if baseline is None: baseline=sig
            assert sig==baseline
            c=sim.operators[0]
            assert preflight_requests(c,policy,{'measurement_end_s':cp['cut']+900})>0
            assert c.config['profile_id']==profile
            del c,sim,requests; gc.collect()


def test_completed_local_common_state_outputs():
    path=Path('stage4/docs/traffic_research/common_state_summary.json')
    if not path.exists():
        pytest.skip('Generated after all prespecified conditions complete')
    summary=json.loads(path.read_text())
    assert summary['status']=='COMPLETE' and len(summary['rows'])==12
    for cut in [37800,63000]:
        rows=[r for r in summary['rows'] if r['cut']==cut]
        assert len({r['checkpoint_sha256'] for r in rows})==1
        assert len({r['common_start']['physical_sha256'] for r in rows})==1
        assert len({r['common_start']['control_sha256'] for r in rows})==1
        assert len({(r['common_start']['acceptance_seed'],r['common_start']['acceptance_rate']) for r in rows})==1
        for row in rows:
            assert row['routing_failures']==0 and row['gate_conservation']
            assert all(v is None for v in row['gammas'].values())
            assert row['cohort_orders']==(379 if cut==37800 else 419)
    comparison=json.loads(Path('stage4/docs/traffic_research/common_state_comparison.json').read_text())
    for row in comparison['pairs']:
        assert row['gained']+row['both_matched']+row['lost']+row['both_unserved']==row['cohort']
        if row['profile']=='A':
            assert row['physical_assignments_identical'] and row['gained']==row['lost']==0
