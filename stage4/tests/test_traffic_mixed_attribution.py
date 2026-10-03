import numpy as np
import pandas as pd
from stage4.analysis.traffic_mixed_attribution import attribution, weighted_stats, paired


def test_attribution_partition_and_policy_tolerance():
    r=pd.Series([0,.08,.01,.08,.03,.05+1e-6,np.nan])
    c=pd.Series([0,.01,.08,.08,.03,0,.08])
    assert attribution(r,c).tolist()==['PASS','RELATIVE_ONLY','CONGESTION_ONLY',
        'BOTH_INDIVIDUALLY','COMBINED_ONLY','PASS','MISSING']


def test_weighted_inverse_cdf_and_missing():
    s=weighted_stats([10,20,30,np.nan],[1,8,1,100])
    assert s['p50']==20 and s['mean']==20 and s['valid_time_s']==10
    assert weighted_stats([np.nan],[1])['mean'] is None


def test_paired_keeps_unserved_and_hv_loss():
    a=pd.DataFrame(dict(order_id=['a','b','c'],matched=[True,False,True],vehicle_type=['HV',None,'AV']))
    b=pd.DataFrame(dict(order_id=['c','a','b'],matched=[True,False,True],vehicle_type=['HV',None,'AV']))
    x=paired(a,b).set_index('order_id')
    assert x.loc['a','outcome']=='LOST' and x.loc['a','type_F']=='HV'
    assert x.loc['b','outcome']=='GAINED' and x.loc['c','type_VT']=='HV'


def test_local_artifacts_reconcile_and_av_identity():
    from pathlib import Path
    import pytest
    from stage4.analysis.traffic_mixed_attribution import OUT, LIMIT
    if not (OUT/'summary.json').exists():
        pytest.skip('Optional local generated artifact QA')
    for date in ['20161025','20161026','20161027','20161031']:
        r=pd.read_parquet(OUT/f'routes_date={date}.parquet')
        assert not r.order_id.duplicated().any()
        assert r.attribution.ne('PASS').equals(r.outside_share.gt(LIMIT))
        assert np.allclose(r.outside_share,r.outside_low_share+r.outside_supported_share)
    joint=pd.read_parquet('stage4/output/traffic_mechanism_v1/offline/test31_joint.parquet')
    for profile in ['C','M']:
        for cut in [37800,63000]:
            x=pd.read_parquet(OUT/f'paired_{profile}_{cut}.parquet')
            assert not x.order_id.duplicated().any()
            assert x.loc[x.attribution.eq('MISSING'),'outcome'].eq('BOTH_UNSERVED').all()
            for variant in ['F','VT']:
                av=x.loc[x[f'type_{variant}'].eq('AV'),['order_id']].merge(
                    joint.loc[joint.profile_id.eq(profile)],on='order_id',validate='one_to_one')
                assert av.selected_route_type.eq('ORIGINAL').all() and av.base_eligible.all()
