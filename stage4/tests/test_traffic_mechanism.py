from stage4.analysis.traffic_independent_diagnosis import joint
from stage4.analysis.traffic_mechanism import solve_metrics, window_config
from stage4.dispatch.solver import AssignmentArc
from stage4.dispatch.exposure import CumulativeExposureState
import pandas as pd


def test_joint_separates_prior_exclusion_and_missing():
    g=pd.DataFrame(dict(outside_share=[.1,.1,.01,None],base_eligible=[True,False,True,False],
                        unknown_share=[.2,.2,.1,None],low_support_share=[.1,.1,0,None]))
    r=joint(g)
    assert r['independent_traffic_rejection']==1
    assert r['already_excluded_with_outside']==1
    assert r['eligible_traffic_pass']==1
    assert r['descriptor_present']==3


def test_graph_capacity_distinct_from_gamma_capacity():
    arcs=[AssignmentArc(1,1,10,False,False,vehicle_type='AV',exposure_dynamic=2),
          AssignmentArc(2,2,12,False,False)]
    k=dict(exposure_state=CumulativeExposureState(),gammas={'dynamic':0},cost_level_enabled=False)
    r=solve_metrics(arcs,k)
    assert r['mixed']['M']==2 and r['constrained_maximum']==1 and r['selected']==1


def test_window_is_bounded_and_profile_specific(tmp_path):
    cfg=dict(budget=.05,window_duration_s=900,patience_s=300,routing_mode='SINGLE_SOURCE_MATRIX',
             scenario_timeout_s=1800,rss_warning_mib=2048)
    c=window_config(tmp_path,cfg,'C',63000,tmp_path/'cp.pkl')
    assert c['measurement_end_s']==63900 and c['last_dispatch_s']==64200
    assert c['baseline_reference'] is None and c['profile_id']=='C'


def test_paired_report_keeps_gained_and_lost_orders():
    from stage4.analysis.traffic_mechanism_compare import compare
    a=pd.DataFrame(dict(order_id=['a','b','c'],matched=[True,True,False],
                        vehicle_type=['HV','AV',None],wait_s=[10.,20.,None]))
    b=pd.DataFrame(dict(order_id=['a','b','c'],matched=[True,False,True],
                        vehicle_type=['AV',None,'HV'],wait_s=[15.,None,30.]))
    r=compare(a,b)
    assert r['gained']==1 and r['lost']==1 and r['both_matched']==1
    assert r['common_matched_wait_difference_s']==5
