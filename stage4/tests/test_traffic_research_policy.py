from types import SimpleNamespace
import pandas as pd
import pytest
from stage4.dispatch.traffic_research_policy import TrafficResearchPolicy, outside_share
from stage4.dispatch.exposure import exposure_excess


def fixture():
    request = SimpleNamespace(request_time=pd.Timestamp("2016-10-31", tz="Asia/Shanghai"), order_id="x",
        profile_id="M", selected_route_type="ORIGINAL", av_smoke_eligible=True, rho_static=1.2, rho_dynamic=9., rho_speed=.8)
    table = pd.DataFrame([dict(date="20161031", order_id="x", profile_id="M", selected_route_reference="ORIGINAL:x",
        outside_share=.05, unknown_share=.4, low_support_share=.2, rho_variability=1.1)])
    return request, table


def test_disabled_does_not_load_table():
    assert TrafficResearchPolicy.from_config({}) is None
    assert TrafficResearchPolicy.from_config({"traffic_research_policy":"FROZEN", "traffic_research_table":"missing"}) is None


def test_budget_boundary_unknown_not_gate_and_no_stacking():
    r, t = fixture(); original = r.__dict__.copy()
    result = TrafficResearchPolicy(t).evaluate(r)
    assert result["traffic_allowed"]
    assert result["exposure"].dynamic == pytest.approx(.1)
    assert result["exposure"].static == exposure_excess(r.rho_static,r.rho_dynamic,r.rho_speed).static
    assert r.__dict__ == original
    t.loc[0,"outside_share"] = .06
    assert not TrafficResearchPolicy(t).evaluate(r)["traffic_allowed"]


def test_identity_and_noneligible_handling():
    r, t = fixture(); r.selected_route_type = "FALLBACK"
    with pytest.raises(ValueError): TrafficResearchPolicy(t).evaluate(r)
    r.av_smoke_eligible=False
    assert TrafficResearchPolicy(t).evaluate(r) is None
    r.av_smoke_eligible=True; r.selected_route_type="ORIGINAL"; r.request_time += pd.Timedelta(days=1)
    with pytest.raises(ValueError): TrafficResearchPolicy(t).evaluate(r)


def test_mixed_mapping_and_nestedness():
    assert outside_share("C",.02,.01,.03) == pytest.approx(.06)
    assert outside_share("M",.02,.01,.03) == .01
    assert outside_share("A",.02,.01,.03) == 0
    with pytest.raises(ValueError): outside_share("X",0,0,0)


def test_variability_only_is_same_exposure_without_traffic_gate():
    from stage4.dispatch.traffic_research_policy import VARIABILITY_MODE
    r,t=fixture(); t.loc[0,"outside_share"] = .5
    v=TrafficResearchPolicy(t,mode=VARIABILITY_MODE).evaluate(r)
    vt=TrafficResearchPolicy(t).evaluate(r)
    assert v["exposure"] == vt["exposure"]
    assert v["traffic_allowed"] and not vt["traffic_allowed"]
    assert v["traffic_policy"] == VARIABILITY_MODE


def test_preflight_ignores_unconsumed_cross_midnight_future():
    from stage4.analysis.traffic_research_window import preflight_requests
    r,t=fixture()
    r.sim_time_s=37800; r.predicted_service_time_s=100.
    t["predicted_time_s"]=100.; t["rho_dynamic_frozen"]=r.rho_dynamic
    future=SimpleNamespace(sim_time_s=86410, av_smoke_eligible=True)
    c=SimpleNamespace(request_by_rid={1:r,2:future})
    assert preflight_requests(c,TrafficResearchPolicy(t),{"measurement_end_s":38700})==1
