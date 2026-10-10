"""The one generic reference may not fit target-day outcomes or task values."""
import pandas as pd

from stage4.analysis.generic_defer_comparison import fit_reference


def test_reference_uses_only_earlier_forecast_customer_site_connections(tmp_path):
    inputs=dict(sites=[dict(site_id="S1"),dict(site_id="S2")],forecast_cohorts=[
        dict(tasks=[dict(job_id="F10_1"),dict(job_id="F17_1")])])
    rows=[dict(origin_id="S1",target_job_id="F10_1",supported=True,compatible_profiles=["HV"],travel_time_s=80,empty_distance_m=800),
        dict(origin_id="S2",target_job_id="F10_1",supported=True,compatible_profiles=["HV"],travel_time_s=40,empty_distance_m=400),
        dict(origin_id="S1",target_job_id="F17_1",supported=True,compatible_profiles=["HV"],travel_time_s=100,empty_distance_m=1000),
        dict(origin_id="S1",target_job_id="A_target_future",supported=True,compatible_profiles=["HV"],travel_time_s=1,empty_distance_m=1),
        dict(origin_id="A_actual_end",target_job_id="F10_1",supported=True,compatible_profiles=["HV"],travel_time_s=2,empty_distance_m=2)]
    pd.DataFrame(rows).to_parquet(tmp_path/"sparse_connections.parquet",index=False)
    result=fit_reference(tmp_path,inputs,dict(reference_max_pickup_s=300,
        history_dates=["20161010","20161017"],reference_rule="fixed_history_rule"))
    assert result["travel_time_s"] == 70 and result["empty_distance_m"] == 700
    assert result["historical_customer_sample_count"] == 2 and not result["target_day_actual_orders_used"]
