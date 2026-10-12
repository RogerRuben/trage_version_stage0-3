"""Four small analysis checks; no native replay, routing or inference."""
import numpy as np
import pandas as pd
import pytest

from stage4.analysis.scheme_a_layout_mechanism import (
    ParkedInventory, activation_decomposition, paired, slot_accounting,
)


def test_first_customer_clocks_retain_unserved_and_finish_after_close():
    working = slot_accounting(100., 400., [(200., 260., 600.)], [(110., 130.)])
    assert working["first_assignment_delay_s"] == 100.
    assert working["first_boarding_delay_s"] == 160.
    assert working["loaded_s"] == 140.
    assert working["pre_first_idle_s"] == 80.
    assert working["pre_first_move_s"] == 20.
    unsuccessful = slot_accounting(100., 400., [], [(110., 130.)])
    assert not unsuccessful["served"]
    assert unsuccessful["first_assignment_delay_s"] is None
    assert unsuccessful["restricted_first_customer_span_s"] == 300.
    assert unsuccessful["pre_first_idle_s"] == 280.
    assert unsuccessful["post_first_idle_s"] == 0.
    outside = slot_accounting(100., 150., [(140., 200., 300.)], [])
    assert outside["served"] and not outside["first_boarding_in_calendar_window"]
    assert outside["loaded_s"] == 0.


def test_paired_extensive_intensive_identity_and_transitions():
    base = pd.DataFrame(dict(served=[True, False, True, False], orders=[1, 0, 2, 0],
        loaded_s=[3600., 0., 7200., 0.], pre_first_idle_s=[100.] * 4, post_first_idle_s=[300.] * 4,
        activation_s=[0.] * 4, admission_end_s=[20000.] * 4, birth_cell=["100_200"] * 4,
        activation_hour=[0] * 4, historical_body_C_count_3days=[10.] * 4,
        historical_body_C_predicted_h_3days=[5.] * 4, template_HV_nearby=[3.] * 4,
        baseline_parked_HV_nearby=[2.] * 4))
    after = base.copy()
    after["served"], after["orders"], after["loaded_s"] = [True, True, False, False], [2, 1, 0, 0], [9000., 3600., 0., 0.]
    out = paired(base, after)
    assert len(out["transitions"]) == 4
    assert sum(row["slots"] for row in out["transitions"]) == 4
    assert out["orders_delta"] == 0 and out["loaded_delta_h"] == .5
    d = activation_decomposition(base, after)
    assert d["working_count_component_h"] == 0.
    assert d["conditional_intensity_component_h"] == .5
    assert not d["causal_component_effects_identified"]
    with pytest.raises(ValueError, match="identities"):
        paired(base, after.iloc[::-1])


def test_stationary_inventory_uses_completed_endpoints_and_excludes_activity():
    plans = pd.DataFrame(dict(native_id=[1], activation_s=[0.], admission_end_s=[100.],
        initial_lon_wgs84=[108.], initial_lat_wgs84=[34.]))
    jobs = pd.DataFrame(dict(native_vehicle_id=[1], simulation_time_s=[20.], service_end_time_s=[40.],
        completion_lon_wgs84=[108.1], completion_lat_wgs84=[34.]))
    moves = pd.DataFrame(dict(native_vehicle_id=[1], start_time_s=[5.], end_time_s=[15.],
        end_lon_wgs84=[108.05], end_lat_wgs84=[34.]))
    inv = ParkedInventory(plans, jobs, moves)
    inv.advance(0.)
    assert inv.nearby((108., 34.)) == 1
    inv.advance(10.)
    assert inv.nearby((108., 34.)) == 0
    inv.advance(15.)
    assert inv.nearby((108.05, 34.)) == 1 and inv.nearby((108., 34.)) == 0
    inv.advance(30.)
    assert inv.nearby((108.1, 34.)) == 0
    inv.advance(40.)
    assert inv.nearby((108.1, 34.)) == 1
    inv.advance(100.)
    assert inv.nearby((108.1, 34.)) == 0


def test_unknown_completed_endpoint_is_not_imputed_as_old_initial_location():
    plans = pd.DataFrame(dict(native_id=[1], activation_s=[0.], admission_end_s=[100.],
        initial_lon_wgs84=[108.], initial_lat_wgs84=[34.]))
    jobs = pd.DataFrame(dict(native_vehicle_id=[1], simulation_time_s=[10.], service_end_time_s=[20.],
        completion_lon_wgs84=[np.nan], completion_lat_wgs84=[34.]))
    moves = pd.DataFrame(columns=["native_vehicle_id", "start_time_s", "end_time_s", "end_lon_wgs84", "end_lat_wgs84"])
    inv = ParkedInventory(plans, jobs, moves)
    inv.advance(20.)
    assert inv.unknown_free == 1 and inv.nearby((108., 34.)) == 0
