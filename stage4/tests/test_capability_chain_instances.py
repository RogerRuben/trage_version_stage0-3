"""Only necessary checks for directional joins and outcome-independent sampling."""
import pandas as pd

from stage4.analysis.capability_chain_instances import choose_tasks, interface_supported


def test_directed_join_does_not_allow_mid_block_reverse_or_distant_jump():
    forward = dict(uid="F", from_node="a", to_node="b", geometry=[(109.0, 34.0), (109.01, 34.0)])
    reverse = dict(uid="RESEARCH_REVERSE:F:R", from_node="b", to_node="a", geometry=list(reversed(forward["geometry"])))
    assert not interface_supported(forward, reverse, (109.005, 34), (109.005, 34), 80)[0]
    assert interface_supported(forward, reverse, (109.01, 34), (109.01, 34), 80)[0]
    disconnected = dict(uid="G", from_node="c", to_node="d", geometry=[(109.01, 34), (109.02, 34)])
    assert not interface_supported(forward, disconnected, (109.01, 34), (109.01, 34), 80)[0]
    assert not interface_supported(forward, forward, (109.005, 34.002), (109.005, 34.002), 80)[0]


def test_diagnostic_selection_is_stable_and_keeps_non_c_tasks():
    cfg = dict(historical_date="20161010", horizon_s=1800, grid_origin_wgs84=[108, 34],
        grid_size_degrees=.02, selection_seed=20261008, maximum_tasks=4, diagnostic_C_task_target=2)
    frame = pd.DataFrame([dict(order_id=f"o{i}", date="20161010", common_eligible=True,
        release_second=27000 + i * 60, compatible_C=i < 3,
        start_lon_wgs84=108.945, start_lat_wgs84=34.225) for i in range(7)])
    sites = [dict(cell="47_11")]
    a, summary = choose_tasks(frame, sites, cfg, 27000)
    b, _ = choose_tasks(frame.iloc[::-1], sites, cfg, 27000)
    assert list(a.order_id) == list(b.order_id)
    assert len(a) == 4 and int(a.compatible_C.sum()) == 2
    assert summary["cell_window_orders"] == 7
    assert len(set(a.job_id)) == 4
