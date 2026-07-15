from pathlib import Path

import pandas as pd
import pytest

from stage4.scripts.compare_fleetpy_cross_validation import compare, fleetpy_metrics, relative_error


def test_relative_error_and_thresholds_are_computed() -> None:
    reference = {
        "completed_orders": 100.0,
        "cancelled_orders": 10.0,
        "mean_pickup_time_sec": 100.0,
        "mean_waiting_time_sec": 120.0,
        "vehicle_busy_time_sec": 1000.0,
        "pickup_distance_m": 5000.0,
        "service_distance_m": 10000.0,
    }
    candidate = dict(reference)
    candidate["mean_waiting_time_sec"] = 133.0
    result = compare(candidate, reference).set_index("metric")
    assert result.loc["completed_orders", "status"] == "PASS"
    assert result.loc["mean_waiting_time_sec", "status"] == "FAIL"
    assert relative_error(0.0, 0.0) == 0.0


def test_missing_fleetpy_outputs_do_not_create_synthetic_pass(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        fleetpy_metrics(tmp_path)


def test_fleetpy_parser_uses_actual_user_and_operator_logs(tmp_path: Path) -> None:
    pd.DataFrame({
        "rq_time": [0, 10], "pickup_time": [20, 40],
        "dropoff_time": [100, None], "operator_id": [0, -1],
    }).to_csv(tmp_path / "1_user_stats.csv", index=False)
    pd.DataFrame({
        "status": ["route", "service"], "start_time": [0, 20], "end_time": [20, 100],
        "driven_distance": [100.0, 1000.0], "rq_on_board": ["", "0"],
        "rq_boarding": ["0", ""],
    }).to_csv(tmp_path / "2_0_op-stats.csv", index=False)
    metrics = fleetpy_metrics(tmp_path)
    assert metrics["completed_orders"] == 1.0
    assert metrics["cancelled_orders"] == 1.0
    assert metrics["mean_waiting_time_sec"] == 20.0
    assert metrics["pickup_distance_m"] == 100.0
    assert metrics["service_distance_m"] == 1000.0
