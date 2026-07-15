import pandas as pd

from stage4.simulator_v3.preassignment.reservation_manager import ReservationManager


def test_reservation_expiry_clears_both_active_maps():
    now = pd.Timestamp("2016-10-23T10:00:00Z")
    manager = ReservationManager()
    manager.create("order-1", "vehicle-1", now, now + pd.Timedelta(minutes=5))

    expired = manager.expire_due(now + pd.Timedelta(minutes=5))

    assert [record.request_id for record in expired] == ["order-1"]
    assert manager.request_to_vehicle == {}
    assert manager.vehicle_to_request == {}
    assert manager.records[0].status == "EXPIRED"
    assert manager.failure_records[0]["failure_event"] == "EXPIRED"
    assert manager.audit()["reservation_invariant_pass"] == "PASS"


def test_reservation_revalidation_invalidates_missed_pickup_deadline():
    now = pd.Timestamp("2016-10-23T10:00:00Z")
    manager = ReservationManager()
    manager.create(
        "order-1",
        "vehicle-1",
        now,
        now + pd.Timedelta(minutes=8),
        expected_release_time=now + pd.Timedelta(minutes=3),
        safe_release_time=now + pd.Timedelta(minutes=4),
        buffer_source="validation_q0.90:global",
        buffer_sample_count=100,
        buffer_quantile=0.9,
        residual_quantile_sec=-60,
    )

    result = manager.revalidate(
        "order-1",
        now + pd.Timedelta(minutes=1),
        safe_release_time=now + pd.Timedelta(minutes=7),
        pickup_eta_sec=120,
        latest_pickup_time=now + pd.Timedelta(minutes=8),
    )

    assert not result.feasible
    assert result.reason == "PLAN_INVALIDATED:PICKUP_DEADLINE"
    assert manager.records[0].status == "INVALIDATED"
    assert manager.request_to_vehicle == {}
    assert manager.vehicle_to_request == {}
    assert manager.audit()["reservation_invariant_pass"] == "PASS"


def test_successful_revalidation_updates_auditable_buffer_metadata():
    now = pd.Timestamp("2016-10-23T10:00:00Z")
    manager = ReservationManager()
    manager.create("order-1", "vehicle-1", now, now + pd.Timedelta(minutes=8))
    metadata = {
        "expected_release_time": now + pd.Timedelta(minutes=2),
        "buffer_source": "validation_q0.90:time-zone-stress",
        "buffer_sample_count": 53,
        "buffer_quantile": 0.9,
        "release_residual_quantile_sec": -75.0,
    }

    result = manager.revalidate(
        "order-1",
        now + pd.Timedelta(seconds=30),
        safe_release_time=now + pd.Timedelta(minutes=3, seconds=15),
        pickup_eta_sec=60,
        latest_pickup_time=now + pd.Timedelta(minutes=8),
        release_metadata=metadata,
    )

    assert result.feasible
    record = manager.active_for_request("order-1")
    assert record is not None
    assert record.buffer_source == metadata["buffer_source"]
    assert record.buffer_sample_count == 53
    assert record.residual_quantile_sec == -75.0

