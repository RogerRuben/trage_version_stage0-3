import pandas as pd

from stage4.simulator_v3.preassignment.safe_release_buffer import (
    RESIDUAL_DEFINITION,
    SafeReleaseBufferResolver,
)


def validation_rows() -> pd.DataFrame:
    records = []
    # Cell q0.9 is deterministic and has enough observations for selection.
    for index in range(40):
        records.append({
            "time_bin": 10,
            "zone": "z0_0",
            "stress_bucket": "high",
            "predicted_service_time_sec": 600 + index,
            "realized_service_time_sec": 700,
        })
    # A separate time cell ensures the table has meaningful fallbacks.
    for index in range(40):
        records.append({
            "time_bin": 11,
            "zone": "z1_0",
            "stress_bucket": "low",
            "predicted_service_time_sec": 900 + index,
            "realized_service_time_sec": 850,
        })
    return pd.DataFrame(records)


def test_validation_q90_safe_release_uses_declared_residual_definition():
    table = SafeReleaseBufferResolver.build_table(
        validation_rows(),
        validation_date="20161022",
        source_dataset="unit_validation",
    )
    resolver = SafeReleaseBufferResolver(table, minimum_samples=30)
    expected = pd.Timestamp("2016-10-23T10:00:00Z")
    resolution = resolver.resolve(expected, 10, "z0_0", "high")

    assert resolution.buffer_source == "validation_q0.90:time-zone-stress"
    assert resolution.buffer_sample_count == 40
    assert resolution.residual_definition == RESIDUAL_DEFINITION
    assert resolution.safe_release_time == expected - pd.Timedelta(
        seconds=resolution.residual_quantile_sec
    )
    assert resolution.to_metadata()["buffer_quantile"] == 0.9


def test_validation_q90_falls_back_hierarchically_without_fixed_seconds():
    table = SafeReleaseBufferResolver.build_table(
        validation_rows(),
        validation_date="20161022",
        source_dataset="unit_validation",
    )
    resolver = SafeReleaseBufferResolver(table, minimum_samples=30)
    resolution = resolver.resolve(
        pd.Timestamp("2016-10-23T10:00:00Z"),
        47,
        "unknown_zone",
        "unknown",
    )
    assert resolution.buffer_source == "validation_q0.90:global"
    assert resolution.buffer_sample_count == 80
    assert resolution.residual_quantile_sec != 30.0

