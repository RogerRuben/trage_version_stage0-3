import numpy as np

from stage0.canonical.intervals import allocate_by_projected_mileage


def test_cross_link_time_and_distance_are_conserved():
    allocated_time, allocated_distance = allocate_by_projected_mileage(
        17.3, 91.7, [10.0, 25.0, 55.0]
    )
    assert np.isclose(allocated_time.sum(), 17.3, atol=1e-12)
    assert np.isclose(allocated_distance.sum(), 91.7, atol=1e-12)
    assert (allocated_time >= 0).all()
    assert (allocated_distance >= 0).all()


def test_zero_projected_mileage_has_explicit_equal_fallback():
    allocated_time, allocated_distance = allocate_by_projected_mileage(9.0, 6.0, [0, 0, 0])
    assert np.allclose(allocated_time, [3, 3, 3])
    assert np.allclose(allocated_distance, [2, 2, 2])

