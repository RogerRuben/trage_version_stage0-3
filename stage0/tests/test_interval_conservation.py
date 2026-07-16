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


def test_time_allocation_conservation_many_links():
    time, _ = allocate_by_projected_mileage(123.456, 500.0, [1.0] * 257)
    assert np.isclose(time.sum(), 123.456, atol=1e-12)
    assert (time >= 0).all()


def test_distance_allocation_conservation_many_links():
    _, distance = allocate_by_projected_mileage(100.0, 9876.543, [1.0] * 257)
    assert np.isclose(distance.sum(), 9876.543, atol=1e-12)
    assert (distance >= 0).all()


def test_time_allocation_conservation():
    time, _ = allocate_by_projected_mileage(10.0, 20.0, [1.0, 2.0])
    assert np.isclose(time.sum(), 10.0)


def test_distance_allocation_conservation():
    _, distance = allocate_by_projected_mileage(10.0, 20.0, [1.0, 2.0])
    assert np.isclose(distance.sum(), 20.0)
