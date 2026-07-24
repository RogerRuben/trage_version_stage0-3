def test_failed_orders_are_explicit():
    expected = {"a", "b", "c"}
    reconstructed = {"a", "b"}
    failed = {"c"}
    assert reconstructed.isdisjoint(failed)
    assert reconstructed | failed == expected
