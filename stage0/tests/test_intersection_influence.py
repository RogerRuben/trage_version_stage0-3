from stage0.canonical.intersection import prorate_to_upstream_influence, upstream_influence_share


def test_long_upstream_link_uses_only_declared_influence_area():
    assert upstream_influence_share(300.0, 75.0) == 0.25
    assert prorate_to_upstream_influence(40.0, 300.0, 75.0) == 10.0


def test_short_link_is_entirely_inside_influence_area():
    assert upstream_influence_share(50.0, 75.0) == 1.0

