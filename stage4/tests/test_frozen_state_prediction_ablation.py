from stage4.analysis import frozen_state_prediction_ablation as ablation


def test_epochs_are_preregistered_and_include_evening():
    assert 8 <= len(ablation.EPOCH_CLOCKS) <= 12
    assert len(set(ablation.EPOCH_CLOCKS)) == len(ablation.EPOCH_CLOCKS)
    assert any(clock.startswith(("17:", "18:")) for clock in ablation.EPOCH_CLOCKS)


def test_variants_and_no_full_day_runner_dependency():
    assert ablation.VARIANTS == ("P", "H", "D0")
    source = open(ablation.__file__, encoding="utf-8").read()
    assert "create_native_simulation" not in source
    assert "solve_lexicographic" in source
    assert "SINGLE_SOURCE_MATRIX" in source


def test_mid_cdf_ties():
    import numpy as np

    values = np.asarray([0.0, 1.0, 1.0, 2.0])
    assert ablation._mid_cdf(1.0, values) == 0.5
