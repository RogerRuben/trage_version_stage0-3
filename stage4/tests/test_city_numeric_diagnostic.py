import pytest

from stage4.analysis.city_numeric_diagnostic import bounded_prefix, DiagnosticPrefixLimit


def test_prefix_stops_before_eleventh_epoch_instead_of_running_full_window():
    values=[]
    with pytest.raises(DiagnosticPrefixLimit, match="ten-epoch"):
        for tick in bounded_prefix(61200,70230,30):
            values.append(tick)
    assert values == list(range(61200,61500,30))
