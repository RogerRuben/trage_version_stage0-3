from scripts.extract_canonical_smoke_raw import stable_rank


def test_stable_order_rank_changes_with_seed_and_date():
    assert stable_rank(1, "20161020", "o") == stable_rank(1, "20161020", "o")
    assert stable_rank(1, "20161020", "o") != stable_rank(2, "20161020", "o")
    assert stable_rank(1, "20161020", "o") != stable_rank(1, "20161022", "o")
