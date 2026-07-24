from stage0.scripts.hmm_viterbi_matcher import geometric_fallback_compatible


def test_geometric_fallback_requires_every_link_in_current_network():
    network = frozenset({"new_a", "new_b"})
    assert geometric_fallback_compatible(["new_a", "new_b"], network)
    assert not geometric_fallback_compatible(["new_a", "old_b"], network)


def test_geometric_fallback_rejects_missing_link_identifier():
    assert not geometric_fallback_compatible(["new_a", None], frozenset({"new_a"}))
