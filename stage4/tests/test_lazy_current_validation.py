"""Four tests of fixed-graph query separation, not business-policy experiments."""
from dataclasses import replace

import pytest

from stage4.dispatch.lazy_current_validation import CheckedAction, select_with_validation
from stage4.dispatch.scheme_a_city_master import CurrentAction


def wait(vid):
    return CurrentAction(f"W{vid}", vid, "WAIT", None)


def service(aid, vid, rid, distance=0., critical=False, carry=False):
    return CurrentAction(aid, vid, "SERVE", rid, critical, carry, distance)


def test_lazy_saves_checks_on_a_noncompetitive_component():
    known = [wait(1), wait(2), service("H", 1, 10, 0., True, True)]
    unknown = [service("C", 2, 10, 0., True, True)]
    callback = lambda ids: {aid: CheckedAction(replace(unknown[0], empty_distance_m=500.), 0) for aid in ids}
    eager = select_with_validation(known, {a.action_id: 0 for a in known},
                                  unknown, {"C": 0}, callback, backend="INTEGER_REFERENCE")
    lazy = select_with_validation(known, {a.action_id: 0 for a in known},
        unknown, {"C": 0}, callback, mode="LAZY_CERTIFIED", backend="INTEGER_REFERENCE")
    assert eager["score"] == lazy["score"]
    assert lazy["full_current_objective_certified"] and lazy["oracle_calls"] == 0
    assert eager["oracle_calls"] == 1


def test_rejected_or_changed_eta_values_do_not_lose_the_optimum():
    known = [wait(1), wait(2), service("H1", 1, 10, 80.), service("H2", 1, 11, 120.)]
    unknown = [service("A", 2, 10), service("B", 2, 11)]
    def callback(ids):
        return {aid: None if aid == "A" else CheckedAction(replace(unknown[1], empty_distance_m=90.), -1)
                for aid in ids}
    kwargs = dict(known=known, known_units={a.action_id: 0 for a in known},
                  unknown_bounds=unknown, upper_units={"A": 3, "B": 2},
                  validate_many=callback, backend="INTEGER_REFERENCE")
    eager = select_with_validation(**kwargs)
    lazy = select_with_validation(**kwargs, mode="LAZY_CERTIFIED")
    assert eager["score"] == lazy["score"]
    assert lazy["all_selected_actions_validated"] and lazy["full_current_objective_certified"]
    assert set(a.action_id for a in lazy["selected"]) == {"H1", "B"}


def test_invalid_bounds_and_unresolved_answers_refuse_certification():
    known = [wait(1)]
    unknown = [service("X", 1, 10, 10.)]
    with pytest.raises(ValueError, match="bound violated"):
        select_with_validation(known, {"W1": 0}, unknown, {"X": 0},
            lambda ids: {"X": CheckedAction(replace(unknown[0], empty_distance_m=5.), 0)},
            mode="LAZY_CERTIFIED", backend="INTEGER_REFERENCE")
    with pytest.raises(TypeError):
        select_with_validation(known, {"W1": 0}, unknown, {"X": 0},
            lambda ids: {"X": "BUDGET_UNKNOWN"}, mode="LAZY_CERTIFIED", backend="INTEGER_REFERENCE")


def test_full_lexicographic_priorities_not_just_matching_count():
    known = [wait(1), wait(2), service("H", 1, 10, 20., False, True)]
    unknown = [service("A", 2, 10, 0., False, True), service("B", 2, 11, 0., True, False)]
    answer = {a.action_id: CheckedAction(replace(a, empty_distance_m=100.), 1) for a in unknown}
    kwargs = dict(known=known, known_units={a.action_id: 0 for a in known},
                  unknown_bounds=unknown, upper_units={a.action_id: 3 for a in unknown},
                  validate_many=lambda ids: {aid: answer[aid] for aid in ids},
                  backend="INTEGER_REFERENCE")
    eager = select_with_validation(**kwargs)
    lazy = select_with_validation(**kwargs, mode="LAZY_CERTIFIED")
    assert eager["score"] == lazy["score"]
    assert lazy["score"][:3] == (1, 2, 1)
    assert lazy["full_current_objective_certified"]
