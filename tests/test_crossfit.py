import numpy as np
import pytest

import experimentlab.crossfit as module
from experimentlab.adjustment_data import frame_arrays, generate_trial
from experimentlab.crossfit import aipw_scores, crossfit_aipw


def trial():
    return frame_arrays(
        generate_trial(n=160, seed=27, propensity=0.3), "outcome", ["x1", "x2", "x3"]
    )


def test_augmented_score_at_unequal_assignment_probability():
    y, w = np.array([1.0, 3.0, 2.0, 7.0]), np.array([0, 0, 1, 1])
    m0, m1 = np.array([0.5, 1, 2, 3]), np.array([2, 2.5, 3, 4])
    # Explicit arm-specific expansion, independent of vectorized implementation.
    expected = np.array([1.5 - 0.5 / 0.7, 1.5 - 2 / 0.7, 1 - 1 / 0.3, 1 + 3 / 0.3])
    np.testing.assert_allclose(aipw_scores(y, w, m0, m1, 0.3), expected)


def test_known_regression_score_cancels_nuisance_in_expectation():
    # Enumerate W for each fixed pair of potential outcomes; nuisance may be wrong.
    y0, y1, m0, m1, p = 2.0, 5.0, -8.0, 12.0, 0.3
    scores = aipw_scores([y0, y1], [0, 1], [m0, m0], [m1, m1], p)
    assert (1 - p) * scores[0] + p * scores[1] == pytest.approx(y1 - y0)


def test_every_model_predicts_only_unseen_users(monkeypatch):
    y, w, x = trial()
    x = np.column_stack([np.arange(len(y)), x])
    predicted, calls = [], []

    class AuditMean:
        def fit(self, features, targets):
            self.users = set(features[:, 0].astype(int))
            self.mean = targets.mean()
            calls.append(self.users)
            return self

        def predict(self, features):
            ids = set(features[:, 0].astype(int))
            assert not self.users & ids
            predicted.extend(ids)
            return np.full(len(features), self.mean)

    monkeypatch.setattr(module, "make_learner", lambda kind, seed: AuditMean())
    result = crossfit_aipw(y, w, x, folds=4, propensity=0.3)
    assert len(calls) == 8
    np.testing.assert_array_equal(np.bincount(predicted), np.full(len(y), 2))
    assert set(result.fold_id) == {0, 1, 2, 3}


def test_heldout_outcomes_cannot_change_their_own_predictions():
    y, w, x = trial()
    a = crossfit_aipw(y, w, x, propensity=0.3, folds=3, seed=17)
    heldout = a.fold_id == 1
    changed = y.copy()
    changed[heldout] += np.linspace(10, 500, heldout.sum())
    b = crossfit_aipw(changed, w, x, propensity=0.3, folds=3, seed=17)
    np.testing.assert_array_equal(
        a.predictions_control[heldout], b.predictions_control[heldout]
    )
    np.testing.assert_array_equal(
        a.predictions_treatment[heldout], b.predictions_treatment[heldout]
    )
    assert not np.allclose(
        a.predictions_control[~heldout], b.predictions_control[~heldout]
    )


@pytest.mark.parametrize("learner", ["ridge", "forest"])
def test_reproducible_estimate_and_influence_se(learner):
    y, w, x = trial()
    a = crossfit_aipw(y, w, x, propensity=0.3, learner=learner)
    b = crossfit_aipw(y, w, x, propensity=0.3, learner=learner)
    assert a.estimate == b.estimate
    assert a.estimate.effect == pytest.approx(a.scores.mean())
    assert a.diagnostics["standard_error"] == pytest.approx(
        a.scores.std(ddof=1) / len(y) ** 0.5
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"propensity": 0},
        {"propensity": 1},
        {"folds": 1},
        {"folds": 1000},
        {"learner": "unknown"},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        crossfit_aipw(*trial(), **kwargs)


def test_insufficient_training_arm_fails_without_retry():
    # Each two-way training split has fewer than three users in at least one arm.
    with pytest.raises(ValueError, match="training fold"):
        crossfit_aipw(
            np.arange(6), np.array([0, 0, 0, 1, 1, 1]), np.arange(6).reshape(-1, 1)
        )
