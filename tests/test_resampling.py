from itertools import combinations

import numpy as np
import pytest
from scipy import stats

from experimentlab.resampling import bootstrap_mean, permutation_mean


def test_permutation_matches_exhaustive_labels():
    a, b = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 8.0])
    pool = np.r_[a, b]
    differences = []
    for selected in combinations(range(6), 3):
        mask = np.zeros(6, dtype=bool)
        mask[list(selected)] = True
        differences.append(pool[mask].mean() - pool[~mask].mean())
    observed = b.mean() - a.mean()
    expected = min(
        1,
        2
        * min(
            np.mean(np.array(differences) >= observed),
            np.mean(np.array(differences) <= observed),
        ),
    )
    assert permutation_mean(a, b, resamples=999)["p_value"] == pytest.approx(expected)


def test_bootstrap_agrees_with_scipy_with_mc_tolerance():
    rng = np.random.default_rng(88)
    a, b = rng.normal(size=50), rng.normal(0.4, size=70)
    actual = bootstrap_mean(a, b, resamples=19999, seed=9)
    reference = stats.bootstrap(
        (a, b),
        lambda x, y, axis: y.mean(axis=axis) - x.mean(axis=axis),
        method="percentile",
        paired=False,
        n_resamples=19999,
        rng=np.random.default_rng(5),
    ).confidence_interval
    # Independent resampling streams, so endpoints have Monte Carlo error.
    assert actual.ci_low == pytest.approx(reference.low, abs=0.035)
    assert actual.ci_high == pytest.approx(reference.high, abs=0.035)
    assert actual.p_value is None


def test_resampling_reproducible_and_nonzero_p():
    a, b = np.arange(10.0), np.arange(10.0) + 100
    assert bootstrap_mean(a, b) == bootstrap_mean(a, b)
    p = permutation_mean(a, b, resamples=199)["p_value"]
    assert p >= 1 / 200
    assert p == permutation_mean(a, b, resamples=199)["p_value"]


@pytest.mark.parametrize("kw", [{"resamples": 0}, {"resamples": 1.2}, {"batch": 0}])
def test_bootstrap_invalid_controls(kw):
    with pytest.raises(ValueError):
        bootstrap_mean([1, 2], [2, 3], **kw)
