import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

from experimentlab.inference import conversion, welch, wilson


@pytest.mark.parametrize("seed", range(8))
def test_welch_matches_scipy(seed):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(2, 1, 23), rng.normal(2.5, 3, 41)
    actual = welch(a, b)
    reference = stats.ttest_ind(b, a, equal_var=False)
    interval = reference.confidence_interval()
    assert actual.p_value == pytest.approx(reference.pvalue, abs=1e-12)
    assert (actual.ci_low, actual.ci_high) == pytest.approx(
        (interval.low, interval.high)
    )


@pytest.mark.parametrize(
    "counts", [(10, 100, 15, 120), (1, 80, 4, 40), (80, 100, 99, 120)]
)
def test_conversion_matches_statsmodels(counts):
    ka, na, kb, nb = counts
    a, b = np.arange(na) < ka, np.arange(nb) < kb
    actual = conversion(a, b)
    _, p = proportions_ztest([kb, ka], [nb, na])
    ci = confint_proportions_2indep(kb, nb, ka, na, method="newcomb")
    assert actual.p_value == pytest.approx(p)
    assert (actual.ci_low, actual.ci_high) == pytest.approx(ci)


@pytest.mark.parametrize("value", [0, 1])
def test_binary_boundaries_have_nonzero_interval(value):
    r = conversion([value] * 20, [value] * 20)
    assert r.p_value == 1
    assert r.ci_low < 0 < r.ci_high
    assert r.warnings


def test_symmetry_and_scaling():
    a, b = [1, 2, 3, 8], [2, 4, 6, 9, 10]
    forward, reverse = welch(a, b), welch(b, a)
    scaled = welch(np.array(a) * 7 + 9, np.array(b) * 7 + 9)
    assert reverse.effect == -forward.effect
    assert reverse.ci_low == pytest.approx(-forward.ci_high)
    assert scaled.p_value == pytest.approx(forward.p_value)
    assert scaled.effect == pytest.approx(7 * forward.effect)


@pytest.mark.parametrize("bad", [[], [1], [1, np.nan], [1, np.inf], [[1, 2], [3, 4]]])
def test_bad_samples_fail(bad):
    with pytest.raises(ValueError):
        welch(bad, [1, 2, 3])


@pytest.mark.parametrize("alpha", [0, 1, -1, np.nan])
def test_bad_alpha_fails(alpha):
    with pytest.raises(ValueError):
        conversion([0, 1], [1, 0], alpha)


def test_degenerate_continuous_fails():
    with pytest.raises(ValueError):
        welch([1, 1], [2, 2])
    with pytest.raises(ValueError):
        conversion([0, 0.5], [0, 1])


def test_wilson_known_and_invalid():
    assert wilson(5, 10) == pytest.approx((0.236593090512564, 0.763406909487436))
    assert wilson(0, 100)[1] > 0
    for args in [(11, 10), (0, 0), (1.2, 3), (True, 3)]:
        with pytest.raises(ValueError):
            wilson(*args)
