import numpy as np
import pytest
from scipy import stats

from experimentlab.fieller import delta_ratio, fieller, quadratic_set


@pytest.mark.parametrize(
    "abc,kind",
    [
        ((1, 0, -1), "bounded"),
        ((-1, 0, 1), "disconnected"),
        ((-1, 0, -1), "all_real"),
        ((1, 0, 1), "empty"),
        ((0, 1, 1), "half_line"),
        ((0, -1, 1), "half_line"),
        ((1, 2, 1), "point"),
        ((0, 0, 0), "all_real"),
    ],
)
def test_quadratic_geometry(abc, kind):
    result = quadratic_set(*abc)
    assert result.kind == kind
    for r in np.linspace(-4, 4, 81):
        assert result.contains(r) == (abc[0] * r * r + abc[1] * r + abc[2] <= 0)


@pytest.mark.parametrize("seed", range(6))
def test_set_membership_equals_direct_t_test(seed):
    rng = np.random.default_rng(seed)
    x, y = rng.multivariate_normal([0.1, 0.05], [[1, 0.4], [0.4, 1]], size=50).T
    result = fieller(x, y)
    for ratio in np.linspace(-20, 20, 201):
        z = x - ratio * y
        assert result.contains(ratio) == (stats.ttest_1samp(z, 0).pvalue >= 0.05)


def test_scale_and_pairing():
    rng = np.random.default_rng(42)
    x, y = rng.multivariate_normal([2, 1], [[1, 0.4], [0.4, 1]], size=100).T
    a, b = fieller(x, y), fieller(-10 * x, -10 * y)
    np.testing.assert_allclose(a.intervals, b.intervals)
    assert delta_ratio(x, y).kind == "bounded"
    assert fieller(x, y[::-1]).intervals != a.intervals


def test_degenerate_covariance_rejected():
    with pytest.raises(ValueError):
        fieller(np.arange(10), np.arange(10))
