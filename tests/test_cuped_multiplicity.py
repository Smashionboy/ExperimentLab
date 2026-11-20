import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from experimentlab.cuped import Cuped
from experimentlab.multiplicity import adjust


@pytest.mark.parametrize("method,ref", [("holm", "holm"), ("bonferroni", "bonferroni")])
def test_adjustment_matches_statsmodels(method, ref):
    p = [0.03, 0, 0.003, 1, 0.2, 0.03]
    assert adjust(p, method) == pytest.approx(multipletests(p, method=ref)[1])


@pytest.mark.parametrize("p", [[], [-0.1], [1.1], [np.nan], [[0.1, 0.2]]])
def test_bad_pvalues(p):
    with pytest.raises(ValueError):
        adjust(p)


def test_cuped_fit_and_population_variance_reduction():
    rng = np.random.default_rng(36)
    x = rng.normal(size=50000)
    y = 0.8 * x + 0.6 * rng.normal(size=x.size)
    model = Cuped.fit(x, y)
    # Independent evaluation data: not evaluating on the fitting sample.
    xe = rng.normal(size=50000)
    ye = 0.8 * xe + 0.6 * rng.normal(size=xe.size)
    adjusted = model.transform(xe, ye)
    assert model.theta == pytest.approx(0.8, abs=0.02)
    assert adjusted.var() / ye.var() == pytest.approx(0.36, abs=0.02)
    assert (model.transform(xe, ye + 3) - adjusted) == pytest.approx(
        np.full(xe.size, 3)
    )


def test_bad_cuped_inputs():
    with pytest.raises(ValueError):
        Cuped.fit([1, 1], [2, 3])
    with pytest.raises(ValueError):
        Cuped.fit([1, 2], [2, 3, 4])
    with pytest.raises(ValueError):
        Cuped(1, 0).transform([1, 2], [1, 2, 3])
