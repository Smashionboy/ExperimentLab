import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from experimentlab.sequential import normal_mixture


def test_mixture_equals_integral():
    d = np.array([0.2, -0.1, 1.0, 0.5])
    r = normal_mixture(d)
    value = quad(
        lambda lam: np.exp(lam * d.sum() - lam * lam * len(d))
        * norm.pdf(lam, scale=0.2),
        -5,
        5,
    )[0]
    assert np.exp(r["log_e"][-1]) == pytest.approx(value, rel=1e-9)


def test_confidence_sequence_inverts_e_process():
    d = np.random.default_rng(4).normal(size=100)
    r = normal_mixture(d)
    for i in [4, 20, 99]:
        boundary = r["cs_high"][i]
        test = normal_mixture(d[: i + 1], null=boundary)
        assert test["log_e"][-1] == pytest.approx(np.log(20))
    shifted = normal_mixture(d + 5, null=5)
    assert shifted["log_e"] == pytest.approx(r["log_e"])
    assert shifted["cs_low"] == pytest.approx(r["cs_low"] + 5)


@pytest.mark.parametrize(
    "kw", [{"variance": 0}, {"prior_scale": -1}, {"null": np.nan}, {"alpha": 1}]
)
def test_invalid_sequential_inputs(kw):
    with pytest.raises(ValueError):
        normal_mixture([1, 2], **kw)
