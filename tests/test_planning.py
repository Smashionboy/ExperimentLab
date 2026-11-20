import numpy as np
import pytest
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import power_proportions_2indep

from experimentlab.planning import (
    conversion_mde,
    conversion_sample_size,
    mean_power,
    mean_sample_size,
    proportion_power,
)


@pytest.mark.parametrize("n,effect", [(30, 0.2), (100, 0.5), (1000, 0.1), (50, 0)])
def test_mean_power_matches_noncentral_t(n, effect):
    expected = TTestIndPower().power(effect, n, alpha=0.05)
    assert mean_power(n, effect, 1) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize(
    "n,p0,p1", [(500, 0.1, 0.12), (200, 0.5, 0.4), (800, 0.3, 0.3)]
)
def test_conversion_power_reference(n, p0, p1):
    expected = power_proportions_2indep(
        p1 - p0, p0, n, ratio=1, alpha=0.05, return_results=False
    )
    assert proportion_power(n, p0, p1) == pytest.approx(expected)


def test_sample_sizes_are_minimal_and_increase_with_precision():
    n = mean_sample_size(0.2, 1)
    assert mean_power(n, 0.2, 1) >= 0.8 > mean_power(n - 1, 0.2, 1)
    assert mean_sample_size(0.1, 1) > n
    n = conversion_sample_size(0.1, 0.02)
    assert proportion_power(n, 0.1, 0.12) >= 0.8 > proportion_power(n - 1, 0.1, 0.12)
    d = conversion_mde(n, 0.1)
    assert proportion_power(n, 0.1, 0.1 + d) == pytest.approx(0.8)
    assert conversion_mde(n * 2, 0.1) < d


@pytest.mark.parametrize(
    "call",
    [
        lambda: mean_sample_size(0, 1),
        lambda: mean_power(2, 1, 0),
        lambda: conversion_sample_size(0.99, 0.1),
        lambda: conversion_mde(2, 0.99),
        lambda: mean_power(2, np.inf, 1),
        lambda: conversion_mde(100, 0.1, target=0.01),
    ],
)
def test_invalid_planning_fails(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize("n", [10000, 63988, 10000000])
def test_far_tail_power_is_finite(n):
    assert 0.999 < mean_power(n, 0.2, 1) <= 1
