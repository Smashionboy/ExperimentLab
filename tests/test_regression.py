import numpy as np
import pytest
import statsmodels.api as sm

from experimentlab.adjustment_data import frame_arrays, generate_trial
from experimentlab.regression import lin_adjusted, ols_hc3


@pytest.mark.parametrize("seed", [7, 101, 1805])
def test_hc3_matches_statsmodels_with_heteroskedastic_noise(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(180, 3))
    a = sm.add_constant(x)
    y = a @ np.array([1, 0.5, -2, 3]) + (1 + abs(x[:, 0])) * rng.normal(size=180)
    actual = ols_hc3(a, y)
    reference = sm.OLS(y, a).fit(cov_type="HC3", use_t=False)
    np.testing.assert_allclose(
        actual.coefficients, reference.params, rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        actual.covariance, reference.cov_params(), rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        actual.leverage, reference.get_influence().hat_matrix_diag, atol=1e-12
    )


def test_lin_interval_and_pvalue_match_interacted_statsmodels():
    y, w, x = frame_arrays(
        generate_trial(n=400, seed=52), "outcome", ["x1", "x2", "x3"]
    )
    z = x - x.mean(axis=0)
    design = np.column_stack([np.ones(len(y)), w, z, w[:, None] * z])
    ref = sm.OLS(y, design).fit(cov_type="HC3", use_t=False)
    actual, diag = lin_adjusted(y, w, x)
    assert actual.effect == pytest.approx(ref.params[1])
    assert actual.p_value == pytest.approx(ref.pvalues[1])
    np.testing.assert_allclose(
        [actual.ci_low, actual.ci_high], ref.conf_int()[1], atol=1e-12
    )
    assert diag["residual_df"] == 392


def test_lin_invariant_to_covariate_units_and_assignment_sign():
    y, w, x = frame_arrays(generate_trial(n=300), "outcome", ["x1", "x2", "x3"])
    a, _ = lin_adjusted(y, w, x)
    b, _ = lin_adjusted(y, w, x * np.array([1e6, -0.01, 2]) + 42)
    c, _ = lin_adjusted(y, 1 - w, x)
    np.testing.assert_allclose(
        [a.effect, a.ci_low, a.ci_high], [b.effect, b.ci_low, b.ci_high], atol=1e-11
    )
    assert c.effect == pytest.approx(-a.effect)
    assert c.ci_low == pytest.approx(-a.ci_high)
    assert c.p_value == pytest.approx(a.p_value)


@pytest.mark.parametrize("design", [np.ones((10, 2)), np.eye(4)])
def test_nonidentifiable_or_saturated_design_rejected(design):
    with pytest.raises(ValueError):
        ols_hc3(design, np.arange(len(design)))


def test_singleton_indicator_rejected_for_unit_leverage():
    a = np.column_stack([np.ones(10), np.arange(10) == 0])
    with pytest.raises(ValueError, match="Leverage"):
        ols_hc3(a, np.arange(10) ** 2)


def test_constant_feature_rejected():
    with pytest.raises(ValueError, match="Constant"):
        lin_adjusted(np.arange(20) ** 2, np.arange(20) % 2, np.ones((20, 1)))
