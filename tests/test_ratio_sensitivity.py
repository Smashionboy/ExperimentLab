import numpy as np
import pytest

from experimentlab.ratio_sensitivity import paired_sample


@pytest.mark.parametrize("distribution", ["normal", "lognormal", "student_t3"])
def test_shift_and_reproducibility(distribution):
    x, y = paired_sample(100, distribution, 42, 0.2)
    a, b = paired_sample(100, distribution, 42, 1)
    np.testing.assert_allclose(a - x, 1.6)
    np.testing.assert_allclose(b - y, 0.8)
    c, d = paired_sample(100, distribution, 42, 0.2)
    np.testing.assert_array_equal(x, c)
    np.testing.assert_array_equal(y, d)


@pytest.mark.parametrize("distribution,denominator", [("unknown", 0.2), ("normal", 0)])
def test_invalid_settings(distribution, denominator):
    with pytest.raises(ValueError):
        paired_sample(100, distribution, 42, denominator)
