"""Equal-allocation, independent-unit, two-sided fixed-horizon planning."""

import numpy as np
from scipy import optimize, stats

from .validation import integer, probability


def mean_power(n, effect, sigma, alpha=0.05):
    """Noncentral t power for equal population SDs; n is per arm.

    This is pooled-t planning, an approximation for subsequent Welch inference.
    """
    n = integer(n, "n", 2)
    alpha = probability(alpha)
    if not np.isfinite([effect, sigma]).all() or sigma <= 0:
        raise ValueError("effect must be finite and sigma finite and positive")
    df, nc = 2 * n - 2, abs(effect) / sigma * np.sqrt(n / 2)
    cutoff = stats.t.ppf(1 - alpha / 2, df)
    # Symmetry avoids numerical NaNs in the far negative CDF tail.
    power = stats.nct.sf(cutoff, df, nc) + stats.nct.sf(cutoff, df, -nc)
    if not np.isfinite(power):
        raise ValueError("Noncentral t power is numerically undefined for these inputs")
    return float(np.clip(power, 0, 1))


def proportion_power(n, baseline, treatment, alpha=0.05):
    """Normal approximation using null and alternative standard errors."""
    n = integer(n, "n", 2)
    p0, p1 = probability(baseline, "baseline"), probability(treatment, "treatment")
    alpha = probability(alpha)
    pbar = (p0 + p1) / 2
    null_se = np.sqrt(2 * pbar * (1 - pbar) / n)
    alt_se = np.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / n)
    cutoff = stats.norm.ppf(1 - alpha / 2) * null_se
    delta = p1 - p0
    return float(
        stats.norm.cdf((-cutoff - delta) / alt_se)
        + stats.norm.sf((cutoff - delta) / alt_se)
    )


def required_n(power_function, target=0.8, maximum=10_000_000):
    """Smallest integer n >= 2 meeting target for a monotone power function."""
    target = probability(target, "target power")
    maximum = integer(maximum, "maximum", 2)
    if power_function(maximum) < target:
        raise ValueError("Target power is unreachable within maximum n")
    low, high = 2, maximum
    while low < high:
        mid = (low + high) // 2
        if power_function(mid) >= target:
            high = mid
        else:
            low = mid + 1
    return low


def mean_sample_size(effect, sigma, target=0.8, alpha=0.05):
    if not np.isfinite(effect) or effect == 0:
        raise ValueError("A nonzero finite effect is required")
    return required_n(lambda n: mean_power(n, effect, sigma, alpha), target)


def conversion_sample_size(baseline, uplift, target=0.8, alpha=0.05):
    if not np.isfinite(uplift) or uplift == 0:
        raise ValueError("uplift must be a nonzero absolute probability difference")
    return required_n(
        lambda n: proportion_power(n, baseline, baseline + uplift, alpha), target
    )


def conversion_mde(n, baseline, target=0.8, alpha=0.05):
    """Positive absolute uplift; e.g. 0.02 means +2 percentage points."""
    n = integer(n, "n", 2)
    baseline = probability(baseline, "baseline")
    target = probability(target, "target power")
    alpha = probability(alpha)
    if target <= alpha:
        raise ValueError("target power must exceed alpha")
    upper = 1 - baseline - 1e-10

    def fun(d):
        return proportion_power(n, baseline, baseline + d, alpha) - target

    if fun(upper) < 0:
        raise ValueError("No feasible positive MDE at this sample size")
    return float(optimize.brentq(fun, 0, upper))
