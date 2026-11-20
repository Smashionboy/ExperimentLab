"""Ratio of totals with independent user-level numerator/denominator pairs."""

import numpy as np
from scipy import stats

from .inference import Estimate
from .validation import probability, sample


def ratio_components(numerator, denominator):
    x, y = sample(numerator), sample(denominator)
    if len(x) != len(y) or (y < 0).any() or y.mean() <= 0:
        raise ValueError(
            "Paired arrays and nonnegative denominator with positive mean required"
        )
    ratio = x.mean() / y.mean()
    # Influence function retains within-user numerator/denominator covariance.
    influence = (x - ratio * y) / y.mean()
    variance = influence.var(ddof=1) / len(x)
    if not np.isfinite([ratio, variance]).all():
        raise ValueError("Ratio or its variance is nonfinite")
    return float(ratio), float(variance), len(x)


def ratio_delta(control_num, control_den, treatment_num, treatment_den, alpha=0.05):
    """Asymptotic normal interval, not Fieller; denominator must be stable."""
    alpha = probability(alpha)
    ra, va, na = ratio_components(control_num, control_den)
    rb, vb, nb = ratio_components(treatment_num, treatment_den)
    se = np.sqrt(va + vb)
    if se <= 0:
        raise ValueError("Positive standard error required for ratio inference")
    effect = rb - ra
    margin = stats.norm.ppf(1 - alpha / 2) * se
    return Estimate(
        "ratio_delta",
        effect,
        float(effect - margin),
        float(effect + margin),
        float(2 * stats.norm.sf(abs(effect / se))),
        na,
        nb,
        (
            "Asymptotic delta method; unstable or near-zero denominators invalidate approximation.",
        ),
    )
