"""Normal-mixture e-process and confidence sequence for known Gaussian variance.

Observations are independent differences D_i = treatment_i - control_i.
Under H0, D_i ~ N(theta0, variance), or obey the corresponding conditional
sub-Gaussian MGF bound. Variance is a known bound, never an estimated sample SD.
"""

import numpy as np

from .validation import probability, sample


def normal_mixture(differences, *, variance=2.0, null=0.0, prior_scale=0.2, alpha=0.05):
    d = sample(differences, minimum=1)
    alpha = probability(alpha)
    if (
        not np.isfinite([variance, null, prior_scale]).all()
        or variance <= 0
        or prior_scale <= 0
    ):
        raise ValueError(
            "Positive finite variance and prior_scale; finite null required"
        )
    t = np.arange(1, len(d) + 1, dtype=float)
    total = np.cumsum(d)
    v = t * variance
    q = prior_scale**2
    denom = 1 + q * v
    log_e = -0.5 * np.log(denom) + q * (total - null * t) ** 2 / (2 * denom)
    radius = np.sqrt(denom * (2 * np.log(1 / alpha) + np.log(denom)) / q) / t
    return {
        "time": t,
        "mean": total / t,
        "log_e": log_e,
        "cs_low": total / t - radius,
        "cs_high": total / t + radius,
        "reject": log_e >= np.log(1 / alpha),
    }
