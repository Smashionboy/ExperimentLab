"""BCa bootstrap: SciPy's bias-corrected and accelerated interval."""

import warnings

import numpy as np
from scipy import stats

from .inference import Estimate
from .validation import integer, probability, sample


def bca_mean(control, treatment, *, resamples=1999, alpha=0.05, seed=42):
    a, b = sample(control), sample(treatment)
    alpha = probability(alpha)
    integer(resamples, "resamples", 99)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", stats.DegenerateDataWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        result = stats.bootstrap(
            (a, b),
            lambda x, y, axis: y.mean(axis=axis) - x.mean(axis=axis),
            vectorized=True,
            paired=False,
            confidence_level=1 - alpha,
            n_resamples=resamples,
            batch=128,
            method="BCa",
            rng=np.random.default_rng(seed),
        )
    lo, hi = result.confidence_interval
    if not np.isfinite([lo, hi]).all():
        raise ValueError("BCa undefined for degenerate samples; no silent fallback")
    return Estimate(
        "bca_bootstrap",
        float(b.mean() - a.mean()),
        float(lo),
        float(hi),
        None,
        len(a),
        len(b),
    )
