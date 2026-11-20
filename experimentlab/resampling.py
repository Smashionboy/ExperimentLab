"""Memory-bounded resampling with explicit random generators."""

import numpy as np
from scipy import stats

from .inference import Estimate
from .validation import integer, probability, sample


def bootstrap_mean(
    control, treatment, *, resamples=1999, alpha=0.05, seed=42, batch=128
):
    """Independent-arm percentile bootstrap; no p-value inferred from this CI."""
    a, b = sample(control), sample(treatment)
    alpha = probability(alpha)
    resamples = integer(resamples, "resamples", 99)
    batch = integer(batch, "batch")
    rng = np.random.default_rng(seed)
    differences = np.empty(resamples)
    for start in range(0, resamples, batch):
        size = min(batch, resamples - start)
        means_a = a[rng.integers(a.size, size=(size, a.size))].mean(axis=1)
        means_b = b[rng.integers(b.size, size=(size, b.size))].mean(axis=1)
        differences[start : start + size] = means_b - means_a
    lo, hi = np.quantile(differences, [alpha / 2, 1 - alpha / 2])
    return Estimate(
        "percentile_bootstrap",
        float(b.mean() - a.mean()),
        float(lo),
        float(hi),
        None,
        a.size,
        b.size,
    )


def permutation_mean(control, treatment, *, resamples=1999, seed=42):
    """SciPy random-label test. Requires exchangeability under the null.

    Tests equality of distributions (or a sharp randomization null), not merely
    equal means under unequal variances. No confidence interval is produced.
    """
    a, b = sample(control), sample(treatment)
    resamples = integer(resamples, "resamples", 99)
    result = stats.permutation_test(
        (b, a),
        lambda x, y, axis: x.mean(axis=axis) - y.mean(axis=axis),
        vectorized=True,
        permutation_type="independent",
        alternative="two-sided",
        n_resamples=resamples,
        batch=128,
        rng=np.random.default_rng(seed),
    )
    return {
        "method": "permutation_mean",
        "effect": float(b.mean() - a.mean()),
        "p_value": float(result.pvalue),
        "resamples_requested": resamples,
    }
