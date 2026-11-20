"""Effects are always treatment minus control; tests are two-sided."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

from .validation import binary, integer, probability, sample


@dataclass(frozen=True)
class Estimate:
    method: str
    effect: float
    ci_low: float
    ci_high: float
    p_value: float | None
    n_control: int
    n_treatment: int
    warnings: tuple[str, ...] = ()

    def to_dict(self):
        return asdict(self)


def welch(control, treatment, alpha=0.05):
    """Welch-Satterthwaite t interval and test for difference in means."""
    a, b = sample(control, "control"), sample(treatment, "treatment")
    alpha = probability(alpha)
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    se2 = va + vb
    if not np.isfinite(se2) or se2 <= 0:
        raise ValueError("Welch inference needs positive, finite standard error")
    df = se2**2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1))
    effect = float(b.mean() - a.mean())
    se = np.sqrt(se2)
    margin = stats.t.ppf(1 - alpha / 2, df) * se
    p = 2 * stats.t.sf(abs(effect / se), df)
    return Estimate(
        "welch",
        effect,
        float(effect - margin),
        float(effect + margin),
        float(p),
        a.size,
        b.size,
    )


def wilson(successes, trials, alpha=0.05):
    """Wilson score interval; also used for Monte Carlo event rates."""
    n = integer(trials, "trials")
    k = integer(successes, "successes", minimum=0)
    if k > n:
        raise ValueError("successes cannot exceed trials")
    alpha = probability(alpha)
    p, z = k / n, stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z / denom * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    # Preserve exact endpoints: roundoff must not exclude the observed 0 or 1.
    low = 0.0 if k == 0 else max(0.0, float(center - radius))
    high = 1.0 if k == n else min(1.0, float(center + radius))
    return low, high


def conversion(control, treatment, alpha=0.05):
    """Pooled z-test and Newcombe independent Wilson difference interval.

    The interval is not an inversion of the pooled z-test: decisions may differ
    near the threshold. Rare-event normal-approximation warnings are retained.
    """
    a, b = binary(control, "control"), binary(treatment, "treatment")
    alpha = probability(alpha)
    pa, pb = a.mean(), b.mean()
    pooled = (a.sum() + b.sum()) / (a.size + b.size)
    se = np.sqrt(pooled * (1 - pooled) * (1 / a.size + 1 / b.size))
    p = 1.0 if se == 0 else float(2 * stats.norm.sf(abs((pb - pa) / se)))
    la, ua = wilson(int(a.sum()), a.size, alpha)
    lb, ub = wilson(int(b.sum()), b.size, alpha)
    effect = float(pb - pa)
    low = effect - np.sqrt((pb - lb) ** 2 + (ua - pa) ** 2)
    high = effect + np.sqrt((ub - pb) ** 2 + (pa - la) ** 2)
    warnings = ()
    if (
        min(
            a.size * pooled,
            a.size * (1 - pooled),
            b.size * pooled,
            b.size * (1 - pooled),
        )
        < 5
    ):
        warnings = ("Sparse expected counts: pooled normal test may be unreliable.",)
    return Estimate(
        "conversion_z_newcombe",
        effect,
        float(low),
        float(high),
        p,
        a.size,
        b.size,
        warnings,
    )
