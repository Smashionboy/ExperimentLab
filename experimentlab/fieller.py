"""Confidence sets by inversion of a paired normal t-test for a ratio of means."""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .validation import probability, sample


@dataclass(frozen=True)
class ConfidenceSet:
    kind: str
    intervals: tuple

    def contains(self, value):
        if not np.isfinite(value):
            raise ValueError("Confidence-set membership requires a finite ratio")
        return any(lo <= value <= hi for lo, hi in self.intervals)

    def to_dict(self):
        # JSON null endpoints denote unboundedness; never emit Infinity.
        return {
            "kind": self.kind,
            "intervals": [
                [
                    float(lo) if np.isfinite(lo) else None,
                    float(hi) if np.isfinite(hi) else None,
                ]
                for lo, hi in self.intervals
            ],
        }


def quadratic_set(a, b, c):
    """Solve a*r^2+b*r+c <= 0, including linear and disconnected cases."""
    coefficients = np.asarray([a, b, c], dtype=float)
    if not np.isfinite(coefficients).all():
        raise ValueError("Finite quadratic coefficients required")
    scale = np.max(np.abs(coefficients))
    if scale == 0:
        return ConfidenceSet("all_real", ((-np.inf, np.inf),))
    a, b, c = coefficients / scale
    if a == 0:
        if b == 0:
            return (
                ConfidenceSet("all_real", ((-np.inf, np.inf),))
                if c <= 0
                else ConfidenceSet("empty", ())
            )
        root = -c / b
        return ConfidenceSet(
            "half_line", ((-np.inf, root),) if b > 0 else ((root, np.inf),)
        )
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return (
            ConfidenceSet("all_real", ((-np.inf, np.inf),))
            if a < 0
            else ConfidenceSet("empty", ())
        )
    if discriminant == 0:
        return (
            ConfidenceSet("all_real", ((-np.inf, np.inf),))
            if a < 0
            else ConfidenceSet("point", ((-b / (2 * a), -b / (2 * a)),))
        )
    # Stable quadratic roots avoid cancellation when |b| is large.
    q = -0.5 * (b + np.copysign(np.sqrt(discriminant), b))
    lo, hi = sorted([q / a, c / q])
    if a > 0:
        return ConfidenceSet("bounded", ((lo, hi),))
    return ConfidenceSet("disconnected", ((-np.inf, lo), (hi, np.inf)))


def _scaled_pairs(numerator, denominator):
    x, y = sample(numerator, minimum=3), sample(denominator, minimum=3)
    if x.size != y.size:
        raise ValueError("Paired numerator and denominator need matching rows")
    # Joint positive rescaling leaves the ratio unchanged and avoids overflow
    # or underflow in sample variances when both variables change units.
    scale = max(np.max(np.abs(x)), np.max(np.abs(y)))
    if scale == 0:
        raise ValueError("All-zero pairs do not identify a ratio")
    return x / scale, y / scale


def fieller(numerator, denominator, alpha=0.05):
    """Exact coverage under iid nondegenerate bivariate normal observations.

    Target E[X]/E[Y], E[Y] != 0. Arbitrary nonnormal data require an
    asymptotic approximation. This is not the difference of two ratios.
    """
    x, y = _scaled_pairs(numerator, denominator)
    alpha = probability(alpha)
    covariance = np.cov(np.stack([x, y]), ddof=1) / len(x)
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError("Positive definite paired covariance required")
    vx, vy, cov = covariance[0, 0], covariance[1, 1], covariance[0, 1]
    mx, my = x.mean(), y.mean()
    critical = stats.t.ppf(1 - alpha / 2, len(x) - 1) ** 2
    return quadratic_set(
        my * my - critical * vy,
        -2 * (mx * my - critical * cov),
        mx * mx - critical * vx,
    )


def delta_ratio(numerator, denominator, alpha=0.05):
    """First-order normal delta interval for a single ratio, for comparison."""
    x, y = _scaled_pairs(numerator, denominator)
    if y.mean() == 0:
        raise ValueError(
            "Matching pairs and nonzero observed denominator mean required"
        )
    alpha = probability(alpha)
    ratio = x.mean() / y.mean()
    se = np.std((x - ratio * y) / y.mean(), ddof=1) / np.sqrt(len(y))
    margin = stats.norm.ppf(1 - alpha / 2) * se
    if not np.isfinite([ratio, margin]).all():
        raise ValueError("Delta ratio interval is numerically undefined")
    return ConfidenceSet("bounded", ((ratio - margin, ratio + margin),))
