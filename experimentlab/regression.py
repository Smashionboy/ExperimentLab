"""OLS/HC3 and Lin-style covariate adjustment for randomized experiments.

Inference is asymptotic and conditional on pretreatment covariates. The
fully interacted coefficient targets the sample-average treatment contrast;
the accompanying simulation uses constant effects to align sample/population ATE.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .inference import Estimate
from .validation import binary, probability, sample


@dataclass(frozen=True)
class OLSFit:
    coefficients: np.ndarray
    covariance: np.ndarray
    residuals: np.ndarray
    leverage: np.ndarray
    rank: int
    condition_number: float


def covariate_arrays(outcome, treatment, covariates):
    y = sample(outcome, "outcome", minimum=6)
    w = binary(treatment, "treatment")
    x = np.asarray(covariates, dtype=float)
    if x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("Covariates must be a nonempty finite 2D matrix")
    if x.shape[0] != y.size or w.size != y.size:
        raise ValueError("Outcome, treatment and covariates need matching rows")
    if min(np.sum(w == 0), np.sum(w == 1)) < 3:
        raise ValueError("At least three users per arm are required")
    return y, w, x


def ols_hc3(design, outcome):
    """SVD least squares and HC3 sandwich without inverting X'X.

    Caller supplies an intercept when required. Rank-deficient designs are
    rejected instead of silently changing the parameter being estimated.
    """
    y = sample(outcome, "outcome")
    a = np.asarray(design, dtype=float)
    if a.ndim != 2 or a.shape[0] != len(y) or not np.isfinite(a).all():
        raise ValueError("Design must be a finite 2D matrix matching outcome")
    n, k = a.shape
    if k == 0 or n <= k:
        raise ValueError("Positive residual degrees of freedom required")
    u, singular, vt = np.linalg.svd(a, full_matrices=False)
    tolerance = max(a.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > tolerance))
    if rank != k:
        raise ValueError("Rank-deficient design; remove collinear covariates")
    pinv = (vt.T / singular) @ u.T
    coefficients = pinv @ y
    residuals = y - a @ coefficients
    leverage = np.sum(u * u, axis=1)
    if np.any(1 - leverage <= 1e-10):
        raise ValueError("Leverage too close to one for HC3 inference")
    weighted = pinv * (residuals / (1 - leverage))
    covariance = weighted @ weighted.T
    if not np.isfinite(covariance).all():
        raise ValueError("Nonfinite HC3 covariance")
    return OLSFit(
        coefficients,
        covariance,
        residuals,
        leverage,
        rank,
        float(singular[0] / singular[-1]),
    )


def lin_adjusted(outcome, treatment, covariates, alpha=0.05):
    """Regress Y on 1, W, centered/scaled X, and W*X; HC3 normal interval.

    Only pretreatment features are permitted. The API cannot establish their
    temporal provenance. Not a high-dimensional or cluster-robust procedure.
    """
    alpha = probability(alpha)
    y, w, x = covariate_arrays(outcome, treatment, covariates)
    scale = x.std(axis=0)
    if np.any(scale <= 0):
        raise ValueError("Constant covariates must be removed")
    z = (x - x.mean(axis=0)) / scale
    a = np.column_stack([np.ones(len(y)), w, z, w[:, None] * z])
    fit = ols_hc3(a, y)
    effect = float(fit.coefficients[1])
    se = float(np.sqrt(fit.covariance[1, 1]))
    if not np.isfinite(se) or se <= 0:
        raise ValueError("Positive standard error required")
    margin = stats.norm.ppf(1 - alpha / 2) * se
    warnings = [
        "HC3 normal approximation; independent users and pretreatment features required."
    ]
    if fit.leverage.max() > 0.2:
        warnings.append("High leverage (>0.2): inspect finite-sample sensitivity.")
    estimate = Estimate(
        "lin_hc3",
        effect,
        effect - margin,
        effect + margin,
        float(2 * stats.norm.sf(abs(effect / se))),
        int(np.sum(w == 0)),
        int(np.sum(w == 1)),
        tuple(warnings),
    )
    diagnostics = {
        "standard_error": se,
        "max_leverage": float(fit.leverage.max()),
        "design_columns": int(a.shape[1]),
        "condition_number": fit.condition_number,
        "residual_df": int(len(y) - fit.rank),
    }
    return estimate, diagnostics
