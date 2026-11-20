"""Synthetic mechanisms with known population moments; no business data claims."""

import numpy as np
import pandas as pd

from .validation import integer, probability


def noise(rng, n, distribution):
    """Population mean zero and variance one for each mechanism."""
    if distribution == "normal":
        return rng.normal(size=n)
    if distribution == "lognormal":
        return (rng.lognormal(0, 1, n) - np.exp(0.5)) / np.sqrt((np.e - 1) * np.e)
    if distribution == "zero_inflated":
        # 20% exponential purchases: E[Y]=.2, Var[Y]=.4-.04=.36.
        return (rng.binomial(1, 0.2, n) * rng.exponential(size=n) - 0.2) / 0.6
    if distribution == "student_t3":
        return rng.standard_t(3, n) / np.sqrt(3)
    raise ValueError(f"Unknown distribution: {distribution}")


def demo_data(n=5000, seed=42):
    """Illustrative randomized product experiment; one row per user.

    Conversion: .10 vs .12. Monetary outcome is zero-inflated lognormal.
    Pre-period score is generated before treatment; revenue correlation comes
    from a shared latent propensity. Not intended as an industry benchmark.
    """
    n = integer(n, "n", 2)
    rng = np.random.default_rng(seed)
    frames = []
    for arm, p in [("control", 0.10), ("treatment", 0.12)]:
        latent = rng.normal(size=n)
        converted = rng.binomial(1, p, n)
        frames.append(
            pd.DataFrame(
                {
                    "user_id": [f"{arm}_{i}" for i in range(n)],
                    "group": arm,
                    "converted": converted,
                    "revenue": converted
                    * np.exp(3 + 0.5 * latent + rng.normal(0, 0.5, n)),
                    "pre_score": latent + rng.normal(0, 0.3, n),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def validate_frame(frame, metric, binary_metric=False):
    if metric in {"user_id", "group"}:
        raise ValueError("Metric cannot be an ID or assignment column")
    if not frame.columns.is_unique:
        raise ValueError("Column names must be unique")
    required = {"user_id", "group", metric}
    if not required.issubset(frame.columns):
        raise ValueError(f"Required columns: {sorted(required)}")
    if frame["user_id"].isna().any() or frame["user_id"].duplicated().any():
        raise ValueError("Each user_id must occur exactly once and cannot be missing")
    if (
        set(frame["group"].dropna()) != {"control", "treatment"}
        or frame["group"].isna().any()
    ):
        raise ValueError("group must contain exactly control and treatment")
    values = pd.to_numeric(frame[metric], errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Metric contains missing or infinite values")
    if binary_metric and not values.isin([0, 1]).all():
        raise ValueError("Binary metric must contain only 0 and 1")
    if frame.groupby("group").size().min() < 2:
        raise ValueError("Each arm needs at least two users")
    return tuple(
        values[frame["group"] == arm].to_numpy(dtype=float)
        for arm in ["control", "treatment"]
    )


def allocation_check(n_control, n_treatment, expected_treatment=0.5):
    """SRM diagnostic, not proof of randomization or absence of bias."""
    from scipy.stats import chisquare

    integer(n_control, "n_control")
    integer(n_treatment, "n_treatment")
    p = probability(expected_treatment, "expected_treatment")
    total = n_control + n_treatment
    return float(
        chisquare([n_control, n_treatment], [total * (1 - p), total * p]).pvalue
    )
