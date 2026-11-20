"""Pretreatment covariates and known-truth randomized trial generators."""

import numpy as np
import pandas as pd

from .validation import integer, probability


def generate_trial(n=800, effect=0.25, mechanism="nonlinear", seed=42, propensity=0.5):
    n = integer(n, "n", 20)
    p = probability(propensity, "propensity")
    if not np.isfinite(effect):
        raise ValueError("effect must be finite")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    if mechanism == "linear":
        baseline = 1.5 * x[:, 0] + x[:, 1] - 0.5 * x[:, 2]
    elif mechanism == "nonlinear":
        baseline = 2 * np.sin(1.5 * x[:, 0]) + 1.5 * (x[:, 1] ** 2 - 1) + 0.75 * x[:, 2]
    else:
        raise ValueError("mechanism must be linear or nonlinear")
    w = rng.binomial(1, p, n)
    # Heteroskedastic outcome noise. Constant effect aligns SATE and PATE.
    sigma = 0.75 + 0.35 * np.abs(x[:, 0])
    y = baseline + effect * w + sigma * rng.normal(size=n)
    return pd.DataFrame(
        {
            "user_id": [f"u{i:06d}" for i in range(n)],
            "group": np.where(w, "treatment", "control"),
            "outcome": y,
            "x1": x[:, 0],
            "x2": x[:, 1],
            "x3": x[:, 2],
        }
    )


def frame_arrays(frame, metric, covariates):
    if not frame.columns.is_unique:
        raise ValueError("Column names must be unique")
    if metric in {"user_id", "group"}:
        raise ValueError("Metric cannot be an ID or assignment column")
    if not covariates or len(set(covariates)) != len(covariates):
        raise ValueError("Provide unique covariate columns")
    if set(covariates) & {"user_id", "group", metric}:
        raise ValueError("Covariates cannot include ID, assignment or outcome columns")
    required = {"user_id", "group", metric, *covariates}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if frame.user_id.isna().any() or frame.user_id.duplicated().any():
        raise ValueError("One nonmissing unique user_id per row required")
    if frame.group.isna().any() or set(frame.group) != {"control", "treatment"}:
        raise ValueError("Expected control and treatment groups")
    y = pd.to_numeric(frame[metric], errors="raise").to_numpy(dtype=float)
    x = frame[covariates].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("Outcome and covariates must be finite with no missing values")
    return y, (frame.group == "treatment").to_numpy(dtype=int), x


def balance_table(treatment, covariates, names):
    rows = []
    for j, name in enumerate(names):
        a, b = covariates[treatment == 0, j], covariates[treatment == 1, j]
        denom = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        rows.append(
            {
                "covariate": name,
                "control_mean": float(a.mean()),
                "treatment_mean": float(b.mean()),
                "standardized_mean_difference": float((b.mean() - a.mean()) / denom)
                if denom > 0
                else None,
            }
        )
    return pd.DataFrame(rows)
