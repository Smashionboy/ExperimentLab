"""Monte Carlo accuracy and paired method comparisons, not business inference."""

import numpy as np
import pandas as pd
from scipy import stats

from .inference import wilson
from .validation import probability

CELL = ["mechanism", "n_total", "true_effect"]


def mean_mc_interval(values):
    """Approximate t interval for a Monte Carlo mean over independent repetitions."""
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("At least two finite Monte Carlo observations required")
    center = float(a.mean())
    margin = float(stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))
    return center, center - margin, center + margin


def summarize(frame, alpha=0.05):
    alpha = probability(alpha)
    required = {
        *CELL,
        "method",
        "replication",
        "estimate",
        "se",
        "ci_low",
        "ci_high",
        "p_value",
    }
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Missing or empty Monte Carlo results")
    if frame[[*CELL, "method", "replication"]].isna().any().any():
        raise ValueError("Missing simulation keys would silently drop replications")
    if frame.duplicated([*CELL, "method", "replication"]).any():
        raise ValueError("Duplicate method/replication within a simulation cell")
    numeric = frame[
        ["estimate", "se", "ci_low", "ci_high", "p_value", "true_effect"]
    ].to_numpy()
    if not np.isfinite(numeric).all() or (frame.se <= 0).any():
        raise ValueError("Nonfinite simulation result or nonpositive standard error")
    if (frame.ci_low > frame.ci_high).any() or not frame.p_value.between(0, 1).all():
        raise ValueError("Invalid confidence interval or p-value")
    summaries, paired = [], []
    for keys, group in frame.groupby([*CELL, "method"], sort=True):
        row = dict(zip([*CELL, "method"], keys))
        error = group.estimate.to_numpy() - row["true_effect"]
        bias, blo, bhi = mean_mc_interval(error)
        mse = float(np.mean(error**2))
        row.update(
            replications=len(group),
            bias=bias,
            bias_mc_low=blo,
            bias_mc_high=bhi,
            empirical_sd=float(group.estimate.std(ddof=1)),
            mean_se=float(group.se.mean()),
            mse=mse,
            rmse=float(np.sqrt(mse)),
            mean_width=float((group.ci_high - group.ci_low).mean()),
        )
        for name, event in [
            (
                "coverage",
                (group.ci_low <= row["true_effect"])
                & (group.ci_high >= row["true_effect"]),
            ),
            ("rejection", group.p_value < alpha),
        ]:
            low, high = wilson(int(event.sum()), len(event))
            row.update(
                {
                    name: float(event.mean()),
                    f"{name}_mc_low": low,
                    f"{name}_mc_high": high,
                }
            )
        summaries.append(row)
    for keys, group in frame.groupby(CELL, sort=True):
        wide = group.pivot(index="replication", columns="method", values="estimate")
        if "welch" not in wide or wide.isna().any().any():
            raise ValueError(
                "Paired comparison requires all methods on exactly the same replications"
            )
        errors = (wide - keys[2]) ** 2
        for method in wide.columns.drop("welch"):
            diff, low, high = mean_mc_interval(errors[method] - errors.welch)
            row = dict(zip(CELL, keys))
            row.update(
                method=method,
                reference="welch",
                replications=len(wide),
                paired_mse_difference=diff,
                mc_low=low,
                mc_high=high,
                mse_ratio=float(errors[method].mean() / errors.welch.mean())
                if errors.welch.mean() > 0
                else None,
            )
            paired.append(row)
    return pd.DataFrame(summaries), pd.DataFrame(paired)
