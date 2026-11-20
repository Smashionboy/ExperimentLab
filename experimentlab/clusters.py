"""Cluster-randomized inference for the average cluster, with equal weights."""

import numpy as np
import pandas as pd

from .inference import Estimate, welch
from .validation import integer, probability


def cluster_means(frame, metric):
    if metric in {"user_id", "cluster_id", "group"}:
        raise ValueError("Metric cannot be an ID or assignment column")
    if not frame.columns.is_unique:
        raise ValueError("Column names must be unique")
    if "user_id" in frame and (
        frame.user_id.isna().any() or frame.user_id.duplicated().any()
    ):
        raise ValueError("user_id must be unique and nonmissing when provided")
    required = {"cluster_id", "group", metric}
    if not required.issubset(frame.columns):
        raise ValueError(f"Required columns: {sorted(required)}")
    if frame[["cluster_id", "group"]].isna().any().any():
        raise ValueError("Missing cluster or group")
    if set(frame.group) != {"control", "treatment"}:
        raise ValueError("Expected control and treatment")
    if (frame.groupby("cluster_id").group.nunique() != 1).any():
        raise ValueError("A cluster cannot belong to both arms")
    data = frame.copy()
    data[metric] = pd.to_numeric(data[metric], errors="raise")
    if not np.isfinite(data[metric]).all():
        raise ValueError("Metric must be finite")
    means = data.groupby(["group", "cluster_id"], sort=False)[metric].mean()
    a, b = (means.loc[g].to_numpy() for g in ["control", "treatment"])
    if min(len(a), len(b)) < 2:
        raise ValueError("At least two independent clusters per arm required")
    return a, b


def cluster_welch(frame, metric, alpha=0.05):
    """Welch on cluster means; sample sizes in result count clusters, not rows.

    With unequal cluster sizes this targets the equally weighted average cluster,
    not the average individual. Duplicate rows cannot be identified without IDs.
    """
    a, b = cluster_means(frame, metric)
    r = welch(a, b, alpha)
    warnings = [
        "Equal cluster weights; n counts clusters. Clusters must be independent."
    ]
    if min(len(a), len(b)) < 20:
        warnings.append("Few clusters: finite-sample coverage may be inaccurate.")
    return Estimate(
        "cluster_mean_welch",
        r.effect,
        r.ci_low,
        r.ci_high,
        r.p_value,
        r.n_control,
        r.n_treatment,
        tuple(warnings),
    )


def design_effect(cluster_size, icc):
    """1+(m-1)*ICC for equal sizes and exchangeable within-cluster correlation."""
    m = integer(cluster_size, "cluster_size")
    rho = probability(icc, "icc", inclusive=True)
    return 1 + (m - 1) * rho
