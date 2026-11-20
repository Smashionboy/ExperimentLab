"""Family-wise error control; no independence assumption needed for Holm."""

import numpy as np


def adjust(pvalues, method="holm"):
    p = np.asarray(pvalues, dtype=float)
    if (
        p.ndim != 1
        or p.size == 0
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError("pvalues must be a nonempty 1D array in [0, 1]")
    if method == "bonferroni":
        return np.minimum(1, p * p.size)
    if method != "holm":
        raise ValueError("method must be holm or bonferroni")
    order = np.argsort(p)
    corrected = np.minimum(
        1, np.maximum.accumulate(p[order] * np.arange(p.size, 0, -1))
    )
    result = np.empty_like(p)
    result[order] = corrected
    return result
