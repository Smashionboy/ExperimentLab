"""Fail early instead of silently dropping invalid observations."""

import numbers

import numpy as np


def probability(value, name="alpha", inclusive=False):
    value = float(value)
    valid = 0 <= value <= 1 if inclusive else 0 < value < 1
    if not np.isfinite(value) or not valid:
        raise ValueError(f"{name} must be {'in [0, 1]' if inclusive else 'in (0, 1)'}")
    return value


def integer(value, name, minimum=1):
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def sample(values, name="sample", minimum=2):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < minimum or not np.isfinite(x).all():
        raise ValueError(
            f"{name} must be a finite 1D array with >= {minimum} observations"
        )
    return x


def binary(values, name="sample"):
    x = sample(values, name)
    if not np.isin(x, [0, 1]).all():
        raise ValueError(f"{name} must contain only 0 and 1")
    return x
