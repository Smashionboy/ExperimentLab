"""Strict config validation prevents accidental partial or misleading studies."""

import json
from pathlib import Path

import numpy as np

from .validation import integer, probability


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "seed",
        "alpha",
        "replications",
        "n_per_arm",
        "standardized_effect",
        "resampling_replications",
        "resampling_n",
        "resamples",
        "looks",
        "metrics",
        "calibration_n",
    }
    if not isinstance(cfg, dict) or set(cfg) != required:
        raise ValueError(f"Config requires exactly: {sorted(required)}")
    integer(cfg["seed"], "seed", 0)
    probability(cfg["alpha"])
    for name in [
        "replications",
        "n_per_arm",
        "resampling_replications",
        "resampling_n",
        "metrics",
        "calibration_n",
    ]:
        integer(cfg[name], name, 2)
    integer(cfg["resamples"], "resamples", 99)
    if not np.isfinite(cfg["standardized_effect"]) or cfg["standardized_effect"] <= 0:
        raise ValueError("standardized_effect must be finite and positive")
    looks = cfg["looks"]
    if not isinstance(looks, list) or not looks:
        raise ValueError("looks must be a nonempty increasing integer list")
    for n in looks:
        integer(n, "look", 2)
    if any(a >= b for a, b in zip(looks, looks[1:])):
        raise ValueError("looks must be strictly increasing")
    return cfg
