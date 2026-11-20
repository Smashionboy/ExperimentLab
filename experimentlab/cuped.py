"""CUPED fitted on an independent historical/calibration sample only."""

from dataclasses import dataclass

import numpy as np

from .validation import sample


@dataclass(frozen=True)
class Cuped:
    theta: float
    center: float

    @classmethod
    def fit(cls, pre_period, outcome):
        x, y = sample(pre_period), sample(outcome)
        if x.size != y.size or x.var(ddof=1) <= 0:
            raise ValueError("Calibration arrays need matching lengths and varying X")
        theta = np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1)
        return cls(float(theta), float(x.mean()))

    def transform(self, pre_period, outcome):
        x, y = sample(pre_period), sample(outcome)
        if x.size != y.size:
            raise ValueError("Pre-period and outcome must be paired by unit")
        return y - self.theta * (x - self.center)
