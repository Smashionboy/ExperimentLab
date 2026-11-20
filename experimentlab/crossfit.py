"""Cross-fitted AIPW for randomized trials with a known assignment probability.

This is NOT observational propensity estimation. Nuisance learners predict
potential outcomes; held-out outcomes never enter their own model's training.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .inference import Estimate
from .regression import covariate_arrays
from .validation import binary, integer, probability, sample


@dataclass(frozen=True)
class CrossFitResult:
    estimate: Estimate
    predictions_control: np.ndarray
    predictions_treatment: np.ndarray
    scores: np.ndarray
    fold_id: np.ndarray
    diagnostics: dict


def make_learner(kind, seed):
    if kind == "ridge":
        # Scaling is fitted exclusively on the training portion of one arm.
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if kind == "forest":
        return RandomForestRegressor(
            n_estimators=40,
            max_depth=6,
            min_samples_leaf=8,
            max_features=1.0,
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError("learner must be ridge or forest")


def aipw_scores(
    outcome, treatment, prediction_control, prediction_treatment, propensity=0.5
):
    y, w = sample(outcome), binary(treatment)
    m0, m1 = sample(prediction_control), sample(prediction_treatment)
    p = probability(propensity, "known propensity")
    if len({len(y), len(w), len(m0), len(m1)}) != 1:
        raise ValueError("AIPW arrays must have matching lengths")
    score = m1 - m0 + w * (y - m1) / p - (1 - w) * (y - m0) / (1 - p)
    if not np.isfinite(score).all():
        raise ValueError("AIPW scores overflowed")
    return score


def crossfit_aipw(
    outcome,
    treatment,
    covariates,
    *,
    propensity=0.5,
    learner="ridge",
    folds=2,
    seed=42,
    alpha=0.05,
):
    y, w, x = covariate_arrays(outcome, treatment, covariates)
    p = probability(propensity, "known propensity")
    alpha = probability(alpha)
    folds = integer(folds, "folds", 2)
    seed = integer(seed, "seed", 0)
    if folds > len(y) // 2:
        raise ValueError("Too many folds for this sample size")
    if seed + 2 * folds - 1 > 2**32 - 1:
        raise ValueError("seed plus fold offsets must fit in a 32-bit unsigned integer")
    # Fold assignment ignores W, X and Y; never retry splits based on results.
    split = KFold(n_splits=folds, shuffle=True, random_state=seed)
    m0, m1 = np.empty(len(y)), np.empty(len(y))
    fold_id = np.full(len(y), -1, dtype=int)
    audit = []
    for fold, (train, test) in enumerate(split.split(x)):
        fold_id[test] = fold
        counts = []
        for arm, dest in [(0, m0), (1, m1)]:
            indices = train[w[train] == arm]
            if len(indices) < 3:
                raise ValueError(
                    "Every training fold needs at least three users per arm"
                )
            model = make_learner(learner, seed + 2 * fold + arm)
            model.fit(x[indices], y[indices])
            prediction = model.predict(x[test])
            if not np.isfinite(prediction).all():
                raise ValueError("Nuisance model produced nonfinite predictions")
            dest[test] = prediction
            counts.append(int(len(indices)))
        audit.append(
            {
                "fold": fold,
                "validation_users": int(len(test)),
                "train_control": counts[0],
                "train_treatment": counts[1],
            }
        )
    scores = aipw_scores(y, w, m0, m1, p)
    effect, se = float(scores.mean()), float(scores.std(ddof=1) / np.sqrt(len(y)))
    if not np.isfinite(se) or se <= 0:
        raise ValueError("Positive standard error required")
    margin = stats.norm.ppf(1 - alpha / 2) * se
    warnings = [
        "Known randomized propensity; asymptotic influence-score interval, not finite-sample exact.",
        "Covariates must be pretreatment; cross-fitting cannot repair treatment leakage or confounding.",
    ]
    if min(p, 1 - p) < 0.1:
        warnings.append(
            "Small assignment probability: large inverse weights can destabilize inference."
        )
    estimate = Estimate(
        f"aipw_{learner}",
        effect,
        effect - margin,
        effect + margin,
        float(2 * stats.norm.sf(abs(effect / se))),
        int(np.sum(w == 0)),
        int(np.sum(w == 1)),
        tuple(warnings),
    )
    residual = y - np.where(w == 1, m1, m0)
    diagnostics = {
        "known_propensity": p,
        "learner": learner,
        "folds": folds,
        "seed": seed,
        "standard_error": se,
        "observed_outcome_oof_rmse": float(np.sqrt(np.mean(residual**2))),
        "oof_plugin_effect": float(np.mean(m1 - m0)),
        "fold_audit": audit,
    }
    return CrossFitResult(estimate, m0, m1, scores, fold_id, diagnostics)
