"""Monte Carlo studies: independent replications and paired method comparisons."""

import numpy as np
import pandas as pd
from scipy import stats

from .cuped import Cuped
from .data import noise
from .inference import conversion, welch, wilson
from .multiplicity import adjust
from .resampling import bootstrap_mean, permutation_mean


def rate_row(study, scenario, method, event, flags, **details):
    k, trials = int(np.sum(flags)), len(flags)
    low, high = wilson(k, trials)
    return {
        "study": study,
        "scenario": scenario,
        "method": method,
        "event": event,
        "events": k,
        "replications": trials,
        "rate": k / trials,
        "mc_ci_low": low,
        "mc_ci_high": high,
        **details,
    }


def fixed_horizon(cfg, rng):
    rows = []
    n, reps, alpha = cfg["n_per_arm"], cfg["replications"], cfg["alpha"]
    for distribution in ["normal", "lognormal", "zero_inflated", "student_t3"]:
        for effect in [0.0, cfg["standardized_effect"]]:
            rejected, covered = [], []
            for _ in range(reps):
                a, b = noise(rng, n, distribution), noise(rng, n, distribution) + effect
                result = welch(a, b, alpha)
                rejected.append(result.p_value < alpha)
                covered.append(result.ci_low <= effect <= result.ci_high)
            scenario = f"{distribution}; effect={effect:g}"
            for event, flags in [
                ("type_I" if effect == 0 else "power", rejected),
                ("coverage", covered),
            ]:
                rows.append(
                    rate_row(
                        "fixed_horizon",
                        scenario,
                        "welch",
                        event,
                        flags,
                        n_per_arm=n,
                        true_effect=effect,
                    )
                )
    for p1 in [0.10, 0.12]:
        rejected, covered = [], []
        for _ in range(reps):
            result = conversion(rng.binomial(1, 0.10, n), rng.binomial(1, p1, n), alpha)
            rejected.append(result.p_value < alpha)
            covered.append(result.ci_low <= p1 - 0.10 <= result.ci_high)
        for event, flags in [
            ("type_I" if p1 == 0.10 else "power", rejected),
            ("coverage", covered),
        ]:
            rows.append(
                rate_row(
                    "conversion",
                    f"p0=.10; p1={p1:g}",
                    "z_newcombe",
                    event,
                    flags,
                    n_per_arm=n,
                    true_effect=p1 - 0.10,
                )
            )
    return rows


def resampling_study(cfg, rng):
    """Separate smaller run: percentile coverage and permutation size/power."""
    rows = []
    n, reps, alpha = cfg["resampling_n"], cfg["resampling_replications"], cfg["alpha"]
    for distribution in ["normal", "lognormal"]:
        for effect in [0.0, cfg["standardized_effect"]]:
            coverage, rejection, welch_coverage, welch_rejection = [], [], [], []
            for _ in range(reps):
                a, b = noise(rng, n, distribution), noise(rng, n, distribution) + effect
                bs = bootstrap_mean(
                    a,
                    b,
                    resamples=cfg["resamples"],
                    alpha=alpha,
                    seed=int(rng.integers(2**32)),
                )
                pt = permutation_mean(
                    a, b, resamples=cfg["resamples"], seed=int(rng.integers(2**32))
                )
                reference = welch(a, b, alpha)
                welch_coverage.append(reference.ci_low <= effect <= reference.ci_high)
                welch_rejection.append(reference.p_value < alpha)
                coverage.append(bs.ci_low <= effect <= bs.ci_high)
                rejection.append(pt["p_value"] < alpha)
            scenario = f"{distribution}; effect={effect:g}"
            for event, values in [
                ("coverage", welch_coverage),
                ("type_I" if effect == 0 else "power", welch_rejection),
            ]:
                rows.append(
                    rate_row(
                        "resampling",
                        scenario,
                        "welch",
                        event,
                        values,
                        n_per_arm=n,
                        true_effect=effect,
                    )
                )
            rows.append(
                rate_row(
                    "resampling",
                    scenario,
                    "percentile_bootstrap",
                    "coverage",
                    coverage,
                    n_per_arm=n,
                    true_effect=effect,
                )
            )
            rows.append(
                rate_row(
                    "resampling",
                    scenario,
                    "permutation",
                    "type_I" if effect == 0 else "power",
                    rejection,
                    n_per_arm=n,
                    true_effect=effect,
                )
            )
    return rows


def peeking_study(cfg, rng):
    """Same A/A realization used for fixed horizon and all interim looks."""
    alpha, reps = cfg["alpha"], cfg["replications"]
    looks = np.array(cfg["looks"])
    horizon = int(looks.max())
    flags = {
        name: [] for name in ["fixed_horizon", "naive_peeking", "bonferroni_looks"]
    }
    for _ in range(reps):
        a, b = rng.normal(size=horizon), rng.normal(size=horizon)
        p = np.array([welch(a[:n], b[:n], alpha).p_value for n in looks])
        flags["fixed_horizon"].append(p[-1] < alpha)
        flags["naive_peeking"].append(p.min() < alpha)
        flags["bonferroni_looks"].append(p.min() < alpha / len(looks))
    return [
        rate_row(
            "peeking",
            f"{len(looks)} planned looks",
            method,
            "type_I",
            values,
            n_per_arm=horizon,
            true_effect=0.0,
        )
        for method, values in flags.items()
    ]


def multiple_study(cfg, rng):
    """Independent null z-statistics; FWER, not per-comparison error."""
    flags = {name: [] for name in ["uncorrected", "holm", "bonferroni"]}
    for _ in range(cfg["replications"]):
        p = 2 * stats.norm.sf(np.abs(rng.normal(size=cfg["metrics"])))
        flags["uncorrected"].append(np.any(p < cfg["alpha"]))
        for method in ["holm", "bonferroni"]:
            flags[method].append(np.any(adjust(p, method) < cfg["alpha"]))
    return [
        rate_row(
            "multiplicity",
            f"{cfg['metrics']} independent null metrics",
            method,
            "FWER",
            values,
        )
        for method, values in flags.items()
    ]


def cuped_study(cfg, rng):
    rows, diagnostics = [], []
    n, reps, alpha = cfg["n_per_arm"], cfg["replications"], cfg["alpha"]
    for rho in [0.0, 0.4, 0.8]:
        # One independent calibration sample per rho, frozen for all replications.
        x_train = rng.normal(size=cfg["calibration_n"])
        y_train = rho * x_train + np.sqrt(1 - rho * rho) * rng.normal(size=x_train.size)
        model = Cuped.fit(x_train, y_train)
        for effect in [0.0, cfg["standardized_effect"]]:
            flags = {name: {"reject": [], "cover": []} for name in ["raw", "cuped"]}
            effects = {name: [] for name in flags}
            for _ in range(reps):
                xa, xb = rng.normal(size=n), rng.normal(size=n)
                ya = rho * xa + np.sqrt(1 - rho * rho) * rng.normal(size=n)
                yb = rho * xb + np.sqrt(1 - rho * rho) * rng.normal(size=n) + effect
                for method, (a, b) in {
                    "raw": (ya, yb),
                    "cuped": (model.transform(xa, ya), model.transform(xb, yb)),
                }.items():
                    result = welch(a, b, alpha)
                    flags[method]["reject"].append(result.p_value < alpha)
                    flags[method]["cover"].append(
                        result.ci_low <= effect <= result.ci_high
                    )
                    effects[method].append(result.effect)
            for method in flags:
                for event, key in [
                    ("type_I" if effect == 0 else "power", "reject"),
                    ("coverage", "cover"),
                ]:
                    rows.append(
                        rate_row(
                            "cuped",
                            f"rho={rho:g}; effect={effect:g}",
                            method,
                            event,
                            flags[method][key],
                            n_per_arm=n,
                            true_effect=effect,
                        )
                    )
            diagnostics.append(
                {
                    "rho": rho,
                    "true_effect": effect,
                    "theta": model.theta,
                    "raw_bias": np.mean(effects["raw"]) - effect,
                    "cuped_bias": np.mean(effects["cuped"]) - effect,
                    "empirical_variance_ratio": np.var(effects["cuped"], ddof=1)
                    / np.var(effects["raw"], ddof=1),
                    "conditional_variance_ratio": 1
                    + model.theta**2
                    - 2 * model.theta * rho,
                    "oracle_variance_ratio": 1 - rho * rho,
                }
            )
    return rows, diagnostics


def run_studies(cfg):
    seeds = np.random.SeedSequence(cfg["seed"]).spawn(5)
    rows = []
    for fun, seed in zip(
        [fixed_horizon, resampling_study, peeking_study, multiple_study], seeds[:4]
    ):
        print(f"Running {fun.__name__}...", flush=True)
        rows.extend(fun(cfg, np.random.default_rng(seed)))
    print("Running cuped_study...", flush=True)
    cuped_rows, diagnostics = cuped_study(cfg, np.random.default_rng(seeds[4]))
    rows.extend(cuped_rows)
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)
