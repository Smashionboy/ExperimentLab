"""Paired Monte Carlo study of regression and cross-fitted RCT adjustment."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .adjustment_data import frame_arrays, generate_trial
from .crossfit import crossfit_aipw
from .evaluation import summarize
from .inference import welch
from .regression import lin_adjusted
from .reporting import COLORS, dump_json, write_manifest
from .validation import integer, probability

METHODS = ["welch", "lin_hc3", "aipw_ridge", "aipw_forest"]
LABELS = {
    "welch": "Welch",
    "lin_hc3": "Lin / HC3",
    "aipw_ridge": "AIPW / Ridge",
    "aipw_forest": "AIPW / Forest",
}


def validate_config(config):
    required = {
        "seed",
        "replications",
        "sample_sizes",
        "mechanisms",
        "effects",
        "propensity",
        "folds",
        "alpha",
    }
    if set(config) != required:
        raise ValueError(f"Config requires exactly {sorted(required)}")
    cfg = dict(config)
    cfg["replications"] = integer(cfg["replications"], "replications", 2)
    cfg["seed"] = integer(cfg["seed"], "seed", 0)
    cfg["folds"] = integer(cfg["folds"], "folds", 2)
    cfg["alpha"] = probability(cfg["alpha"])
    cfg["propensity"] = probability(cfg["propensity"], "propensity")
    for field in ["sample_sizes", "mechanisms", "effects"]:
        if (
            not isinstance(cfg[field], list)
            or not cfg[field]
            or len(set(cfg[field])) != len(cfg[field])
        ):
            raise ValueError(f"{field} must be a nonempty list without duplicates")
    cfg["sample_sizes"] = [integer(n, "n_total", 40) for n in cfg["sample_sizes"]]
    if set(cfg["mechanisms"]) - {"linear", "nonlinear"}:
        raise ValueError("Unknown mechanism")
    if not all(np.isfinite(t) for t in cfg["effects"]):
        raise ValueError("All effects must be finite")
    if cfg["folds"] > min(cfg["sample_sizes"]) // 2:
        raise ValueError("Too many folds for the sample size")
    return cfg


def simulate(config, progress=False):
    cfg = validate_config(config)
    rows = []
    cells = list(
        itertools.product(cfg["mechanisms"], cfg["sample_sizes"], cfg["effects"])
    )
    # Cell identities (and therefore seeds) use the explicit ordered config.
    for cell, (mechanism, n, effect) in enumerate(cells):
        for rep in range(cfg["replications"]):
            seeds = np.random.SeedSequence([cfg["seed"], cell, rep]).generate_state(2)
            data_seed, split_seed = [int(v % (2**31 - 100)) for v in seeds]
            y, w, x = frame_arrays(
                generate_trial(n, effect, mechanism, data_seed, cfg["propensity"]),
                "outcome",
                ["x1", "x2", "x3"],
            )
            raw = welch(y[w == 0], y[w == 1], cfg["alpha"])
            raw_se = float(
                np.sqrt(
                    y[w == 0].var(ddof=1) / (w == 0).sum()
                    + y[w == 1].var(ddof=1) / (w == 1).sum()
                )
            )
            adjusted, diagnostic = lin_adjusted(y, w, x, cfg["alpha"])
            fits = [(raw, raw_se, None), (adjusted, diagnostic["standard_error"], None)]
            for learner in ["ridge", "forest"]:
                result = crossfit_aipw(
                    y,
                    w,
                    x,
                    propensity=cfg["propensity"],
                    learner=learner,
                    folds=cfg["folds"],
                    seed=split_seed,
                    alpha=cfg["alpha"],
                )
                fits.append(
                    (
                        result.estimate,
                        result.diagnostics["standard_error"],
                        result.diagnostics["observed_outcome_oof_rmse"],
                    )
                )
            for fit, se, prediction_rmse in fits:
                rows.append(
                    {
                        "mechanism": mechanism,
                        "n_total": n,
                        "true_effect": effect,
                        "replication": rep,
                        "data_seed": data_seed,
                        "split_seed": split_seed,
                        "method": fit.method,
                        "estimate": fit.effect,
                        "se": se,
                        "ci_low": fit.ci_low,
                        "ci_high": fit.ci_high,
                        "p_value": fit.p_value,
                        "n_control": fit.n_control,
                        "n_treatment": fit.n_treatment,
                        "prediction_rmse": prediction_rmse,
                    }
                )
            if progress and (rep + 1) % 50 == 0:
                print(
                    f"Cell {cell + 1}/{len(cells)}: {mechanism}, n={n}, effect={effect}, {rep + 1} replications",
                    flush=True,
                )
    return pd.DataFrame(rows)


def write_figures(raw, summary, paired, cfg, output):
    from matplotlib import pyplot as plt

    directory = Path(output) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    mechanisms = cfg["mechanisms"]
    effect = max(cfg["effects"])
    for field, name, title, ylabel in [
        (
            "rejection",
            "power",
            f"Rejection rate at true effect {effect:g}",
            "Rejection probability",
        ),
        (
            "rmse",
            "rmse",
            f"Estimation error at true effect {effect:g}",
            "RMSE of treatment effect",
        ),
    ]:
        fig, axes = plt.subplots(
            1, len(mechanisms), figsize=(6 * len(mechanisms), 4.5), squeeze=False
        )
        for ax, mechanism in zip(axes.flat, mechanisms):
            for method, color in zip(METHODS, COLORS):
                subset = summary[
                    (summary.mechanism == mechanism)
                    & (summary.method == method)
                    & (summary.true_effect == effect)
                ].sort_values("n_total")
                errors = (
                    None
                    if field == "rmse"
                    else np.vstack(
                        [
                            subset[field] - subset[f"{field}_mc_low"],
                            subset[f"{field}_mc_high"] - subset[field],
                        ]
                    )
                )
                ax.errorbar(
                    subset.n_total,
                    subset[field],
                    yerr=errors,
                    marker="o",
                    capsize=3,
                    color=color,
                    label=LABELS[method],
                )
            ax.set(
                title=mechanism.capitalize(),
                xlabel="Total randomized users",
                ylabel=ylabel,
                xticks=cfg["sample_sizes"],
            )
            ax.set_ylim(bottom=0, top=1.02 if field == "rejection" else None)
            ax.grid(alpha=0.2)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
        fig.suptitle(
            title + ("; bars = 95% Wilson MC intervals" if field == "rejection" else "")
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.94))
        fig.savefig(directory / f"{name}.png", dpi=150)
        plt.close(fig)
    cells = list(itertools.product(mechanisms, cfg["effects"]))
    for field in ["coverage", "paired_mse"]:
        fig, axes = plt.subplots(
            len(mechanisms),
            len(cfg["effects"]),
            figsize=(6 * len(cfg["effects"]), 3.8 * len(mechanisms)),
            squeeze=False,
        )
        for ax, (mechanism, delta) in zip(axes.flat, cells):
            source = summary if field == "coverage" else paired
            for method, color in zip(METHODS, COLORS):
                if field == "paired_mse" and method == "welch":
                    continue
                s = source[
                    (source.mechanism == mechanism)
                    & (source.method == method)
                    & (source.true_effect == delta)
                ].sort_values("n_total")
                center = s.coverage if field == "coverage" else s.paired_mse_difference
                lo, hi = (
                    (s.coverage_mc_low, s.coverage_mc_high)
                    if field == "coverage"
                    else (s.mc_low, s.mc_high)
                )
                ax.errorbar(
                    s.n_total,
                    center,
                    yerr=np.vstack([center - lo, hi - center]),
                    marker="o",
                    capsize=3,
                    label=LABELS[method],
                    color=color,
                )
            ax.axhline(
                1 - cfg["alpha"] if field == "coverage" else 0,
                linestyle="--",
                color="gray",
                lw=1,
            )
            ax.set(
                title=f"{mechanism.capitalize()}, true effect = {delta:g}",
                xlabel="Total randomized users",
                xticks=cfg["sample_sizes"],
                ylabel="Interval coverage"
                if field == "coverage"
                else "MSE(method) - MSE(Welch)",
            )
            ax.grid(alpha=0.2)
            if field == "coverage":
                lower = source.coverage_mc_low.min()
                ax.set_ylim(max(0, min(0.85, lower - 0.02)), 1.005)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
        fig.suptitle(
            "Coverage; bars = 95% Wilson Monte Carlo intervals"
            if field == "coverage"
            else "Paired MSE differences; approximate 95% MC t intervals; negative favors adjustment"
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        fig.savefig(directory / f"{field}.png", dpi=150)
        plt.close(fig)
    chosen_mechanism = "nonlinear" if "nonlinear" in mechanisms else mechanisms[0]
    s = raw[
        (raw.mechanism == chosen_mechanism)
        & (raw.n_total == max(cfg["sample_sizes"]))
        & (raw.true_effect == effect)
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(s.estimate.min(), s.estimate.max(), 28)
    for method, color in zip(METHODS, COLORS):
        ax.hist(
            s.loc[s.method == method, "estimate"],
            bins=bins,
            histtype="step",
            linewidth=1.8,
            density=True,
            color=color,
            label=LABELS[method],
        )
    ax.axvline(effect, linestyle="--", color="black", label="True effect")
    ax.set(
        title=f"Sampling distributions: {chosen_mechanism}, n={max(cfg['sample_sizes'])}, R={cfg['replications']}",
        xlabel="Estimated treatment effect",
        ylabel="Density (shared histogram bins)",
    )
    ax.legend()
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(directory / "sampling_distributions.png", dpi=150)
    plt.close(fig)


def run(config, output, progress=False):
    cfg = validate_config(config)
    frame = simulate(cfg, progress)
    summary, paired = summarize(frame, cfg["alpha"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "replications.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    paired.to_csv(output / "paired_comparisons.csv", index=False)
    dump_json(output / "config.json", cfg)
    write_figures(frame, summary, paired, cfg, output)
    artifacts = [
        output / name
        for name in [
            "replications.csv",
            "summary.csv",
            "paired_comparisons.csv",
            "config.json",
        ]
    ]
    artifacts += sorted((output / "figures").glob("*.png"))
    write_manifest(
        output,
        artifacts,
        {
            "data_origin": "synthetic randomized experiments; constant treatment effect",
            "independent_trials": int(len(frame) // len(METHODS)),
            "estimator_results": len(frame),
            "failed_replications": 0,
            "failure_policy": "Fail-fast; no dropping/retrying failures",
            "n_definition": "total users; Bernoulli assignment yields random arm counts",
            "comparisons": "Methods paired within trials; cells independent. MC intervals are pointwise, not simultaneous.",
            "learner_settings": {
                "ridge": {"alpha": 1.0, "scaling": "train-fold only"},
                "forest": {
                    "n_estimators": 40,
                    "max_depth": 6,
                    "min_samples_leaf": 8,
                    "max_features": 1.0,
                    "n_jobs": 1,
                },
            },
        },
    )
    return summary, paired


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/adjustment_full.json")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/adjustment"))
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        run(config, args.output, progress=True)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
