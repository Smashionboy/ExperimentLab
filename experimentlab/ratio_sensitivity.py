"""Stress test ratio inference outside the Gaussian model."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import mean_mc_interval
from .fieller import delta_ratio, fieller
from .inference import wilson
from .reporting import write_manifest
from .validation import integer


def paired_sample(n, distribution, seed, denominator=0.2):
    integer(n, "n", 3)
    if not np.isfinite(denominator) or denominator == 0:
        raise ValueError("Finite nonzero denominator required")
    rng = np.random.default_rng(seed)
    if distribution == "normal":
        u, v = rng.normal(size=(2, n))
    elif distribution == "lognormal":
        u, v = (rng.lognormal(size=(2, n)) - np.exp(0.5)) / np.sqrt(np.e * (np.e - 1))
    elif distribution == "student_t3":
        u, v = rng.standard_t(3, size=(2, n)) / np.sqrt(3)
    else:
        raise ValueError("Unknown distribution")
    return 2 * denominator + u, denominator + 0.5 * u + np.sqrt(0.75) * v


def run(output="reports/ratio_sensitivity", replications=1000, seed=1955):
    integer(replications, "replications", 2)
    integer(seed, "seed", 0)
    rows = []
    for index, distribution in enumerate(["normal", "lognormal", "student_t3"]):
        for n in [100, 500]:
            for rep in range(replications):
                x, y = paired_sample(
                    n, distribution, np.random.SeedSequence([seed, index, n, rep])
                )
                for name, method in [("fieller", fieller), ("delta", delta_ratio)]:
                    result = method(x, y)
                    rows.append(
                        dict(
                            distribution=distribution,
                            n=n,
                            replication=rep,
                            method=name,
                            covered=result.contains(2),
                            unbounded=result.kind not in ["bounded", "point", "empty"],
                        )
                    )
    raw = pd.DataFrame(rows)
    summaries, paired = [], []
    for (distribution, n), cell in raw.groupby(["distribution", "n"]):
        wide = cell.pivot(
            index="replication", columns="method", values="covered"
        ).astype(int)
        mean, lo, hi = mean_mc_interval(wide.fieller - wide.delta)
        paired.append(
            dict(
                distribution=distribution,
                n=n,
                coverage_difference=mean,
                mc_low=lo,
                mc_high=hi,
            )
        )
        for method, g in cell.groupby("method"):
            row = dict(distribution=distribution, n=n, method=method)
            for event in ["covered", "unbounded"]:
                lo, hi = wilson(int(g[event].sum()), len(g))
                row.update(
                    {event: g[event].mean(), event + "_low": lo, event + "_high": hi}
                )
            summaries.append(row)
    summary = pd.DataFrame(summaries)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "replications.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "paired_coverage.csv", index=False)
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, event in zip(axes, ["covered", "unbounded"]):
        for offset, method in enumerate(["delta", "fieller"]):
            s = summary[summary.method == method].sort_values(["distribution", "n"])
            ax.errorbar(
                np.arange(len(s)) + (offset - 0.5) * 0.16,
                s[event],
                yerr=np.maximum(
                    0, [s[event] - s[event + "_low"], s[event + "_high"] - s[event]]
                ),
                fmt="o",
                capsize=3,
                label=method,
            )
        ax.set_xticks(
            np.arange(len(s)),
            [f"{r.distribution}\nn={r.n}" for r in s.itertuples()],
            rotation=25,
            ha="right",
        )
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("Coverage" if event == "covered" else "Unbounded confidence sets")
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
    axes[0].axhline(0.95, color="gray", linestyle="--")
    fig.suptitle("Same population means and covariance; 95% Wilson MC intervals")
    fig.tight_layout()
    fig.savefig(out / "sensitivity.png", dpi=150)
    plt.close(fig)
    write_manifest(
        out,
        [
            out / name
            for name in [
                "replications.csv",
                "summary.csv",
                "paired_coverage.csv",
                "sensitivity.png",
            ]
        ],
        dict(
            seed=seed,
            replications_per_cell=replications,
            cells=6,
            true_ratio=2,
            denominator_mean=0.2,
            data="synthetic iid paired observations",
        ),
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/ratio_sensitivity")
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1955)
    run(**vars(parser.parse_args()))
