"""Reproducible weak-denominator experiment; python -m experimentlab.fieller_study."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .fieller import delta_ratio, fieller
from .inference import wilson
from .reporting import write_manifest
from .validation import integer


def run(output="reports/fieller", replications=2000, seed=1954):
    integer(replications, "replications", 2)
    rng = np.random.default_rng(seed)
    rows = []
    for n in [30, 100, 500]:
        for denominator in [0.05, 0.2, 1.0]:
            for rep in range(replications):
                x, y = rng.multivariate_normal(
                    [2 * denominator, denominator], [[1, 0.5], [0.5, 1]], size=n
                ).T
                for name, method in [("fieller", fieller), ("delta", delta_ratio)]:
                    result = method(x, y)
                    rows.append(
                        dict(
                            n=n,
                            denominator_mean=denominator,
                            replication=rep,
                            method=name,
                            covered=result.contains(2),
                            unbounded=result.kind not in ["bounded", "point", "empty"],
                            kind=result.kind,
                        )
                    )
    raw = pd.DataFrame(rows)
    summary = []
    for keys, group in raw.groupby(["n", "denominator_mean", "method"]):
        row = dict(zip(["n", "denominator_mean", "method"], keys))
        for event in ["covered", "unbounded"]:
            lo, hi = wilson(int(group[event].sum()), len(group))
            row.update(
                {event: group[event].mean(), event + "_low": lo, event + "_high": hi}
            )
        summary.append(row)
    summary = pd.DataFrame(summary)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "replications.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    from matplotlib import pyplot as plt

    for event, target in [("covered", 0.95), ("unbounded", None)]:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        for ax, n in zip(axes, [30, 100, 500]):
            for method in ["fieller", "delta"]:
                s = summary[(summary.n == n) & (summary.method == method)]
                ax.errorbar(
                    s.denominator_mean,
                    s[event],
                    yerr=np.maximum(
                        0, [s[event] - s[event + "_low"], s[event + "_high"] - s[event]]
                    ),
                    label=method,
                    marker="o",
                    capsize=3,
                )
            if target is not None:
                ax.axhline(target, color="gray", linestyle="--")
            ax.set(
                xscale="log",
                xlabel="True denominator mean",
                title=f"n = {n}",
                ylim=(-0.02, 1.02),
            )
            ax.grid(alpha=0.2)
        axes[0].set_ylabel(
            "Coverage" if event == "covered" else "Probability of an unbounded set"
        )
        axes[-1].legend()
        fig.suptitle("Paired Gaussian ratios; 95% Wilson Monte Carlo intervals")
        fig.tight_layout()
        fig.savefig(out / f"{event}.png", dpi=150)
        plt.close(fig)
    write_manifest(
        out,
        [
            out / name
            for name in [
                "replications.csv",
                "summary.csv",
                "covered.png",
                "unbounded.png",
            ]
        ],
        dict(
            seed=seed,
            replications_per_cell=replications,
            n_cells=9,
            true_ratio=2,
            data="synthetic iid bivariate Gaussian; covariance [[1,.5],[.5,1]]",
        ),
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/fieller")
    parser.add_argument("--replications", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1954)
    run(**vars(parser.parse_args()))
