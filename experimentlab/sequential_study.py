"""Anytime validity experiment with known Gaussian variance, paired paths."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .reporting import dump_json, rate_plot, write_manifest
from .sequential import normal_mixture
from .studies import rate_row
from .validation import integer


def run(output="reports/sequential", replications=2000, horizon=1000, seed=180650):
    integer(replications, "replications", 2)
    integer(horizon, "horizon", 20)
    integer(seed, "seed", 0)
    rng = np.random.default_rng(seed)
    rows = []
    for effect in [0.0, 0.2]:
        flags = {k: [] for k in ["fixed_z", "naive_z_peeking", "mixture_e"]}
        covered = []
        stop_times = []
        for _ in range(replications):
            d = rng.normal(effect, np.sqrt(2), horizon)
            r = normal_mixture(d)
            z = np.abs(r["mean"]) * np.sqrt(r["time"] / 2)
            flags["fixed_z"].append(z[-1] > stats.norm.ppf(0.975))
            flags["naive_z_peeking"].append(np.any(z[19:] > stats.norm.ppf(0.975)))
            flags["mixture_e"].append(np.any(r["reject"][19:]))
            covered.append(np.all((r["cs_low"] <= effect) & (effect <= r["cs_high"])))
            crossed = np.flatnonzero(r["reject"][19:])
            stop_times.append(int(crossed[0] + 20) if len(crossed) else horizon)
        for method, values in flags.items():
            rows.append(
                rate_row(
                    "sequential",
                    f"effect={effect}; horizon={horizon}",
                    method,
                    "type_I" if effect == 0 else "power_by_horizon",
                    values,
                    n_per_arm=horizon,
                )
            )
        rows.append(
            rate_row(
                "sequential",
                f"effect={effect}; horizon={horizon}",
                "mixture_cs",
                "simultaneous_coverage",
                covered,
                n_per_arm=horizon,
            )
        )
        rows[-1]["mean_capped_stopping_time"] = float(np.mean(stop_times))
    out = Path(output)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "rates.csv", index=False)
    rate_plot(
        frame[frame.event == "type_I"],
        "Known variance: repeated monitoring",
        0.05,
        out / "figures/error.png",
    )
    import matplotlib.pyplot as plt

    path = normal_mixture(
        np.random.default_rng(seed + 1).normal(0.2, np.sqrt(2), horizon)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    t = path["time"][19:]
    ax.fill_between(
        t,
        path["cs_low"][19:],
        path["cs_high"][19:],
        alpha=0.2,
        label="95% confidence sequence",
    )
    ax.plot(t, path["mean"][19:], label="Running mean")
    ax.axhline(0.2, color="black", linestyle="--", label="True effect")
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set(
        xlabel="Pairs observed",
        ylabel="Treatment minus control",
        title="One illustrative path; no claim of typical stopping time",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures/confidence_sequence.png", dpi=150)
    plt.close(fig)
    dump_json(
        out / "config.json",
        dict(
            replications=replications,
            horizon=horizon,
            seed=seed,
            variance=2.0,
            prior_scale=0.2,
            alpha=0.05,
            first_look=20,
            data="synthetic Gaussian",
        ),
    )
    write_manifest(
        out,
        [
            out / "rates.csv",
            out / "config.json",
            *sorted((out / "figures").glob("*.png")),
        ],
        {"study": "sequential"},
    )
    print(frame.to_string(index=False))
    return frame


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="reports/sequential")
    p.add_argument("--replications", type=int, default=2000)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--seed", type=int, default=180650)
    run(**vars(p.parse_args()))
