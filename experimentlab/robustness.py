"""Additional studies: dependence, ratio estimands, and bootstrap coverage."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .advanced_resampling import bca_mean
from .clusters import design_effect
from .data import noise
from .inference import welch
from .ratio import ratio_delta
from .reporting import dump_json, rate_plot, write_manifest
from .resampling import bootstrap_mean
from .studies import rate_row
from .validation import integer


def cluster_study(reps, rng):
    rows = []
    clusters, size = 40, 20
    for icc in [0.0, 0.1, 0.3]:
        flags = {m: [] for m in ["naive_rows", "cluster_means"]}
        for _ in range(reps):
            a = np.sqrt(icc) * rng.normal(size=(clusters, 1)) + np.sqrt(
                1 - icc
            ) * rng.normal(size=(clusters, size))
            b = np.sqrt(icc) * rng.normal(size=(clusters, 1)) + np.sqrt(
                1 - icc
            ) * rng.normal(size=(clusters, size))
            flags["naive_rows"].append(welch(a.ravel(), b.ravel()).p_value < 0.05)
            flags["cluster_means"].append(
                welch(a.mean(axis=1), b.mean(axis=1)).p_value < 0.05
            )
        for method, values in flags.items():
            rows.append(
                rate_row(
                    "clustering",
                    f"ICC={icc}; 40 clusters x 20 rows",
                    method,
                    "type_I",
                    values,
                    n_per_arm=clusters * size,
                    design_effect=design_effect(size, icc),
                )
            )
    return rows


def ratio_study(reps, rng):
    rows = []
    n = 300
    for effect in [0.0, 0.3]:
        flags = {
            m: {"reject": [], "cover": []} for m in ["user_delta", "naive_poisson"]
        }
        for _ in range(reps):
            da, db = rng.poisson(4, n) + 1, rng.poisson(4, n) + 1
            # User-level heterogeneity: within-user events are not iid Poisson.
            xa = rng.poisson(da * 2 * rng.lognormal(-0.32, 0.8, n))
            xb = rng.poisson(db * (2 + effect) * rng.lognormal(-0.32, 0.8, n))
            r = ratio_delta(xa, da, xb, db)
            flags["user_delta"]["reject"].append(r.p_value < 0.05)
            flags["user_delta"]["cover"].append(r.ci_low <= effect <= r.ci_high)
            delta = xb.sum() / db.sum() - xa.sum() / da.sum()
            se = np.sqrt(xa.sum() / da.sum() ** 2 + xb.sum() / db.sum() ** 2)
            flags["naive_poisson"]["reject"].append(
                abs(delta / se) > stats.norm.ppf(0.975)
            )
            flags["naive_poisson"]["cover"].append(
                abs(delta - effect) <= stats.norm.ppf(0.975) * se
            )
        for method, events in flags.items():
            for event, key in [
                ("type_I" if effect == 0 else "power", "reject"),
                ("coverage", "cover"),
            ]:
                rows.append(
                    rate_row(
                        "ratio",
                        f"ratio effect={effect}; user heterogeneity",
                        method,
                        event,
                        events[key],
                        n_per_arm=n,
                    )
                )
    return rows


def bootstrap_study(reps, resamples, rng):
    rows = []
    for distribution in ["normal", "lognormal"]:
        flags = {m: [] for m in ["welch", "percentile", "BCa"]}
        for _ in range(reps):
            a, b = noise(rng, 80, distribution), noise(rng, 80, distribution)
            outputs = {
                "welch": welch(a, b),
                "percentile": bootstrap_mean(
                    a, b, resamples=resamples, seed=int(rng.integers(2**32))
                ),
                "BCa": bca_mean(
                    a, b, resamples=resamples, seed=int(rng.integers(2**32))
                ),
            }
            for method, r in outputs.items():
                flags[method].append(r.ci_low <= 0 <= r.ci_high)
        for method, values in flags.items():
            rows.append(
                rate_row(
                    "bootstrap_comparison",
                    distribution,
                    method,
                    "coverage",
                    values,
                    n_per_arm=80,
                )
            )
    return rows


def run(
    output, replications=2000, bootstrap_replications=500, resamples=1999, seed=650
):
    integer(replications, "replications", 2)
    integer(bootstrap_replications, "bootstrap_replications", 2)
    integer(resamples, "resamples", 99)
    integer(seed, "seed", 0)
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    print("Cluster randomization...", flush=True)
    rows = cluster_study(replications, streams[0])
    print("User-level ratios...", flush=True)
    rows += ratio_study(replications, streams[1])
    print("Bootstrap interval coverage...", flush=True)
    rows += bootstrap_study(bootstrap_replications, resamples, streams[2])
    frame = pd.DataFrame(rows)
    out = Path(output)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "rates.csv", index=False)
    for name, subset, target in [
        ("clustering", frame[frame.study == "clustering"], 0.05),
        (
            "ratio_type_I",
            frame[(frame.study == "ratio") & (frame.event == "type_I")],
            0.05,
        ),
        ("bootstrap_coverage", frame[frame.study == "bootstrap_comparison"], 0.95),
    ]:
        rate_plot(
            subset,
            name.replace("_", " ").title(),
            target,
            out / "figures" / f"{name}.png",
        )
    config = {
        "replications": replications,
        "bootstrap_replications": bootstrap_replications,
        "resamples": resamples,
        "seed": seed,
        "alpha": 0.05,
        "data": "synthetic",
    }
    dump_json(out / "config.json", config)

    write_manifest(
        out,
        [
            out / "rates.csv",
            out / "config.json",
            *sorted((out / "figures").glob("*.png")),
        ],
        {},
    )
    print(f"Saved {len(frame)} estimates: {out}")
    return frame


def main():
    p = argparse.ArgumentParser(description="Dependence, ratios and BCa research")
    p.add_argument("--output", default="reports/robustness")
    p.add_argument("--replications", type=int, default=2000)
    p.add_argument("--bootstrap-replications", type=int, default=500)
    p.add_argument("--resamples", type=int, default=1999)
    p.add_argument("--seed", type=int, default=650)
    args = p.parse_args()
    try:
        run(**vars(args))
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
