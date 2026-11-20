"""Portable report artifacts; Matplotlib uses a headless backend."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import __version__
from .planning import (
    conversion_mde,
    conversion_sample_size,
    mean_power,
    mean_sample_size,
)

COLORS = ["#176B87", "#E67E22", "#6B4C9A", "#36946B"]


def dump_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def rate_plot(frame, title, target, path):
    labels = [
        f"{r.scenario}\n{r.method}"
        + (f" (n={int(r.n_per_arm)}/arm)" if np.isfinite(r.n_per_arm) else "")
        for r in frame.itertuples()
    ]
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(10, max(3.5, len(frame) * 0.58)))
    ax.errorbar(
        frame.rate,
        y,
        xerr=np.vstack([frame.rate - frame.mc_ci_low, frame.mc_ci_high - frame.rate]),
        fmt="o",
        color=COLORS[0],
        capsize=4,
    )
    ax.set_yticks(y, labels)
    ax.axvline(target, color=COLORS[1], linestyle="--", label=f"Reference {target:.0%}")
    ax.set_xlim(
        (max(0, min(0.8, float(frame.mc_ci_low.min()) - 0.02)), 1)
        if target > 0.9
        else (0, 0.3)
        if frame.mc_ci_high.max() < 0.3
        else (0, 1)
    )
    ax.invert_yaxis()
    ax.set_xlabel(
        f"Rate; bars = 95% Wilson Monte Carlo intervals; dashed = {target:.0%}"
    )
    ax.set_title(title, loc="left", fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(rates, cuped, cfg, output):
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    rates.to_csv(output / "rates.csv", index=False)
    cuped.to_csv(output / "cuped_diagnostics.csv", index=False)
    dump_json(output / "config.json", cfg)
    for name, subset, title, target in [
        (
            "type_I",
            rates[
                (rates.study.isin(["fixed_horizon", "conversion"]))
                & (rates.event == "type_I")
            ],
            "Fixed horizon: false positive rates",
            cfg["alpha"],
        ),
        (
            "coverage",
            rates[
                (rates.study.isin(["fixed_horizon", "resampling"]))
                & (rates.event == "coverage")
                & (rates.true_effect == 0)
            ],
            "Interval coverage under zero effect",
            1 - cfg["alpha"],
        ),
        (
            "peeking",
            rates[rates.study == "peeking"],
            "Repeated looks at the same A/A experiment",
            cfg["alpha"],
        ),
        (
            "multiplicity",
            rates[rates.study == "multiplicity"],
            "Probability of at least one false discovery",
            cfg["alpha"],
        ),
        (
            "cuped_power",
            rates[(rates.study == "cuped") & (rates.event == "power")],
            "CUPED: power on independent calibration data",
            0.8,
        ),
    ]:
        rate_plot(subset, title, target, output / "figures" / f"{name}.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sizes = np.arange(20, 1001, 20)
    for d, color in zip([0.1, 0.2, 0.3], COLORS):
        ax.plot(
            sizes,
            [mean_power(int(n), d, 1, cfg["alpha"]) for n in sizes],
            label=f"Effect / SD = {d}",
            color=color,
        )
    ax.axhline(0.8, color="gray", linestyle="--")
    ax.set(
        xlabel="Users per arm",
        ylabel="Power",
        ylim=(0, 1),
        title="Planning: equal-SD noncentral t approximation",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "figures" / "power_curve.png", dpi=150)
    plt.close(fig)
    planning = {
        "alpha": cfg["alpha"],
        "target_power": 0.8,
        "mean_effect_0_2_sd_n_per_arm": mean_sample_size(0.2, 1, alpha=cfg["alpha"]),
        "conversion_10_to_12_percent_n_per_arm": conversion_sample_size(
            0.1, 0.02, alpha=cfg["alpha"]
        ),
        "conversion_mde_n_5000_baseline_10_percent": conversion_mde(
            5000, 0.1, alpha=cfg["alpha"]
        ),
    }
    dump_json(output / "planning.json", planning)
    paths = [
        output / "rates.csv",
        output / "cuped_diagnostics.csv",
        output / "config.json",
        output / "planning.json",
    ]
    paths.extend(sorted((output / "figures").glob("*.png")))
    write_manifest(
        output,
        paths,
        {
            "data_origin": "synthetic Monte Carlo; no real business data",
            "intervals": "Pointwise Wilson 95%; independent replications, paired methods. Not simultaneous.",
        },
    )


def write_manifest(output, artifacts, metadata=None):
    """Seal only explicit outputs; record runtime and source snapshot per run."""
    output = Path(output)
    source = Path(__file__).parent
    payload = dict(metadata or {})
    payload.setdefault("data_origin", "synthetic Monte Carlo; no real business data")
    payload.update(
        project_version=__version__,
        python=platform.python_version(),
        platform=platform.system(),
        versions={
            name: importlib.metadata.version(name)
            for name in ["numpy", "scipy", "pandas", "matplotlib", "scikit-learn"]
        },
        artifact_sha256={
            Path(p).relative_to(output).as_posix(): hashlib.sha256(
                Path(p).read_bytes()
            ).hexdigest()
            for p in artifacts
        },
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.glob("*.py"))
        },
    )
    dump_json(output / "manifest.json", payload)
