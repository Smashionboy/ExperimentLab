import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from experimentlab.advanced import main
from experimentlab.advanced_resampling import bca_mean
from experimentlab.clusters import cluster_welch, design_effect
from experimentlab.inference import welch
from experimentlab.ratio import ratio_components, ratio_delta
from experimentlab.robustness import run


def cluster_frame():
    return pd.DataFrame(
        {
            "cluster_id": np.repeat(["a", "b", "c", "d"], 3),
            "group": ["control"] * 6 + ["treatment"] * 6,
            "y": [1, 2, 3, 2, 3, 4, 3, 4, 5, 6, 7, 8],
        }
    )


def test_ratio_covariance_formula():
    x, y = np.array([2.0, 8, 4, 12, 7]), np.array([1.0, 2, 1, 5, 3])
    ratio, variance, n = ratio_components(x, y)
    cov = np.cov(x, y, ddof=1)
    gradient = np.array([1 / y.mean(), -x.mean() / y.mean() ** 2])
    assert ratio == pytest.approx(x.sum() / y.sum())
    assert variance == pytest.approx(gradient @ cov @ gradient / n)
    assert ratio != pytest.approx(np.mean(x / y))


def test_ratio_scaling_symmetry():
    x, y = np.array([1.0, 7, 3, 9]), np.array([1.0, 2, 2, 3])
    r = ratio_delta(x, y, x + 2, y)
    s = ratio_delta(x + 2, y, x, y)
    scaled = ratio_delta(x * 10, y * 10, (x + 2) * 10, y * 10)
    assert r.effect == -s.effect
    assert s.ci_low == pytest.approx(-r.ci_high)
    assert scaled.p_value == pytest.approx(r.p_value)


@pytest.mark.parametrize(
    "num,den",
    [([1, 2], [0, 0]), ([1, 2], [-1, 3]), ([1, 2], [1, 2, 3]), ([1, np.nan], [1, 2])],
)
def test_ratio_invalid(num, den):
    with pytest.raises(ValueError):
        ratio_components(num, den)


def test_cluster_matches_independent_cluster_means():
    r = cluster_welch(cluster_frame(), "y")
    expected = welch([2, 3], [4, 7])
    assert r.effect == expected.effect
    assert r.p_value == expected.p_value
    assert r.n_control == 2
    assert len(r.warnings) == 2
    # Replicating rows inside each cluster adds no independent information.
    repeated = cluster_welch(pd.concat([cluster_frame()] * 4), "y")
    assert repeated == r


def test_cluster_weights_are_equal_even_when_sizes_differ():
    f = cluster_frame()
    f = pd.concat([f, f[f.cluster_id == "d"]] * 1)
    assert cluster_welch(f, "y").effect == pytest.approx(3.0)


@pytest.mark.parametrize("kind", ["mixed", "missing", "nonfinite", "one_cluster"])
def test_cluster_invalid(kind):
    f = cluster_frame()
    if kind == "mixed":
        f.loc[6, "cluster_id"] = "a"
    if kind == "missing":
        f.loc[0, "cluster_id"] = None
    if kind == "nonfinite":
        f["y"] = f["y"].astype(float)
        f.loc[0, "y"] = np.inf
    if kind == "one_cluster":
        f.loc[:5, "cluster_id"] = "a"
    with pytest.raises(ValueError):
        cluster_welch(f, "y")


def test_design_effect():
    assert design_effect(20, 0.3) == pytest.approx(6.7)
    assert design_effect(20, 0) == 1
    with pytest.raises(ValueError):
        design_effect(20, 1.1)


def test_bca_matches_reference_and_reproducible():
    rng = np.random.default_rng(9)
    a, b = rng.normal(size=20), rng.lognormal(size=30)
    r = bca_mean(a, b, resamples=999, seed=88)
    ref = stats.bootstrap(
        (a, b),
        lambda x, y, axis: y.mean(axis=axis) - x.mean(axis=axis),
        method="BCa",
        n_resamples=999,
        batch=128,
        rng=np.random.default_rng(88),
    )
    assert (r.ci_low, r.ci_high) == pytest.approx(ref.confidence_interval)
    assert r == bca_mean(a, b, resamples=999, seed=88)
    with pytest.raises(ValueError, match="degenerate"):
        bca_mean([1, 1, 1], [2, 2, 2])


def test_advanced_csv_cli(tmp_path):
    f = cluster_frame()
    p = tmp_path / "data.csv"
    f.to_csv(p, index=False)
    out = tmp_path / "result.json"
    main(["cluster", str(p), "--metric", "y", "--output", str(out)])
    assert json.loads(out.read_text())["n_control"] == 2
    f["user_id"] = np.arange(len(f))
    f["sessions"] = np.tile([1, 2, 3], 4)
    f.to_csv(p, index=False)
    main(
        [
            "ratio",
            str(p),
            "--numerator",
            "y",
            "--denominator",
            "sessions",
            "--output",
            str(out),
        ]
    )
    assert json.loads(out.read_text())["method"] == "ratio_delta"


def test_robustness_pipeline(tmp_path):
    r = run(tmp_path, replications=3, bootstrap_replications=2, resamples=99)
    assert len(r) == 20
    assert len(list((tmp_path / "figures").glob("*.png"))) == 3
    assert (r.mc_ci_low <= r.rate + 1e-15).all()


def test_bca_cli_flag(tmp_path):
    from experimentlab.cli import main as basic_main

    frame = pd.DataFrame(
        {
            "user_id": list("abcdefgh"),
            "group": ["control"] * 4 + ["treatment"] * 4,
            "y": [1, 2, 3, 5, 2, 4, 6, 9],
        }
    )
    source = tmp_path / "data.csv"
    frame.to_csv(source, index=False)
    dest = tmp_path / "result.json"
    basic_main(
        [
            "analyze",
            str(source),
            "--kind",
            "continuous",
            "--metric",
            "y",
            "--bca",
            "--resamples",
            "199",
            "--output",
            str(dest),
        ]
    )
    assert [e["method"] for e in json.loads(dest.read_text())["estimates"]] == [
        "welch",
        "percentile_bootstrap",
        "bca_bootstrap",
    ]
