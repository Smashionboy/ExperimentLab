import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from experimentlab.adjustment import analyze, main
from experimentlab.adjustment_data import frame_arrays, generate_trial
from experimentlab.adjustment_study import run, simulate, validate_config


def config():
    return dict(
        seed=6036,
        replications=2,
        sample_sizes=[80],
        mechanisms=["linear"],
        effects=[0.0, 0.25],
        propensity=0.5,
        folds=2,
        alpha=0.05,
    )


def test_generator_preserves_fixed_effect_and_covariates():
    a, b = generate_trial(effect=0), generate_trial(effect=0.7)
    pd.testing.assert_frame_equal(a.drop(columns="outcome"), b.drop(columns="outcome"))
    np.testing.assert_allclose(
        b.outcome - a.outcome, 0.7 * (a.group == "treatment"), atol=1e-14
    )


@pytest.mark.parametrize("failure", ["outcome", "group", "id", "missing", "duplicate"])
def test_input_leakage_and_invalid_rows(failure):
    frame = generate_trial(n=80)
    covariates = ["x1", "x2", "x3"]
    if failure in {"outcome", "group", "id"}:
        covariates.append("user_id" if failure == "id" else failure)
    elif failure == "missing":
        frame.loc[0, "x1"] = np.nan
    else:
        frame.loc[0, "user_id"] = frame.loc[1, "user_id"]
    with pytest.raises(ValueError):
        frame_arrays(frame, "outcome", covariates)


def test_demo_and_all_method_csv_roundtrip(tmp_path):
    source, output = tmp_path / "trial.csv", tmp_path / "result"
    main(["demo", "--n", "160", "--output", str(source)])
    assert (
        json.loads(source.with_suffix(".metadata.json").read_text())["data_origin"]
        == "synthetic"
    )
    main(
        [
            "analyze",
            str(source),
            "--metric",
            "outcome",
            "--covariates",
            "x1",
            "x2",
            "x3",
            "--propensity",
            ".5",
            "--output",
            str(output),
        ]
    )
    payload = json.loads((output / "summary.json").read_text())
    assert len(payload["estimates"]) == 4
    assert len(pd.read_csv(output / "balance.csv")) == 3
    for name in ["aipw_ridge", "aipw_forest"]:
        oof = pd.read_csv(output / f"{name}_oof.csv")
        assert len(oof) == 160
        assert oof.user_id.is_unique
        assert set(oof.fold) == {0, 1}


def test_failed_analysis_does_not_create_output(tmp_path):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        analyze(generate_trial(), "outcome", ["outcome"], output)
    assert not output.exists()


def test_full_adjustment_pipeline_and_manifest(tmp_path):
    summary, paired = run(config(), tmp_path)
    assert len(summary) == 8
    assert len(paired) == 6
    assert len(list((tmp_path / "figures").glob("*.png"))) == 5
    raw = pd.read_csv(tmp_path / "replications.csv")
    assert len(raw) == 16
    pd.testing.assert_frame_equal(
        raw, simulate(config()), check_dtype=False, atol=1e-12
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    for relative, digest in manifest["artifact_sha256"].items():
        assert hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest() == digest
    assert manifest["failed_replications"] == 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("replications", 1),
        ("sample_sizes", [80, 80]),
        ("mechanisms", ["bad"]),
        ("effects", [float("nan")]),
        ("propensity", 0),
        ("folds", 1),
    ],
)
def test_config_validation(key, value):
    cfg = config()
    cfg[key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)
