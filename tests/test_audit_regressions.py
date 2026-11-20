import json

import numpy as np
import pandas as pd
import pytest

from experimentlab.adjustment import analyze, main
from experimentlab.adjustment_data import generate_trial
from experimentlab.clusters import cluster_welch
from experimentlab.data import validate_frame
from experimentlab.fieller import delta_ratio, fieller
from experimentlab.fieller_study import run as fieller_run
from experimentlab.inference import wilson
from experimentlab.ratio_sensitivity import run as sensitivity_run
from experimentlab.sequential_study import run as sequential_run
from scripts.verify_release import verify


@pytest.mark.parametrize("n", [2, 10, 100, 1000, 2000])
def test_wilson_contains_boundary_without_roundoff(n):
    assert wilson(0, n)[0] == 0
    assert wilson(n, n)[1] == 1


def test_adjustment_csv_preserves_string_ids(tmp_path):
    frame = generate_trial(n=80)
    ids = [f"{i:04d}" for i in range(80)]
    ids[1] = "0"  # distinct from 0000
    frame.user_id = ids
    csv = tmp_path / "input.csv"
    frame.to_csv(csv, index=False)
    out = tmp_path / "analysis"
    main(
        [
            "analyze",
            str(csv),
            "--metric",
            "outcome",
            "--covariates",
            "x1",
            "x2",
            "x3",
            "--propensity",
            ".5",
            "--output",
            str(out),
        ]
    )
    result = pd.read_csv(out / "aipw_ridge_oof.csv", dtype={"user_id": "string"})
    assert result.user_id.tolist() == ids


def test_rerun_removes_only_stale_owned_oof_files(tmp_path):
    frame = generate_trial(n=80)
    analyze(frame, "outcome", ["x1", "x2", "x3"], tmp_path)
    marker = tmp_path / "notes.txt"
    marker.write_text("keep")
    analyze(frame, "outcome", ["x1", "x2", "x3"], tmp_path, method="lin")
    assert not list(tmp_path.glob("*_oof.csv"))
    assert marker.read_text() == "keep"
    assert len(json.loads((tmp_path / "summary.json").read_text())["estimates"]) == 2


def test_reserved_identifier_cannot_be_metric():
    frame = generate_trial(n=80)
    frame.user_id = np.arange(80)
    with pytest.raises(ValueError, match="ID"):
        validate_frame(frame, "user_id")


def test_cluster_api_rejects_duplicate_users():
    frame = pd.DataFrame(
        {
            "user_id": ["a", "a", "b", "c"],
            "cluster_id": [1, 2, 3, 4],
            "group": ["control", "control", "treatment", "treatment"],
            "value": [1, 2, 3, 4],
        }
    )
    with pytest.raises(ValueError, match="unique"):
        cluster_welch(frame, "value")


@pytest.mark.parametrize("scale", [1e-160, 1e160])
@pytest.mark.parametrize("method", [fieller, delta_ratio])
def test_ratio_intervals_survive_joint_unit_changes(scale, method):
    x, y = (
        np.random.default_rng(8)
        .multivariate_normal([2, 1], [[1, 0.5], [0.5, 1]], size=100)
        .T
    )
    a, b = method(x, y), method(x * scale, y * scale)
    assert a.kind == b.kind
    np.testing.assert_allclose(a.intervals, b.intervals, rtol=1e-12)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_fieller_membership_requires_finite_ratio(value):
    with pytest.raises(ValueError, match="finite"):
        fieller([1, 2, 4], [0, 3, 2]).contains(value)


@pytest.mark.parametrize("runner", [fieller_run, sensitivity_run, sequential_run])
def test_study_writes_fresh_manifest(tmp_path, runner):
    if runner == sequential_run:
        runner(tmp_path, replications=2, horizon=20)
    else:
        runner(tmp_path, replications=2)
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["project_version"] == "1.6.0"
    assert "scikit-learn" in data["versions"]
    import hashlib

    for name, expected in data["artifact_sha256"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == expected


def test_release_verifier_detects_tampering(tmp_path, monkeypatch):
    import hashlib

    import scripts.verify_release as module

    monkeypatch.setattr(module, "STUDIES", ["."])
    artifact = tmp_path / "rates.csv"
    artifact.write_text("original")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_sha256": {
                    "rates.csv": hashlib.sha256(artifact.read_bytes()).hexdigest()
                }
            }
        )
    )
    assert verify(tmp_path) == 1
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="Checksum"):
        verify(tmp_path)
