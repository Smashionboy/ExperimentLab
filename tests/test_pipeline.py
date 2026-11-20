import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experimentlab.cli import main
from experimentlab.config import load_config
from experimentlab.data import allocation_check, demo_data, validate_frame
from experimentlab.inference import welch
from experimentlab.studies import run_studies


def tiny_config():
    c = load_config(Path(__file__).parents[1] / "configs/quick.json")
    c.update(
        replications=4,
        resampling_replications=2,
        resamples=99,
        looks=[20, 40],
        n_per_arm=20,
        resampling_n=10,
    )
    return c


def test_study_determinism_and_intervals():
    a, diag = run_studies(tiny_config())
    b, diag2 = run_studies(tiny_config())
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(diag, diag2)
    assert (a.mc_ci_low <= a.rate + 1e-15).all()
    assert (a.rate <= a.mc_ci_high + 1e-15).all()
    assert set(a.study) == {
        "fixed_horizon",
        "conversion",
        "resampling",
        "peeking",
        "multiplicity",
        "cuped",
    }


def test_normal_null_calibration():
    rng = np.random.default_rng(2026)
    rejections = sum(
        welch(rng.normal(size=70), rng.normal(size=70)).p_value < 0.05
        for _ in range(1500)
    )
    # Broad, predetermined 4-sigma MC envelope; not a fitted pass criterion.
    assert abs(rejections / 1500 - 0.05) < 4 * np.sqrt(0.05 * 0.95 / 1500)


def test_user_validation_and_srm():
    frame = demo_data(100)
    a, b = validate_frame(frame, "converted", True)
    assert len(a) == len(b) == 100
    assert allocation_check(100, 100) == 1
    assert allocation_check(10, 100) < 0.001
    for bad in [
        pd.concat([frame, frame.iloc[:1]]),
        frame.assign(revenue=np.nan),
        frame.assign(group="oops"),
    ]:
        with pytest.raises(ValueError):
            validate_frame(bad, "revenue")


def test_cli_end_to_end(tmp_path):
    csv = tmp_path / "demo.csv"
    main(["demo", "--n", "100", "--output", str(csv)])
    for kind, metric in [("binary", "converted"), ("continuous", "revenue")]:
        dest = tmp_path / f"{kind}.json"
        main(
            [
                "analyze",
                str(csv),
                "--metric",
                metric,
                "--kind",
                kind,
                "--resamples",
                "99",
                "--output",
                str(dest),
            ]
        )
        payload = json.loads(dest.read_text())
        assert payload["effect_direction"] == "treatment - control"
        assert len(payload["estimates"]) == (1 if kind == "binary" else 2)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(tiny_config()))
    output = tmp_path / "report"
    main(["run", "--config", str(config), "--output", str(output)])
    assert len(list((output / "figures").glob("*.png"))) == 6
    assert json.loads((output / "manifest.json").read_text())["data_origin"].startswith(
        "synthetic"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"looks": [10, 5]},
        {"alpha": 1},
        {"replications": 1.5},
        {"seed": -1},
        {"surprise": 4},
    ],
)
def test_invalid_config(tmp_path, change):
    c = tiny_config()
    c.update(change)
    path = tmp_path / "c.json"
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        load_config(path)


def test_cli_parameterized_planning(tmp_path):
    dest = tmp_path / "plan.json"
    main(
        [
            "plan",
            "--baseline",
            ".2",
            "--uplift",
            ".03",
            "--effect",
            ".4",
            "--sigma",
            "2",
            "--power",
            ".9",
            "--output",
            str(dest),
        ]
    )
    result = json.loads(dest.read_text())
    from experimentlab.planning import proportion_power

    assert result["target_power"] == 0.9
    assert proportion_power(result["conversion_n_per_arm"], 0.2, 0.23) >= 0.9


def test_cli_invalid_input_does_not_write_report(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("user_id,group,revenue\na,control,2\na,treatment,3\n")
    output = tmp_path / "analysis.json"
    with pytest.raises(SystemExit, match="exactly once"):
        main(
            [
                "analyze",
                str(path),
                "--metric",
                "revenue",
                "--kind",
                "continuous",
                "--output",
                str(output),
            ]
        )
    assert not output.exists()
