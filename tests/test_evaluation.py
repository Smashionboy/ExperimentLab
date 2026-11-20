import numpy as np
import pandas as pd
import pytest

from experimentlab.evaluation import summarize


def results():
    rows = []
    for method, estimates in [
        ("welch", [0.0, 2.0, 0.0, 2.0]),
        ("adjusted", [0.5, 1.5, 0.5, 1.5]),
    ]:
        for rep, estimate in enumerate(estimates):
            rows.append(
                dict(
                    mechanism="linear",
                    n_total=100,
                    true_effect=1.0,
                    method=method,
                    replication=rep,
                    estimate=estimate,
                    se=0.5,
                    ci_low=estimate - 1.1,
                    ci_high=estimate + 1.1,
                    p_value=0.4,
                )
            )
    return pd.DataFrame(rows)


def test_known_bias_mse_and_paired_squared_error_difference():
    summary, paired = summarize(results())
    by_method = summary.set_index("method")
    assert by_method.loc["welch", "bias"] == 0
    assert by_method.loc["welch", "rmse"] == 1
    assert by_method.loc["adjusted", "rmse"] == 0.5
    assert paired.iloc[0].paired_mse_difference == -0.75
    assert paired.iloc[0].mc_low == -0.75
    assert paired.iloc[0].mc_high == -0.75
    assert (summary.coverage == 1).all()
    # Exact finite Monte Carlo decomposition uses (R-1)/R times sample variance.
    for row in summary.itertuples():
        assert row.mse == pytest.approx(
            row.bias**2
            + (row.replications - 1) / row.replications * row.empirical_sd**2
        )


def test_order_does_not_change_pairing():
    a, b = summarize(results())
    c, d = summarize(results().sample(frac=1, random_state=9))
    pd.testing.assert_frame_equal(a, c)
    pd.testing.assert_frame_equal(b, d)


@pytest.mark.parametrize("failure", ["duplicate", "unpaired", "nonfinite"])
def test_invalid_replications_rejected(failure):
    frame = results()
    if failure == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif failure == "unpaired":
        frame = frame.iloc[:-1]
    else:
        frame.loc[0, "estimate"] = np.nan
    with pytest.raises(ValueError):
        summarize(frame)
