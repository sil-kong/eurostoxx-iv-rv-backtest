import numpy as np
import pandas as pd

from eurostoxx_iv_rv_backtest.scripts.build_signals import add_iv_rv_signal


def test_add_iv_rv_signal_does_not_depend_on_forward_rv() -> None:
    df = pd.DataFrame(
        {
            "iv": [0.20, 0.21, 0.22, 0.30, 0.10],
            "rv_20d": [0.20, 0.20, 0.20, 0.20, 0.20],
            "rv_fwd_20d": [0.10, 0.20, 0.30, 0.40, 0.50],
        }
    )
    changed_forward = df.copy()
    changed_forward["rv_fwd_20d"] = [10.0, 9.0, 8.0, 7.0, 6.0]

    result = add_iv_rv_signal(df, lookback=3, z_entry=0.5)
    changed_result = add_iv_rv_signal(changed_forward, lookback=3, z_entry=0.5)

    pd.testing.assert_series_equal(
        result["iv_minus_rv"],
        changed_result["iv_minus_rv"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["iv_rv_zscore"],
        changed_result["iv_rv_zscore"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["signal_vol"],
        changed_result["signal_vol"],
        check_names=False,
    )


def test_add_iv_rv_signal_at_t_does_not_depend_on_future_iv_or_rv() -> None:
    df = pd.DataFrame(
        {
            "iv": [0.20, 0.21, 0.22, 0.24, 0.18, 0.19, 0.20],
            "rv_20d": [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20],
        }
    )
    changed_future = df.copy()
    changed_future.loc[5:, "iv"] = [1.50, 2.00]
    changed_future.loc[5:, "rv_20d"] = [0.01, 0.02]

    result = add_iv_rv_signal(df, lookback=3, z_entry=0.5)
    changed_result = add_iv_rv_signal(changed_future, lookback=3, z_entry=0.5)

    target_idx = 4
    assert result.loc[target_idx, "iv_minus_rv"] == changed_result.loc[
        target_idx, "iv_minus_rv"
    ]
    assert result.loc[target_idx, "iv_rv_zscore"] == changed_result.loc[
        target_idx, "iv_rv_zscore"
    ]
    assert result.loc[target_idx, "signal_vol"] == changed_result.loc[
        target_idx, "signal_vol"
    ]


def test_add_iv_rv_signal_applies_zscore_threshold_rule() -> None:
    df = pd.DataFrame(
        {
            "iv": [0.20, 0.20, 0.20, 0.30, 0.10],
            "rv_20d": [0.20, 0.20, 0.20, 0.20, 0.20],
        }
    )

    result = add_iv_rv_signal(df, lookback=2, z_entry=0.5)

    assert np.isclose(result.loc[3, "iv_rv_zscore"], 0.7071067811865475)
    assert result.loc[3, "signal_vol"] == -1
    assert np.isclose(result.loc[4, "iv_rv_zscore"], -0.7071067811865475)
    assert result.loc[4, "signal_vol"] == 1
