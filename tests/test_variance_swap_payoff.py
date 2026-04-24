import numpy as np
import pandas as pd

from eurostoxx_iv_rv_backtest.features.iv_rv_variance_swap import (
    backtest_iv_rv_variance_swap,
)


def test_backtest_variance_swap_zero_signal_has_zero_pnl() -> None:
    df = pd.DataFrame(
        {
            "iv": [0.20, 0.30],
            "rv_fwd_20d": [0.40, 0.10],
            "signal_vol": [0, 0],
        }
    )

    result = backtest_iv_rv_variance_swap(df)

    assert (result["pnl_varswap"] == 0.0).all()
    assert (result["equity_varswap"] == 0.0).all()


def test_backtest_variance_swap_long_and_short_signs() -> None:
    df = pd.DataFrame(
        {
            "iv": [0.20, 0.20],
            "rv_fwd_20d": [0.30, 0.30],
            "signal_vol": [1, -1],
        }
    )

    result = backtest_iv_rv_variance_swap(df)
    spread = 0.30**2 - 0.20**2

    assert np.isclose(result.loc[0, "pnl_varswap"], spread)
    assert np.isclose(result.loc[1, "pnl_varswap"], -spread)


def test_backtest_variance_swap_missing_iv_or_forward_rv_produces_zero_pnl() -> None:
    df = pd.DataFrame(
        {
            "iv": [0.20, np.nan, 0.20],
            "rv_fwd_20d": [np.nan, 0.30, 0.30],
            "signal_vol": [1, 1, 1],
        }
    )

    result = backtest_iv_rv_variance_swap(df)

    assert result.loc[0, "pnl_varswap"] == 0.0
    assert result.loc[1, "pnl_varswap"] == 0.0
    assert np.isclose(result.loc[2, "pnl_varswap"], 0.30**2 - 0.20**2)


def test_backtest_variance_swap_equity_is_cumulative_pnl() -> None:
    df = pd.DataFrame(
        {
            "iv": [0.20, 0.20, 0.20],
            "rv_fwd_20d": [0.30, 0.10, 0.20],
            "signal_vol": [1, 1, -1],
        }
    )

    result = backtest_iv_rv_variance_swap(df)

    pd.testing.assert_series_equal(
        result["equity_varswap"],
        result["pnl_varswap"].cumsum(),
        check_names=False,
    )
