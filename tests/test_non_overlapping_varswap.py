import numpy as np
import pandas as pd

from eurostoxx_iv_rv_backtest.backtesting.non_overlapping_varswap import (
    backtest_non_overlapping_varswap,
)


def test_non_overlapping_varswap_books_payoff_at_maturity_only() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=7, freq="B"),
            "iv": [0.20] * 7,
            "rv_fwd_2d": [0.30, 0.40, 0.50, 0.10, 0.20, 0.30, np.nan],
            "signal_vol": [1, 1, 1, -1, -1, 1, 1],
        }
    )

    result = backtest_non_overlapping_varswap(df, rv_fwd_col="rv_fwd_2d", horizon=2)

    trade_rows = result[result["trade_id"].notna()]
    assert trade_rows["trade_id"].tolist() == [1, 2]
    assert trade_rows.index.tolist() == [2, 5]
    assert result.loc[0, "trade_pnl"] == 0.0
    assert np.isclose(result.loc[2, "trade_pnl"], 0.30**2 - 0.20**2)
    assert np.isclose(result.loc[5, "trade_pnl"], -(0.10**2 - 0.20**2))
    assert np.isclose(
        result.loc[5, "cumulative_trade_pnl"],
        result["trade_pnl"].sum(),
    )


def test_non_overlapping_varswap_skips_missing_forward_rv_and_tail_trades() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="B"),
            "iv": [0.20, 0.20, 0.20, 0.20],
            "rv_fwd_2d": [np.nan, 0.30, 0.40, np.nan],
            "signal_vol": [1, 1, 1, 1],
        }
    )

    result = backtest_non_overlapping_varswap(df, rv_fwd_col="rv_fwd_2d", horizon=2)

    assert result["trade_id"].notna().sum() == 1
    assert result.loc[3, "trade_id"] == 1
    assert np.isclose(result.loc[3, "trade_pnl"], 0.30**2 - 0.20**2)


def test_non_overlapping_varswap_applies_simple_trade_cost() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="B"),
            "iv": [0.20, 0.20, 0.20],
            "rv_fwd_2d": [0.30, 0.30, np.nan],
            "signal_vol": [1, 1, 1],
        }
    )

    result = backtest_non_overlapping_varswap(
        df,
        rv_fwd_col="rv_fwd_2d",
        horizon=2,
        cost_per_trade=0.01,
    )

    assert np.isclose(result.loc[2, "trade_pnl"], 0.30**2 - 0.20**2 - 0.01)
