import numpy as np
import pandas as pd

from eurostoxx_iv_rv_backtest.analytics.performance import (
    compute_drawdown,
    compute_regime_stats,
    compute_summary_stats,
    compute_worst_drawdown_periods,
    compute_yearly_stats,
)


def test_compute_drawdown_returns_absolute_underwater_curve() -> None:
    equity = pd.Series([0.0, 2.0, 1.0, 3.0, 2.5])

    drawdown = compute_drawdown(equity)

    pd.testing.assert_series_equal(
        drawdown,
        pd.Series([0.0, 0.0, -1.0, 0.0, -0.5]),
    )


def test_compute_summary_stats_aggregates_pnl_signal_and_drawdown() -> None:
    pnl = pd.Series([1.0, -0.5, 0.25, 0.0])
    equity = pnl.cumsum()
    signal = pd.Series([1, -1, 0, 1])

    stats = compute_summary_stats(pnl, equity, signal)

    assert stats["total_pnl"] == 0.75
    assert stats["max_drawdown"] == -0.5
    assert stats["nb_days"] == 4
    assert stats["days_in_position"] == 3
    assert stats["pct_in_market"] == 75.0
    assert stats["long_vol_pnl"] == 1.0
    assert stats["short_vol_pnl"] == -0.5
    assert np.isfinite(stats["sharpe"])


def test_compute_regime_stats_groups_flat_long_and_short() -> None:
    df = pd.DataFrame(
        {
            "pnl_varswap": [1.0, -0.5, 0.25, 0.0],
            "signal_vol": [1, -1, 0, 1],
        }
    )

    result = compute_regime_stats(df)

    by_regime = result.set_index("regime")
    assert by_regime.loc["long_vol", "days"] == 2
    assert by_regime.loc["long_vol", "total_pnl"] == 1.0
    assert by_regime.loc["short_vol", "total_pnl"] == -0.5
    assert by_regime.loc["flat", "total_pnl"] == 0.25


def test_compute_yearly_stats_resets_equity_by_calendar_year() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-03"]),
            "pnl_varswap": [1.0, -0.5, 0.25],
            "signal_vol": [1, -1, 0],
        }
    )

    result = compute_yearly_stats(df)

    by_year = result.set_index("year")
    assert by_year.loc[2023, "total_pnl"] == 1.0
    assert by_year.loc[2024, "total_pnl"] == -0.25
    assert by_year.loc[2024, "max_drawdown"] == 0.0


def test_compute_worst_drawdown_periods_returns_largest_periods() -> None:
    equity = pd.Series([0.0, 2.0, 1.0, 0.5, 3.0, 2.0])
    dates = pd.Series(pd.date_range("2024-01-01", periods=len(equity), freq="D"))

    result = compute_worst_drawdown_periods(equity, dates=dates, top_n=1)

    assert len(result) == 1
    assert result.loc[0, "max_drawdown"] == -1.5
    assert result.loc[0, "start_date"] == dates.iloc[2]
    assert result.loc[0, "trough_date"] == dates.iloc[3]
