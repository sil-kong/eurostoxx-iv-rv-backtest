import numpy as np
import pandas as pd

from eurostoxx_iv_rv_backtest.backtesting.benchmarks import (
    always_flat,
    always_long_vol,
    always_short_vol,
    run_benchmark_backtests,
)


def test_always_short_vol_returns_minus_one_on_valid_rows() -> None:
    df = pd.DataFrame(
        {
            "iv": [0.20, np.nan, 0.25, 0.30],
            "rv_fwd_20d": [0.15, 0.18, np.nan, 0.35],
        }
    )

    signal = always_short_vol(df)

    assert signal.tolist() == [-1, 0, 0, -1]


def test_always_long_vol_returns_plus_one_on_valid_rows() -> None:
    df = pd.DataFrame(
        {
            "iv": [0.20, np.nan, 0.25, 0.30],
            "rv_fwd_20d": [0.15, 0.18, np.nan, 0.35],
        }
    )

    signal = always_long_vol(df)

    assert signal.tolist() == [1, 0, 0, 1]


def test_always_flat_returns_zero() -> None:
    df = pd.DataFrame({"iv": [0.20, 0.30]})

    signal = always_flat(df)

    assert signal.tolist() == [0, 0]


def test_benchmark_payoff_signs_on_artificial_dataset() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2, freq="B"),
            "iv": [0.20, 0.20],
            "rv_fwd_20d": [0.30, 0.10],
            "signal_vol": [1, -1],
        }
    )

    comparison, equity = run_benchmark_backtests(df)
    by_strategy = comparison.set_index("strategy")
    spread_sum = (0.30**2 - 0.20**2) + (0.10**2 - 0.20**2)

    assert np.isclose(by_strategy.loc["always_long_vol", "total_pnl"], spread_sum)
    assert np.isclose(by_strategy.loc["always_short_vol", "total_pnl"], -spread_sum)
    assert by_strategy.loc["always_flat", "total_pnl"] == 0.0
    assert {"iv_rv_signal", "always_short_vol", "always_long_vol", "always_flat"}.issubset(
        equity.columns
    )
