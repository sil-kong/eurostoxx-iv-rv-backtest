import numpy as np
import pandas as pd

from eurostoxx_iv_rv_backtest.backtesting.benchmarks import (
    always_flat,
    always_long_vol,
    always_short_vol,
    non_overlap_sharpe,
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


def test_benchmarks_are_restricted_to_signal_sample() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="B"),
            "iv": [0.20, 0.20, 0.20, 0.20],
            "rv_fwd_20d": [0.10, 0.10, 0.30, 0.10],
            "signal_vol": [0, 0, 1, -1],
            "iv_rv_zscore": [np.nan, np.nan, -1.0, 1.0],
        }
    )

    comparison, _ = run_benchmark_backtests(df, horizon=1)
    by_strategy = comparison.set_index("strategy")
    expected_short = -((0.30**2 - 0.20**2) + (0.10**2 - 0.20**2))

    assert np.isclose(by_strategy.loc["always_short_vol", "total_pnl"], expected_short)
    assert by_strategy.loc["always_short_vol", "days_in_position"] == 2


def test_non_overlap_sharpe_matches_manual_sampling() -> None:
    pnl = pd.Series([1.0, 9.0, 3.0, 9.0, 2.0, 9.0, 4.0, 9.0])

    # horizon=2: offset 0 -> [1,3,2,4], offset 1 -> [9,9,9,9] (zero std, skipped)
    sample = pd.Series([1.0, 3.0, 2.0, 4.0])
    expected = sample.mean() / sample.std(ddof=1) * np.sqrt(252 / 2)

    assert np.isclose(non_overlap_sharpe(pnl, horizon=2), expected)
