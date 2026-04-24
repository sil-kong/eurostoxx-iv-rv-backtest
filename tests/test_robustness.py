import pandas as pd

from eurostoxx_iv_rv_backtest.analytics.robustness import run_robustness_grid


def test_run_robustness_grid_returns_one_row_per_parameter_combination() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8, freq="B"),
            "close": [100.0, 101.0, 99.0, 102.0, 100.0, 103.0, 101.0, 104.0],
            "iv": [0.20, 0.21, 0.19, 0.23, 0.18, 0.24, 0.20, 0.22],
        }
    )

    grid = run_robustness_grid(
        df,
        z_entries=(0.5, 1.0),
        lookbacks=(2,),
        horizons=(2, 3),
    )

    assert len(grid) == 4
    assert set(grid["horizon"]) == {2.0, 3.0}
    assert set(grid["z_entry"]) == {0.5, 1.0}
    assert {"total_pnl", "sharpe", "max_drawdown", "pct_in_market"}.issubset(
        grid.columns
    )
