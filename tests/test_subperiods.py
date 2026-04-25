import pandas as pd

from eurostoxx_iv_rv_backtest.analytics.subperiods import compute_subperiod_performance


def test_compute_subperiod_performance_returns_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2007-12-31", "2008-01-02", "2011-06-01", "2020-03-16", "2021-01-04"]
            ),
            "pnl_varswap": [1.0, -0.5, 0.25, -0.1, 0.4],
            "signal_vol": [1, -1, -1, 1, 0],
        }
    )

    result = compute_subperiod_performance(df)

    assert {"pre_gfc", "gfc", "euro_crisis", "covid", "post_covid"}.issubset(
        set(result["subperiod"])
    )
    assert {
        "total_pnl",
        "annualized_mean",
        "annualized_vol",
        "sharpe_approx",
        "max_drawdown",
        "hit_ratio",
        "days_in_position",
        "long_vol_pnl",
        "short_vol_pnl",
    }.issubset(result.columns)
    pre_gfc = result.set_index("subperiod").loc["pre_gfc"]
    assert pre_gfc["total_pnl"] == 1.0
