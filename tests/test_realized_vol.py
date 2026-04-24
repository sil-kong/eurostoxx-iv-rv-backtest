import numpy as np
import pandas as pd

from eurostoxx_iv_rv_backtest.features.realized_vol import (
    add_forward_realized_vol,
    add_realized_vol,
)


def test_add_realized_vol_computes_log_returns_and_annualized_rv() -> None:
    returns = np.array([0.01, -0.02, 0.03])
    prices = 100.0 * np.exp(np.r_[0.0, returns].cumsum())
    df = pd.DataFrame({"close": prices})

    result = add_realized_vol(
        df,
        price_col="close",
        windows=(3,),
        trading_days_per_year=252,
    )

    expected_log_ret = pd.Series([np.nan, *returns])
    pd.testing.assert_series_equal(
        result["log_ret"],
        expected_log_ret,
        check_names=False,
    )

    expected_rv = returns.std(ddof=1) * np.sqrt(252)
    assert np.isclose(result.loc[3, "rv_3d"], expected_rv)
    assert np.isclose(result.loc[3, "rv_3d_pct"], expected_rv * 100.0)
    assert result["rv_3d"].iloc[:3].isna().all()


def test_add_forward_realized_vol_uses_strictly_future_returns() -> None:
    returns = np.array([0.50, 0.01, 0.02, 0.04, -0.01])
    prices = 100.0 * np.exp(np.r_[0.0, returns].cumsum())
    df = pd.DataFrame({"close": prices})

    result = add_forward_realized_vol(
        df,
        price_col="close",
        window=3,
        trading_days_per_year=252,
    )

    # At index 1, the current return is 0.50. The forward RV must use only
    # returns at indices 2, 3 and 4: 0.01, 0.02 and 0.04.
    expected = np.array([0.01, 0.02, 0.04]).std(ddof=1) * np.sqrt(252)
    excluded_current = np.array([0.50, 0.01, 0.02]).std(ddof=1) * np.sqrt(252)

    assert np.isclose(result.loc[1, "rv_fwd_3d"], expected)
    assert not np.isclose(result.loc[1, "rv_fwd_3d"], excluded_current)
    assert np.isclose(result.loc[1, "rv_fwd_3d_pct"], expected * 100.0)


def test_add_forward_realized_vol_last_rows_without_future_data_are_nan() -> None:
    returns = np.array([0.01, 0.02, 0.03, 0.04])
    prices = 100.0 * np.exp(np.r_[0.0, returns].cumsum())
    df = pd.DataFrame({"close": prices})

    result = add_forward_realized_vol(df, window=2)

    assert result["rv_fwd_2d"].tail(2).isna().all()
