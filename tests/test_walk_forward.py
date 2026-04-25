import numpy as np
import pandas as pd

from eurostoxx_iv_rv_backtest.validation.walk_forward import (
    make_walk_forward_folds,
    run_walk_forward_validation,
)


def _sample_walk_forward_data() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="B")
    returns = 0.0002 + 0.01 * np.sin(np.arange(len(dates)) / 9.0)
    close = 100.0 * np.exp(np.cumsum(returns))
    iv = 0.22 + 0.03 * np.cos(np.arange(len(dates)) / 17.0)
    return pd.DataFrame({"date": dates, "close": close, "iv": iv})


def test_walk_forward_folds_are_chronological() -> None:
    df = _sample_walk_forward_data()

    folds = make_walk_forward_folds(df, train_years=1, test_years=1)

    assert folds
    for fold in folds:
        assert fold.train_start <= fold.train_end
        assert fold.train_end < fold.test_start
        assert fold.test_start <= fold.test_end


def test_walk_forward_outputs_expected_columns() -> None:
    df = _sample_walk_forward_data()

    results, selected, equity = run_walk_forward_validation(
        df,
        train_years=1,
        test_years=1,
        z_entry_grid=(0.5, 1.0),
        lookback_grid=(5,),
        horizon_grid=(2, 3),
        min_days_in_position=0,
    )

    expected = {
        "fold_id",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "selected_z_entry",
        "selected_lookback",
        "selected_horizon",
        "train_metric",
        "test_total_pnl",
        "test_sharpe_approx",
        "test_max_drawdown",
        "test_days_in_position",
    }
    assert expected.issubset(results.columns)
    assert expected.intersection(selected.columns)
    assert {"date", "fold_id", "walk_forward_equity"}.issubset(equity.columns)
    assert set(results["selected_z_entry"]).issubset({0.5, 1.0})
    assert set(results["selected_horizon"]).issubset({2, 3})


def test_walk_forward_first_fold_selection_does_not_use_test_window() -> None:
    df = _sample_walk_forward_data()
    folds = make_walk_forward_folds(df, train_years=1, test_years=1)
    first_test_start = folds[0].test_start

    baseline, _, _ = run_walk_forward_validation(
        df,
        train_years=1,
        test_years=1,
        z_entry_grid=(0.5, 1.0),
        lookback_grid=(5,),
        horizon_grid=(2, 3),
        min_days_in_position=0,
    )

    changed_future = df.copy()
    future_mask = changed_future["date"] >= first_test_start
    changed_future.loc[future_mask, "close"] = np.linspace(500.0, 1000.0, future_mask.sum())
    changed_future.loc[future_mask, "iv"] = 0.90

    changed, _, _ = run_walk_forward_validation(
        changed_future,
        train_years=1,
        test_years=1,
        z_entry_grid=(0.5, 1.0),
        lookback_grid=(5,),
        horizon_grid=(2, 3),
        min_days_in_position=0,
    )

    first_cols = ["selected_z_entry", "selected_lookback", "selected_horizon", "train_metric"]
    pd.testing.assert_series_equal(
        baseline.loc[0, first_cols],
        changed.loc[0, first_cols],
        check_names=False,
    )
