from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from eurostoxx_iv_rv_backtest.analytics.performance import compute_summary_stats
from eurostoxx_iv_rv_backtest.features.iv_rv_variance_swap import (
    backtest_iv_rv_variance_swap,
)
from eurostoxx_iv_rv_backtest.features.realized_vol import (
    add_forward_realized_vol,
    add_realized_vol,
)
from eurostoxx_iv_rv_backtest.scripts.build_signals import add_iv_rv_signal


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def run_walk_forward_validation(
    df: pd.DataFrame,
    train_years: int = 5,
    test_years: int = 1,
    z_entry_grid: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
    lookback_grid: Sequence[int] = (63, 126, 252, 504),
    horizon_grid: Sequence[int] = (10, 20, 30),
    selection_metric: str = "sharpe_approx",
    min_days_in_position: int = 20,
    date_col: str = "date",
    price_col: str = "close",
    iv_col: str = "iv",
    trading_days_per_year: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Select IV/RV parameters on rolling train windows and test them out of sample.

    The signal for each parameter set is computed chronologically from spot and
    IV data only. Forward RV is used only after the fact to score train and test
    payoff windows.
    """
    if train_years <= 0 or test_years <= 0:
        raise ValueError("train_years and test_years must be positive.")
    if min_days_in_position < 0:
        raise ValueError("min_days_in_position must be non-negative.")

    _require_columns(df, {date_col, price_col, iv_col})
    work = df.copy().sort_values(date_col).reset_index(drop=True)
    work[date_col] = pd.to_datetime(work[date_col])

    folds = make_walk_forward_folds(
        work,
        train_years=train_years,
        test_years=test_years,
        date_col=date_col,
    )
    backtest_cache = _build_parameter_backtests(
        work,
        z_entry_grid=z_entry_grid,
        lookback_grid=lookback_grid,
        horizon_grid=horizon_grid,
        date_col=date_col,
        price_col=price_col,
        iv_col=iv_col,
        trading_days_per_year=trading_days_per_year,
    )

    result_rows: list[dict[str, float | int | pd.Timestamp]] = []
    selected_rows: list[dict[str, float | int | pd.Timestamp]] = []
    equity_parts: list[pd.DataFrame] = []
    running_equity_offset = 0.0

    for fold in folds:
        train_mask = (work[date_col] >= fold.train_start) & (work[date_col] <= fold.train_end)
        test_mask = (work[date_col] >= fold.test_start) & (work[date_col] <= fold.test_end)
        if not train_mask.any() or not test_mask.any():
            continue

        candidates: list[dict[str, float | int]] = []
        for params, bt in backtest_cache.items():
            train_eval_mask = _payoff_window_mask(
                bt,
                base_mask=train_mask,
                horizon=params[2],
                window_end=fold.train_end,
                date_col=date_col,
            )
            train_stats = _window_stats(
                bt.loc[train_eval_mask],
                trading_days_per_year=trading_days_per_year,
            )
            candidates.append(
                {
                    "z_entry": params[0],
                    "lookback": params[1],
                    "horizon": params[2],
                    **train_stats,
                }
            )

        candidates_df = pd.DataFrame(candidates)
        eligible = candidates_df.loc[candidates_df["days_in_position"] >= min_days_in_position]
        if eligible.empty:
            eligible = candidates_df

        metric_col = _normalize_selection_metric(selection_metric)
        if metric_col not in eligible.columns:
            raise ValueError(
                f"Unsupported selection_metric '{selection_metric}'. "
                f"Available columns include: {sorted(eligible.columns)}"
            )

        selected = (
            eligible.assign(_selection_value=eligible[metric_col].replace([np.inf, -np.inf], np.nan))
            .sort_values("_selection_value", ascending=False, na_position="last")
            .iloc[0]
        )
        params_key = (
            float(selected["z_entry"]),
            int(selected["lookback"]),
            int(selected["horizon"]),
        )
        selected_bt = backtest_cache[params_key]
        test_eval_mask = _payoff_window_mask(
            selected_bt,
            base_mask=test_mask,
            horizon=params_key[2],
            window_end=fold.test_end,
            date_col=date_col,
        )
        test_window = selected_bt.loc[test_eval_mask].copy()
        test_stats = _window_stats(test_window, trading_days_per_year=trading_days_per_year)

        result_rows.append(
            {
                "fold_id": fold.fold_id,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "selected_z_entry": params_key[0],
                "selected_lookback": params_key[1],
                "selected_horizon": params_key[2],
                "train_metric": selected[metric_col],
                "test_total_pnl": test_stats["total_pnl"],
                "test_sharpe_approx": test_stats["sharpe_approx"],
                "test_max_drawdown": test_stats["max_drawdown"],
                "test_days_in_position": test_stats["days_in_position"],
            }
        )
        selected_rows.append(
            {
                "fold_id": fold.fold_id,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "selected_z_entry": params_key[0],
                "selected_lookback": params_key[1],
                "selected_horizon": params_key[2],
                "selection_metric": metric_col,
                "train_metric": selected[metric_col],
                "train_total_pnl": selected["total_pnl"],
                "train_sharpe_approx": selected["sharpe_approx"],
                "train_max_drawdown": selected["max_drawdown"],
                "train_days_in_position": selected["days_in_position"],
            }
        )

        if not test_window.empty:
            fold_equity = test_window[[date_col, "pnl_varswap", "signal_vol"]].copy()
            fold_equity["fold_id"] = fold.fold_id
            fold_equity["selected_z_entry"] = params_key[0]
            fold_equity["selected_lookback"] = params_key[1]
            fold_equity["selected_horizon"] = params_key[2]
            fold_equity["walk_forward_equity"] = running_equity_offset + fold_equity[
                "pnl_varswap"
            ].cumsum()
            running_equity_offset = float(fold_equity["walk_forward_equity"].iloc[-1])
            equity_parts.append(fold_equity)

    results = pd.DataFrame(result_rows, columns=_RESULT_COLUMNS)
    selected_params = pd.DataFrame(selected_rows, columns=_SELECTED_COLUMNS)
    equity = (
        pd.concat(equity_parts, ignore_index=True)
        if equity_parts
        else pd.DataFrame(columns=_EQUITY_COLUMNS)
    )
    return results, selected_params, equity


def make_walk_forward_folds(
    df: pd.DataFrame,
    train_years: int,
    test_years: int,
    date_col: str = "date",
) -> list[WalkForwardFold]:
    """Create chronological train/test calendar folds."""
    _require_columns(df, {date_col})
    dates = pd.to_datetime(df[date_col]).sort_values().reset_index(drop=True)
    if dates.empty:
        return []

    folds: list[WalkForwardFold] = []
    train_start = dates.iloc[0].normalize()
    last_date = dates.iloc[-1]
    fold_id = 1

    while True:
        test_start = train_start + pd.DateOffset(years=train_years)
        test_end_exclusive = test_start + pd.DateOffset(years=test_years)
        if test_start > last_date:
            break

        train_mask = (dates >= train_start) & (dates < test_start)
        test_mask = (dates >= test_start) & (dates < test_end_exclusive)
        if train_mask.any() and test_mask.any():
            folds.append(
                WalkForwardFold(
                    fold_id=fold_id,
                    train_start=dates.loc[train_mask].iloc[0],
                    train_end=dates.loc[train_mask].iloc[-1],
                    test_start=dates.loc[test_mask].iloc[0],
                    test_end=dates.loc[test_mask].iloc[-1],
                )
            )
            fold_id += 1

        train_start = train_start + pd.DateOffset(years=test_years)
        if train_start + pd.DateOffset(years=train_years) > last_date:
            break

    return folds


def _build_parameter_backtests(
    df: pd.DataFrame,
    z_entry_grid: Sequence[float],
    lookback_grid: Sequence[int],
    horizon_grid: Sequence[int],
    date_col: str,
    price_col: str,
    iv_col: str,
    trading_days_per_year: int,
) -> dict[tuple[float, int, int], pd.DataFrame]:
    cache: dict[tuple[float, int, int], pd.DataFrame] = {}
    for horizon in horizon_grid:
        feature_df = add_realized_vol(
            df,
            price_col=price_col,
            windows=(int(horizon),),
            trading_days_per_year=trading_days_per_year,
        )
        feature_df = add_forward_realized_vol(
            feature_df,
            price_col=price_col,
            window=int(horizon),
            trading_days_per_year=trading_days_per_year,
        )
        rv_col = f"rv_{int(horizon)}d"
        rv_fwd_col = f"rv_fwd_{int(horizon)}d"

        for lookback in lookback_grid:
            for z_entry in z_entry_grid:
                signal_df = add_iv_rv_signal(
                    feature_df,
                    iv_col=iv_col,
                    rv_col=rv_col,
                    lookback=int(lookback),
                    z_entry=float(z_entry),
                )
                bt = backtest_iv_rv_variance_swap(
                    signal_df,
                    iv_col=iv_col,
                    rv_fwd_col=rv_fwd_col,
                    signal_col="signal_vol",
                    notional=1.0,
                )
                cache[(float(z_entry), int(lookback), int(horizon))] = bt.sort_values(
                    date_col
                ).reset_index(drop=True)
    return cache


def _window_stats(
    df: pd.DataFrame,
    trading_days_per_year: int,
) -> dict[str, float]:
    pnl = df["pnl_varswap"].fillna(0.0).astype(float)
    equity = pnl.cumsum()
    signal = df["signal_vol"].fillna(0.0).astype(float)
    return compute_summary_stats(
        pnl,
        equity,
        signal,
        trading_days_per_year=trading_days_per_year,
        initial_equity=0.0,
    )


def _payoff_window_mask(
    df: pd.DataFrame,
    base_mask: pd.Series,
    horizon: int,
    window_end: pd.Timestamp,
    date_col: str,
) -> pd.Series:
    """Keep only rows whose forward payoff horizon ends inside the same window."""
    exit_dates = pd.to_datetime(df[date_col]).shift(-int(horizon))
    return base_mask.to_numpy() & (exit_dates <= window_end).fillna(False).to_numpy()


def _normalize_selection_metric(selection_metric: str) -> str:
    if selection_metric == "sharpe":
        return "sharpe_approx"
    return selection_metric


def _require_columns(df: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")


_RESULT_COLUMNS = [
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
]

_SELECTED_COLUMNS = [
    "fold_id",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "selected_z_entry",
    "selected_lookback",
    "selected_horizon",
    "selection_metric",
    "train_metric",
    "train_total_pnl",
    "train_sharpe_approx",
    "train_max_drawdown",
    "train_days_in_position",
]

_EQUITY_COLUMNS = [
    "date",
    "pnl_varswap",
    "signal_vol",
    "fold_id",
    "selected_z_entry",
    "selected_lookback",
    "selected_horizon",
    "walk_forward_equity",
]
