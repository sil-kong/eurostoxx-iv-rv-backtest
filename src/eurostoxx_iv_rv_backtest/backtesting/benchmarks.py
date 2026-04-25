from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from eurostoxx_iv_rv_backtest.analytics.performance import compute_summary_stats
from eurostoxx_iv_rv_backtest.features.iv_rv_variance_swap import (
    backtest_iv_rv_variance_swap,
)


def always_short_vol(
    df: pd.DataFrame,
    iv_col: str = "iv",
    rv_fwd_col: str = "rv_fwd_20d",
) -> pd.Series:
    """Return -1 on rows with valid IV and forward RV, otherwise 0."""
    signal = pd.Series(0, index=df.index, dtype="int64", name="always_short_vol")
    signal.loc[_valid_payoff_mask(df, iv_col, rv_fwd_col)] = -1
    return signal


def always_long_vol(
    df: pd.DataFrame,
    iv_col: str = "iv",
    rv_fwd_col: str = "rv_fwd_20d",
) -> pd.Series:
    """Return +1 on rows with valid IV and forward RV, otherwise 0."""
    signal = pd.Series(0, index=df.index, dtype="int64", name="always_long_vol")
    signal.loc[_valid_payoff_mask(df, iv_col, rv_fwd_col)] = 1
    return signal


def always_flat(df: pd.DataFrame) -> pd.Series:
    """Return an all-flat benchmark signal."""
    return pd.Series(0, index=df.index, dtype="int64", name="always_flat")


def random_signals(
    df: pd.DataFrame,
    seed: int,
    iv_col: str = "iv",
    rv_fwd_col: str = "rv_fwd_20d",
) -> pd.Series:
    """
    Return seeded random -1/0/+1 signals for diagnostic checks only.

    This is deliberately excluded from the default benchmark report because it
    has no economic interpretation.
    """
    rng = np.random.default_rng(seed)
    signal = pd.Series(0, index=df.index, dtype="int64", name="random_signals")
    valid = _valid_payoff_mask(df, iv_col, rv_fwd_col)
    signal.loc[valid] = rng.choice([-1, 0, 1], size=int(valid.sum()))
    return signal


def run_benchmark_backtests(
    df: pd.DataFrame,
    date_col: str = "date",
    iv_col: str = "iv",
    rv_fwd_col: str = "rv_fwd_20d",
    signal_col: str = "signal_vol",
    trading_days_per_year: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare the IV/RV signal with simple variance-risk-premium benchmarks.

    Returns:
        comparison: one row per strategy with summary diagnostics.
        equity_curves: date-indexed cumulative normalized PnL columns.
    """
    required = {date_col, iv_col, rv_fwd_col, signal_col}
    _require_columns(df, required)

    work = df.copy().sort_values(date_col).reset_index(drop=True)
    work[date_col] = pd.to_datetime(work[date_col])

    strategies: Mapping[str, pd.Series] = {
        "iv_rv_signal": work[signal_col].fillna(0).astype(float),
        "always_short_vol": always_short_vol(work, iv_col, rv_fwd_col),
        "always_long_vol": always_long_vol(work, iv_col, rv_fwd_col),
        "always_flat": always_flat(work),
    }

    metric_rows: list[dict[str, float | str]] = []
    equity_curves = pd.DataFrame({date_col: work[date_col]})

    for strategy, signal in strategies.items():
        strategy_df = work.copy()
        strategy_df["_benchmark_signal"] = signal.to_numpy()
        bt = backtest_iv_rv_variance_swap(
            strategy_df,
            iv_col=iv_col,
            rv_fwd_col=rv_fwd_col,
            signal_col="_benchmark_signal",
            notional=1.0,
        )
        metric_rows.append(
            summarize_benchmark_result(
                bt,
                strategy=strategy,
                date_col=date_col,
                pnl_col="pnl_varswap",
                equity_col="equity_varswap",
                signal_col="_benchmark_signal",
                trading_days_per_year=trading_days_per_year,
            )
        )
        equity_curves[strategy] = bt["equity_varswap"]

    return pd.DataFrame(metric_rows), equity_curves


def summarize_benchmark_result(
    df: pd.DataFrame,
    strategy: str,
    date_col: str = "date",
    pnl_col: str = "pnl_varswap",
    equity_col: str = "equity_varswap",
    signal_col: str = "signal_vol",
    trading_days_per_year: int = 252,
) -> dict[str, float | str]:
    """Summarize one stylized payoff series in research-report-friendly columns."""
    _require_columns(df, {date_col, pnl_col, equity_col, signal_col})
    stats = compute_summary_stats(
        df[pnl_col],
        df[equity_col],
        df[signal_col],
        trading_days_per_year=trading_days_per_year,
    )
    yearly = _yearly_pnl(df, date_col=date_col, pnl_col=pnl_col)
    best_year = yearly.idxmax() if len(yearly) else np.nan
    worst_year = yearly.idxmin() if len(yearly) else np.nan

    return {
        "strategy": strategy,
        "total_pnl": stats["total_pnl"],
        "annualized_mean": stats["annualized_mean"],
        "annualized_vol": stats["annualized_vol"],
        "sharpe_approx": stats["sharpe_approx"],
        "max_drawdown": stats["max_drawdown"],
        "hit_ratio": stats["hit_ratio"],
        "days_in_position": stats["days_in_position"],
        "exposure_fraction": stats["exposure_fraction"],
        "best_year": float(best_year) if pd.notna(best_year) else np.nan,
        "worst_year": float(worst_year) if pd.notna(worst_year) else np.nan,
        "long_vol_pnl": stats["long_vol_pnl"],
        "short_vol_pnl": stats["short_vol_pnl"],
    }


def _yearly_pnl(df: pd.DataFrame, date_col: str, pnl_col: str) -> pd.Series:
    work = df[[date_col, pnl_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work[pnl_col] = work[pnl_col].fillna(0.0).astype(float)
    return work.groupby(work[date_col].dt.year)[pnl_col].sum()


def _valid_payoff_mask(df: pd.DataFrame, iv_col: str, rv_fwd_col: str) -> pd.Series:
    _require_columns(df, {iv_col, rv_fwd_col})
    return df[iv_col].notna() & df[rv_fwd_col].notna()


def _require_columns(df: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
