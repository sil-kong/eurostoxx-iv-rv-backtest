from __future__ import annotations

import pandas as pd

from eurostoxx_iv_rv_backtest.analytics.performance import compute_summary_stats


DEFAULT_SUBPERIODS: tuple[tuple[str, str | None, str | None], ...] = (
    ("pre_gfc", None, "2007-12-31"),
    ("gfc", "2008-01-01", "2009-12-31"),
    ("euro_crisis", "2010-01-01", "2012-12-31"),
    ("low_vol_2013_2019", "2013-01-01", "2019-12-31"),
    ("covid", "2020-01-01", "2020-12-31"),
    ("post_covid", "2021-01-01", None),
)


def compute_subperiod_performance(
    df: pd.DataFrame,
    subperiods: tuple[tuple[str, str | None, str | None], ...] = DEFAULT_SUBPERIODS,
    date_col: str = "date",
    pnl_col: str = "pnl_varswap",
    signal_col: str = "signal_vol",
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """Compute stylized payoff diagnostics by named calendar subperiod."""
    _require_columns(df, {date_col, pnl_col, signal_col})
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    rows: list[dict[str, float | str | None]] = []

    for name, start, end in subperiods:
        mask = pd.Series(True, index=work.index)
        if start is not None:
            mask &= work[date_col] >= pd.Timestamp(start)
        if end is not None:
            mask &= work[date_col] <= pd.Timestamp(end)

        period = work.loc[mask].copy()
        if period.empty:
            continue

        pnl = period[pnl_col].fillna(0.0).astype(float)
        stats = compute_summary_stats(
            pnl,
            pnl.cumsum(),
            period[signal_col].fillna(0.0),
            trading_days_per_year=trading_days_per_year,
            initial_equity=0.0,
        )
        rows.append(
            {
                "subperiod": name,
                "start_date": period[date_col].iloc[0].date().isoformat(),
                "end_date": period[date_col].iloc[-1].date().isoformat(),
                "total_pnl": stats["total_pnl"],
                "annualized_mean": stats["annualized_mean"],
                "annualized_vol": stats["annualized_vol"],
                "sharpe_approx": stats["sharpe_approx"],
                "max_drawdown": stats["max_drawdown"],
                "hit_ratio": stats["hit_ratio"],
                "days": stats["nb_days"],
                "days_in_position": stats["days_in_position"],
                "exposure_fraction": stats["exposure_fraction"],
                "long_vol_pnl": stats["long_vol_pnl"],
                "short_vol_pnl": stats["short_vol_pnl"],
            }
        )

    return pd.DataFrame(rows, columns=_SUBPERIOD_COLUMNS)


def _require_columns(df: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")


_SUBPERIOD_COLUMNS = [
    "subperiod",
    "start_date",
    "end_date",
    "total_pnl",
    "annualized_mean",
    "annualized_vol",
    "sharpe_approx",
    "max_drawdown",
    "hit_ratio",
    "days",
    "days_in_position",
    "exposure_fraction",
    "long_vol_pnl",
    "short_vol_pnl",
]
