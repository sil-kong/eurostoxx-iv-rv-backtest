from __future__ import annotations

import numpy as np
import pandas as pd


def compute_drawdown(equity: pd.Series) -> pd.Series:
    """Return absolute drawdown from the running maximum of an equity curve."""
    equity_float = equity.astype(float)
    return equity_float - equity_float.cummax()


def compute_summary_stats(
    pnl: pd.Series,
    equity: pd.Series,
    signal: pd.Series,
    trading_days_per_year: int = 252,
) -> dict[str, float]:
    """
    Compute summary statistics for the stylized daily IV/RV payoff.

    Values are expressed in the same normalized units as the input PnL. The
    Sharpe is an approximate daily-to-annual scaling, not a live portfolio
    performance claim.
    """
    df = pd.DataFrame(
        {
            "pnl": pnl.astype(float),
            "equity": equity.astype(float),
            "signal": signal.fillna(0.0).astype(float),
        }
    ).dropna(subset=["pnl", "equity", "signal"])

    if df.empty:
        return {
            "total_pnl": 0.0,
            "annualized_mean": np.nan,
            "annualized_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "nb_days": 0.0,
            "days_in_position": 0.0,
            "pct_in_market": np.nan,
            "long_vol_pnl": 0.0,
            "short_vol_pnl": 0.0,
        }

    mean_daily = float(df["pnl"].mean())
    std_daily = float(df["pnl"].std(ddof=1))
    annualized_mean = mean_daily * trading_days_per_year
    annualized_vol = std_daily * np.sqrt(trading_days_per_year)
    sharpe = annualized_mean / annualized_vol if annualized_vol > 0 else np.nan
    nb_days = len(df)
    days_in_position = int((df["signal"] != 0).sum())

    return {
        "total_pnl": float(df["equity"].iloc[-1]),
        "annualized_mean": annualized_mean,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": float(compute_drawdown(df["equity"]).min()),
        "nb_days": float(nb_days),
        "days_in_position": float(days_in_position),
        "pct_in_market": 100.0 * days_in_position / nb_days,
        "long_vol_pnl": float(df.loc[df["signal"] == 1, "pnl"].sum()),
        "short_vol_pnl": float(df.loc[df["signal"] == -1, "pnl"].sum()),
    }


def compute_regime_stats(
    df: pd.DataFrame,
    pnl_col: str = "pnl_varswap",
    signal_col: str = "signal_vol",
) -> pd.DataFrame:
    """Aggregate PnL and time in market by flat, long-vol and short-vol regimes."""
    required = {pnl_col, signal_col}
    _require_columns(df, required)

    work = df[[pnl_col, signal_col]].copy()
    work[signal_col] = work[signal_col].fillna(0).astype(int)
    work[pnl_col] = work[pnl_col].fillna(0.0).astype(float)
    work["regime"] = work[signal_col].map({-1: "short_vol", 0: "flat", 1: "long_vol"})
    grouped = (
        work.groupby("regime", dropna=False)
        .agg(days=(pnl_col, "size"), total_pnl=(pnl_col, "sum"), mean_pnl=(pnl_col, "mean"))
        .reset_index()
    )
    grouped["pct_days"] = 100.0 * grouped["days"] / len(work) if len(work) else np.nan
    return grouped.sort_values("regime").reset_index(drop=True)


def compute_yearly_stats(
    df: pd.DataFrame,
    date_col: str = "date",
    pnl_col: str = "pnl_varswap",
    signal_col: str = "signal_vol",
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """Compute calendar-year diagnostics for the stylized daily payoff."""
    _require_columns(df, {date_col, pnl_col, signal_col})
    work = df[[date_col, pnl_col, signal_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work[pnl_col] = work[pnl_col].fillna(0.0).astype(float)
    work[signal_col] = work[signal_col].fillna(0).astype(int)
    work["year"] = work[date_col].dt.year

    rows: list[dict[str, float]] = []
    for year, group in work.groupby("year"):
        pnl = group[pnl_col]
        equity = pnl.cumsum()
        stats = compute_summary_stats(pnl, equity, group[signal_col], trading_days_per_year)
        rows.append(
            {
                "year": int(year),
                "total_pnl": stats["total_pnl"],
                "annualized_mean": stats["annualized_mean"],
                "annualized_vol": stats["annualized_vol"],
                "sharpe": stats["sharpe"],
                "max_drawdown": stats["max_drawdown"],
                "days": stats["nb_days"],
                "days_in_position": stats["days_in_position"],
                "pct_in_market": stats["pct_in_market"],
                "long_vol_pnl": stats["long_vol_pnl"],
                "short_vol_pnl": stats["short_vol_pnl"],
            }
        )
    return pd.DataFrame(rows)


def compute_worst_drawdown_periods(
    equity: pd.Series,
    dates: pd.Series | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """Identify the largest underwater periods in an equity curve."""
    if top_n <= 0:
        raise ValueError("top_n doit être strictement positif.")

    equity_float = equity.astype(float).reset_index(drop=True)
    drawdown = compute_drawdown(equity_float)
    if dates is None:
        date_values = pd.Series(equity.index)
    else:
        date_values = pd.to_datetime(dates).reset_index(drop=True)

    underwater = drawdown < 0
    periods: list[dict[str, object]] = []
    start_idx: int | None = None

    for idx, is_underwater in enumerate(underwater):
        if is_underwater and start_idx is None:
            start_idx = idx
        recovered = start_idx is not None and (not is_underwater or idx == len(underwater) - 1)
        if recovered:
            end_idx = idx if not is_underwater else idx
            segment = drawdown.iloc[start_idx : end_idx + 1]
            trough_idx = int(segment.idxmin())
            periods.append(
                {
                    "start_date": date_values.iloc[start_idx],
                    "trough_date": date_values.iloc[trough_idx],
                    "end_date": date_values.iloc[end_idx],
                    "max_drawdown": float(segment.min()),
                    "duration_days": int(end_idx - start_idx + 1),
                }
            )
            start_idx = None

    if not periods:
        return pd.DataFrame(
            columns=["start_date", "trough_date", "end_date", "max_drawdown", "duration_days"]
        )

    return (
        pd.DataFrame(periods)
        .sort_values("max_drawdown", ascending=True)
        .head(top_n)
        .reset_index(drop=True)
    )


def _require_columns(df: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {sorted(missing)}")
