from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_non_overlapping_varswap(
    df: pd.DataFrame,
    date_col: str = "date",
    iv_col: str = "iv",
    rv_fwd_col: str = "rv_fwd_20d",
    signal_col: str = "signal_vol",
    horizon: int = 20,
    notional: float = 1.0,
    cost_per_trade: float = 0.0,
) -> pd.DataFrame:
    """
    Backtest a stylized non-overlapping variance payoff layer.

    Convention:
        - At entry date t, if ``signal_t != 0``, open one trade.
        - Hold for exactly ``horizon`` rows/trading days.
        - Do not open another trade until the previous one has matured.
        - Book the payoff at the exit row, not at the entry row.
        - Payoff = notional * signal_t * (RV_realized^2 - IV_t^2).

    ``rv_fwd_col`` is the ex-post realized volatility over the same horizon,
    computed with the repository's no-look-ahead convention. This module uses it
    only for maturity payoff evaluation, never for signal construction.

    ``cost_per_trade`` is a simple normalized friction deducted once at entry and
    booked together with the maturity payoff. It is not a full execution model.
    """
    if horizon <= 0:
        raise ValueError("horizon doit être strictement positif.")
    if cost_per_trade < 0:
        raise ValueError("cost_per_trade doit être positif ou nul.")

    required = {date_col, iv_col, rv_fwd_col, signal_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {sorted(missing)}")

    result = df.copy().sort_values(date_col).reset_index(drop=True)
    result[date_col] = pd.to_datetime(result[date_col])

    result["trade_id"] = pd.Series([pd.NA] * len(result), dtype="Int64")
    result["entry_date"] = pd.NaT
    result["exit_date"] = pd.NaT
    result["entry_iv"] = np.nan
    result["entry_signal"] = np.nan
    result["realized_var"] = np.nan
    result["implied_var"] = np.nan
    result["trade_pnl"] = 0.0

    next_entry_idx = 0
    trade_id = 1
    n_rows = len(result)

    while next_entry_idx < n_rows:
        row = result.loc[next_entry_idx]
        signal = float(0.0 if pd.isna(row[signal_col]) else row[signal_col])
        exit_idx = next_entry_idx + horizon

        can_open = (
            signal != 0.0
            and exit_idx < n_rows
            and pd.notna(row[iv_col])
            and pd.notna(row[rv_fwd_col])
        )

        if not can_open:
            next_entry_idx += 1
            continue

        entry_iv = float(row[iv_col])
        realized_vol = float(row[rv_fwd_col])
        implied_var = entry_iv**2
        realized_var = realized_vol**2
        trade_pnl = notional * signal * (realized_var - implied_var) - cost_per_trade

        result.loc[exit_idx, "trade_id"] = trade_id
        result.loc[exit_idx, "entry_date"] = row[date_col]
        result.loc[exit_idx, "exit_date"] = result.loc[exit_idx, date_col]
        result.loc[exit_idx, "entry_iv"] = entry_iv
        result.loc[exit_idx, "entry_signal"] = signal
        result.loc[exit_idx, "realized_var"] = realized_var
        result.loc[exit_idx, "implied_var"] = implied_var
        result.loc[exit_idx, "trade_pnl"] = trade_pnl

        trade_id += 1
        next_entry_idx = exit_idx + 1

    result["cumulative_trade_pnl"] = result["trade_pnl"].cumsum()
    return result
