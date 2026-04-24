# src/eurostoxx_iv_rv_backtest/features/realized_vol.py

from typing import Sequence

import numpy as np
import pandas as pd


def add_realized_vol(
    df: pd.DataFrame,
    price_col: str = "close",
    windows: Sequence[int] = (20, 30),
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """
    Add historical realized volatility columns from daily close prices.

    Log-return convention:
        log_ret_t = log(S_t / S_{t-1})

    At date t, ``rv_{w}d`` is the annualized rolling standard deviation of the
    latest ``w`` log-returns ending at t. It is therefore observable after the
    close at t and can be used by same-date signal construction in this
    stylized research pipeline.
    """
    df = df.copy()

    if price_col not in df.columns:
        raise ValueError(f"Colonne '{price_col}' absente du DataFrame.")
    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year doit être strictement positif.")

    df["log_ret"] = np.log(df[price_col] / df[price_col].shift(1))

    for w in windows:
        if w <= 0:
            raise ValueError("Les fenêtres de RV doivent être strictement positives.")

        col_rv = f"rv_{w}d"
        col_rv_pct = f"rv_{w}d_pct"

        rolling_std = df["log_ret"].rolling(w).std()
        df[col_rv] = rolling_std * np.sqrt(trading_days_per_year)
        df[col_rv_pct] = df[col_rv] * 100.0

    return df


def add_forward_realized_vol(
    df: pd.DataFrame,
    price_col: str = "close",
    window: int = 20,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """
    Add ex-post forward realized volatility over strictly future returns.

    Log-return convention:
        log_ret_t = log(S_t / S_{t-1})

    At index i, corresponding to date t, ``rv_fwd_{window}d`` is computed from:
        log_ret.iloc[i + 1 : i + 1 + window]

    Example with ``window=3``:
        rv_fwd_3d at date t uses returns from t+1, t+2 and t+3.
        It does not use log_ret_t.

    This column is for ex-post payoff evaluation only. It must not be used as
    an input to same-date signal construction.
    """
    df = df.copy()

    if price_col not in df.columns:
        raise ValueError(f"Colonne '{price_col}' absente du DataFrame.")
    if window <= 0:
        raise ValueError("window doit être strictement positif.")
    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year doit être strictement positif.")

    log_ret = np.log(df[price_col] / df[price_col].shift(1))

    # Explicit no-look-ahead convention:
    # rolling std ending at i+window, shifted back by window rows, so the value
    # stored at i uses log_ret[i+1] ... log_ret[i+window].
    rolling_std_fwd = log_ret.rolling(window).std().shift(-window)

    rv_fwd = rolling_std_fwd * np.sqrt(trading_days_per_year)
    df[f"rv_fwd_{window}d"] = rv_fwd
    df[f"rv_fwd_{window}d_pct"] = rv_fwd * 100.0

    return df
