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
    df = df.copy()

    if price_col not in df.columns:
        raise ValueError(f"Colonne '{price_col}' absente du DataFrame.")

    df["log_ret"] = np.log(df[price_col] / df[price_col].shift(1))

    for w in windows:
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
    Ajoute une vol réalisée *future* sur 'window' jours :
    à la date t, rv_fwd est calculée sur les rendements t+1 ... t+window.

    Utile comme RV dans un payoff type variance swap
    (on connaît IV_t, et RV_fwd(t) est la réalisation future).
    """
    df = df.copy()

    if price_col not in df.columns:
        raise ValueError(f"Colonne '{price_col}' absente du DataFrame.")

    log_ret = np.log(df[price_col] / df[price_col].shift(1))

    # On veut une fenêtre "forward": tu peux simplement décaler la série
    rolling_std_fwd = (
        log_ret[::-1].rolling(window).std()[::-1]
    )  # reverse / rolling / reverse trick

    rv_fwd = rolling_std_fwd * np.sqrt(trading_days_per_year)
    df[f"rv_fwd_{window}d"] = rv_fwd
    df[f"rv_fwd_{window}d_pct"] = rv_fwd * 100.0

    return df
