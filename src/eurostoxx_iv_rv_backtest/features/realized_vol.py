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
    Ajoute des colonnes de volatilité réalisée (RV) à un DataFrame de prix.

    Paramètres
    ----------
    df : DataFrame
        Doit contenir au minimum une colonne `price_col`.
    price_col : str
        Nom de la colonne de prix de clôture (par défaut "close").
    windows : séquence d'int
        Fenêtres (en jours) pour lesquelles on calcule la RV (ex: [20, 30]).
    trading_days_per_year : int
        Nombre de jours de bourse pour annualiser la vol (252 par défaut).

    Retour
    ------
    DataFrame
        Le DataFrame d'entrée, avec en plus :
        - log_ret
        - rv_<N>d (décimal, ex: 0.20 pour 20 %)
        - rv_<N>d_pct (en %)
    """
    df = df.copy()

    if price_col not in df.columns:
        raise ValueError(f"Colonne '{price_col}' absente du DataFrame.")

    # 1) Log-rendements journaliers
    df["log_ret"] = np.log(df[price_col] / df[price_col].shift(1))

    # 2) RV pour chaque fenêtre demandée
    for w in windows:
        col_rv = f"rv_{w}d"
        col_rv_pct = f"rv_{w}d_pct"

        rolling_std = df["log_ret"].rolling(w).std()
        df[col_rv] = rolling_std * np.sqrt(trading_days_per_year)
        df[col_rv_pct] = df[col_rv] * 100.0

    return df
