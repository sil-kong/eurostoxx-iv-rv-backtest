# src/eurostoxx_iv_rv_backtest/features/iv_rv_variance_swap.py

import pandas as pd


def backtest_iv_rv_variance_swap(
    df: pd.DataFrame,
    iv_col: str = "iv",
    rv_fwd_col: str = "rv_fwd_20d",
    signal_col: str = "signal_vol",
    notional: float = 1.0,
) -> pd.DataFrame:
    """
    Compute a stylized IV/RV variance-swap diagnostic payoff.

    Formula:
        PnL_t = notional * signal_t * (RV_fwd_t^2 - IV_t^2)

    Conventions:
        - IV_t and RV_fwd_t are annualized decimal volatilities, e.g. 0.20.
        - signal_t is -1 / 0 / +1 for short / flat / long volatility.
        - Missing IV or forward RV produces zero PnL for that row.

    This is a normalized, fee-free, hold-to-expiry-style diagnostic payoff. It
    has no mark-to-market dynamics, no maturity term structure, no transaction
    costs and no desk-level variance-swap replication. It is useful for testing
    whether the signal is aligned with subsequent realized variance, not for
    claiming tradable PnL.
    """

    df = df.copy()

    for col in (iv_col, rv_fwd_col, signal_col):
        if col not in df.columns:
            raise ValueError(f"Colonne manquante pour le backtest : {col}")

    iv = df[iv_col].astype(float)
    rv_fwd = df[rv_fwd_col].astype(float)
    signal = df[signal_col].fillna(0.0).astype(float)

    # Variances
    var_iv = iv**2
    var_rv = rv_fwd**2

    # Série PnL initialisée à 0
    pnl = pd.Series(0.0, index=df.index)

    # On ne trade que là où IV et RV_fwd existent
    mask = iv.notna() & rv_fwd.notna()
    pnl.loc[mask] = notional * signal.loc[mask] * (var_rv.loc[mask] - var_iv.loc[mask])

    df["pnl_varswap"] = pnl
    df["equity_varswap"] = pnl.cumsum()

    return df
