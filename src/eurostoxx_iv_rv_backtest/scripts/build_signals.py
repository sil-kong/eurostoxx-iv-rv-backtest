# src/eurostoxx_iv_rv_backtest/scripts/build_signals.py

import pandas as pd

from eurostoxx_iv_rv_backtest.config import OUTPUTS
from eurostoxx_iv_rv_backtest.features.realized_vol import add_forward_realized_vol


def add_iv_rv_signal(
    df: pd.DataFrame,
    iv_col: str = "iv",
    rv_col: str = "rv_20d",
    lookback: int = 252,
    z_entry: float = 0.5,
) -> pd.DataFrame:
    """
    Ajoute :
      - iv_minus_rv = iv - rv
      - iv_rv_zscore = (iv_minus_rv - moyenne) / sigma sur 'lookback' jours
      - signal_vol :
          +1 = long vol (IV sous-évalue la RV)
          -1 = short vol (IV surévalue la RV)
           0 = neutre (écart limité)
    """

    df = df.copy()

    if iv_col not in df.columns or rv_col not in df.columns:
        raise ValueError("Colonnes IV/RV manquantes pour le signal.")

    # Écart IV - RV (en vol annualisée)
    df["iv_minus_rv"] = df[iv_col] - df[rv_col]

    # Stats glissantes sur l'écart
    rolling_mean = df["iv_minus_rv"].rolling(lookback).mean()
    rolling_std = df["iv_minus_rv"].rolling(lookback).std()

    df["iv_rv_zscore"] = (df["iv_minus_rv"] - rolling_mean) / rolling_std

    # Signal discret : +1 / -1 / 0
    z = df["iv_rv_zscore"]
    signal = pd.Series(0, index=df.index, dtype="int64")
    signal = signal.mask(z > z_entry, -1)  # IV >> RV → short vol
    signal = signal.mask(z < -z_entry, 1)  # IV << RV → long vol

    df["signal_vol"] = signal

    return df


def main() -> None:
    """
    Construit RV forward + signaux IV-RV et écrit :

      outputs/SXE50_with_IV_RV_daily_20y_with_signals.csv
    """

    input_path = OUTPUTS / "SXE50_with_IV_RV_daily_20y.csv"
    output_path = OUTPUTS / "SXE50_with_IV_RV_daily_20y_with_signals.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Fichier d'entrée introuvable : {input_path}\n"
            "Tu dois d'abord lancer build_rv.py."
        )

    print(f">>> Lecture de {input_path}")
    df = (
        pd.read_csv(input_path, parse_dates=["date"])
        .sort_values(by="date")
        .reset_index(drop=True)
    )

    # Sanity check
    required = ("close", "iv", "rv_20d")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Colonnes manquantes pour build_signals : {missing} "
            f"(colonnes dispo = {list(df.columns)})"
        )

    # 1) RV forward 20 jours
    df = add_forward_realized_vol(
        df,
        price_col="close",
        window=20,
        trading_days_per_year=252,
    )

    # 2) Signal IV - RV
    df = add_iv_rv_signal(
        df,
        iv_col="iv",
        rv_col="rv_20d",
        lookback=252,
        z_entry=0.5,
    )

    # Aperçu console
    cols = [
        "date",
        "iv",
        "rv_20d",
        "rv_fwd_20d",
        "iv_minus_rv",
        "iv_rv_zscore",
        "signal_vol",
    ]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].head(10))

    df.to_csv(output_path, index=False)
    print(f"\n✅ Fichier avec RV forward + signaux exporté dans : {output_path}")


if __name__ == "__main__":
    main()
