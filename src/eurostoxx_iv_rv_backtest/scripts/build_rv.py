# src/eurostoxx_iv_rv_backtest/scripts/build_rv.py

import pandas as pd
from eurostoxx_iv_rv_backtest.config import DATA_RAW, OUTPUTS

from eurostoxx_iv_rv_backtest.features.realized_vol import add_realized_vol


def main() -> None:
    input_path = DATA_RAW / "SXE50_with_IV_daily_20y.csv"
    output_path = OUTPUTS / "SXE50_with_IV_RV_daily_20y.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Fichier d'entrée introuvable : {input_path}\n"
            "Tu as bien lancé data/raw/getdata.py avant ?"
        )

    print(f">>> Chargement du fichier de travail : {input_path}")
    df = pd.read_csv(input_path)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df_rv = add_realized_vol(
        df,
        price_col="close",
        windows=(20, 30),
        trading_days_per_year=252,
    )

    print(df_rv[["date", "close", "iv", "rv_20d", "rv_30d"]].head(10))

    df_rv.to_csv(output_path, index=False)
    print(f"\n✅ Fichier enrichi avec RV exporté dans : {output_path}")


if __name__ == "__main__":
    main()
