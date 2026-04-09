# src/eurostoxx_iv_rv_backtest/scripts/analyze_backtest.py

import numpy as np
import pandas as pd

from eurostoxx_iv_rv_backtest.config import OUTPUTS


def main() -> None:
    csv_path = OUTPUTS / "SXE50_iv_rv_varswap_backtest.csv"
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")

    # On garde seulement les lignes où tout est défini
    df = df.dropna(subset=["pnl_varswap", "equity_varswap", "signal_vol"])

    # PnL journalier
    pnl = df["pnl_varswap"]
    equity = df["equity_varswap"]

    # 1) PnL total
    total_pnl = float(equity.iloc[-1])

    # 2) PnL moyen et annualisé
    mean_daily = float(pnl.mean())
    std_daily = float(pnl.std(ddof=1))
    ann_factor = np.sqrt(252)

    ann_return = mean_daily * 252
    ann_vol = std_daily * ann_factor
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    # 3) Max drawdown
    roll_max = equity.cummax()
    drawdown = equity - roll_max
    max_dd = float(drawdown.min())

    # 4) Temps en position
    nb_days = len(df)
    nb_days_in_market = int((df["signal_vol"] != 0).sum())
    pct_in_market = 100.0 * nb_days_in_market / nb_days

    # 5) Stats par régime
    pnl_long_vol = pnl[df["signal_vol"] == 1].sum()
    pnl_short_vol = pnl[df["signal_vol"] == -1].sum()

    print("=== Résumé backtest IV vs RV (variance swap) ===\n")
    print(f"Total PnL       : {total_pnl:.3f}")
    print(f"Annualisé (moy) : {ann_return:.4f}")
    print(f"Annualisé (vol) : {ann_vol:.4f}")
    print(f"Sharpe approx   : {sharpe:.2f}")
    print(f"Max drawdown    : {max_dd:.3f}")
    print()
    print(f"Nb jours       : {nb_days}")
    print(f"Nb jours en position : {nb_days_in_market} ({pct_in_market:.1f} %)")
    print()
    print(f"PnL long vol  : {pnl_long_vol:.3f}")
    print(f"PnL short vol : {pnl_short_vol:.3f}")


if __name__ == "__main__":
    main()
