import argparse

import pandas as pd

from eurostoxx_iv_rv_backtest.analytics.performance import compute_summary_stats
from eurostoxx_iv_rv_backtest.config import OUTPUTS


def main() -> None:
    _parse_args()
    csv_path = OUTPUTS / "SXE50_iv_rv_varswap_backtest.csv"
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")

    stats = compute_summary_stats(
        df["pnl_varswap"],
        df["equity_varswap"],
        df["signal_vol"],
    )

    print("=== Résumé backtest IV vs RV (variance swap) ===\n")
    print(f"Total PnL       : {stats['total_pnl']:.3f}")
    print(f"Annualisé (moy) : {stats['annualized_mean']:.4f}")
    print(f"Annualisé (vol) : {stats['annualized_vol']:.4f}")
    print(f"Sharpe approx   : {stats['sharpe']:.2f}")
    print(f"Max drawdown    : {stats['max_drawdown']:.3f}")
    print()
    print(f"Nb jours       : {stats['nb_days']:.0f}")
    print(
        "Nb jours en position : "
        f"{stats['days_in_position']:.0f} ({stats['pct_in_market']:.1f} %)"
    )
    print()
    print(f"PnL long vol  : {stats['long_vol_pnl']:.3f}")
    print(f"PnL short vol : {stats['short_vol_pnl']:.3f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print summary diagnostics for the daily backtest.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
