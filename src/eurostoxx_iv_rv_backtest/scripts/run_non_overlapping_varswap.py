import argparse

import pandas as pd

from eurostoxx_iv_rv_backtest.backtesting.non_overlapping_varswap import (
    backtest_non_overlapping_varswap,
)
from eurostoxx_iv_rv_backtest.config import OUTPUTS


def main() -> None:
    """Run the non-overlapping stylized 20-day variance payoff layer."""
    _parse_args()
    input_path = OUTPUTS / "SXE50_with_IV_RV_daily_20y_with_signals.csv"
    output_path = OUTPUTS / "SXE50_iv_rv_non_overlapping_varswap_backtest.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Fichier d'entrée introuvable : {input_path}\n"
            "Lance d'abord build_signals.py."
        )

    df = pd.read_csv(input_path, parse_dates=["date"]).sort_values("date")
    result = backtest_non_overlapping_varswap(
        df,
        date_col="date",
        iv_col="iv",
        rv_fwd_col="rv_fwd_20d",
        signal_col="signal_vol",
        horizon=20,
        notional=1.0,
    )
    result.to_csv(output_path, index=False)

    trade_rows = result[result["trade_id"].notna()]
    print(trade_rows[["trade_id", "entry_date", "exit_date", "entry_signal", "trade_pnl"]].tail())
    print(f"\nNb trades non-overlapping : {len(trade_rows)}")
    print(f"Cumulative trade PnL      : {result['cumulative_trade_pnl'].iloc[-1]:.3f}")
    print(f"\nBacktest non-overlapping sauvegardé dans : {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the non-overlapping stylized variance-payoff layer."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
