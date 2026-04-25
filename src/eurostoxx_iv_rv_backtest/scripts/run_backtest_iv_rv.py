# src/eurostoxx_iv_rv_backtest/scripts/run_backtest_iv_rv.py

import argparse

import pandas as pd

from eurostoxx_iv_rv_backtest.config import OUTPUTS
from eurostoxx_iv_rv_backtest.features.iv_rv_variance_swap import (
    backtest_iv_rv_variance_swap,
)


def main() -> None:
    _parse_args()
    csv_path = OUTPUTS / "SXE50_with_IV_RV_daily_20y_with_signals.csv"
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")

    df_bt = backtest_iv_rv_variance_swap(
        df,
        iv_col="iv",
        rv_fwd_col="rv_fwd_20d",
        signal_col="signal_vol",
        notional=1.0,
    )

    print(
        df_bt[["date", "iv", "rv_20d", "rv_fwd_20d", "signal_vol", "pnl_varswap"]].tail(
            10
        )
    )
    print("\nEquity final :", df_bt["equity_varswap"].iloc[-1])

    out_path = OUTPUTS / "SXE50_iv_rv_varswap_backtest.csv"
    df_bt.to_csv(out_path, index=False)
    print(f"\n✅ Backtest sauvegardé dans : {out_path}")
    print("\n=== NaN check ===")
    print(
        df_bt[["iv", "rv_fwd_20d", "signal_vol", "pnl_varswap", "equity_varswap"]]
        .isna()
        .sum()
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the daily stylized IV/RV variance-payoff diagnostic."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
