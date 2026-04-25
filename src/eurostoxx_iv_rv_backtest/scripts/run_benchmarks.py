from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eurostoxx_iv_rv_backtest.backtesting.benchmarks import run_benchmark_backtests
from eurostoxx_iv_rv_backtest.config import OUTPUTS


def main() -> None:
    """Run IV/RV signal benchmarks against naive volatility exposures."""
    args = _parse_args()
    input_path = Path(args.input)
    comparison_path = Path(args.comparison_output)
    equity_path = Path(args.equity_output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing input file: {input_path}\n"
            "Run eurostoxx-build-signals before eurostoxx-run-benchmarks."
        )

    df = pd.read_csv(input_path, parse_dates=["date"]).sort_values("date")
    comparison, equity_curves = run_benchmark_backtests(
        df,
        date_col="date",
        iv_col=args.iv_col,
        rv_fwd_col=args.rv_fwd_col,
        signal_col=args.signal_col,
    )

    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    equity_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)
    equity_curves.to_csv(equity_path, index=False)

    print(f"Benchmark comparison exported to: {comparison_path}")
    print(f"Benchmark equity curves exported to: {equity_path}")
    print(comparison)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare IV/RV signal payoff with naive volatility benchmarks."
    )
    parser.add_argument(
        "--input",
        default=str(OUTPUTS / "SXE50_with_IV_RV_daily_20y_with_signals.csv"),
        help="Signal-ready CSV produced by eurostoxx-build-signals.",
    )
    parser.add_argument("--iv-col", default="iv", help="Annualized decimal IV column.")
    parser.add_argument(
        "--rv-fwd-col",
        default="rv_fwd_20d",
        help="Forward realized volatility column used for ex-post payoff.",
    )
    parser.add_argument(
        "--signal-col",
        default="signal_vol",
        help="IV/RV signal column to compare against benchmarks.",
    )
    parser.add_argument(
        "--comparison-output",
        default=str(OUTPUTS / "benchmark_comparison.csv"),
        help="Output CSV for benchmark summary metrics.",
    )
    parser.add_argument(
        "--equity-output",
        default=str(OUTPUTS / "benchmark_equity_curves.csv"),
        help="Output CSV for cumulative benchmark equity curves.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
