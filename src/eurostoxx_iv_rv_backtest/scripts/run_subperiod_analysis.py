from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eurostoxx_iv_rv_backtest.analytics.subperiods import compute_subperiod_performance
from eurostoxx_iv_rv_backtest.config import OUTPUTS


def main() -> None:
    """Run default subperiod and crisis-period diagnostics."""
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing input file: {input_path}\n"
            "Run eurostoxx-run-varswap-backtest before eurostoxx-run-subperiods."
        )

    df = pd.read_csv(input_path, parse_dates=["date"]).sort_values("date")
    subperiods = compute_subperiod_performance(
        df,
        date_col="date",
        pnl_col=args.pnl_col,
        signal_col=args.signal_col,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subperiods.to_csv(output_path, index=False)

    print(f"Subperiod performance exported to: {output_path}")
    print(subperiods)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute IV/RV payoff diagnostics across named subperiods."
    )
    parser.add_argument(
        "--input",
        default=str(OUTPUTS / "SXE50_iv_rv_varswap_backtest.csv"),
        help="Daily backtest CSV produced by eurostoxx-run-varswap-backtest.",
    )
    parser.add_argument("--pnl-col", default="pnl_varswap")
    parser.add_argument("--signal-col", default="signal_vol")
    parser.add_argument(
        "--output",
        default=str(OUTPUTS / "subperiod_performance.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
