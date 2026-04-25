from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eurostoxx_iv_rv_backtest.config import OUTPUTS
from eurostoxx_iv_rv_backtest.validation.walk_forward import run_walk_forward_validation


def main() -> None:
    """Run walk-forward parameter selection and out-of-sample diagnostics."""
    args = _parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing input file: {input_path}\n"
            "Run eurostoxx-build-rv before eurostoxx-run-walk-forward."
        )

    df = pd.read_csv(input_path, parse_dates=["date"]).sort_values("date")
    results, selected_params, equity = run_walk_forward_validation(
        df,
        train_years=args.train_years,
        test_years=args.test_years,
        z_entry_grid=_parse_float_grid(args.z_entry_grid),
        lookback_grid=_parse_int_grid(args.lookback_grid),
        horizon_grid=_parse_int_grid(args.horizon_grid),
        selection_metric=args.selection_metric,
        min_days_in_position=args.min_days_in_position,
    )

    results_path = Path(args.results_output)
    selected_path = Path(args.selected_output)
    equity_path = Path(args.equity_output)
    for output_path, frame in (
        (results_path, results),
        (selected_path, selected_params),
        (equity_path, equity),
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)

    print(f"Walk-forward results exported to: {results_path}")
    print(f"Selected parameters exported to: {selected_path}")
    print(f"Walk-forward equity exported to: {equity_path}")
    print(results)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run walk-forward validation for IV/RV signal parameters."
    )
    parser.add_argument(
        "--input",
        default=str(OUTPUTS / "SXE50_with_IV_RV_daily_20y.csv"),
        help="Base RV CSV produced by eurostoxx-build-rv.",
    )
    parser.add_argument("--train-years", type=int, default=5)
    parser.add_argument("--test-years", type=int, default=1)
    parser.add_argument("--z-entry-grid", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--lookback-grid", default="63,126,252,504")
    parser.add_argument("--horizon-grid", default="10,20,30")
    parser.add_argument(
        "--selection-metric",
        default="sharpe_approx",
        choices=["sharpe_approx", "sharpe", "total_pnl"],
    )
    parser.add_argument("--min-days-in-position", type=int, default=20)
    parser.add_argument(
        "--results-output",
        default=str(OUTPUTS / "walk_forward_results.csv"),
    )
    parser.add_argument(
        "--selected-output",
        default=str(OUTPUTS / "walk_forward_selected_params.csv"),
    )
    parser.add_argument(
        "--equity-output",
        default=str(OUTPUTS / "walk_forward_equity.csv"),
    )
    return parser.parse_args()


def _parse_float_grid(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _parse_int_grid(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


if __name__ == "__main__":
    main()
