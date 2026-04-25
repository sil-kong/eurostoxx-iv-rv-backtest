from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from eurostoxx_iv_rv_backtest.analytics.performance import compute_summary_stats
from eurostoxx_iv_rv_backtest.config import OUTPUTS
from eurostoxx_iv_rv_backtest.features.iv_rv_variance_swap import (
    backtest_iv_rv_variance_swap,
)


DEFAULT_COST_LEVELS = (0.0, 0.0001, 0.0005, 0.001, 0.0025, 0.005)


def run_cost_sensitivity(
    df: pd.DataFrame,
    cost_levels: Sequence[float] = DEFAULT_COST_LEVELS,
    iv_col: str = "iv",
    rv_fwd_col: str = "rv_fwd_20d",
    signal_col: str = "signal_vol",
) -> pd.DataFrame:
    """Run the stylized daily payoff under a grid of signal-change costs."""
    required = {iv_col, rv_fwd_col, signal_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    rows: list[dict[str, float]] = []
    signal = df[signal_col].fillna(0.0).astype(float)
    valid = df[iv_col].notna() & df[rv_fwd_col].notna()
    signal_delta = (signal - signal.shift(1).fillna(0.0)).abs()
    turnover_proxy = float(signal_delta.loc[valid].sum())
    number_signal_changes = float((signal_delta.loc[valid] > 0).sum())

    for cost in cost_levels:
        bt = backtest_iv_rv_variance_swap(
            df,
            iv_col=iv_col,
            rv_fwd_col=rv_fwd_col,
            signal_col=signal_col,
            notional=1.0,
            cost_per_signal_change=float(cost),
        )
        stats = compute_summary_stats(
            bt["pnl_varswap"],
            bt["equity_varswap"],
            bt[signal_col],
        )
        rows.append(
            {
                "cost_per_signal_change": float(cost),
                "total_pnl": stats["total_pnl"],
                "sharpe_approx": stats["sharpe_approx"],
                "max_drawdown": stats["max_drawdown"],
                "days_in_position": stats["days_in_position"],
                "turnover_proxy": turnover_proxy,
                "number_signal_changes": number_signal_changes,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Run stylized transaction-cost sensitivity diagnostics."""
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing input file: {input_path}\n"
            "Run eurostoxx-build-signals before eurostoxx-run-cost-sensitivity."
        )

    df = pd.read_csv(input_path, parse_dates=["date"]).sort_values("date")
    sensitivity = run_cost_sensitivity(
        df,
        cost_levels=_parse_float_grid(args.cost_levels),
        iv_col=args.iv_col,
        rv_fwd_col=args.rv_fwd_col,
        signal_col=args.signal_col,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(output_path, index=False)

    print(f"Cost sensitivity exported to: {output_path}")
    print(
        "Costs are stylized deductions per unit of absolute signal change; "
        "they are not bid/ask or execution costs."
    )
    print(sensitivity)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stylized signal-change cost sensitivity diagnostics."
    )
    parser.add_argument(
        "--input",
        default=str(OUTPUTS / "SXE50_with_IV_RV_daily_20y_with_signals.csv"),
    )
    parser.add_argument("--iv-col", default="iv")
    parser.add_argument("--rv-fwd-col", default="rv_fwd_20d")
    parser.add_argument("--signal-col", default="signal_vol")
    parser.add_argument("--cost-levels", default="0.0,0.0001,0.0005,0.001,0.0025,0.005")
    parser.add_argument("--output", default=str(OUTPUTS / "cost_sensitivity.csv"))
    return parser.parse_args()


def _parse_float_grid(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


if __name__ == "__main__":
    main()
