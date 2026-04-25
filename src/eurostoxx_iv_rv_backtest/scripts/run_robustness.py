from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eurostoxx_iv_rv_backtest.analytics.robustness import (
    plot_robustness_metric_heatmap,
    plot_robustness_heatmap,
    run_robustness_grid,
)
from eurostoxx_iv_rv_backtest.config import OUTPUTS


def main() -> None:
    """Run IV/RV parameter robustness diagnostics and export CSV/heatmap outputs."""
    args = _parse_args()
    input_path = Path(args.input)
    grid_path = Path(args.output)
    figure_dir = Path(args.figure_dir)
    figure_path = figure_dir / "robustness_heatmap.png"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Fichier d'entrée introuvable : {input_path}\n"
            "Lance d'abord build_rv.py."
        )

    df = pd.read_csv(input_path, parse_dates=["date"]).sort_values("date")
    grid = run_robustness_grid(
        df,
        z_entries=_parse_float_grid(args.z_entry_grid),
        lookbacks=_parse_int_grid(args.lookback_grid),
        horizons=_parse_int_grid(args.horizon_grid),
    )
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(grid_path, index=False)
    plot_robustness_heatmap(grid, figure_path, metric="sharpe")
    plot_robustness_metric_heatmap(
        grid,
        figure_dir / "robustness_sharpe_heatmap.png",
        metric="sharpe",
        horizon=args.horizon,
    )
    plot_robustness_metric_heatmap(
        grid,
        figure_dir / "robustness_total_pnl_heatmap.png",
        metric="total_pnl",
        horizon=args.horizon,
    )
    plot_robustness_metric_heatmap(
        grid,
        figure_dir / "robustness_drawdown_heatmap.png",
        metric="max_drawdown",
        horizon=args.horizon,
    )

    print(f"Robustness grid exported to: {grid_path}")
    print(f"Robustness heatmap exported to: {figure_path}")
    print("\nTop rows by Sharpe, for diagnostic inspection only, not parameter selection:")
    print(grid.sort_values("sharpe", ascending=False).head(10))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run IV/RV parameter robustness diagnostics and heatmaps."
    )
    parser.add_argument(
        "--input",
        default=str(OUTPUTS / "SXE50_with_IV_RV_daily_20y.csv"),
        help="Base RV CSV produced by eurostoxx-build-rv.",
    )
    parser.add_argument("--z-entry-grid", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--lookback-grid", default="63,126,252,504")
    parser.add_argument("--horizon-grid", default="10,20,30")
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Horizon to use for single-metric robustness heatmaps.",
    )
    parser.add_argument("--output", default=str(OUTPUTS / "robustness_grid.csv"))
    parser.add_argument("--figure-dir", default=str(OUTPUTS / "figures"))
    return parser.parse_args()


def _parse_float_grid(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _parse_int_grid(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


if __name__ == "__main__":
    main()
