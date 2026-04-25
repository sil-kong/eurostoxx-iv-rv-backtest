from __future__ import annotations

import argparse
import shutil
import warnings
from pathlib import Path

import pandas as pd

from eurostoxx_iv_rv_backtest.analytics.figures import (
    plot_benchmark_equity_curves,
    plot_drawdown_curve,
    plot_equity_curve,
    plot_iv_vs_rv,
    plot_robustness,
    plot_subperiod_performance,
    plot_walk_forward_equity,
    plot_yearly_pnl,
    plot_zscore_signal,
)
from eurostoxx_iv_rv_backtest.analytics.performance import compute_yearly_stats
from eurostoxx_iv_rv_backtest.analytics.robustness import (
    plot_robustness_metric_heatmap,
    run_robustness_grid,
)
from eurostoxx_iv_rv_backtest.config import DOCS_FIGURES, OUTPUTS


def main() -> None:
    """Generate static research figures under outputs/figures."""
    args = _parse_args()
    figures_dir = Path(args.figure_dir)
    signals_path = OUTPUTS / "SXE50_with_IV_RV_daily_20y_with_signals.csv"
    rv_path = OUTPUTS / "SXE50_with_IV_RV_daily_20y.csv"
    backtest_path = OUTPUTS / "SXE50_iv_rv_varswap_backtest.csv"
    yearly_path = OUTPUTS / "yearly_performance.csv"
    robustness_path = OUTPUTS / "robustness_grid.csv"
    benchmark_equity_path = OUTPUTS / "benchmark_equity_curves.csv"
    subperiod_path = OUTPUTS / "subperiod_performance.csv"
    walk_forward_equity_path = OUTPUTS / "walk_forward_equity.csv"

    if not signals_path.exists():
        raise FileNotFoundError(f"Missing signals file: {signals_path}")
    if not rv_path.exists():
        raise FileNotFoundError(f"Missing RV file: {rv_path}")
    if not backtest_path.exists():
        raise FileNotFoundError(f"Missing backtest file: {backtest_path}")

    signals = pd.read_csv(signals_path, parse_dates=["date"]).sort_values("date")
    rv = pd.read_csv(rv_path, parse_dates=["date"]).sort_values("date")
    backtest = pd.read_csv(backtest_path, parse_dates=["date"]).sort_values("date")

    if yearly_path.exists():
        yearly = pd.read_csv(yearly_path)
    else:
        yearly = compute_yearly_stats(backtest)
        yearly.to_csv(yearly_path, index=False)

    if robustness_path.exists():
        robustness = pd.read_csv(robustness_path)
    else:
        robustness = run_robustness_grid(rv)
        robustness.to_csv(robustness_path, index=False)

    outputs = {
        "iv_vs_rv": figures_dir / "iv_vs_rv.png",
        "zscore_signal": figures_dir / "zscore_signal.png",
        "equity_curve": figures_dir / "equity_curve.png",
        "drawdown": figures_dir / "drawdown.png",
        "drawdown_curve": figures_dir / "drawdown_curve.png",
        "yearly_pnl": figures_dir / "yearly_pnl.png",
        "robustness_heatmap": figures_dir / "robustness_heatmap.png",
        "robustness_sharpe_heatmap": figures_dir / "robustness_sharpe_heatmap.png",
        "robustness_total_pnl_heatmap": figures_dir / "robustness_total_pnl_heatmap.png",
        "robustness_drawdown_heatmap": figures_dir / "robustness_drawdown_heatmap.png",
    }

    plot_iv_vs_rv(rv, outputs["iv_vs_rv"])
    plot_zscore_signal(signals, outputs["zscore_signal"])
    plot_equity_curve(backtest, outputs["equity_curve"])
    plot_drawdown_curve(backtest, outputs["drawdown"])
    plot_drawdown_curve(backtest, outputs["drawdown_curve"])
    plot_yearly_pnl(yearly, outputs["yearly_pnl"])
    plot_robustness(robustness, outputs["robustness_heatmap"])
    plot_robustness_metric_heatmap(
        robustness,
        outputs["robustness_sharpe_heatmap"],
        metric="sharpe",
        horizon=args.horizon,
    )
    plot_robustness_metric_heatmap(
        robustness,
        outputs["robustness_total_pnl_heatmap"],
        metric="total_pnl",
        horizon=args.horizon,
    )
    plot_robustness_metric_heatmap(
        robustness,
        outputs["robustness_drawdown_heatmap"],
        metric="max_drawdown",
        horizon=args.horizon,
    )

    if benchmark_equity_path.exists():
        benchmark_equity = pd.read_csv(benchmark_equity_path, parse_dates=["date"])
        outputs["benchmark_equity_curves"] = figures_dir / "benchmark_equity_curves.png"
        plot_benchmark_equity_curves(benchmark_equity, outputs["benchmark_equity_curves"])
    else:
        warnings.warn(f"Skipping benchmark equity figure; missing {benchmark_equity_path}")

    if subperiod_path.exists():
        subperiods = pd.read_csv(subperiod_path)
        outputs["subperiod_performance"] = figures_dir / "subperiod_performance.png"
        plot_subperiod_performance(subperiods, outputs["subperiod_performance"])
    else:
        warnings.warn(f"Skipping subperiod figure; missing {subperiod_path}")

    if walk_forward_equity_path.exists():
        walk_forward_equity = pd.read_csv(walk_forward_equity_path, parse_dates=["date"])
        outputs["walk_forward_equity"] = figures_dir / "walk_forward_equity.png"
        plot_walk_forward_equity(walk_forward_equity, outputs["walk_forward_equity"])
    else:
        warnings.warn(f"Skipping walk-forward equity figure; missing {walk_forward_equity_path}")

    print("Generated figures:")
    for output_path in outputs.values():
        print(f"- {output_path}")

    if args.publish_docs_figures:
        copied = publish_docs_figures(figures_dir=figures_dir, docs_figures_dir=DOCS_FIGURES)
        print("\nPublished curated docs figures:")
        for output_path in copied:
            print(f"- {output_path}")


def publish_docs_figures(
    figures_dir: Path = OUTPUTS / "figures",
    docs_figures_dir: Path = DOCS_FIGURES,
) -> list[Path]:
    """Copy selected generated figures into docs/figures for GitHub rendering."""
    selected = [
        "equity_curve.png",
        "drawdown.png",
        "benchmark_equity_curves.png",
        "robustness_heatmap.png",
        "robustness_sharpe_heatmap.png",
        "robustness_total_pnl_heatmap.png",
        "robustness_drawdown_heatmap.png",
        "subperiod_performance.png",
        "iv_vs_rv.png",
        "walk_forward_equity.png",
    ]
    docs_figures_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for filename in selected:
        source = figures_dir / filename
        destination = docs_figures_dir / filename
        if not source.exists():
            warnings.warn(f"Skipping docs figure; missing {source}")
            continue
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and optionally publish research figures.")
    parser.add_argument("--figure-dir", default=str(OUTPUTS / "figures"))
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Horizon to use for single-metric robustness heatmaps.",
    )
    parser.add_argument(
        "--publish-docs-figures",
        action="store_true",
        help="Copy selected figures from outputs/figures to docs/figures.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
