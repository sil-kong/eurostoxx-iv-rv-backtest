"""Reusable analytics for stylized volatility backtests."""

from eurostoxx_iv_rv_backtest.analytics.performance import (
    compute_drawdown,
    compute_regime_stats,
    compute_summary_stats,
    compute_worst_drawdown_periods,
    compute_yearly_stats,
)
from eurostoxx_iv_rv_backtest.analytics.robustness import (
    plot_robustness_heatmap,
    run_robustness_grid,
)

__all__ = [
    "compute_drawdown",
    "compute_regime_stats",
    "compute_summary_stats",
    "compute_worst_drawdown_periods",
    "compute_yearly_stats",
    "plot_robustness_heatmap",
    "run_robustness_grid",
]
