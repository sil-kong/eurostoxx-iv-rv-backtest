"""Reusable analytics for stylized volatility backtests."""

from eurostoxx_iv_rv_backtest.analytics.figures import (
    plot_drawdown_curve,
    plot_equity_curve,
    plot_iv_vs_rv,
    plot_robustness,
    plot_yearly_pnl,
    plot_zscore_signal,
)
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
    "plot_drawdown_curve",
    "plot_equity_curve",
    "plot_iv_vs_rv",
    "plot_robustness",
    "plot_robustness_heatmap",
    "plot_yearly_pnl",
    "plot_zscore_signal",
    "run_robustness_grid",
]
