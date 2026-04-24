import pandas as pd

from eurostoxx_iv_rv_backtest.analytics.figures import (
    plot_drawdown_curve,
    plot_equity_curve,
    plot_iv_vs_rv,
    plot_robustness,
    plot_yearly_pnl,
    plot_zscore_signal,
)
from eurostoxx_iv_rv_backtest.analytics.performance import compute_yearly_stats
from eurostoxx_iv_rv_backtest.analytics.robustness import run_robustness_grid
from eurostoxx_iv_rv_backtest.config import OUTPUTS


def main() -> None:
    """Generate static research figures under outputs/figures."""
    figures_dir = OUTPUTS / "figures"
    signals_path = OUTPUTS / "SXE50_with_IV_RV_daily_20y_with_signals.csv"
    rv_path = OUTPUTS / "SXE50_with_IV_RV_daily_20y.csv"
    backtest_path = OUTPUTS / "SXE50_iv_rv_varswap_backtest.csv"
    yearly_path = OUTPUTS / "yearly_performance.csv"
    robustness_path = OUTPUTS / "robustness_grid.csv"

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
        "drawdown_curve": figures_dir / "drawdown_curve.png",
        "yearly_pnl": figures_dir / "yearly_pnl.png",
        "robustness_heatmap": figures_dir / "robustness_heatmap.png",
    }

    plot_iv_vs_rv(rv, outputs["iv_vs_rv"])
    plot_zscore_signal(signals, outputs["zscore_signal"])
    plot_equity_curve(backtest, outputs["equity_curve"])
    plot_drawdown_curve(backtest, outputs["drawdown_curve"])
    plot_yearly_pnl(yearly, outputs["yearly_pnl"])
    plot_robustness(robustness, outputs["robustness_heatmap"])

    print("Generated figures:")
    for output_path in outputs.values():
        print(f"- {output_path}")


if __name__ == "__main__":
    main()
