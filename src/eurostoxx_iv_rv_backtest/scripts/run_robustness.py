import pandas as pd

from eurostoxx_iv_rv_backtest.analytics.robustness import (
    plot_robustness_heatmap,
    run_robustness_grid,
)
from eurostoxx_iv_rv_backtest.config import OUTPUTS


def main() -> None:
    """Run IV/RV parameter robustness diagnostics and export CSV/heatmap outputs."""
    input_path = OUTPUTS / "SXE50_with_IV_RV_daily_20y.csv"
    grid_path = OUTPUTS / "robustness_grid.csv"
    figure_path = OUTPUTS / "figures" / "robustness_heatmap.png"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Fichier d'entrée introuvable : {input_path}\n"
            "Lance d'abord build_rv.py."
        )

    df = pd.read_csv(input_path, parse_dates=["date"]).sort_values("date")
    grid = run_robustness_grid(
        df,
        z_entries=(0.5, 1.0, 1.5, 2.0),
        lookbacks=(63, 126, 252, 504),
        horizons=(10, 20, 30),
    )
    grid.to_csv(grid_path, index=False)
    plot_robustness_heatmap(grid, figure_path, metric="sharpe")

    print(f"Robustness grid exported to: {grid_path}")
    print(f"Robustness heatmap exported to: {figure_path}")
    print(grid.sort_values("sharpe", ascending=False).head(10))


if __name__ == "__main__":
    main()
