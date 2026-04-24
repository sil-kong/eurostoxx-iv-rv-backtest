from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from eurostoxx_iv_rv_backtest.analytics.performance import compute_summary_stats
from eurostoxx_iv_rv_backtest.features.iv_rv_variance_swap import (
    backtest_iv_rv_variance_swap,
)
from eurostoxx_iv_rv_backtest.features.realized_vol import (
    add_forward_realized_vol,
    add_realized_vol,
)
from eurostoxx_iv_rv_backtest.scripts.build_signals import add_iv_rv_signal

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "eurostoxx_iv_rv_mpl"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
_XDG_CACHE_HOME = _MPLCONFIGDIR / "xdg"
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def run_robustness_grid(
    df: pd.DataFrame,
    z_entries: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
    lookbacks: Sequence[int] = (63, 126, 252, 504),
    horizons: Sequence[int] = (10, 20, 30),
    price_col: str = "close",
    iv_col: str = "iv",
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """
    Run IV/RV signal sensitivity across thresholds, lookbacks and horizons.

    This is a robustness diagnostic, not parameter optimization. Each grid point
    recomputes realized volatility and forward realized volatility using the
    same no-look-ahead convention as the main pipeline.
    """
    required = {price_col, iv_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la robustesse: {sorted(missing)}")

    rows: list[dict[str, float]] = []
    base = df.copy().sort_values("date").reset_index(drop=True) if "date" in df else df.copy()

    for horizon in horizons:
        feature_df = add_realized_vol(
            base,
            price_col=price_col,
            windows=(horizon,),
            trading_days_per_year=trading_days_per_year,
        )
        feature_df = add_forward_realized_vol(
            feature_df,
            price_col=price_col,
            window=horizon,
            trading_days_per_year=trading_days_per_year,
        )
        rv_col = f"rv_{horizon}d"
        rv_fwd_col = f"rv_fwd_{horizon}d"

        for lookback in lookbacks:
            for z_entry in z_entries:
                signal_df = add_iv_rv_signal(
                    feature_df,
                    iv_col=iv_col,
                    rv_col=rv_col,
                    lookback=lookback,
                    z_entry=z_entry,
                )
                bt = backtest_iv_rv_variance_swap(
                    signal_df,
                    iv_col=iv_col,
                    rv_fwd_col=rv_fwd_col,
                    signal_col="signal_vol",
                    notional=1.0,
                )
                stats = compute_summary_stats(
                    bt["pnl_varswap"],
                    bt["equity_varswap"],
                    bt["signal_vol"],
                    trading_days_per_year=trading_days_per_year,
                )
                rows.append(
                    {
                        "horizon": float(horizon),
                        "lookback": float(lookback),
                        "z_entry": float(z_entry),
                        "total_pnl": stats["total_pnl"],
                        "annualized_mean": stats["annualized_mean"],
                        "annualized_vol": stats["annualized_vol"],
                        "sharpe": stats["sharpe"],
                        "max_drawdown": stats["max_drawdown"],
                        "days_in_position": stats["days_in_position"],
                        "pct_in_market": stats["pct_in_market"],
                        "long_vol_pnl": stats["long_vol_pnl"],
                        "short_vol_pnl": stats["short_vol_pnl"],
                    }
                )

    return pd.DataFrame(rows)


def plot_robustness_heatmap(
    grid: pd.DataFrame,
    output_path: Path,
    metric: str = "sharpe",
) -> None:
    """Plot one heatmap panel per horizon for a selected robustness metric."""
    required = {"horizon", "lookback", "z_entry", metric}
    missing = required.difference(grid.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la heatmap: {sorted(missing)}")

    horizons = sorted(grid["horizon"].dropna().unique())
    ncols = len(horizons)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.5), squeeze=False)
    values = grid[metric].replace([np.inf, -np.inf], np.nan)
    finite_values = values[np.isfinite(values)]
    vmin = float(finite_values.min()) if len(finite_values) else None
    vmax = float(finite_values.max()) if len(finite_values) else None

    for ax, horizon in zip(axes[0], horizons):
        subset = grid.loc[grid["horizon"] == horizon]
        pivot = subset.pivot(index="lookback", columns="z_entry", values=metric).sort_index()
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"Horizon {int(horizon)}d")
        ax.set_xlabel("z_entry")
        ax.set_ylabel("lookback")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{value:g}" for value in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{int(value)}" for value in pivot.index])

    fig.suptitle(f"Robustness heatmap: {metric}")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
