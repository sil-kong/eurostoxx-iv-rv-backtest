from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from eurostoxx_iv_rv_backtest.analytics.performance import compute_drawdown

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "eurostoxx_iv_rv_mpl"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
_XDG_CACHE_HOME = _MPLCONFIGDIR / "xdg"
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_iv_vs_rv(df: pd.DataFrame, output_path: Path) -> None:
    """Plot VSTOXX implied volatility versus 20-day realized volatility."""
    _require_columns(df, {"date", "iv", "rv_20d"})
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(df["date"], df["iv"] * 100.0, label="VSTOXX proxy IV", linewidth=1.4)
    ax.plot(df["date"], df["rv_20d"] * 100.0, label="20d realized vol", linewidth=1.2)
    ax.set_title("Euro STOXX 50 implied vs realized volatility")
    ax.set_ylabel("Annualized volatility (%)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, output_path)


def plot_zscore_signal(df: pd.DataFrame, output_path: Path, z_entry: float = 0.5) -> None:
    """Plot IV/RV z-score with long/short threshold lines."""
    _require_columns(df, {"date", "iv_rv_zscore", "signal_vol"})
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(df["date"], df["iv_rv_zscore"], label="IV-RV z-score", linewidth=1.2)
    ax.axhline(z_entry, color="red", linestyle="--", linewidth=1.0, label="short-vol threshold")
    ax.axhline(-z_entry, color="blue", linestyle="--", linewidth=1.0, label="long-vol threshold")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("IV/RV z-score and signal thresholds")
    ax.set_ylabel("z-score")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, output_path)


def plot_equity_curve(df: pd.DataFrame, output_path: Path) -> None:
    """Plot cumulative normalized stylized daily payoff."""
    _require_columns(df, {"date", "equity_varswap"})
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(df["date"], df["equity_varswap"], color="#1f77b4", linewidth=1.6)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("Stylized IV/RV variance payoff equity curve")
    ax.set_ylabel("Cumulative normalized PnL")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.25)
    _save(fig, output_path)


def plot_drawdown_curve(df: pd.DataFrame, output_path: Path) -> None:
    """Plot absolute drawdown of the stylized daily payoff equity curve."""
    _require_columns(df, {"date", "equity_varswap"})
    fig, ax = plt.subplots(figsize=(11, 5.5))
    drawdown = compute_drawdown(df["equity_varswap"])
    ax.fill_between(df["date"], drawdown, 0.0, color="#d62728", alpha=0.35)
    ax.plot(df["date"], drawdown, color="#8c1d18", linewidth=1.0)
    ax.set_title("Stylized IV/RV payoff drawdown")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.25)
    _save(fig, output_path)


def plot_yearly_pnl(yearly: pd.DataFrame, output_path: Path) -> None:
    """Plot yearly total normalized PnL as a bar chart."""
    _require_columns(yearly, {"year", "total_pnl"})
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#1f77b4" if value >= 0 else "#d62728" for value in yearly["total_pnl"]]
    ax.bar(yearly["year"].astype(str), yearly["total_pnl"], color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Yearly stylized IV/RV payoff")
    ax.set_ylabel("Normalized PnL")
    ax.set_xlabel("Year")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_path)


def plot_robustness(grid: pd.DataFrame, output_path: Path) -> None:
    """Plot the robustness Sharpe heatmap panels."""
    from eurostoxx_iv_rv_backtest.analytics.robustness import plot_robustness_heatmap

    plot_robustness_heatmap(grid, output_path, metric="sharpe")


def _require_columns(df: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la figure: {sorted(missing)}")


def _save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
