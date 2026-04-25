import argparse

import pandas as pd

from eurostoxx_iv_rv_backtest.analytics.performance import (
    compute_regime_stats,
    compute_summary_stats,
    compute_worst_drawdown_periods,
    compute_yearly_stats,
)
from eurostoxx_iv_rv_backtest.config import OUTPUTS


CRISIS_PERIODS = {
    "2008": ("2008-01-01", "2008-12-31"),
    "2011": ("2011-01-01", "2011-12-31"),
    "2020": ("2020-01-01", "2020-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
}


def main() -> None:
    """Export yearly, regime and drawdown diagnostics for the daily payoff layer."""
    _parse_args()
    input_path = OUTPUTS / "SXE50_iv_rv_varswap_backtest.csv"
    yearly_path = OUTPUTS / "yearly_performance.csv"
    regime_path = OUTPUTS / "regime_performance.csv"
    drawdown_path = OUTPUTS / "drawdown_periods.csv"
    crisis_path = OUTPUTS / "crisis_periods.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Fichier d'entrée introuvable : {input_path}\n"
            "Lance d'abord run_backtest_iv_rv.py."
        )

    df = pd.read_csv(input_path, parse_dates=["date"]).sort_values("date")

    yearly = compute_yearly_stats(df)
    regime = compute_regime_stats(df)
    drawdowns = compute_worst_drawdown_periods(
        df["equity_varswap"],
        dates=df["date"],
        top_n=10,
    )
    crisis = _compute_crisis_periods(df)

    yearly.to_csv(yearly_path, index=False)
    regime.to_csv(regime_path, index=False)
    drawdowns.to_csv(drawdown_path, index=False)
    crisis.to_csv(crisis_path, index=False)

    print(f"Yearly performance exported to: {yearly_path}")
    print(f"Regime performance exported to: {regime_path}")
    print(f"Drawdown periods exported to: {drawdown_path}")
    print(f"Crisis-period diagnostics exported to: {crisis_path}")
    print("\nYearly performance tail:")
    print(yearly.tail())
    print("\nRegime performance:")
    print(regime)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export yearly, regime, drawdown and crisis diagnostics."
    )
    return parser.parse_args()


def _compute_crisis_periods(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for label, (start, end) in CRISIS_PERIODS.items():
        mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        period = df.loc[mask]
        if period.empty:
            continue
        stats = compute_summary_stats(
            period["pnl_varswap"],
            period["pnl_varswap"].cumsum(),
            period["signal_vol"],
        )
        rows.append(
            {
                "period": label,
                "start_date": start,
                "end_date": end,
                "total_pnl": stats["total_pnl"],
                "annualized_mean": stats["annualized_mean"],
                "annualized_vol": stats["annualized_vol"],
                "sharpe": stats["sharpe"],
                "max_drawdown": stats["max_drawdown"],
                "days": stats["nb_days"],
                "days_in_position": stats["days_in_position"],
                "pct_in_market": stats["pct_in_market"],
                "long_vol_pnl": stats["long_vol_pnl"],
                "short_vol_pnl": stats["short_vol_pnl"],
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
