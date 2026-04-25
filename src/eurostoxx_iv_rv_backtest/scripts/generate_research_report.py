from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from eurostoxx_iv_rv_backtest.analytics.performance import compute_summary_stats
from eurostoxx_iv_rv_backtest.config import DOCS, OUTPUTS


def main() -> None:
    """Generate Markdown research reports from the current CSV outputs."""
    args = _parse_args()
    report, generation_warnings = build_research_report(outputs_dir=Path(args.outputs_dir))

    output_path = Path(args.output)
    docs_path = Path(args.docs_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    docs_path.write_text(report, encoding="utf-8")

    for message in generation_warnings:
        warnings.warn(message)

    print(f"Research report exported to: {output_path}")
    print(f"Curated docs report exported to: {docs_path}")


def build_research_report(outputs_dir: Path = OUTPUTS) -> tuple[str, list[str]]:
    """Build the Markdown report text and return non-fatal generation warnings."""
    warnings_list: list[str] = []
    files = {
        "daily": outputs_dir / "SXE50_iv_rv_varswap_backtest.csv",
        "non_overlap": outputs_dir / "SXE50_iv_rv_non_overlapping_varswap_backtest.csv",
        "benchmark": outputs_dir / "benchmark_comparison.csv",
        "walk_forward": outputs_dir / "walk_forward_results.csv",
        "selected_params": outputs_dir / "walk_forward_selected_params.csv",
        "subperiod": outputs_dir / "subperiod_performance.csv",
        "robustness": outputs_dir / "robustness_grid.csv",
        "cost": outputs_dir / "cost_sensitivity.csv",
    }
    frames = {
        name: _read_optional_csv(path, warnings_list)
        for name, path in files.items()
    }

    sections = [
        "# Euro STOXX 50 IV/RV Volatility Research Report",
        "",
        "## Research Question",
        (
            "Does the IV/RV spread contain information about future realized variance "
            "beyond naive variance-risk-premium exposure?"
        ),
        "",
        "## Data",
        "- SXE50 / Euro STOXX 50 spot data.",
        "- VSTOXX / V2TX implied volatility proxy data.",
        "- IV is represented as decimal annualized volatility, for example 0.20 for 20%.",
        (
            "- VSTOXX is a broad implied-volatility proxy, not an option chain, "
            "volatility surface or variance-swap quote."
        ),
        "",
        "## Methodology",
        "- Historical RV is computed from close-to-close log returns and annualized.",
        "- Forward RV is strictly ex-post and reserved for payoff evaluation.",
        "- The signal uses the IV/RV spread and a rolling z-score threshold.",
        "- Daily payoff convention: `signal * (RV_fwd^2 - IV^2)`.",
        "- The non-overlapping layer opens at most one fixed-horizon trade at a time.",
        "- No-look-ahead policy: forward RV is never used to build the signal.",
        "",
        "## Main Results",
        _main_results_section(frames["daily"], frames["non_overlap"]),
        "",
        "## Benchmark Comparison",
        _benchmark_section(frames["benchmark"]),
        "",
        "## Walk-Forward Validation",
        _walk_forward_section(frames["walk_forward"], frames["selected_params"]),
        "",
        "## Robustness",
        _robustness_section(frames["robustness"]),
        "",
        "## Subperiods / Regimes",
        _subperiod_section(frames["subperiod"]),
        "",
        "## Cost Sensitivity",
        _cost_section(frames["cost"]),
        "",
        "## Interpretation",
        _interpretation_section(frames),
        "",
        "## Professional Limitations",
        (
            "This project is strong as an empirical research diagnostic, but it is not:"
        ),
        "- a tradable variance swap backtester;",
        "- an option-chain backtester;",
        "- a volatility surface model;",
        "- a transaction-cost-aware execution simulator;",
        "- a broker-ready trading system.",
        "",
        "## Limitations",
        "- VSTOXX proxy, no option chain.",
        "- No volatility surface.",
        "- No bid/ask or execution model.",
        "- Stylized variance payoff.",
        "- Not live trading alpha.",
        "- No margin/funding.",
        "- No desk-level variance swap pricing.",
        "",
        "## Next Steps",
        "- Real option-chain extension.",
        "- Delta-hedged straddle proxy.",
        "- Variance swap maturity ladder.",
        "- Richer regime classifier.",
        "- Bootstrap confidence intervals.",
    ]

    if warnings_list:
        sections.extend(["", "## Generation Warnings"])
        sections.extend(f"- {message}" for message in warnings_list)

    return "\n".join(sections) + "\n", warnings_list


def _main_results_section(daily: pd.DataFrame | None, non_overlap: pd.DataFrame | None) -> str:
    lines: list[str] = []
    if _has_columns(daily, {"pnl_varswap", "equity_varswap", "signal_vol"}):
        stats = compute_summary_stats(
            daily["pnl_varswap"],
            daily["pnl_varswap"].fillna(0.0).cumsum(),
            daily["signal_vol"],
        )
        rows = [
            {
                "layer": "Daily IV/RV signal",
                "total_pnl": stats["total_pnl"],
                "sharpe_approx": stats["sharpe_approx"],
                "max_drawdown": stats["max_drawdown"],
                "days_in_position": stats["days_in_position"],
            }
        ]
        lines.append(_markdown_table(rows, ["layer", "total_pnl", "sharpe_approx", "max_drawdown", "days_in_position"]))
    else:
        lines.append("Daily backtest summary unavailable. Run `eurostoxx-run-varswap-backtest`.")

    if _has_columns(non_overlap, {"trade_id", "trade_pnl", "cumulative_trade_pnl"}):
        trade_rows = non_overlap.loc[non_overlap["trade_id"].notna()]
        cumulative = (
            float(non_overlap["cumulative_trade_pnl"].dropna().iloc[-1])
            if non_overlap["cumulative_trade_pnl"].notna().any()
            else float(non_overlap["trade_pnl"].sum())
        )
        lines.append(
            f"Non-overlapping layer: {len(trade_rows)} trades, cumulative normalized PnL "
            f"{_fmt(cumulative)}."
        )
    else:
        lines.append(
            "Non-overlapping trade summary unavailable. "
            "Run `eurostoxx-run-non-overlap-varswap`."
        )

    lines.append(
        "All values are normalized diagnostics, not returns, EUR PnL or executable trading results."
    )
    return "\n\n".join(lines)


def _benchmark_section(benchmark: pd.DataFrame | None) -> str:
    if not _has_columns(
        benchmark,
        {"strategy", "total_pnl", "sharpe_approx", "max_drawdown", "days_in_position"},
    ):
        return "Benchmark comparison unavailable. Run `eurostoxx-run-benchmarks`."

    columns = ["strategy", "total_pnl", "sharpe_approx", "max_drawdown", "days_in_position"]
    text = _markdown_table(benchmark.to_dict("records"), columns)
    conclusion = _benchmark_conclusion(benchmark)
    return f"{text}\n\n{conclusion}"


def _walk_forward_section(
    walk_forward: pd.DataFrame | None,
    selected_params: pd.DataFrame | None,
) -> str:
    if not _has_columns(walk_forward, {"test_total_pnl", "test_sharpe_approx"}):
        return "Walk-forward results unavailable. Run `eurostoxx-run-walk-forward`."

    avg_pnl = float(walk_forward["test_total_pnl"].mean())
    positive_folds = int((walk_forward["test_total_pnl"] > 0).sum())
    total_folds = len(walk_forward)
    avg_sharpe = float(walk_forward["test_sharpe_approx"].mean())
    lines = [
        f"Average test PnL: {_fmt(avg_pnl)}.",
        f"Positive test folds: {positive_folds} / {total_folds}.",
        f"Average test Sharpe approximation: {_fmt(avg_sharpe)}.",
    ]
    if _has_columns(selected_params, {"selected_z_entry", "selected_lookback", "selected_horizon"}):
        dispersion = selected_params[
            ["selected_z_entry", "selected_lookback", "selected_horizon"]
        ].nunique()
        lines.append(
            "Selected-parameter dispersion: "
            f"{int(dispersion['selected_z_entry'])} z thresholds, "
            f"{int(dispersion['selected_lookback'])} lookbacks, "
            f"{int(dispersion['selected_horizon'])} horizons."
        )
    return "\n".join(f"- {line}" for line in lines)


def _robustness_section(robustness: pd.DataFrame | None) -> str:
    if not _has_columns(robustness, {"z_entry", "lookback", "horizon", "total_pnl", "max_drawdown"}):
        return "Robustness grid unavailable. Run `eurostoxx-run-robustness`."

    sharpe_col = "sharpe_approx" if "sharpe_approx" in robustness.columns else "sharpe"
    rows = [
        {"metric": "total_pnl_min", "value": robustness["total_pnl"].min()},
        {"metric": "total_pnl_max", "value": robustness["total_pnl"].max()},
        {"metric": f"{sharpe_col}_min", "value": robustness[sharpe_col].min()},
        {"metric": f"{sharpe_col}_max", "value": robustness[sharpe_col].max()},
        {"metric": "max_drawdown_worst", "value": robustness["max_drawdown"].min()},
        {"metric": "grid_points", "value": len(robustness)},
    ]
    return (
        _markdown_table(rows, ["metric", "value"])
        + "\n\nThe grid is a stability diagnostic, not a parameter-mining exercise."
    )


def _subperiod_section(subperiod: pd.DataFrame | None) -> str:
    if not _has_columns(subperiod, {"subperiod", "total_pnl", "sharpe_approx"}):
        return "Subperiod diagnostics unavailable. Run `eurostoxx-run-subperiods`."
    columns = ["subperiod", "total_pnl", "sharpe_approx", "max_drawdown", "days_in_position"]
    text = _markdown_table(subperiod.to_dict("records"), columns)
    top = subperiod.loc[subperiod["total_pnl"].abs().idxmax()]
    return f"{text}\n\nLargest absolute contribution: `{top['subperiod']}`."


def _cost_section(cost: pd.DataFrame | None) -> str:
    if not _has_columns(cost, {"cost_per_signal_change", "total_pnl", "sharpe_approx"}):
        return "Cost sensitivity unavailable. Run `eurostoxx-run-cost-sensitivity`."
    columns = ["cost_per_signal_change", "total_pnl", "sharpe_approx", "max_drawdown"]
    text = _markdown_table(cost.to_dict("records"), columns)
    return (
        f"{text}\n\nCosts are stylized signal-change frictions, not bid/ask, "
        "market impact or financing costs."
    )


def _interpretation_section(frames: dict[str, pd.DataFrame | None]) -> str:
    bullets: list[str] = []
    benchmark = frames["benchmark"]
    daily = frames["daily"]
    walk_forward = frames["walk_forward"]
    subperiod = frames["subperiod"]
    cost = frames["cost"]

    if _has_columns(benchmark, {"strategy", "total_pnl"}):
        bullets.append(_benchmark_conclusion(benchmark))
    else:
        bullets.append("Benchmark evidence is not available yet, so the VRP comparison is unresolved.")

    if _has_columns(daily, {"pnl_varswap", "signal_vol"}):
        short_pnl = float(daily.loc[daily["signal_vol"] == -1, "pnl_varswap"].sum())
        long_pnl = float(daily.loc[daily["signal_vol"] == 1, "pnl_varswap"].sum())
        if short_pnl > long_pnl:
            bullets.append(
                "The short-vol leg contributes more than the long-vol leg in the daily diagnostic."
            )
        else:
            bullets.append(
                "The long-vol leg contributes at least as much as the short-vol leg in the daily diagnostic."
            )

    if _has_columns(walk_forward, {"test_total_pnl"}):
        positive = int((walk_forward["test_total_pnl"] > 0).sum())
        total = len(walk_forward)
        bullets.append(
            f"Walk-forward validation has {positive} positive test folds out of {total}, "
            "which is the relevant out-of-sample stability check."
        )

    if _has_columns(subperiod, {"subperiod", "total_pnl"}):
        concentration = subperiod.loc[subperiod["total_pnl"].abs().idxmax()]
        bullets.append(
            f"Subperiod attribution should be read carefully because `{concentration['subperiod']}` "
            "is the largest absolute contributor."
        )

    if _has_columns(cost, {"cost_per_signal_change", "total_pnl"}):
        first = cost.sort_values("cost_per_signal_change").iloc[0]
        last = cost.sort_values("cost_per_signal_change").iloc[-1]
        bullets.append(
            f"Raising stylized costs from {_fmt(first['cost_per_signal_change'])} to "
            f"{_fmt(last['cost_per_signal_change'])} changes total PnL from "
            f"{_fmt(first['total_pnl'])} to {_fmt(last['total_pnl'])}."
        )

    bullets.extend(
        [
            "The payoff is useful for signal diagnostics, not for claiming live alpha.",
            "The VSTOXX proxy keeps the study reproducible but limits desk-level interpretation.",
        ]
    )
    return "\n".join(f"- {bullet}" for bullet in bullets[:10])


def _benchmark_conclusion(benchmark: pd.DataFrame) -> str:
    indexed = benchmark.set_index("strategy")
    if {"iv_rv_signal", "always_short_vol"}.issubset(indexed.index):
        signal_pnl = float(indexed.loc["iv_rv_signal", "total_pnl"])
        short_pnl = float(indexed.loc["always_short_vol", "total_pnl"])
        if signal_pnl > short_pnl:
            return (
                "In this sample, the IV/RV signal has higher total PnL than the always-short-vol "
                "benchmark."
            )
        if signal_pnl < short_pnl:
            return (
                "In this sample, the always-short-vol benchmark has higher total PnL than the "
                "IV/RV signal, so the result appears substantially related to naive VRP exposure."
            )
        return "In this sample, the IV/RV signal and always-short-vol benchmark tie on total PnL."
    return "Benchmark rows are incomplete, so the signal-versus-VRP comparison is inconclusive."


def _read_optional_csv(path: Path, warnings_list: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        warnings_list.append(f"Missing optional output: {path.name}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive guard for corrupt local files
        warnings_list.append(f"Could not read {path.name}: {exc}")
        return None


def _has_columns(df: pd.DataFrame | None, columns: set[str]) -> bool:
    return df is not None and columns.issubset(df.columns) and not df.empty


def _markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "_No rows available._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_fmt(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "n/a"
        return f"{float(value):.4g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Markdown research report.")
    parser.add_argument("--outputs-dir", default=str(OUTPUTS))
    parser.add_argument("--output", default=str(OUTPUTS / "research_report.md"))
    parser.add_argument("--docs-output", default=str(DOCS / "research_report.md"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
