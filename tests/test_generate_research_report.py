import pandas as pd

from eurostoxx_iv_rv_backtest.scripts.generate_research_report import build_research_report


def test_research_report_contains_main_sections_with_minimal_outputs(tmp_path) -> None:
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="B"),
            "pnl_varswap": [0.1, -0.05, 0.02],
            "equity_varswap": [0.1, 0.05, 0.07],
            "signal_vol": [1, -1, 0],
        }
    ).to_csv(tmp_path / "SXE50_iv_rv_varswap_backtest.csv", index=False)
    pd.DataFrame(
        {
            "strategy": ["iv_rv_signal", "always_short_vol", "always_long_vol", "always_flat"],
            "total_pnl": [0.07, 0.10, -0.10, 0.0],
            "sharpe_approx": [0.5, 0.7, -0.7, float("nan")],
            "max_drawdown": [-0.05, -0.03, -0.2, 0.0],
            "days_in_position": [2, 3, 3, 0],
        }
    ).to_csv(tmp_path / "benchmark_comparison.csv", index=False)

    report, warnings = build_research_report(outputs_dir=tmp_path)

    assert "# Euro STOXX 50 IV/RV Volatility Research Report" in report
    assert "## Research Question" in report
    assert "## Benchmark Comparison" in report
    assert "## Walk-Forward Validation" in report
    assert "## Professional Limitations" in report
    assert "always-short-vol benchmark has higher total PnL" in report
    assert warnings


def test_research_report_handles_missing_optional_files(tmp_path) -> None:
    report, warnings = build_research_report(outputs_dir=tmp_path)

    assert "Daily backtest summary unavailable" in report
    assert "Benchmark comparison unavailable" in report
    assert "## Generation Warnings" in report
    assert warnings
