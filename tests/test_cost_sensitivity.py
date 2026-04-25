import pandas as pd

from eurostoxx_iv_rv_backtest.scripts.run_cost_sensitivity import run_cost_sensitivity


def test_run_cost_sensitivity_reduces_pnl_as_costs_increase() -> None:
    df = pd.DataFrame(
        {
            "iv": [0.20, 0.20, 0.20],
            "rv_fwd_20d": [0.30, 0.30, 0.30],
            "signal_vol": [0, 1, -1],
        }
    )

    result = run_cost_sensitivity(df, cost_levels=(0.0, 0.01))

    by_cost = result.set_index("cost_per_signal_change")
    assert by_cost.loc[0.01, "total_pnl"] < by_cost.loc[0.0, "total_pnl"]
    assert by_cost.loc[0.01, "turnover_proxy"] == 3.0
    assert by_cost.loc[0.01, "number_signal_changes"] == 2.0
