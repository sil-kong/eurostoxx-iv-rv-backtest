import numpy as np
import pytest

from eurostoxx_iv_rv_backtest.options.payoffs import (
    call_payoff,
    option_pnl_at_expiry,
    put_payoff,
    straddle_payoff,
)
from eurostoxx_iv_rv_backtest.options.straddle import atm_straddle_pnl_at_expiry


def test_basic_option_payoffs() -> None:
    spots = np.array([80.0, 100.0, 120.0])

    np.testing.assert_allclose(call_payoff(spots, 100.0), [0.0, 0.0, 20.0])
    np.testing.assert_allclose(put_payoff(spots, 100.0), [20.0, 0.0, 0.0])
    np.testing.assert_allclose(straddle_payoff(spots, 100.0), [20.0, 0.0, 20.0])


def test_option_pnl_at_expiry_uses_position_sign() -> None:
    long_call = option_pnl_at_expiry(120.0, 100.0, 8.0, "call", position=1.0)
    short_put = option_pnl_at_expiry(80.0, 100.0, 7.0, "put", position=-1.0)

    assert long_call == pytest.approx(12.0)
    assert short_put == pytest.approx(-13.0)


def test_option_pnl_at_expiry_rejects_invalid_option_type() -> None:
    with pytest.raises(ValueError, match="option_type"):
        option_pnl_at_expiry(100.0, 100.0, 1.0, "forward")


def test_atm_straddle_pnl_at_expiry_returns_expected_components() -> None:
    result = atm_straddle_pnl_at_expiry(
        spot_initial=100.0,
        spot_at_expiry=115.0,
        time_to_maturity=30.0 / 252.0,
        risk_free_rate=0.0,
        volatility=0.20,
    )

    assert result["strike"] == 100.0
    assert result["payoff"] == 15.0
    assert result["initial_cost"] == pytest.approx(
        result["call_cost"] + result["put_cost"]
    )
    assert result["pnl"] == pytest.approx(15.0 - result["initial_cost"])
