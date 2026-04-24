import math

import pytest

from eurostoxx_iv_rv_backtest.options.black_scholes import (
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_price,
    black_scholes_theta,
    black_scholes_vega,
    call_price,
    implied_volatility_bisection,
    put_price,
)


def test_black_scholes_prices_match_reference_values() -> None:
    call = call_price(100.0, 100.0, 1.0, 0.05, 0.20)
    put = put_price(100.0, 100.0, 1.0, 0.05, 0.20)

    assert call == pytest.approx(10.4506, rel=1e-4)
    assert put == pytest.approx(5.5735, rel=1e-4)


def test_black_scholes_put_call_parity() -> None:
    spot = 100.0
    strike = 95.0
    maturity = 0.75
    rate = 0.03
    vol = 0.25

    call = black_scholes_price(spot, strike, maturity, rate, vol, "call")
    put = black_scholes_price(spot, strike, maturity, rate, vol, "put")

    assert call - put == pytest.approx(
        spot - strike * math.exp(-rate * maturity),
        rel=1e-10,
    )


def test_black_scholes_greeks_are_reasonable() -> None:
    delta_call = black_scholes_delta(100.0, 100.0, 1.0, 0.05, 0.20, "call")
    delta_put = black_scholes_delta(100.0, 100.0, 1.0, 0.05, 0.20, "put")
    gamma = black_scholes_gamma(100.0, 100.0, 1.0, 0.05, 0.20)
    vega = black_scholes_vega(100.0, 100.0, 1.0, 0.05, 0.20)
    theta = black_scholes_theta(100.0, 100.0, 1.0, 0.05, 0.20, "call")

    assert delta_call == pytest.approx(0.6368, rel=1e-4)
    assert delta_put == pytest.approx(-0.3632, rel=1e-4)
    assert gamma > 0.0
    assert vega > 0.0
    assert theta < 0.0


def test_black_scholes_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="option_type"):
        black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.20, "digital")
    with pytest.raises(ValueError, match="spot"):
        call_price(0.0, 100.0, 1.0, 0.05, 0.20)
    with pytest.raises(ValueError, match="volatility"):
        put_price(100.0, 100.0, 1.0, 0.05, 0.0)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_implied_volatility_bisection_recovers_input_vol(option_type: str) -> None:
    price = black_scholes_price(100.0, 100.0, 1.0, 0.02, 0.20, option_type)

    implied_vol = implied_volatility_bisection(
        price,
        100.0,
        100.0,
        1.0,
        0.02,
        option_type,
    )

    assert implied_vol == pytest.approx(0.20, abs=1e-7)


def test_implied_volatility_bisection_rejects_unreachable_price() -> None:
    with pytest.raises(ValueError, match="hors des bornes"):
        implied_volatility_bisection(500.0, 100.0, 100.0, 1.0, 0.02, "call")
