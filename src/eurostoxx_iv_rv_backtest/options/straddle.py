from eurostoxx_iv_rv_backtest.options.black_scholes import call_price, put_price
from eurostoxx_iv_rv_backtest.options.payoffs import straddle_payoff


def atm_straddle_pnl_at_expiry(
    spot_initial: float,
    spot_at_expiry: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> dict[str, float]:
    """
    Build a stylized ATM straddle and compute its expiry PnL.

    Convention:
        - strike = spot_initial
        - initial cost = Black-Scholes call + put price using volatility at t
        - expiry payoff = max(S_T - K, 0) + max(K - S_T, 0)
        - PnL = payoff - initial cost

    This is a pedagogical link between implied volatility, realized movement and
    option payoff. It does not use an option chain, bid/ask, transaction costs,
    financing, early unwind, margin, or a volatility surface.
    """
    strike = spot_initial
    call_cost = call_price(
        spot_initial,
        strike,
        time_to_maturity,
        risk_free_rate,
        volatility,
        dividend_yield,
    )
    put_cost = put_price(
        spot_initial,
        strike,
        time_to_maturity,
        risk_free_rate,
        volatility,
        dividend_yield,
    )
    initial_cost = call_cost + put_cost
    payoff = float(straddle_payoff(spot_at_expiry, strike))

    return {
        "strike": strike,
        "call_cost": call_cost,
        "put_cost": put_cost,
        "initial_cost": initial_cost,
        "payoff": payoff,
        "pnl": payoff - initial_cost,
    }
