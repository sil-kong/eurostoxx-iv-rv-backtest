import numpy as np


def call_payoff(spot: float | np.ndarray, strike: float) -> float | np.ndarray:
    """Return the expiry payoff max(S_T - K, 0) for a European call."""
    return np.maximum(np.asarray(spot) - strike, 0.0)


def put_payoff(spot: float | np.ndarray, strike: float) -> float | np.ndarray:
    """Return the expiry payoff max(K - S_T, 0) for a European put."""
    return np.maximum(strike - np.asarray(spot), 0.0)


def straddle_payoff(spot: float | np.ndarray, strike: float) -> float | np.ndarray:
    """Return the expiry payoff of one call plus one put at the same strike."""
    return call_payoff(spot, strike) + put_payoff(spot, strike)


def option_pnl_at_expiry(
    spot_at_expiry: float | np.ndarray,
    strike: float,
    premium: float,
    option_type: str,
    position: float = 1.0,
) -> float | np.ndarray:
    """
    Return expiry PnL for a simple European option position.

    ``position`` is positive for long options and negative for short options.
    The function ignores funding, bid/ask spreads, margin and interim
    mark-to-market. It is a payoff utility, not a trading simulator.
    """
    option = option_type.lower()
    if option == "call":
        payoff = call_payoff(spot_at_expiry, strike)
    elif option == "put":
        payoff = put_payoff(spot_at_expiry, strike)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'.")
    return position * (payoff - premium)
