from math import erf, exp, log, pi, sqrt


OptionType = str


def _validate_option_type(option_type: OptionType) -> str:
    option = option_type.lower()
    if option not in {"call", "put"}:
        raise ValueError("option_type doit être 'call' ou 'put'.")
    return option


def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} doit être strictement positif.")


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _d1_d2(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    _validate_positive(spot, "spot")
    _validate_positive(strike, "strike")
    _validate_positive(time_to_maturity, "time_to_maturity")
    _validate_positive(volatility, "volatility")

    vol_sqrt_t = volatility * sqrt(time_to_maturity)
    d1 = (
        log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_maturity
    ) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> float:
    """
    Price a European Black-Scholes option with continuous dividend yield.

    Conventions:
        - ``spot`` and ``strike`` are strictly positive.
        - ``time_to_maturity`` is in years and strictly positive.
        - ``risk_free_rate``, ``dividend_yield`` and ``volatility`` are annual.
        - ``volatility`` is a decimal annualized volatility, e.g. 0.20.

    Formulas:
        d1 = [ln(S/K) + (r - q + 0.5 sigma^2) T] / [sigma sqrt(T)]
        d2 = d1 - sigma sqrt(T)
    """
    option = _validate_option_type(option_type)
    d1, d2 = _d1_d2(
        spot,
        strike,
        time_to_maturity,
        risk_free_rate,
        volatility,
        dividend_yield,
    )

    discounted_spot = spot * exp(-dividend_yield * time_to_maturity)
    discounted_strike = strike * exp(-risk_free_rate * time_to_maturity)

    if option == "call":
        return discounted_spot * _norm_cdf(d1) - discounted_strike * _norm_cdf(d2)
    return discounted_strike * _norm_cdf(-d2) - discounted_spot * _norm_cdf(-d1)


def call_price(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Price a European call using Black-Scholes."""
    return black_scholes_price(
        spot,
        strike,
        time_to_maturity,
        risk_free_rate,
        volatility,
        "call",
        dividend_yield,
    )


def put_price(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Price a European put using Black-Scholes."""
    return black_scholes_price(
        spot,
        strike,
        time_to_maturity,
        risk_free_rate,
        volatility,
        "put",
        dividend_yield,
    )


def black_scholes_delta(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> float:
    """Return Black-Scholes spot delta for a European call or put."""
    option = _validate_option_type(option_type)
    d1, _ = _d1_d2(
        spot,
        strike,
        time_to_maturity,
        risk_free_rate,
        volatility,
        dividend_yield,
    )
    discount = exp(-dividend_yield * time_to_maturity)
    if option == "call":
        return discount * _norm_cdf(d1)
    return discount * (_norm_cdf(d1) - 1.0)


def black_scholes_gamma(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Return Black-Scholes gamma, shared by calls and puts."""
    d1, _ = _d1_d2(
        spot,
        strike,
        time_to_maturity,
        risk_free_rate,
        volatility,
        dividend_yield,
    )
    return (
        exp(-dividend_yield * time_to_maturity)
        * _norm_pdf(d1)
        / (spot * volatility * sqrt(time_to_maturity))
    )


def black_scholes_vega(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Return Black-Scholes vega per 1.00 volatility point.

    Example: a value of 40 means the option price changes by about 0.40 for a
    one percentage point volatility move.
    """
    d1, _ = _d1_d2(
        spot,
        strike,
        time_to_maturity,
        risk_free_rate,
        volatility,
        dividend_yield,
    )
    return spot * exp(-dividend_yield * time_to_maturity) * _norm_pdf(d1) * sqrt(
        time_to_maturity
    )


def black_scholes_theta(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> float:
    """Return annual Black-Scholes theta for a European call or put."""
    option = _validate_option_type(option_type)
    d1, d2 = _d1_d2(
        spot,
        strike,
        time_to_maturity,
        risk_free_rate,
        volatility,
        dividend_yield,
    )
    first_term = (
        -spot
        * exp(-dividend_yield * time_to_maturity)
        * _norm_pdf(d1)
        * volatility
        / (2.0 * sqrt(time_to_maturity))
    )
    if option == "call":
        return (
            first_term
            - risk_free_rate
            * strike
            * exp(-risk_free_rate * time_to_maturity)
            * _norm_cdf(d2)
            + dividend_yield
            * spot
            * exp(-dividend_yield * time_to_maturity)
            * _norm_cdf(d1)
        )
    return (
        first_term
        + risk_free_rate
        * strike
        * exp(-risk_free_rate * time_to_maturity)
        * _norm_cdf(-d2)
        - dividend_yield
        * spot
        * exp(-dividend_yield * time_to_maturity)
        * _norm_cdf(-d1)
    )


def implied_volatility_bisection(
    market_price: float,
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    vol_lower: float = 1e-6,
    vol_upper: float = 5.0,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> float:
    """
    Infer Black-Scholes implied volatility by robust bisection.

    The market price must lie between the Black-Scholes prices produced by
    ``vol_lower`` and ``vol_upper``. If not, the function raises ``ValueError``
    instead of returning a misleading number.
    """
    _validate_option_type(option_type)
    _validate_positive(market_price, "market_price")
    _validate_positive(vol_lower, "vol_lower")
    _validate_positive(vol_upper, "vol_upper")
    if vol_lower >= vol_upper:
        raise ValueError("vol_lower doit être strictement inférieur à vol_upper.")
    if tolerance <= 0:
        raise ValueError("tolerance doit être strictement positive.")
    if max_iterations <= 0:
        raise ValueError("max_iterations doit être strictement positif.")

    low_price = black_scholes_price(
        spot,
        strike,
        time_to_maturity,
        risk_free_rate,
        vol_lower,
        option_type,
        dividend_yield,
    )
    high_price = black_scholes_price(
        spot,
        strike,
        time_to_maturity,
        risk_free_rate,
        vol_upper,
        option_type,
        dividend_yield,
    )
    if market_price < low_price - tolerance or market_price > high_price + tolerance:
        raise ValueError(
            "market_price est hors des bornes compatibles avec les volatilités "
            "fournies."
        )

    low = vol_lower
    high = vol_upper
    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        mid_price = black_scholes_price(
            spot,
            strike,
            time_to_maturity,
            risk_free_rate,
            mid,
            option_type,
            dividend_yield,
        )
        error = mid_price - market_price
        if abs(error) <= tolerance:
            return mid
        if error > 0:
            high = mid
        else:
            low = mid

    raise ValueError("Volatilité implicite non trouvée dans max_iterations.")
