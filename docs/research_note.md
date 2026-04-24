# Euro STOXX 50 IV/RV Volatility Research Note

## 1. Motivation

This project studies a simple implied-versus-realized volatility signal on Euro
STOXX 50. The economic motivation is the variance risk premium: implied
volatility often trades above subsequently realized volatility, but the
relationship is time-varying and regime-dependent.

The objective is not to build a tradable options strategy. The objective is to
construct a reproducible research pipeline that tests whether a transparent IV/RV
signal is directionally aligned with future realized variance.

## 2. Data

The dataset combines:

- Euro STOXX 50 spot data from Yahoo Finance (`^STOXX50E`),
- VSTOXX / V2TX historical values from STOXX,
- daily date alignment between spot and implied volatility.

VSTOXX is used as a broad implied volatility proxy. It is not a volatility
surface and does not contain strike, maturity, bid/ask or option-chain details.

## 3. Feature Construction

Daily log returns are computed from close-to-close SX5E prices:

$$
r_t = \ln(S_t / S_{t-1})
$$

Historical realized volatility over a window \(w\) is the rolling standard
deviation of log returns, annualized by \(\sqrt{252}\).

Forward realized volatility is computed strictly from future returns. At date
\(t\), `rv_fwd_20d_t` uses returns from \(t+1\) to \(t+20\). It is reserved for
ex-post payoff evaluation.

## 4. Signal Construction

The signal uses the spread:

$$
s_t = IV_t - RV_{t}
$$

and a rolling z-score:

$$
z_t = (s_t - \mu_t) / \sigma_t
$$

The default rule is:

- short volatility when IV is high relative to RV (`z > z_entry`),
- long volatility when IV is low relative to RV (`z < -z_entry`),
- flat otherwise.

The signal is intentionally simple. This makes the timing convention and
robustness analysis easier to audit.

## 5. Payoff Convention

The first payoff layer is a daily diagnostic:

$$
PnL_t = signal_t \cdot (RV_{fwd,t}^2 - IV_t^2)
$$

The second layer is a non-overlapping hold-to-maturity version:

- open at most one trade at a time,
- hold for a fixed 20-trading-day horizon,
- book payoff at maturity,
- skip overlapping entries while a trade is active.

Both layers are normalized stylized payoffs. They do not model transaction
costs, bid/ask spreads, margin, funding, daily mark-to-market or true variance
swap replication.

The daily diagnostic layer includes an optional simplified friction parameter
charged per unit of absolute signal change. This is only a sensitivity check; it
is not a bid/ask, slippage or execution model.

## 6. No-Look-Ahead Convention

The project enforces the following time convention:

- `rv_20d_t` may use returns up to date \(t\),
- `signal_vol_t` uses IV and historical RV available at \(t\),
- `rv_fwd_*_t` uses strictly future returns,
- forward RV is never used to construct the signal.

Unit tests explicitly check that changing future data does not alter the signal
at \(t\), and that forward RV only affects the ex-post payoff layer.

## 7. Results Summary

On the current local sample, the daily stylized payoff produces:

- total normalized PnL: 8.222,
- approximate Sharpe: 0.75,
- max drawdown: -4.784,
- days in position: 47.1%,
- long-vol PnL: -5.993,
- short-vol PnL: 14.214.

The non-overlapping 20-day layer produces:

- 164 trades,
- cumulative normalized trade PnL: 0.474.

These values should be read as signal diagnostics, not as live trading results.

## 8. Robustness

The robustness grid varies:

- z-score threshold,
- rolling lookback,
- realized-volatility and forward horizon.

The goal is to inspect sensitivity across plausible assumptions. It is not a
parameter search and should not be used to cherry-pick the best Sharpe ratio.

## 9. Limitations

Main limitations:

- VSTOXX is a proxy, not a calibrated options surface.
- No option-chain data, strike dimension or maturity term structure.
- No bid/ask, slippage, liquidity, funding or margin.
- Payoff layers are stylized and normalized.
- The Black-Scholes utilities are educational and do not imply a real options
  trading backtest.

## 10. Next Extensions

Reasonable next steps:

- overlapping maturity ladder for variance trades,
- carefully documented delta-hedged ATM straddle experiment,
- subperiod and bootstrap stability analysis,
- richer crisis-period attribution,
- optional report generation from the CSV and figure outputs.
