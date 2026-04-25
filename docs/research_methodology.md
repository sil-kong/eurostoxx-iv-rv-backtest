# Research Methodology

## Research Question

Does the implied-realized volatility spread on Euro STOXX 50 contain predictive
information about subsequent realized variance, beyond a naive
variance-risk-premium exposure?

## Data Sources

- Euro STOXX 50 / SX5E spot data from Yahoo Finance ticker `^STOXX50E`.
- VSTOXX / V2TX historical data from STOXX.
- VSTOXX is converted to decimal annualized implied volatility as
  `iv = vstoxx_close / 100`.

VSTOXX is used as a broad implied-volatility proxy. The project does not use an
option chain, strike dimension, maturity ladder, bid/ask quotes or a volatility
surface.

## Time Alignment

The pipeline aligns spot and implied-volatility observations by calendar date.
Rows are sorted chronologically before feature construction. Historical RV is
observable only after the close of the current row. Forward RV is strictly
future information and is used only for ex-post payoff evaluation.

## No-Look-Ahead Conventions

At date `t`:

- `rv_*d_t` may use returns up to and including `t`.
- `iv_t` and `rv_*d_t` can be used by `signal_vol_t`.
- `rv_fwd_*d_t` uses returns from `t+1` through `t+horizon`.
- `rv_fwd_*d_t` is never used to construct the signal.
- Walk-forward parameter selection excludes train rows whose payoff horizon
  would cross into the test window.

## Realized Volatility

Daily log returns are:

```text
r_t = log(S_t / S_{t-1})
```

Historical realized volatility over window `w` is the annualized rolling
standard deviation of close-to-close log returns:

```text
RV_{w,t} = std(r_{t-w+1}, ..., r_t) * sqrt(252)
```

Forward realized volatility is computed from strictly future returns:

```text
RV_fwd_{w,t} = std(r_{t+1}, ..., r_{t+w}) * sqrt(252)
```

## Signal Construction

The spread is:

```text
spread_t = IV_t - RV_t
```

The signal uses a rolling z-score of this spread:

```text
z_t = (spread_t - rolling_mean_t) / rolling_std_t
```

Default rule:

- `signal_vol = -1` when `z_t > z_entry`, interpreted as short-vol bias.
- `signal_vol = +1` when `z_t < -z_entry`, interpreted as long-vol bias.
- `signal_vol = 0` otherwise.

## Payoff Convention

The daily diagnostic payoff is:

```text
PnL_t = signal_t * (RV_fwd_t^2 - IV_t^2)
```

All values are normalized diagnostic units. They are not returns, EUR PnL,
desk marks or executable strategy performance.

## Benchmark Definitions

The benchmark layer uses the same payoff convention:

- `iv_rv_signal`: the repository's z-score signal.
- `always_short_vol`: `signal = -1` whenever IV and forward RV are valid.
- `always_long_vol`: `signal = +1` whenever IV and forward RV are valid.
- `always_flat`: `signal = 0`.

This directly tests whether the IV/RV signal adds information beyond a naive
variance-risk-premium exposure.

## Non-Overlapping Trades

The non-overlapping layer opens at most one stylized variance trade at a time.
If a signal is active, it opens a fixed-horizon trade and books the payoff at
maturity. It does not open another trade while one is active.

## Walk-Forward Validation

The walk-forward framework:

1. Splits data into chronological train and test windows.
2. Scores each parameter set on the train window only.
3. Enforces a minimum exposure constraint when selecting parameters.
4. Evaluates the selected parameter set on the following test window.
5. Exports fold-level results, selected parameters and out-of-sample equity.

Parameter selection is designed to avoid using test-window payoff information.

## Robustness Grid

The robustness grid varies:

- z-score entry threshold,
- rolling spread lookback,
- realized-volatility and forward-payoff horizon.

The grid is a stability diagnostic. It is not evidence that the best parameter
combination was knowable ahead of time.

## Cost Model

Cost sensitivity uses a stylized deduction per unit of absolute signal change.
For example, changing from `+1` to `-1` has a turnover proxy of two units. This
is not a bid/ask, slippage, market-impact, margin or funding model.

## Limitations

This project is intentionally scoped as empirical volatility research. It is
not:

- a tradable variance swap backtester;
- an option-chain backtester;
- a volatility surface model;
- a transaction-cost-aware execution simulator;
- a broker-ready trading system.

The framework is best read as a reproducible diagnostic for studying whether an
IV/RV spread contains information about future realized variance.
