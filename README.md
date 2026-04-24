# eurostoxx-iv-rv-backtest

Backtest of an implied vs realized volatility trading strategy on Euro STOXX 50 (SX5E / VSTOXX), built around a reproducible data pipeline, volatility feature engineering, and a stylized variance-swap type payoff.

## Overview

This project studies a simple volatility trading idea on Euro STOXX 50:

- use **VSTOXX (V2TX)** as a proxy for implied volatility,
- compute **realized volatility** on SX5E spot prices,
- compare IV vs RV through a rolling signal,
- generate long / short volatility regimes,
- evaluate the strategy through a stylized variance-based PnL.

The repo is meant as a clean research pipeline rather than a production-ready trading engine.
The focus is on:

- market data ingestion,
- volatility feature construction,
- signal design,
- backtesting,
- visual analysis.

It also serves as a base for later extensions toward more realistic variance products or option-based strategies.

---

## Strategy idea

The economic intuition is straightforward:

- if **implied volatility** trades well above **realized volatility**, volatility may be overpriced → short vol bias,
- if **implied volatility** trades well below **realized volatility**, volatility may be underpriced → long vol bias.

The signal is built from the IV-RV spread and the backtest evaluates a stylized payoff of the form:

$$
\text{PnL}_t \propto \text{signal}_t \cdot \left(RV^{2}_{fwd,t} - IV_t^2\right)
$$

This should be interpreted as a signal-oriented approximation of variance trading, not as a full desk-level variance swap valuation framework.

---

## Data

### Euro STOXX 50 spot data

- Source: `yfinance`
- Ticker: `^STOXX50E`
- Frequency: daily

Main fields used:

- `date`
- `open`
- `high`
- `low`
- `close`
- `adj_close`
- `volume`

### Implied volatility data

- Source: STOXX historical text file
- File: `h_v2tx.txt`
- Index: **V2TX / VSTOXX**

The raw STOXX series is converted into:

- `vstoxx_close`
- `iv = vstoxx_close / 100`

So for example:

- `V2TX = 20.0` becomes `iv = 0.20`

---

## Project structure

```text
.
├── data
│   └── raw
│       ├── getdata.py
│       └── ...
├── outputs
│   ├── SXE50_with_IV_RV_daily_20y.csv
│   ├── SXE50_with_IV_RV_daily_20y_with_signals.csv
│   └── SXE50_iv_rv_varswap_backtest.csv
└── src
    └── eurostoxx_iv_rv_backtest
        ├── __init__.py
        ├── config.py
        ├── features
        │   ├── __init__.py
        │   ├── realized_vol.py
        │   └── iv_rv_variance_swap.py
        ├── options
        │   ├── __init__.py
        │   ├── black_scholes.py
        │   ├── payoffs.py
        │   └── straddle.py
        └── scripts
            ├── build_rv.py
            ├── build_signals.py
            ├── run_backtest_iv_rv.py
            ├── animate_iv_rv.py
            ├── animate_equity.py
            └── analyze_backtest.py
```

The project is organized as a small research workflow, with:

- raw / intermediate data in `data/raw`,
- generated outputs in `outputs`,
- reusable code and scripts under `src`.

---

## Methodology

### 1. Log-returns

From spot prices $S_t$, daily log-returns are computed as:

$$
r_t = \ln\left(\frac{S_t}{S_{t-1}}\right)
$$

These returns are the base input for realized volatility estimation.

### 2. Historical realized volatility

For a rolling window $w$ (typically 20d and 30d), realized volatility is computed as:

1. rolling standard deviation of log-returns,
2. annualization using $\sqrt{252}$.

This produces:

- `rv_20d`
- `rv_20d_pct`
- `rv_30d`
- `rv_30d_pct`

### 3. Forward realized volatility

The project also computes a forward realized volatility measure:

- `rv_fwd_20d`
- `rv_fwd_20d_pct`

At date $t$, this uses future returns over the interval $[t+1, t+20]$.
It is meant to proxy the realized volatility entering the forward-looking payoff.

### No-look-ahead policy

The time convention is explicit:

- at date \(t\), the signal uses only fields available at \(t\),
- `rv_20d_t` may use log-returns up to and including \(t\), because it is observable after the close,
- `rv_fwd_20d_t` uses strictly future returns \(t+1\) to \(t+20\),
- `rv_fwd_20d_t` is reserved for ex-post payoff evaluation,
- `signal_vol_t` must not use `rv_fwd_20d_t`.

Unit tests cover this convention directly, including a deterministic forward-RV example where the current return is deliberately different from the future returns.

### 4. IV-RV spread and z-score

A simple spread is defined as:

$$
s_t = IV_t - RV_{20d,t}
$$

This spread is normalized through a rolling z-score:

$$
z_t = \frac{(IV_t - RV_{20d,t}) - \mu_t}{\sigma_t}
$$

with:

- rolling lookback = 252 days,
- default threshold = 0.5.

### 5. Trading signal

The trading rule is intentionally simple:

- `signal_vol = -1` if `z > z_entry` → short vol
- `signal_vol = +1` if `z < -z_entry` → long vol
- `signal_vol = 0` otherwise

### 6. Stylized payoff

The daily PnL is defined as:

$$
\text{PnL}_t = \text{signal}_t \cdot \left(RV^{2}_{fwd,t} - IV_t^2\right)
$$

with normalized notional equal to 1.

The cumulative PnL is stored as:

- `pnl_varswap`
- `equity_varswap`

This is a stylized payoff useful to evaluate the signal, but not a full mark-to-market implementation of a live variance swap book.

### 7. Options pricing extension

The repo also includes a small option-pricing utility layer:

- European Black-Scholes call and put prices,
- call and put delta, gamma, vega and annual theta,
- implied volatility inversion by robust bisection,
- call, put and straddle expiry payoffs,
- a simplified ATM straddle expiry PnL helper.

This extension is intentionally educational. It links the IV/RV research idea to option pricing and option payoff mechanics, but it does not build a real options book or calibrate an options surface.

---

## How to run

Run all scripts from the project root.

### 1. Download and clean market data

```bash
env PYTHONPATH=src .venv/bin/python data/raw/getdata.py
```

This script:

- downloads SX5E spot data from Yahoo Finance,
- downloads V2TX history from STOXX,
- cleans both series,
- aligns dates,
- exports a merged dataset in:

```text
data/raw/SXE50_with_IV_daily_20y.csv
```

### 2. Build realized volatility features

```bash
env PYTHONPATH=src .venv/bin/python src/eurostoxx_iv_rv_backtest/scripts/build_rv.py
```

This creates:

```text
outputs/SXE50_with_IV_RV_daily_20y.csv
```

with historical realized vol measures.

### 3. Build forward RV and signals

```bash
env PYTHONPATH=src .venv/bin/python src/eurostoxx_iv_rv_backtest/scripts/build_signals.py
```

This creates:

```text
outputs/SXE50_with_IV_RV_daily_20y_with_signals.csv
```

with:

- forward RV,
- IV-RV spread,
- z-score,
- trading signal.

### 4. Run the backtest

```bash
env PYTHONPATH=src .venv/bin/python src/eurostoxx_iv_rv_backtest/scripts/run_backtest_iv_rv.py
```

This creates:

```text
outputs/SXE50_iv_rv_varswap_backtest.csv
```

with:

- `pnl_varswap`
- `equity_varswap`

### 5. Visualize IV vs RV

```bash
env PYTHONPATH=src .venv/bin/python src/eurostoxx_iv_rv_backtest/scripts/animate_iv_rv.py
```

This animation shows:

- implied vol,
- realized vol,
- colored volatility regimes:
  - red = short vol regime
  - blue = long vol regime

### 6. Visualize the equity curve

```bash
env PYTHONPATH=src .venv/bin/python src/eurostoxx_iv_rv_backtest/scripts/animate_equity.py
```

### 7. Print backtest summary statistics

```bash
env PYTHONPATH=src .venv/bin/python src/eurostoxx_iv_rv_backtest/scripts/analyze_backtest.py
```

### 8. Run tests

```bash
.venv/bin/python -m pytest
```

The tests are configured through `pytest.ini`, so running from the project root is enough.

---

## Current results

Current backtest summary:

```text
=== Résumé backtest IV vs RV (variance swap) ===

Total PnL       : 8.222
Annualisé (moy) : 0.4379
Annualisé (vol) : 0.5812
Sharpe approx   : 0.75
Max drawdown    : -4.784

Nb jours       : 4731
Nb jours en position : 2227 (47.1 %)

PnL long vol  : -5.993
PnL short vol : 14.214
```
### Basic interpretation

- In the current sample and with the current stylized assumptions, the strategy delivers a positive cumulative normalized payoff.
- The Sharpe ratio (~0.75) is a descriptive statistic for this simplified payoff, not a live trading performance claim.
- Most of the performance comes from the short-vol leg, which is consistent with the standard variance risk premium intuition: implied volatility tends to trade above realized volatility on average.
- The long-vol leg is negative over the full sample in this run, while still acting differently during stressed volatility regimes.

### About the equity curve

The equity curve is the cumulative sum of the daily stylized payoff.

It should **not** be interpreted as:

- a return in %,
- a monetary PnL in EUR,
- a fully realistic desk PnL.

It should be read as:

- a normalized cumulative payoff,
- with `notional = 1`,
- useful to compare signal quality through time and across regimes.

---

## Strengths of the project

What this repo already does reasonably well:

- clean market data ingestion from two different sources,
- proper volatility feature engineering (historical and forward RV),
- simple but structured signal generation,
- reproducible backtest pipeline,
- useful visual diagnostics,
- clear base for more advanced volatility or option strategies.

---

## What this project is not

This project is intentionally simple. It is not:

- a production trading engine,
- a real options surface or calibration framework,
- an options-chain backtest with bid/ask quotes,
- an order-book or execution simulator,
- a transaction-cost-aware portfolio engine,
- a live variance-swap book or desk-level PnL system.

## Limitations

### 1. Stylized variance payoff

The backtest uses a payoff based on:

$$
RV_{fwd}^2 - IV^2
$$

This is useful to study the signal, but it is not a full product valuation framework.

### 2. No explicit maturity book

The current version does not model:

- non-overlapping 20-day swaps,
- an overlapping variance swap book with explicit notionals,
- mark-to-market dynamics.

### 3. No execution frictions

The backtest ignores:

- transaction costs,
- bid/ask spreads,
- slippage,
- liquidity constraints.

### 4. No options surface

IV is proxied through VSTOXX only:

- no strike dimension,
- no expiry structure,
- no options chain calibration.

### 5. Stylized option payoff layer

The option extension uses Black-Scholes and simple expiry payoffs. The ATM straddle helper uses `strike = spot_t` and a Black-Scholes initial cost, then compares that cost with the expiry intrinsic payoff. It ignores:

- bid/ask spreads,
- dividends beyond an optional continuous yield parameter,
- financing and margin,
- volatility surface dynamics,
- option-chain availability,
- early unwind and mark-to-market.

---

## Natural extensions

This repo is meant to be a base, not an endpoint.

Reasonable next steps would be:

### 1. Non-overlapping variance swap backtest

Open one 20-day trade, hold it to maturity, and only book payoff at expiry.

### 2. Overlapping book with explicit exposure

Introduce a ladder of positions, notionals, and a more realistic exposure management.

### 3. Delta-hedged option strategy

Use the same SX5E / VSTOXX universe to backtest:

- ATM straddles,
- Black-Scholes pricing,
- daily delta hedging,
- PnL decomposition.

### 4. Parameter robustness

Study sensitivity to:

- rolling lookback,
- z-score threshold,
- RV window length,
- signal design.

---

## Why this project exists

This project was built as a first serious volatility research project, with three goals in mind:

1. work on something closer to market data and derivatives than pure textbook exercises,
2. practice building a clean quant pipeline from raw data to backtest results,
3. create a reusable foundation for more advanced volatility / options projects later on.

It is somewhere between a research notebook and a small structured quant repo, which is exactly what it is supposed to be.

---

## Author note

This project is best read as:

- a structured IV vs RV research pipeline,
- a stylized long / short volatility backtest,
- and a solid stepping stone toward more realistic volatility trading models.
