# eurostoxx-iv-rv-backtest

Backtest of an implied vs realized volatility strategy on Euro STOXX 50
(SX5E / VSTOXX), with a reproducible research pipeline, explicit
no-look-ahead conventions, robustness diagnostics, and stylized variance-swap
payoff analysis.

This is a quant research project, not a production trading engine.

## Project Pitch

The project studies whether a simple IV/RV signal on Euro STOXX 50 contains
useful information about subsequent realized variance. It combines:

- SX5E spot data from Yahoo Finance,
- VSTOXX / V2TX as a broad proxy for implied volatility,
- historical realized volatility,
- forward realized volatility for ex-post payoff evaluation,
- a rolling IV/RV z-score signal,
- stylized variance-swap payoff diagnostics,
- Black-Scholes and option payoff utilities,
- parameter robustness, yearly diagnostics and research figures.

The goal is to be transparent and testable enough for a quant interview or
junior research portfolio discussion, while staying honest about simplifications.

## What The Project Does

The pipeline:

1. downloads and cleans SX5E and VSTOXX data,
2. builds historical RV features,
3. builds strictly future forward RV for ex-post evaluation,
4. constructs an IV/RV z-score signal,
5. runs a daily stylized variance payoff,
6. runs a more conservative non-overlapping 20-day payoff layer,
7. exports performance, robustness and regime diagnostics,
8. generates research figures.

## What Is Quant-Grade About It

- Explicit time conventions and no-look-ahead tests.
- Reusable analytics instead of duplicated script logic.
- Unit tests for volatility features, signals, payoff conventions, options and
  performance statistics.
- Parameter sensitivity analysis across thresholds, lookbacks and horizons.
- Clear limitations: no option chain, no volatility surface, no desk-level
  variance-swap valuation and no execution model.
- Standard Python packaging with editable installation and `python -m` commands.

## Methodology

Daily log returns are defined as:

$$
r_t = \ln\left(\frac{S_t}{S_{t-1}}\right)
$$

Historical realized volatility over a window \(w\) is:

$$
RV_{w,t} = \mathrm{std}(r_{t-w+1}, \ldots, r_t)\sqrt{252}
$$

The signal uses the IV/RV spread:

$$
s_t = IV_t - RV_{w,t}
$$

and a rolling z-score:

$$
z_t = \frac{s_t - \mu_t}{\sigma_t}
$$

Default rule:

- `signal_vol = -1` if `z > z_entry` -> short volatility bias,
- `signal_vol = +1` if `z < -z_entry` -> long volatility bias,
- `signal_vol = 0` otherwise.

## Data Sources

Spot:

- Yahoo Finance ticker `^STOXX50E`
- daily OHLCV fields

Implied volatility:

- STOXX historical V2TX text file
- `iv = vstoxx_close / 100`

VSTOXX is used as a broad implied-volatility proxy. It is not treated as a full
option surface and does not provide strike or expiry dimensions.

## No-Look-Ahead Policy

At date \(t\):

- the signal uses only data available at \(t\),
- `rv_20d_t` may use returns up to and including \(t\), observable after close,
- `rv_fwd_20d_t` uses strictly future returns from \(t+1\) to \(t+20\),
- `rv_fwd_*` is only for ex-post payoff evaluation,
- `signal_vol_t` must not use `rv_fwd_*`.

Dedicated tests verify that changing future IV/RV or forward RV does not change
the signal at \(t\), and that missing forward RV produces controlled payoff
behavior.

## Backtest Layers

### 1. Daily Diagnostic Stylized Payoff

The daily diagnostic payoff is:

$$
PnL_t = signal_t \cdot (RV_{fwd,t}^2 - IV_t^2)
$$

It is normalized, fee-free by default, and useful for studying signal alignment
with future realized variance. It is not a realistic daily mark-to-market of a
live product.

A simplified `cost_per_signal_change` parameter is available for sensitivity
checks. It is charged per unit of absolute signal change, e.g. `0 -> +1` costs
one unit and `+1 -> -1` costs two units, only on rows with valid IV and forward
RV. It is a placeholder friction model, not a bid/ask or execution simulator.

### 2. Non-Overlapping 20-Day Stylized Variance Trade

The non-overlapping layer opens at most one trade at a time:

- if `signal_vol_t != 0`, open a trade at \(t\),
- hold for exactly 20 trading rows,
- do not open another trade until maturity,
- book payoff at maturity,
- use the same stylized variance payoff convention.

This is more conservative than the daily diagnostic layer because it avoids
implicitly opening a new overlapping trade every day.

## Results

Current daily diagnostic run on the local sample:

```text
Total PnL       : 8.222
Annualized mean : 0.4379
Annualized vol  : 0.5812
Sharpe approx   : 0.75
Max drawdown    : -4.784
Days in position: 2227 / 4731 (47.1%)
Long-vol PnL    : -5.993
Short-vol PnL   : 14.214
```

Current non-overlapping 20-day run:

```text
Trades          : 164
Cumulative PnL : 0.474
```

These are normalized research diagnostics. They are not EUR PnL, portfolio
returns, live trading results or desk-level variance-swap marks.

## Robustness Analysis

The robustness script runs a grid over:

- `z_entry`: 0.5, 1.0, 1.5, 2.0
- `lookback`: 63, 126, 252, 504
- `rv_window / forward horizon`: 10, 20, 30

For each combination it exports:

- total PnL,
- annualized mean and volatility,
- approximate Sharpe,
- max drawdown,
- days in position,
- long-vol and short-vol PnL.

The purpose is sensitivity analysis, not parameter optimization. The grid is
intended to show whether the signal remains broadly stable across reasonable
assumptions, not to cherry-pick the highest Sharpe.

## Options Pricing Extension

The `options/` package contains:

- European Black-Scholes call and put prices,
- delta, gamma, vega and annual theta,
- implied volatility inversion by bisection,
- call, put and straddle expiry payoffs,
- a simplified ATM straddle expiry PnL helper.

This connects the IV/RV research idea to option pricing mechanics, but it still
does not use real option chains, bid/ask quotes or volatility-surface dynamics.

## How To Run

Install in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Build the full local pipeline:

```bash
python -m eurostoxx_iv_rv_backtest.scripts.getdata
python -m eurostoxx_iv_rv_backtest.scripts.build_rv
python -m eurostoxx_iv_rv_backtest.scripts.build_signals
python -m eurostoxx_iv_rv_backtest.scripts.run_backtest_iv_rv
python -m eurostoxx_iv_rv_backtest.scripts.run_non_overlapping_varswap
python -m eurostoxx_iv_rv_backtest.scripts.analyze_backtest
python -m eurostoxx_iv_rv_backtest.scripts.run_diagnostics
python -m eurostoxx_iv_rv_backtest.scripts.run_robustness
python -m eurostoxx_iv_rv_backtest.scripts.generate_figures
```

Run tests and quality checks:

```bash
python -m pytest
python -m compileall src
python -m ruff check .
```

The historical wrapper still works after editable installation:

```bash
python data/raw/getdata.py --help
```

## Outputs

Main generated files:

- `data/raw/SXE50_with_IV_daily_20y.csv`
- `outputs/SXE50_with_IV_RV_daily_20y.csv`
- `outputs/SXE50_with_IV_RV_daily_20y_with_signals.csv`
- `outputs/SXE50_iv_rv_varswap_backtest.csv`
- `outputs/SXE50_iv_rv_non_overlapping_varswap_backtest.csv`
- `outputs/yearly_performance.csv`
- `outputs/regime_performance.csv`
- `outputs/drawdown_periods.csv`
- `outputs/crisis_periods.csv`
- `outputs/robustness_grid.csv`
- `outputs/figures/*.png`

Generated data and figures are intentionally ignored by Git.

## Repository Structure

```text
.
├── data/raw/getdata.py
├── docs/
├── outputs/
│   └── figures/
├── src/eurostoxx_iv_rv_backtest/
│   ├── analytics/
│   ├── backtesting/
│   ├── data/
│   ├── features/
│   ├── options/
│   └── scripts/
└── tests/
```
### Basic interpretation

- In the current sample and with the current stylized assumptions, the strategy delivers a positive cumulative normalized payoff.
- The Sharpe ratio is a descriptive statistic for this simplified payoff, not a live trading performance claim.
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
- proper volatility feature engineering,
- explicit historical and forward RV conventions,
- simple but structured signal generation,
- reproducible backtest pipeline,
- no-look-ahead tests,
- robustness diagnostics,
- useful visual diagnostics,
- clear base for more advanced volatility or option strategies.

---

## What This Project Is Not

This project is not:

- a production trading engine,
- a real options surface or calibration framework,
- an options-chain backtest,
- an order-book or execution simulator,
- a transaction-cost-aware portfolio system,
- a live variance-swap book,
- desk-level variance-swap valuation.

## Limitations

- VSTOXX is a broad proxy for implied volatility, not a full surface.
- The payoff is stylized and normalized.
- No bid/ask, slippage, financing, margin or liquidity constraints are modeled.
- The non-overlapping backtest is still a simplified hold-to-expiry diagnostic.
- The options layer is educational and Black-Scholes based.
- Robustness grids are sensitivity checks, not a license to tune parameters.

## Future Extensions

- Add an overlapping variance trade book with explicit maturity ladders.
- Add a carefully documented delta-hedged straddle experiment.
- Add bootstrap or subperiod stability analysis.
- Add richer drawdown and crisis-period attribution.
- Add optional notebooks or a short report generated from the CSV outputs.
