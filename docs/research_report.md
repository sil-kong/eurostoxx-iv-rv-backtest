# Euro STOXX 50 IV/RV Volatility Research Report

## Research Question
Does the IV/RV spread contain information about future realized variance beyond naive variance-risk-premium exposure?

## Data
- SXE50 / Euro STOXX 50 spot data.
- VSTOXX / V2TX implied volatility proxy data.
- IV is represented as decimal annualized volatility, for example 0.20 for 20%.
- VSTOXX is a broad implied-volatility proxy, not an option chain, volatility surface or variance-swap quote.

## Methodology
- Historical RV is computed from close-to-close log returns and annualized.
- Forward RV is strictly ex-post and reserved for payoff evaluation.
- The signal uses the IV/RV spread and a rolling z-score threshold.
- Daily payoff convention: `signal * (RV_fwd^2 - IV^2)`.
- The non-overlapping layer opens at most one fixed-horizon trade at a time.
- No-look-ahead policy: forward RV is never used to build the signal.

## Main Results
| layer | total_pnl | sharpe_approx | max_drawdown | days_in_position |
| --- | --- | --- | --- | --- |
| Daily IV/RV signal | 12.72 | 0.831 | -4.906 | 2730 |

Non-overlapping layer: 203 trades, cumulative normalized PnL 1.239.

All values are normalized diagnostics, not returns, EUR PnL or executable trading results.

## Benchmark Comparison
| strategy | total_pnl | sharpe_approx | sharpe_non_overlap | max_drawdown | days_in_position |
| --- | --- | --- | --- | --- | --- |
| iv_rv_signal | 12.72 | 0.831 | 0.1969 | -4.906 | 2730 |
| always_short_vol | 48.37 | 2.802 | 0.6501 | -11.39 | 4643 |
| always_long_vol | -48.37 | -2.802 | -0.6501 | -58.68 | 4643 |
| always_flat | 0 | n/a | n/a | -0 | 0 |

`sharpe_approx` annualizes overlapping multi-day payoffs as if they were independent daily returns and is inflated. `sharpe_non_overlap` samples one payoff per horizon and is the more honest risk-adjusted comparison. All strategies are evaluated on the sample where the IV/RV signal is defined.

In this sample, the always-short-vol benchmark has higher total PnL than the IV/RV signal, so the result appears substantially related to naive VRP exposure.

## Walk-Forward Validation
- Average test PnL: 0.3603.
- Positive test folds: 11 / 15.
- Average test Sharpe approximation: 1.539.
- Selected-parameter dispersion: 3 z thresholds, 4 lookbacks, 3 horizons.

## Robustness
| metric | value |
| --- | --- |
| total_pnl_min | -5.463 |
| total_pnl_max | 21.35 |
| sharpe_approx_min | -1.148 |
| sharpe_approx_max | 1.413 |
| max_drawdown_worst | -8.132 |
| grid_points | 48 |

The grid is a stability diagnostic, not a parameter-mining exercise.

## Subperiods / Regimes
| subperiod | total_pnl | sharpe_approx | max_drawdown | days_in_position |
| --- | --- | --- | --- | --- |
| pre_gfc | 0 | n/a | -0 | 0 |
| gfc | 7.76 | 2.531 | -2.369 | 230 |
| euro_crisis | 1.934 | 0.9285 | -4.748 | 445 |
| low_vol_2013_2019 | -0.4397 | -0.1968 | -2.552 | 1056 |
| covid | 0.1426 | 0.07062 | -4.784 | 160 |
| post_covid | 3.325 | 1.415 | -1.672 | 839 |

Largest absolute contribution: `gfc`.

## Cost Sensitivity
| cost_per_signal_change | total_pnl | sharpe_approx | max_drawdown |
| --- | --- | --- | --- |
| 0 | 12.72 | 0.831 | -4.906 |
| 0.0001 | 12.63 | 0.8249 | -4.911 |
| 0.0005 | 12.25 | 0.8003 | -4.929 |
| 0.001 | 11.78 | 0.7695 | -4.953 |
| 0.0025 | 10.37 | 0.6772 | -5.023 |
| 0.005 | 8.027 | 0.5234 | -5.141 |

Costs are stylized signal-change frictions, not bid/ask, market impact or financing costs.

## Interpretation
- In this sample, the always-short-vol benchmark has higher total PnL than the IV/RV signal, so the result appears substantially related to naive VRP exposure.
- The short-vol leg contributes more than the long-vol leg in the daily diagnostic.
- Walk-forward validation has 11 positive test folds out of 15, which is the relevant out-of-sample stability check.
- Subperiod attribution should be read carefully because `gfc` is the largest absolute contributor.
- Raising stylized costs from 0 to 0.005 changes total PnL from 12.72 to 8.027.
- The payoff is useful for signal diagnostics, not for claiming live alpha.
- The VSTOXX proxy keeps the study reproducible but limits desk-level interpretation.

## Professional Limitations
This project is strong as an empirical research diagnostic, but it is not:
- a tradable variance swap backtester;
- an option-chain backtester;
- a volatility surface model;
- a transaction-cost-aware execution simulator;
- a broker-ready trading system.

## Limitations
- VSTOXX proxy, no option chain.
- No volatility surface.
- No bid/ask or execution model.
- Stylized variance payoff.
- Not live trading alpha.
- No margin/funding.
- No desk-level variance swap pricing.

## Next Steps
- Real option-chain extension.
- Delta-hedged straddle proxy.
- Variance swap maturity ladder.
- Richer regime classifier.
- Bootstrap confidence intervals.
