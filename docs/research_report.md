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
| Daily IV/RV signal | 8.222 | 0.7535 | -4.784 | 2227 |

Non-overlapping layer: 164 trades, cumulative normalized PnL 0.4741.

All values are normalized diagnostics, not returns, EUR PnL or executable trading results.

## Benchmark Comparison
| strategy | total_pnl | sharpe_approx | max_drawdown | days_in_position |
| --- | --- | --- | --- | --- |
| iv_rv_signal | 8.222 | 0.7535 | -4.784 | 2227 |
| always_short_vol | 47.83 | 2.789 | -11.39 | 4705 |
| always_long_vol | -47.83 | -2.789 | -55.74 | 4705 |
| always_flat | 0 | n/a | -0 | 0 |

In this sample, the always-short-vol benchmark has higher total PnL than the IV/RV signal, so the result appears substantially related to naive VRP exposure.

## Walk-Forward Validation
- Average test PnL: 0.4243.
- Positive test folds: 11 / 14.
- Average test Sharpe approximation: 1.859.
- Selected-parameter dispersion: 3 z thresholds, 4 lookbacks, 2 horizons.

## Robustness
| metric | value |
| --- | --- |
| total_pnl_min | -5.621 |
| total_pnl_max | 19.28 |
| sharpe_approx_min | -1.389 |
| sharpe_approx_max | 1.328 |
| max_drawdown_worst | -7.586 |
| grid_points | 48 |

The grid is a stability diagnostic, not a parameter-mining exercise.

## Subperiods / Regimes
| subperiod | total_pnl | sharpe_approx | max_drawdown | days_in_position |
| --- | --- | --- | --- | --- |
| pre_gfc | 0 | n/a | -0 | 0 |
| gfc | 0 | n/a | 0 | 0 |
| euro_crisis | 5.116 | 3.652 | -1.321 | 287 |
| low_vol_2013_2019 | -0.4397 | -0.1968 | -2.552 | 1056 |
| covid | 0.1426 | 0.07062 | -4.784 | 160 |
| post_covid | 3.403 | 1.606 | -1.672 | 724 |

Largest absolute contribution: `euro_crisis`.

## Cost Sensitivity
| cost_per_signal_change | total_pnl | sharpe_approx | max_drawdown |
| --- | --- | --- | --- |
| 0 | 8.222 | 0.7535 | -4.784 |
| 0.0001 | 8.147 | 0.7467 | -4.785 |
| 0.0005 | 7.851 | 0.7193 | -4.79 |
| 0.001 | 7.48 | 0.6852 | -4.797 |
| 0.0025 | 6.367 | 0.5828 | -4.816 |
| 0.005 | 4.512 | 0.4122 | -4.849 |

Costs are stylized signal-change frictions, not bid/ask, market impact or financing costs.

## Interpretation
- In this sample, the always-short-vol benchmark has higher total PnL than the IV/RV signal, so the result appears substantially related to naive VRP exposure.
- The short-vol leg contributes more than the long-vol leg in the daily diagnostic.
- Walk-forward validation has 11 positive test folds out of 14, which is the relevant out-of-sample stability check.
- Subperiod attribution should be read carefully because `euro_crisis` is the largest absolute contributor.
- Raising stylized costs from 0 to 0.005 changes total PnL from 8.222 to 4.512.
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
