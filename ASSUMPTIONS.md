# Assumptions: every placeholder default

Nothing in this file is an estimate. Each value is a placeholder chosen so the engine runs and the sensitivity grid has a sensible center. No value was measured, calibrated, or verified against live data, and none was tuned on results. The source column says where the value comes from, and where it says no source, none exists. Every one of these is a config value, so a reader can change it, and results must be read across the sensitivity grid.

A test checks that every numeric field of the backtest and portfolio configs is listed here.

## Execution and cost defaults (backtest config)

| field | default | source or reason |
|---|---|---|
| delay | 1 | Convention: trade at the close after the signal date so same-day information is never used. Not an estimate. |
| rebalance_every | 21 | Pre-registered, roughly monthly. Chosen before any result. |
| quantile | 0.2 | Pre-registered, quintile sorts are the common convention in the literature. Chosen before any result. |
| aum | 10000000 | Placeholder round number. It only scales market impact. No source. |
| commission_bps | 0.5 | Placeholder. Order of magnitude guess for institutional equity commissions. No source consulted. |
| half_spread_bps | 2.0 | Placeholder. Order of magnitude guess for large cap quoted half spreads. Real spreads vary by name and date and were not measured. |
| slippage_bps | 1.0 | Placeholder for execution shortfall beyond the spread. No source. |
| impact_coeff | 0.1 | Placeholder coefficient of a square root impact model, impact equals coefficient times daily volatility times the square root of trade over average dollar volume. The functional form is common, this specific value is not calibrated. |
| max_position | 0.03 | Design choice for diversification, share of NAV per name. Not an estimate. |
| max_gross | 2.0 | Design choice, long one and short one. Not an estimate. |
| max_participation | 0.05 | Diagnostic threshold, a common convention for share of average daily volume. Not enforced as a constraint. |
| borrow_rate_annual | 0.01 | ASSUMPTION. Flat rate on the short book. No free data source for borrow costs exists. Hard to borrow names can cost far more, so the grid runs from 0 to 5 percent. |
| borrow_grid | 0, 0.005, 0.01, 0.02, 0.05 | Spread of plausible levels so that no single borrow assumption carries a conclusion. |
| cost_multipliers | 0.5, 1, 2, 4 | Scales all linear costs and impact together, to show how fragile a result is to cost misspecification. |
| delisting_return | -0.30 | ASSUMPTION. Tied to the milder survivorship scenario in the data quality report. It is the terminal return applied adversely to a held position in a name that vanishes. Not an estimate. The upper bound scenario uses a larger loss. |
| missing_return_treatment | delisting_adverse | Conservative default. Option zero is non-conservative and exists for comparison only. |

## Portfolio construction defaults (portfolio config)

| field | default | source or reason |
|---|---|---|
| ic_assumed | 0.02 | ASSUMPTION. Scales composite scores into expected returns for mean-variance. Not estimated from data. If the real information coefficient is lower, mean-variance overstates its edge. |
| cov_window | 252 | Pre-registered, one year of daily returns. Convention. |
| cov_min_obs | 252 | Pre-registered, names need the full window. |
| risk_aversion | 10 | Placeholder, sets how much variance is penalized against alpha. Chosen before any result. |
| leg_quantile | 0.2 | Pre-registered, same as the baseline quantile. |
| composite_min_signals | 4 | Pre-registered, minimum available signals for a composite score. |
| min_names | 10 | Pre-registered, below this the rebalance holds zero weights. |
| exposure_tol | 0.10 | Placeholder tolerance on factor exposure in standardized units. Chosen before any result. |
| turnover_cap_one_way | 0.5 | Placeholder cap on one way turnover per rebalance. Chosen before any result. |

## Estimation windows inside the cost model

The cost model uses a trailing 21 day volatility and average dollar volume, lagged one day, with at least 10 observations. These are conventions, not estimates, and the names without enough history get zero impact and are counted in the output.
