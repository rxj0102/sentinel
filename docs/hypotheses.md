# Pre-registration of hypotheses

Status: LOCKED on 2026-10-15, before any real data was loaded. The git history of this file is the timestamp. Nothing in this file was chosen after looking at a result, because no real result exists.

## Rules

1. The sign of every signal at every horizon is locked in the table below. A sign may not be flipped after real data has been seen. If real data contradicts a locked sign, the contradiction is the finding and it is reported as such.
2. Every parameter below is fixed. No lookback, window, quantile, or rebalance frequency may be tuned on results. A different parameter set is a new trial, is logged in the trial registry, and counts in the deflated Sharpe.
3. Any amendment is a new dated entry in the amendments section. The locked table is never edited silently, and a test compares the code registry with this table.
4. Primary horizon is 21 trading days. Rebalance every 21 trading days. Quantile is 0.2 (top quintile long, bottom quintile short). Positions open at the close one day after the signal date.
5. Horizons examined for IC and decay: 1, 5, 10, 21, 63 trading days.

## Locked table

Signs are the expected sign of the relation between the signal value and the forward return. The plus sign means higher signal, higher return.

| signal | parameters | h1 | h5 | h10 | h21 | h63 |
|---|---|---|---|---|---|---|
| max_lottery | window=21 k=5 | - | - | - | - | - |
| high52 | window=252 | + | + | + | + | + |
| reversal | window=21 | - | - | - | - | - |
| overhang | vol_window=252 | + | + | + | + | + |
| realized_skew | window=63 | - | - | - | - | - |
| abnormal_volume | short=5 long=252 | + | + | + | + | - |

## Rationale for the abnormal volume signs

Attention driven buying by retail investors is expected to push prices up over roughly the next month, so the signs are positive from 1 to 21 days. Over a quarter, the price pressure is expected to reverse, so the sign at 63 days is negative. This is the only signal with a sign change across horizons, and it was fixed here before any data was seen.

## Rationale for the other signs

- max_lottery: lottery preference makes high MAX stocks overpriced. Negative at every horizon.
- high52: anchoring causes underreaction near the 52 week high. Positive at every horizon, with the effect expected to persist for months.
- reversal: overreaction and liquidity provision. Negative at every horizon, with the effect expected to fade at long horizons. The locked sign remains negative at 63 days, so a zero at that horizon counts as a non-confirmation, not a confirmation.
- overhang: the disposition effect. Positive at every horizon. Lower confidence because shares outstanding are not available, see the model card.
- realized_skew: lottery preference through skewness. Negative at every horizon.

## Deferred and excluded

- Analyst herding is excluded: no free point in time analyst data exists.
- Option implied skew is forward collection only and is not backtested until at least a year of snapshots exists.
- News tone from GDELT and Google Trends attention are not in the core registry. If added later they require an amendment with a locked sign before their first use.

## Amendments

Each amendment is dated and committed before the code it governs. The locked table above is not edited.

### Amendment 1 (2026-10-15), committed before any Phase 4 code

A. Headline rule for cost cells. The headline result of any backtest uses the pre-registered default cost cell only: the default borrow rate and a cost multiplier of 1 from the config. Every sensitivity cell is reported next to it. The best cell is never chosen as the headline. If any cell is ever used to choose a configuration, that choice is a new trial and is registered as one.

B. Missing returns. A held position whose price series ends (the name vanished) must not be treated as flat. Config option missing_return_treatment, default delisting_adverse: on the first missing day after the last valid price, the position suffers the adverse delisting return, then the position is closed. The magnitude is tied to the milder survivorship scenario in the data quality report, minus 30 percent. This is an assumption, not an estimate. Option delisting_stock_return applies the same return to the stock itself (a short would gain). Option zero keeps the old flat treatment and is allowed only for comparison. A gap in prices that later resumes is treated as a flat day and counted in the output.

C. Universe. Signals are masked by point in time index membership before any ranking or weighting. Before this amendment the backtest ranked every ticker with prices, including names that had not yet joined the index.

D. Phase 4 portfolio construction. All numbers below are fixed now and are not tuned on results. Three constructions (mean-variance, leg-wise risk parity, leg-wise HRP) share the same composite alpha, the same universe, the same covariance estimate and the same constraints, so the comparison isolates the weighting rule.

| parameter | value |
|---|---|
| composite_signals | all six registered signals |
| composite_weights | equal |
| composite_min_signals | 4 |
| winsorize_quantiles | 0.01 0.99 |
| ic_assumed | 0.02 |
| cov_window | 252 |
| cov_min_obs | 252 |
| cov_estimator | ledoit_wolf |
| risk_aversion | 10 |
| leg_quantile | 0.2 |
| min_names | 10 |
| max_gross | 2.0 |
| max_position | 0.03 |
| exposure_tol | 0.10 |
| turnover_cap_one_way | 0.5 |
| delisting_return | -0.30 |

Definitions:

- Composite score: each signal is winsorized across names at the quantiles above, z-scored, multiplied by its locked 21 day sign, and the available scores are averaged with equal weights. At least composite_min_signals must be available. The overhang signal is included at full weight even though it is marked lower confidence. This is stated, not hidden.
- Alpha for mean-variance: monthly expected return equals ic_assumed times the monthly volatility of the name times the composite score. The value of ic_assumed is an assumption, not an estimate from data.
- Covariance: Ledoit-Wolf shrinkage as implemented in scikit-learn, shrinking toward a scaled identity with the analytic intensity, so there is no shrinkage parameter to tune. Estimated on the cov_window trailing daily returns through the signal date and scaled to 21 days. Names with fewer than cov_min_obs observations are excluded from that rebalance.
- Mean-variance: maximize alpha times weights minus half of risk_aversion times weights, covariance, weights, subject to the common constraints.
- Leg-wise risk parity: names are selected as the top and bottom leg_quantile of the composite score, as in the baseline. Within each leg, weights equalize risk contributions using the Ledoit-Wolf covariance. Legs are scaled to plus one and minus one.
- Leg-wise HRP: same legs. Within each leg, the hierarchical risk parity allocation of Lopez de Prado (2016) is applied, that is single linkage clustering on the distance sqrt of half of one minus correlation, quasi-diagonalization, and recursive bisection on cluster variance, as in Lopez de Prado (2016).
- Common constraints, applied to all three: dollar neutral, gross at most max_gross, each name at most max_position in absolute weight, absolute exposure to log dollar volume, market beta and 12 minus 1 momentum each at most exposure_tol, and one way turnover at most turnover_cap_one_way per rebalance. Leg-wise risk parity and leg-wise HRP targets pass through the same constrained projection (least change in L2 norm), mean-variance enforces the constraints natively.
- Baseline for comparison: the equal weight quantile portfolio on the same composite score.
- Trials: composite signal with equal weights (signal), mean-variance, leg-wise risk parity and leg-wise HRP (each a portfolio variant). Each is registered and counted.

### Amendment 2 (2026-10-15), committed before the code it governs

Amendment 1 wording was edited in this commit for names only, to use the leg-wise names below. No parameter or rule in it changed.

A. Names and model description. The two risk based variants are named leg-wise risk parity and leg-wise HRP in code, tables and docs. The unqualified names are not used anywhere, and a CI check enforces that in prose. They are not true long-short risk parity. A true long-short version would equalize the risk contributions of one combined signed book. Here, names are first chosen by the composite score, a long only rule is applied inside each leg separately, the legs are scaled to plus one and minus one, and the result is constrained. The comparison isolates the weighting rule inside the legs and does not test those methods in their textbook form.

B. Replacing the assumed information coefficient. The assumed value stays the default and stays listed in ASSUMPTIONS.md as an assumption. It may be replaced only by the following procedure, which is fixed now.

| parameter | value |
|---|---|
| ic_source | assumed |
| ic_horizon | 21 |
| ic_min_dates | 24 |

- Estimator: the mean of per-date Spearman rank correlations between the composite score at date s and the forward return over ic_horizon trading days (entry at the close one day after s), over point in time index members. Dates s are spaced ic_horizon trading days apart so that labels do not overlap. The standard error is Newey-West and is reported next to the estimate.
- Data: the development window only, 2005 to 2022, never holdout data. The inputs pass through the holdout guard, and the code audit covers the path.
- Point in time use: at rebalance date t, only dates s whose label ends on or before t are used, so no label extends past t. This is an expanding window estimate. A single full-window estimate is not allowed for a backtest over the same window, because it would be in sample for the parameter.
- Minimum history: if fewer than ic_min_dates usable dates exist at t, the assumed value is used for that rebalance, and the fallback is counted and reported.
- Floor: a negative estimate is replaced by zero, because the signs are locked and a negative coefficient would flip them. Each floored rebalance is counted and reported.
- Counting: switching ic_source to the estimate is a new trial of the mean-variance variant. The trial parameters include ic_source, so it is registered and counted when applied, and it is applied at most once.
- Reporting: the estimate, its standard error and the number of dates are reported next to the assumed value. The assumed value is never silently replaced.

C. Feasibility before any real run. With a per name cap and a gross target, each leg needs at least the reciprocal of the cap, times half the gross, in names for the cap to allow full gross. The universe at each rebalance must therefore hold at least that many names per leg divided by the leg quantile. The code computes and prints both numbers from the config. Real data runs always enforce this check and fail loudly with the observed minimum and the date. Synthetic runs report the check and label it a mechanism test only. The check can be switched off only for synthetic runs.

D. Infeasible constraint events. Every portfolio report header states, per construction, the number of rebalances attempted, the number that fell back to held weights because the constraints were infeasible, and the number skipped for lack of inputs. A warning line appears whenever any fallback occurred. There is no threshold below which the counts are hidden.
