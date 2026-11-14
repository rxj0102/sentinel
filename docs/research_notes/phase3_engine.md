# Phase 3 notes: backtest engine and validation framework

Status: engineering only. Every output so far comes from the synthetic fixture and is labeled synthetic. No performance number is quoted anywhere in the docs, and the REAL-DATA GATE stays open.

## What was built

- Vectorized daily engine with a fixed timing rule: weights decided at close t trade at close t plus delay and first earn the return of day t plus delay plus 1. Costs are booked on the trade date.
- Costs from config: commission, half spread, slippage, square root impact, a flat borrow rate on the short book, per name position cap, gross cap, and a participation diagnostic. Defaults live in configs and are assumptions, not measurements.
- Sensitivity grid built in: borrow rate by cost multiplier. The borrow rate has no free data source and is labeled as an assumption in the config, the config validator, and the results.
- Holdout guard: the engine asks the guard before touching any date. Holdout dates need one explicit unlock with a reason, a locked end date in the config, and no earlier unlock in the log. Every denied or granted access is appended to a log with a timestamp and commit hash. The real log is holdout/access_log.csv and is committed. Synthetic runs log under results/synthetic [SYNTHETIC-ENGINEERING-ONLY].
- Trial registry: the engine will not run without a trial spec, and every run is registered first. Distinct counted trials feed the deflated Sharpe. Identical reruns are flagged, not hidden. Sensitivity cells are logged separately and excluded from the count.
- Statistics: Sharpe, probabilistic Sharpe, deflated Sharpe, Holm and Benjamini-Hochberg adjustments.
- Walk-forward and purged k-fold with purge and embargo. Every split is checked for label overlap before it is yielded, and tests re-check it with an independent brute force method.

## Assumptions

1. Labels span delay plus horizon, so the purge gap is that span plus the embargo.
2. Trials are treated as independent in the deflation. Correlated trials mean the effective count is lower, so the deflation is conservative.
3. Sensitivity cells reprice the same weights, so they are not alternative strategies. This is a judgment and it is stated in the output.
4. Impact coefficient, spread, slippage, commission and borrow are placeholders for a liquid large cap universe.

## What could be wrong

- Turnover and costs use absolute weight changes and ignore drift between rebalances.
- Missing returns are treated as zero, so a position in a vanished name is flat. This understates delisting losses.
- No cash return or financing on the long book.
- The participation cap is a diagnostic, not a constraint. Trades are not scaled down.
- Walk-forward and purged folds on rule based signals show stability over time, not out of sample skill, because nothing is fitted.
- Cost inputs need 10 days of history. Early trades have zero impact and are counted in the engine output.

## Bugs found by review and tests during this phase

1. Phase 2 gross_ls_return shifted weights by delay only, so a trade at the signal close earned the next return. Fixed to delay plus one. No Phase 2 output used this function.
2. A single holdout unlock was consumed by each backtest call, so a multi-signal run hit the once-only rule. Fixed with a grant object that covers one run, with every read still logged.
3. The synthetic holdout log path resolved against the working directory instead of the results folder. Fixed.
4. Mutation checks: removing the holding lag in the engine, removing the embargo from the walk-forward gap, and bypassing the holdout check each make tests fail.

## Tested

Known answer cost and borrow math, timing, limits, no look ahead under perturbation, monotone sensitivity grid, holdout refusal without the flag, grant rules, once-only unlock across guard instances, trial counting across processes, DSR against independent formulas, multiple testing known answers and ordering properties, and no overlap including embargo for both split schemes.

## Corrections after Phase 3 review (Amendment 1)

5. Missing returns were counted as zero, so a held position in a vanished name looked flat. The engine now applies a conservative adverse delisting return by default (config option, tied to the milder survivorship scenario, an assumption) and closes the position. A test shows a delisted name does not look flat, for longs and shorts. A price gap that resumes stays flat and is counted.
6. Signals were ranked over every ticker with prices, including names that had not yet joined the index. Signals are now masked to point in time membership before ranking.
7. The new holdout audit found that the membership change log was read without the guard, and a change dated inside the holdout would reveal holdout period information. The log is now clipped to the development window before use.
8. Cost sensitivity rule: the headline result uses the pre-registered default cell only, every cell is reported, and using a cell to choose a configuration makes it a trial.
