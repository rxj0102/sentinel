# Phase 4 notes: portfolio construction

Status: engineering only. Every output so far comes from the synthetic fixture and is labeled synthetic. No performance number is quoted anywhere in the docs, and the REAL-DATA GATE stays open. Phase 5 has not started.

## What was built

- Ledoit-Wolf covariance through scikit-learn, shrinking toward a scaled identity with the analytic intensity, so there is no shrinkage parameter to tune.
- Mean-variance with the shrunk covariance and an assumed information coefficient to scale the composite score into expected returns.
- Risk parity (equal risk contribution) by coordinate descent, and hierarchical risk parity with single linkage on the correlation distance, quasi-diagonalization and recursive bisection.
- One shared constraint set for all three: dollar neutral, gross, per name cap, factor exposure bounds on size, market beta and momentum, and a one way turnover cap that also counts forced closings. Mean-variance enforces it natively. Risk parity and hierarchical risk parity targets pass through a least change projection onto the same set.
- A composite score from all six pre-registered signals with equal weights and locked signs.
- A comparison that reports the pre-registered default cost cell as the headline, every sensitivity cell next to it, per construction diagnostics (effective number of names, risk concentration, realized exposures, infeasible rebalances, shrinkage), and deflated Sharpe with Holm and Benjamini-Hochberg adjustments.
- The equal weight quantile portfolio on the same composite score is the baseline.

## Design choice, and why

Risk parity and hierarchical risk parity are long only constructions and use only a covariance matrix. To compare them with mean-variance on equal terms, names are chosen by the same composite score (top and bottom quintile), the legs are weighted by each rule within the leg, and the result goes through the same constraints. This isolates the weighting rule. It is not the same as a true long-short risk parity, and the comparison says so.

## Assumptions

1. The information coefficient used to scale alpha is an assumption, not an estimate. If the true value is lower, mean-variance overstates its edge.
2. All three constructions use the same covariance, so differences come from the rule and not from the estimator.
3. Constraint settings are pre-registered in docs/hypotheses.md Amendment 1 and are not tuned. Each is listed in ASSUMPTIONS.md.

## What could be wrong

- Mean-variance is sensitive to estimation error in expected returns. The shrunk covariance helps with risk, not with alpha.
- The scaled identity shrinkage target ignores factor structure in returns.
- The per name cap makes gross exposure small in a small universe. A cap of a few percent needs on the order of thirty or more names per leg to reach full gross, so the constraint binds on the synthetic fixture and is a tested mechanism, not a statement about real data.
- If the constraints cannot be met, the previous weights are held and the rebalance is flagged. Flags are counted in the diagnostics and are never silently relaxed.
- Turnover caps slow the initial build of the book.
- Exposure bounds are on standardized exposures within the universe each rebalance.

## Tested

Ledoit-Wolf against an independent implementation of the published formulas, positive definiteness with more assets than observations, trace preservation, and the limits of the shrinkage intensity. Risk parity against closed forms (diagonal, two asset, equal correlation) and equal risk contributions on random matrices. Hierarchical risk parity against a hand computed three asset example, plus permutation invariance. Mean-variance against the unconstrained closed form, constraint satisfaction on random inputs, and variance falling as risk aversion rises. Projection leaves feasible targets unchanged and flags infeasible problems. Weights satisfy every constraint at every rebalance, appear only on rebalance rows after warm-up, hold only index members, and do not change when future data is perturbed. The headline uses only the default cell and matches its own cell in the grid. Every portfolio variant is registered as a trial, cells are not, and adopting a cell as a configuration is counted.
