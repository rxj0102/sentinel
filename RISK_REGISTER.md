# Risk register

Living file. Phase 1 entries only so far. Likelihood and impact are the builder judgement, not measured.

| ID | Risk | Likelihood | Impact | Mitigation | Status |
|---|---|---|---|---|---|
| R1 | Survivorship bias from missing delisted prices inflates every return estimate | High | High | Per year missing price share and sensitivity scenarios are a required output of the data quality report. All alpha is reported as an upper bound. | Open, measured on real data only after local rebuild |
| R2 | Look-ahead through revised or late data | Medium | High | available_date on every table, strict as-of joins, leakage tests | Mitigated in code, re-tested per phase |
| R3 | Data snooping across signals, horizons and portfolio methods | High | High | Pre-registration, trial counting, deflated Sharpe, one time holdout | Planned, Phase 3 |
| R4 | Unofficial source breakage prevents reproduction | High | Medium | Parsers separated from fetchers with offline contract tests, local rebuild script, graceful degradation | Open |
| R5 | Synthetic CI data hides real data problems | Medium | Medium | Synthetic generator reproduces delistings and absent histories. Real data checks run locally. | Open |
| R6 | Wikipedia change log gaps distort membership | Medium | Medium | Reconcile member count to the expected index size each month and report gaps | Planned, needs real data |
