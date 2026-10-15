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
| R7 | Synthetic engineering results leak into docs as findings | Medium | High | REAL-DATA GATE (issue 1), sidecar labels, CI docs-check | Mitigated in code |
| R8 | Rebuild scripts fail on live endpoints because they are untested there | High | Medium | Offline parser tests, doctor command, sponsor returns logs, fixes land in parsers | Open |
| R9 | Ticker reuse misattributes prices to index members (survivorship risk) | Medium | High | Data quality report flags tickers whose first price date is late against the index add date, tickers with no prices, and tickers that left and returned. Flags are reviewed before the REAL-DATA GATE closes. | Open, measured on real data only |
| R10 | Mean-variance amplifies error in the assumed information coefficient | Medium | Medium | The coefficient is an assumption fixed in the pre-registration. Results are compared with simpler constructions and read across the cost grid. | Open |
| R11 | Portfolio constraints become infeasible and weights are held | Medium | Low | Fallback holds previous weights, flags and counts every event, never relaxes silently | Mitigated in code |
| R12 | Cherry-picking the best cost cell as a headline | Medium | High | Headline is the pre-registered default cell only, every cell is reported, adopting a cell as a configuration is registered as a trial. Structural tests enforce this. | Mitigated in code |
