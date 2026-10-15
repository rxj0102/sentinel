# Limitations and known failure modes

Data layer, Phase 1.

1. Yahoo has no delisted tickers, so histories of index members that later vanished are absent. Every return estimate from the free price source is biased upward. The data quality report measures the missing share per year and runs sensitivity scenarios. Those scenarios are assumptions, not estimates, because the missing prices are unknowable from free data.
2. The delisting terminal return scenario is conservative. It is applied to any member whose history ends before the sample end, even when the last print already contains the terminal move, so it can double count.
3. Index membership is rebuilt from a public change log that can have gaps. Effective dates are used as if known on the effective date. Announcement dates would be more accurate for point in time use and are not available in this table.
4. Availability lags are fixed calendar offsets, a simplification. Ken French factors use a conservative sixty day lag because the library updates with delay and revisions.
5. The equal weight survivorship sensitivity ignores trading costs and uses lagged membership as weights. It is a diagnostic of bias, not a strategy.
6. The synthetic market validates code paths and invariants, not economic conclusions. Nothing computed on it may be cited as a finding.
