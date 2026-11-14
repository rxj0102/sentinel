# Model card: reversal

## Economic hypothesis

Overreaction and liquidity provision: last month winners underperform next month, losers outperform.

Expected sign of the relation to next period returns at the 21 day primary horizon: negative.
Signs for every horizon are locked in docs/hypotheses.md.
Pre-registered before any real data was touched.

## Construction

Return over the past 21 trading days.

Minimum history in trading days: 22. Data: prices and volume (yfinance).
Timing: the signal at date t uses data through the close of t.
Positions open at the close of t plus one day.

## Reference

Jegadeesh (1990)

## Confidence

Normal.

## Known caveats

- Very high turnover, likely killed by costs. Kept as a cost honesty case.

## Results

PENDING REAL-DATA GATE. IC, decay, turnover, capacity and orthogonalization results
are produced by the signals research command and may be added here only after the gate
in docs/REAL_DATA_GATE.md is closed.
