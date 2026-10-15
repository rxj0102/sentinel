# Model card: realized_skew

## Economic hypothesis

Lottery preference through skewness: positively skewed stocks are overpriced.

Expected sign of the relation to next period returns at the 21 day primary horizon: negative.
Signs for every horizon are locked in docs/hypotheses.md.
Pre-registered before any real data was touched.

## Construction

Skewness of daily returns over 63 trading days.

Minimum history in trading days: 64. Data: prices and volume (yfinance).
Timing: the signal at date t uses data through the close of t.
Positions open at the close of t plus one day.

## Reference

Amaya, Christoffersen, Jacobs, Vasquez (2015)

## Confidence

Normal.

## Known caveats

- Backtestable substitute for option implied skew, which is forward only.

## Results

PENDING REAL-DATA GATE. IC, decay, turnover, capacity and orthogonalization results
are produced by the signals research command and may be added here only after the gate
in docs/REAL_DATA_GATE.md is closed.
