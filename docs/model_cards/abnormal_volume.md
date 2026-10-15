# Model card: abnormal_volume

## Economic hypothesis

Attention driven retail buying pushes prices up over the next few weeks, then reverses at longer horizons. Signs are locked per horizon in docs/hypotheses.md.

Expected sign of the relation to next period returns at the 21 day primary horizon: positive.
Signs for every horizon are locked in docs/hypotheses.md.
Pre-registered before any real data was touched.

## Construction

Log of the 5 day over 252 day average volume.

Minimum history in trading days: 253. Data: prices and volume (yfinance).
Timing: the signal at date t uses data through the close of t.
Positions open at the close of t plus one day.

## Reference

Barber and Odean (2008)

## Confidence

Normal.

## Known caveats

- Google Trends version is best effort and not in the core pipeline.

## Results

PENDING REAL-DATA GATE. IC, decay, turnover, capacity and orthogonalization results
are produced by the signals research command and may be added here only after the gate
in docs/REAL_DATA_GATE.md is closed.
