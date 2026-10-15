# Model card: high52

## Economic hypothesis

Investors anchor on the 52 week high and underreact to good news for stocks near it, so proximity to the high predicts higher returns.

Expected sign of the relation to next period returns at the 21 day primary horizon: positive.
Signs for every horizon are locked in docs/hypotheses.md.
Pre-registered before any real data was touched.

## Construction

Adjusted close divided by the trailing 252 day maximum close.

Minimum history in trading days: 253. Data: prices and volume (yfinance).
Timing: the signal at date t uses data through the close of t.
Positions open at the close of t plus one day.

## Reference

George and Hwang (2004)

## Confidence

Normal.

## Known caveats

- Replaces analyst herding, see deviations in README.

## Results

PENDING REAL-DATA GATE. IC, decay, turnover, capacity and orthogonalization results
are produced by the signals research command and may be added here only after the gate
in docs/REAL_DATA_GATE.md is closed.
