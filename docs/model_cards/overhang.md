# Model card: overhang

## Economic hypothesis

Disposition effect: unrealized gains create selling pressure that depresses price, which later drifts up toward fundamentals, so high overhang predicts higher returns.

Expected sign of the relation to next period returns at the 21 day primary horizon: positive.
Signs for every horizon are locked in docs/hypotheses.md.
Pre-registered before any real data was touched.

## Construction

Price minus a volume weighted reference price, divided by price.

Minimum history in trading days: 253. Data: prices and volume (yfinance).
Timing: the signal at date t uses data through the close of t.
Positions open at the close of t plus one day.

## Reference

Grinblatt and Han (2005)

## Confidence

LOWER CONFIDENCE. Free data has no shares outstanding history, so the turnover weight in the reference price is a volume ratio proxy. The proxy can misstate the reference price, so any result for this signal carries lower confidence than the others.

## Known caveats

- Turnover weights are approximated without shares outstanding.

## Results

PENDING REAL-DATA GATE. IC, decay, turnover, capacity and orthogonalization results
are produced by the signals research command and may be added here only after the gate
in docs/REAL_DATA_GATE.md is closed.
