# Model card: overhang

## Economic hypothesis

Disposition effect: unrealized gains create selling pressure that depresses price, which later drifts up toward fundamentals, so high overhang predicts higher returns.

Expected sign of the relation to next period returns: positive.
Pre-registered before any real data was touched.

## Construction

Price minus a volume weighted reference price, divided by price.

Minimum history in trading days: 253. Data: prices and volume (yfinance).
Timing: the signal at date t uses data through the close of t.
Positions open at the close of t plus one day.

## Reference

Grinblatt and Han (2005)

## Known caveats

- Turnover weights are approximated without shares outstanding.

## Results

PENDING REAL-DATA GATE. IC, decay, turnover, capacity and orthogonalization results
are produced by the signals research command and may be added here only after the gate
in docs/REAL_DATA_GATE.md is closed.
