# Model card: max_lottery

## Economic hypothesis

Investors overpay for lottery payoffs, so stocks with extreme recent up days are overpriced and earn lower future returns. Limits to arbitrage via short constraints.

Expected sign of the relation to next period returns at the 21 day primary horizon: negative.
Signs for every horizon are locked in docs/hypotheses.md.
Pre-registered before any real data was touched.

## Construction

Mean of the five largest daily returns in the past 21 trading days.

Minimum history in trading days: 22. Data: prices and volume (yfinance).
Timing: the signal at date t uses data through the close of t.
Positions open at the close of t plus one day.

## Reference

Bali, Cakici, Whitelaw (2011)

## Confidence

Normal.

## Known caveats

- Effect concentrates in small illiquid names, weaker in large caps.

## Results

PENDING REAL-DATA GATE. IC, decay, turnover, capacity and orthogonalization results
are produced by the signals research command and may be added here only after the gate
in docs/REAL_DATA_GATE.md is closed.
