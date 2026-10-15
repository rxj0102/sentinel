# Pre-registration of hypotheses

Status: LOCKED on 2026-10-15, before any real data was loaded. The git history of this file is the timestamp. Nothing in this file was chosen after looking at a result, because no real result exists.

## Rules

1. The sign of every signal at every horizon is locked in the table below. A sign may not be flipped after real data has been seen. If real data contradicts a locked sign, the contradiction is the finding and it is reported as such.
2. Every parameter below is fixed. No lookback, window, quantile, or rebalance frequency may be tuned on results. A different parameter set is a new trial, is logged in the trial registry, and counts in the deflated Sharpe.
3. Any amendment is a new dated entry in the amendments section. The locked table is never edited silently, and a test compares the code registry with this table.
4. Primary horizon is 21 trading days. Rebalance every 21 trading days. Quantile is 0.2 (top quintile long, bottom quintile short). Positions open at the close one day after the signal date.
5. Horizons examined for IC and decay: 1, 5, 10, 21, 63 trading days.

## Locked table

Signs are the expected sign of the relation between the signal value and the forward return. The plus sign means higher signal, higher return.

| signal | parameters | h1 | h5 | h10 | h21 | h63 |
|---|---|---|---|---|---|---|
| max_lottery | window=21 k=5 | - | - | - | - | - |
| high52 | window=252 | + | + | + | + | + |
| reversal | window=21 | - | - | - | - | - |
| overhang | vol_window=252 | + | + | + | + | + |
| realized_skew | window=63 | - | - | - | - | - |
| abnormal_volume | short=5 long=252 | + | + | + | + | - |

## Rationale for the abnormal volume signs

Attention driven buying by retail investors is expected to push prices up over roughly the next month, so the signs are positive from 1 to 21 days. Over a quarter, the price pressure is expected to reverse, so the sign at 63 days is negative. This is the only signal with a sign change across horizons, and it was fixed here before any data was seen.

## Rationale for the other signs

- max_lottery: lottery preference makes high MAX stocks overpriced. Negative at every horizon.
- high52: anchoring causes underreaction near the 52 week high. Positive at every horizon, with the effect expected to persist for months.
- reversal: overreaction and liquidity provision. Negative at every horizon, with the effect expected to fade at long horizons. The locked sign remains negative at 63 days, so a zero at that horizon counts as a non-confirmation, not a confirmation.
- overhang: the disposition effect. Positive at every horizon. Lower confidence because shares outstanding are not available, see the model card.
- realized_skew: lottery preference through skewness. Negative at every horizon.

## Deferred and excluded

- Analyst herding is excluded: no free point in time analyst data exists.
- Option implied skew is forward collection only and is not backtested until at least a year of snapshots exists.
- News tone from GDELT and Google Trends attention are not in the core registry. If added later they require an amendment with a locked sign before their first use.

## Amendments

None.
