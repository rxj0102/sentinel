# REAL-DATA GATE

Tracked in GitHub issue 1 of this repo.

Status: OPEN

Scope, added by Amendment 3 of docs/hypotheses.md on 2026-11-15: the gate covers every stock-level result. The first real rebuild failed both conditions below, so it stays open. Findings from the survivorship report itself and from the published portfolios track live under evidence/ and may be cited while the gate is open. Stock-level backtests were run once on the survivor-only data on 2026-11-15 to exercise the pipeline; they are registered in validation/trial_log.csv and are not reported anywhere.

While the status above is OPEN, no signal result, figure, or number from Phase 2 onward may appear in any doc. CI enforces this with sentinel docs-check.

The gate closes only when both hold, and only the sponsor changes the status line:

1. The real-data survivorship report exists, produced by a local rebuild on real data.
2. The monthly index membership count reconciles to about 500, with gaps documented.

How findings are cited once the gate is closed: write a line tagged with the finding marker and a src path to a real result file under results, for example the marker followed by src equals results/some_result.json. The result must have a sidecar saying it is not synthetic. Outputs from the synthetic fixture live under results/synthetic and may only be mentioned on lines tagged with the synthetic engineering only marker.
