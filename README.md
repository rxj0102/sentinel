# Sentinel

Behavioral alpha and risk platform built only on free data. It researches behavioral signals, builds portfolios from them, stress tests those portfolios, and monitors them as if live.

Status: Phase 1 code gate closed pending CI and the REAL-DATA GATE. Phase 2 to Phase 4 engineering (signals, backtest engine, validation framework, portfolio construction) run on synthetic data only. Phase 5 waits for the real-data logs.

Where to look first: ASSUMPTIONS.md (every placeholder default), docs/hypotheses.md (pre-registration and amendments), docs/REAL_DATA_GATE.md (what blocks results), docs/REAL_DATA_RUNBOOK.md (first real run). Pre-registration is locked in docs/hypotheses.md. No empirical result on real data exists yet. Every number that will appear in this repo is written by code into results/ and nothing is quoted by hand.

Related work: [behavioral-pricing-kernel](https://github.com/rxj0102/behavioral-pricing-kernel), a separate study of the empirical pricing kernel. Sentinel is standalone and never imports it.

## Python version

Sentinel needs Python 3.11 or 3.12 (the pin is below 3.13). On a Mac, either of these works.

With uv:

    brew install uv
    uv venv --python 3.11 .venv
    source .venv/bin/activate
    uv pip install -e ".[dev,live]"

With pyenv:

    brew install pyenv
    pyenv install 3.11
    pyenv local 3.11
    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev,live]"

## Reproduce
    make all          # lint, types, tests, synthetic quality report, docs gate check

CI runs on a deterministic synthetic market, needs no network and no secrets. No Yahoo derived data is published anywhere.

## Rebuilding real data locally

UNTESTED AGAINST LIVE ENDPOINTS. Parsers have offline contract tests, the network code does not. Run on a machine with network access and send back the full log and any errors.

    export SENTINEL_USER_AGENT="sentinel-research your-email@example.com"
    sentinel data doctor                         # connectivity check for every host
    sentinel data rebuild --dry-run              # universe only, no price download
    sentinel data rebuild --limit 40             # small smoke test
    sentinel data rebuild                        # full rebuild, log in logs/rebuild.log
    sentinel data quality --fixture data/curated # real data survivorship report

The rebuild stops at the end of the development window, so the holdout is not downloaded.
Optional: set FRED_API_KEY (free key, no card) for macro series.

If the first run fails, see docs/REAL_DATA_RUNBOOK.md for the five most likely failure points and the log line each produces.

## REAL-DATA GATE

No signal result, figure, or number from Phase 2 onward may appear in any doc until the real-data survivorship report exists and the monthly membership count reconciles to about 500. See docs/REAL_DATA_GATE.md. Outputs from the synthetic fixture are written under results/synthetic [SYNTHETIC-ENGINEERING-ONLY], carry a synthetic label, and CI fails if docs cite them as findings (sentinel docs-check).

## Layout

See docs/PHASE0.md for the architecture, data inventory, hypotheses, and decision log. COSTS.md lists every dependency and its terms. RISK_REGISTER.md tracks risks.

## Deviations from the original brief

Both were approved by the sponsor and are repeated in the methodology paper.

1. Analyst herding is replaced by 52 week high anchoring, because free sources have no point in time analyst forecast history.
2. Option implied skew is forward collection only. Realized skewness is the backtestable substitute.
