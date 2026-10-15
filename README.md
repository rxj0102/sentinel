# Sentinel

Behavioral alpha and risk platform built only on free data. It researches behavioral signals, builds portfolios from them, stress tests those portfolios, and monitors them as if live.

Status: Phase 1 (data layer) in progress. No empirical result on real data exists yet. Every number that will appear in this repo is written by code into results/ and nothing is quoted by hand.

Related work: [behavioral-pricing-kernel](https://github.com/rxj0102/behavioral-pricing-kernel), a separate study of the empirical pricing kernel. Sentinel is standalone and never imports it.

## Reproduce

    make setup
    make all          # lint, types, tests, then the synthetic end to end quality report

CI runs on a deterministic synthetic market, needs no network and no secrets. Real data is rebuilt locally from source (scripts to come in Phase 1b). No Yahoo derived data is published anywhere.

## Layout

See docs/PHASE0.md for the architecture, data inventory, hypotheses, and decision log. COSTS.md lists every dependency and its terms. RISK_REGISTER.md tracks risks.

## Deviations from the original brief

Both were approved by the sponsor and are repeated in the methodology paper.

1. Analyst herding is replaced by 52 week high anchoring, because free sources have no point in time analyst forecast history.
2. Option implied skew is forward collection only. Realized skewness is the backtestable substitute.
