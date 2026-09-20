# Phase 8H Post-Blind Recovery — 4/6

Date: 2026-09-20

## Genuine blind baseline

Frozen parent architecture:
- phase8g-composition-6of6
- commit ef3be7c

Phase 8H genuine blind baseline:
- 0/6

Blind cohort SHA256:
- eb9fdf1e6955f69730e5890fe438d0e96773453b59fe812d995e997629204fba

Blind summary SHA256:
- 91660d5ecf09354054048f3dcfc09be88594c9fe4c27dd1d8c1807acca400921

## First architecture increment

Reusable stochastic-process composition recovered:

- t1-crypto-funding-rate-basis-carry: reward 1.0
- t1-geometric-mean-reverting-jd: reward 1.0

Milestone:
- phase8h-process-recovery-2of6
- commit aba7a3d959bf69f3feae0b24027a749e5a64ba00

Process recovery summary SHA256:
- 9da08adbd29b8b7234c909f1d81d8c8e568c5dae6df69ad6d14b35a6607bac57

## Second architecture increment

Added reusable derivatives and Monte Carlo capabilities:

- historical close-to-close volatility calibration
- Black-Scholes pricing and Greeks
- forward-start ATM option pricing
- cliquet aggregation
- GBM simulation from reusable normal draws
- common-random-number finite-difference Greeks
- pathwise Greeks
- likelihood-ratio Greeks
- European and arithmetic-Asian Monte Carlo pricing
- Greek surfaces
- convergence studies

Recovered:

- t1-cliquet-ratchet-pricing: reward 1.0
- t1-mc-greek-surface-1: reward 1.0

Final derivatives summary SHA256:
- 304d4d9610d31f379d537af30e7bb1760e4bdea316662ed56bbf6c6298608646

The initial Cliquet recovery attempt produced reward 0.0 because the
calibration helper used scipy skew/kurtosis with bias=False. After inspecting
the post-blind checker failure, the generic calibration convention was changed
to scipy's default moment convention. Pricing and forward-start calculations
were already correct.

## Current result

Genuine blind score:
- 0/6

Current post-blind recovery:
- 4/6

Recovered tasks:
1. t1-crypto-funding-rate-basis-carry
2. t1-geometric-mean-reverting-jd
3. t1-cliquet-ratchet-pricing
4. t1-mc-greek-surface-1

Remaining:
1. t1-multimodal-alpha-fusion-edgar-cot-gdelt
2. t1-polars-api-migration

## Regression

Full suite:
- 393 passed
- 38 subtests passed
- 2 existing Phase-7 OHLC warnings

These results are post-blind recovery and must not be reported as the blind score.
