# Phase 8H Stochastic-Process Recovery

Date: 2026-09-20

## Frozen blind baseline

Parent architecture:
- phase8g-composition-6of6
- commit ef3be7c

Phase 8H genuine blind baseline:
- 0/6

Blind cohort SHA256:
- eb9fdf1e6955f69730e5890fe438d0e96773453b59fe812d995e997629204fba

Blind summary SHA256:
- 91660d5ecf09354054048f3dcfc09be88594c9fe4c27dd1d8c1807acca400921

## First post-blind architecture increment

Added reusable stochastic-process capabilities rather than task-ID-specific solvers:

- OU plug-in Euler calibration
- exact AR(1) to continuous-time OU calibration
- exact OU transition simulation
- ADF and autocorrelation diagnostics
- residual-based jump detection
- log-OU compound-Poisson conditional moments
- Monte Carlo process simulation
- empirical carry VaR/CVaR
- safer semantic matching with token boundaries

Initial executable compositions:

- funding OU carry analysis
- log-OU jump-diffusion analysis

## Recovery result

t1-crypto-funding-rate-basis-carry:
- solver_exit = 0
- checker_exit = 0
- reward = 1.0

t1-geometric-mean-reverting-jd:
- solver_exit = 0
- checker_exit = 0
- reward = 1.0

Current post-blind recovery:
- 2/6

Recovery summary SHA256:
- 9da08adbd29b8b7234c909f1d81d8c8e568c5dae6df69ad6d14b35a6607bac57

The blind baseline was frozen before inspecting these task contracts.
These results are post-blind recovery and are not part of the blind score.
