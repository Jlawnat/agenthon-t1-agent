# Phase 8H Unseen Generalization — Final Recovery

## Experimental protocol

Phase 8H used a frozen six-task unseen cohort.

The genuine blind baseline was evaluated before inspecting task-specific
instructions, checkers, reference outputs, or datasets.

Genuine blind baseline:

- 0 / 6 solver completions
- tag: phase8h-blind-0of6-completion

All subsequent work is explicitly classified as post-blind recovery.

## Frozen cohort

1. t1-cliquet-ratchet-pricing
2. t1-crypto-funding-rate-basis-carry
3. t1-geometric-mean-reverting-jd
4. t1-mc-greek-surface-1
5. t1-multimodal-alpha-fusion-edgar-cot-gdelt
6. t1-polars-api-migration

## Post-blind recovery

Recovered through reusable capability additions:

- stochastic-process composition: 2 / 6
- derivatives / Monte Carlo composition: 4 / 6
- multimodal alpha composition: 5 / 6
- generic Polars source-code migration: 6 / 6

Final post-blind recovery:

**6 / 6**

This does not alter the genuine blind score, which remains:

**0 / 6**

## Final Polars migration task

Task:

`t1-polars-api-migration`

The final capability is a generic, task-ID-independent Polars source migration
skill registered directly in the offline runtime.

It migrates legacy Polars 0.x source constructs to Polars 1.x-compatible
constructs and executes the migrated pipeline without network access.

Final evaluator result:

- generated migration script: success
- evaluator execution: success
- Polars evaluator version: 1.39.3
- checker: pass
- reward: 1.0
- checker result: 57 passed, 1 harmless pytest cache warning

Valid evaluation summary SHA256:

`7af86cb83728f1eed713f205b44d40c1d5e90f92f3352ba7274ab82a69705b17`

Earlier local reward=0 attempts used incorrect evaluation/checker mounts or the
wrong input dataset and are not treated as valid benchmark results.

## Final regression

Full repository regression:

`403 passed, 2 warnings, 38 subtests passed in 15.22s`

The two warnings are the previously known Phase 7 OHLC NumPy warnings:

- Mean of empty slice
- invalid value encountered in scalar divide

No new regression warnings were introduced.
