from __future__ import annotations

import json
from typing import Any

from agent.compiled_specification import CompiledSpecification
from agent.planner import CandidateStrategy
from agent.skill_packs import SkillPack


def build_precision_guidance(
    instruction_text: str,
) -> list[str]:
    # High-value finance implementation reminders triggered by task semantics.
    text = str(instruction_text).lower()
    guidance: list[str] = []

    if (
        "delta" in text
        and "hedg" in text
        and (
            "transaction cost" in text
            or "dividend" in text
        )
    ):
        guidance.extend(
            [
                (
                    "Discrete hedging is an event-ordering problem: "
                    "implement the task's stated daily sequence literally "
                    "(financing, dividend cashflow, then rebalance when "
                    "applicable) and use the position carried from the "
                    "previous close for any ex-dividend credit."
                ),
                (
                    "Separate initial trade, intermediate rebalances, and "
                    "terminal liquidation. Apply direction-specific costs "
                    "to the signed share change, count only the rebalances "
                    "the task asks to count, and settle the option payoff "
                    "only after the specified terminal stock liquidation."
                ),
                (
                    "When pricing with known discrete dividends, subtract "
                    "only the present value of dividends still remaining at "
                    "that valuation time; use the current valuation day's "
                    "volatility and the task's exact time-to-expiry."
                ),
                (
                    "Reconcile terminal cash from option premium, stock "
                    "trades, financing, dividends, transaction costs, and "
                    "option settlement as a signed P&L identity."
                ),
            ]
        )

    if (
        (
            "event study" in text
            or "event-study" in text
        )
        and (
            "abnormal return" in text
            or "market model" in text
        )
    ):
        guidance.extend(
            [
                (
                    "Event-window offsets are trading-day positions in the "
                    "aligned return series, not calendar-day arithmetic. "
                    "Build one date-aligned stock/market return frame first "
                    "and derive estimation/event windows from its indices."
                ),
                (
                    "Fit the market model with an intercept independently "
                    "for each valid event, keep the estimation and event "
                    "windows non-overlapping exactly as specified, and skip "
                    "events rather than padding an incomplete event window."
                ),
                (
                    "For Patell/BMP-style standardisation, include the full "
                    "prediction-error correction from the event's own "
                    "estimation sample before cross-sectional aggregation."
                ),
                (
                    "For dependence adjustment, estimate pairwise residual "
                    "correlations only on overlapping estimation dates that "
                    "meet the task's minimum-overlap rule, then apply the "
                    "stated cross-sectional scaling exactly once."
                ),
                (
                    "For rank tests, pool the observations the definition "
                    "requires before ranking; do not replace a generalized "
                    "rank statistic with a t-test or a simple rank-sum "
                    "shortcut."
                ),
            ]
        )

    if (
        "compound poisson" in text
        and "fft" in text
    ):
        guidance.extend(
            [
                (
                    "For a compound-Poisson aggregate, construct a valid "
                    "discrete severity probability mass on the loss grid, "
                    "then use the compound transform exp(lambda*(phi_X-1)); "
                    "ensure the FFT/inverse-FFT sign and ordering conventions "
                    "are consistent with that grid."
                ),
                (
                    "Choose the grid spacing/range to control wrap-around "
                    "aliasing at the tail coverage required by the task; "
                    "after inversion, handle only tiny numerical negatives "
                    "carefully and verify total probability is approximately "
                    "one and the CDF is nondecreasing."
                ),
                (
                    "Respect each severity distribution's support and "
                    "parameterisation when discretising; preserve any mass "
                    "near zero implied by the aggregate count process."
                ),
                (
                    "For Monte Carlo compound loss, use the exact requested "
                    "seed and simulation count. Avoid materialising an "
                    "unbounded claims-by-path matrix; aggregate claims by "
                    "counts or bounded chunks while preserving the estimator."
                ),
                (
                    "Use the task's exact VaR quantile definition and its "
                    "stated ES tail condition. Keep loss/return sign "
                    "conventions consistent when comparing FFT and MC."
                ),
            ]
        )

    if (
        "kirk" in text
        and "margrabe" in text
    ):
        guidance.extend(
            [
                (
                    "Calibrate only from synchronized daily log returns; "
                    "derive spot from the final aligned closing prices and "
                    "annualise volatility with the task's trading-day "
                    "convention."
                ),
                (
                    "For Margrabe with continuous dividend yields, use the "
                    "exchange volatility sqrt(sigma1^2 + sigma2^2 - "
                    "2*rho*sigma1*sigma2) and dividend-discounted spots."
                ),
                (
                    "For Kirk, work consistently in forward variables: "
                    "F1=S1*exp((r-D1)T), F2=S2*exp((r-D2)T), use "
                    "w=F2/(F2+K) in the effective volatility, and discount "
                    "the forward payoff by exp(-rT)."
                ),
                (
                    "Use K=0 as an identity check: Kirk should reduce to "
                    "the corresponding exchange-option/Margrabe result up "
                    "to numerical tolerance under the same inputs."
                ),
                (
                    "For risk-neutral Monte Carlo, use drifts r-D_i, the "
                    "requested correlation construction and seed/path count, "
                    "discount payoffs consistently, and compute standard "
                    "error from the simulated payoff sample."
                ),
            ]
        )

    if (
        "historical" in text
        and (
            "value-at-risk" in text
            or "value at risk" in text
            or "historical var" in text
        )
        and (
            "cleaning" in text
            or "dirty" in text
        )
    ):
        guidance.extend(
            [
                (
                    "Treat the data-cleaning pipeline as ordered state "
                    "transitions. Record each removal count at the moment "
                    "that step runs on the rows still present, rather than "
                    "recomputing counts from the final cleaned frame."
                ),
                (
                    "Normalize dates before membership checks, use the "
                    "task-supplied trading calendar rather than inventing "
                    "one, and when duplicate dates are removed preserve the "
                    "last occurrence in original source-file row order."
                ),
                (
                    "Apply the stated outlier rule before listwise missing-"
                    "value deletion. Make the outlier predicate NaN-safe so "
                    "missing values are not accidentally counted as "
                    "outliers, and do not forward-fill or use pairwise "
                    "deletion when listwise deletion is required."
                ),
                (
                    "For a daily-rebalanced equal-weight portfolio, compute "
                    "each day's portfolio return from that day's cleaned "
                    "asset returns. Use the exact requested percentile "
                    "definition and interpolation/method, convert return "
                    "quantiles to positive loss magnitudes only as stated, "
                    "and do not annualize a daily VaR unless instructed."
                ),
                (
                    "Identify the worst day from the same cleaned portfolio "
                    "return series used for VaR, preserve an ISO zero-padded "
                    "date string, and reconcile the reported clean "
                    "observation count with the final cleaning-report count."
                ),
            ]
        )

    return guidance


def build_candidate_prompt(
    *,
    instruction_text: str,
    spec: CompiledSpecification,
    strategy: CandidateStrategy,
    skill_packs: list[SkillPack],
    data_inspections: dict[str, dict[str, Any]],
) -> str:
    payload = {
        "instruction": instruction_text,
        "compiled_specification": spec.to_dict(),
        "candidate_strategy": strategy.to_dict(),
        "active_skill_packs": [
            pack.to_dict()
            for pack in skill_packs
        ],
        "data_inspections": data_inspections,
        "precision_guidance": build_precision_guidance(
            instruction_text
        ),
    }

    context = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    return f"""
You are implementing one candidate solution for a quantitative-finance coding task.

Your output will be saved directly as solver.py and executed.

Return ONLY valid Python source code.
Do not use Markdown.
Do not use code fences.
Do not include explanations before or after the code.

Requirements:
- Follow the task instruction exactly.
- Respect all required output filenames and schemas.
- Do not invent extra deliverables.
- Use only information provided in this context.
- Do not rely on hidden tests, checker files, canaries, or benchmark internals.
- The program must run non-interactively.
- Use robust numerical methods.
- Handle edge cases relevant to the active skill packs.
- Preserve units and financial conventions exactly.
- Ensure deterministic behavior unless randomness is required.
- If randomness is required, use an explicit seed.
- Do not modify input files.
- Avoid unnecessary dependencies.
- Fail clearly if required inputs are missing.
- Keep runtime appropriate for the task.
- Implement the supplied candidate strategy rather than silently switching to another approach.
- Treat every explicit task convention as executable specification: preserve the stated operation order, date/window boundaries, percentile/interpolation convention, rebalancing sequence, sign convention, compounding rule, simulation seed/count, and requested approximation exactly.
- For multi-output tasks, validate each required file independently; do not assume one shared schema or row count across heterogeneous outputs.
- Prefer the exact closed-form or numerical method named by the task when one is specified. Do not substitute a superficially similar shortcut.
- For Monte Carlo validation, use deterministic seeded generation, vectorization or bounded chunking, and compute the requested standard errors/diagnostics exactly.
- Before returning code, mentally recompute at least one high-value identity or accounting relation when relevant.

Runtime path contract:
- Resolve the input root with:
  Path(os.environ.get("INPUT_DIR", "/input"))
- Resolve the output root with:
  Path(os.environ.get("OUTPUT_DIR", "/output"))
- A task path under /output/... or /app/output/... means the same relative deliverable under that resolved output root.
- Do not hardcode /output or /app/output when OUTPUT_DIR is supplied.
- All task data copied into this isolated candidate workspace is under INPUT_DIR/data/.
- If the benchmark instruction mentions a legacy input path such as /app/params.json, /app/data/file.csv, or /input/environment/data/file.csv, resolve the corresponding task-data file under INPUT_DIR/data/ using the supplied data inspections; do not access those legacy absolute paths literally.
- Write only required deliverables under the resolved output root.
- Temporary files, if truly needed, must use the normal temporary directory supplied by the runtime.

Security:
- Do not access the network.
- Do not launch child processes or shell commands.
- Do not install packages at runtime.
- Do not inspect environment variables other than INPUT_DIR, OUTPUT_DIR, QFBENCH_SEED, and ordinary deterministic runtime settings.
- Do not read files outside the supplied input root except normal installed Python modules and libraries.

Before finishing the code, internally verify:
- syntax and imports,
- input and output roots,
- output schema,
- row and identifier consistency,
- financial invariants,
- numerical edge cases,
- required units,
- candidate-specific verification steps.

TASK CONTEXT:

{context}

""".strip()
