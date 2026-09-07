from __future__ import annotations
import re
from agent.compiled_specification import CompiledSpecification
from agent.specification import TaskSpecification


CATEGORY_REVIEW_PACKS = {
    "derivatives-pricing": [
        "software",
        "numerical",
        "derivatives",
    ],
    "fixed-income": [
        "software",
        "numerical",
        "fixed-income",
    ],
    "credit": [
        "software",
        "numerical",
        "credit",
    ],
    "factor-research": [
        "software",
        "data-causality",
        "factor-research",
    ],
    "backtesting": [
        "software",
        "data-causality",
        "accounting",
        "backtesting",
    ],
    "risk-management": [
        "software",
        "numerical",
        "risk-management",
    ],
    "microstructure": [
        "software",
        "data-causality",
        "accounting",
        "microstructure",
    ],
    "fx": [
        "software",
        "numerical",
        "fx",
    ],
    "nlp-on-finance": [
        "software",
        "data-causality",
        "nlp-finance",
    ],
    "cross-domain": [
        "software",
        "data-causality",
        "numerical",
        "accounting",
        "cross-domain",
    ],
}
CATEGORY_ALIASES = {
    "derivatives": "derivatives-pricing",
    "derivatives_pricing": "derivatives-pricing",
    "derivative-pricing": "derivatives-pricing",

    "credit-risk": "credit",
    "credit-analysis": "credit",

    "factor-models": "factor-research",

    "execution": "microstructure",

    "fx-pricing": "fx",

    "risk-modeling": "risk-management",
}


CATEGORY_ROUTE_OVERRIDES = {
    "performance-attribution": [
        "software",
        "numerical",
        "accounting",
        "factor-research",
    ],

    "dependence-modeling": [
        "software",
        "numerical",
        "risk-management",
    ],

    "cross-sectional-strategies": [
        "software",
        "data-causality",
        "accounting",
        "factor-research",
        "backtesting",
    ],

    "cross-asset-analysis": [
        "software",
        "data-causality",
        "numerical",
        "factor-research",
    ],

    "portfolio-analysis": [
        "software",
        "numerical",
        "accounting",
        "risk-management",
    ],

    "fixed-income-nlp": [
        "software",
        "data-causality",
        "numerical",
        "fixed-income",
        "nlp-finance",
    ],

    "fx-strategy": [
        "software",
        "data-causality",
        "numerical",
        "accounting",
        "backtesting",
        "fx",
    ],

    "interest-rate-derivatives": [
        "software",
        "numerical",
        "derivatives",
        "fixed-income",
    ],

    "predictive-alpha-modeling": [
        "software",
        "data-causality",
        "numerical",
        "factor-research",
        "microstructure",
    ],

    "cross-currency-rates": [
        "software",
        "numerical",
        "fixed-income",
        "fx",
    ],

    "volatility-modeling": [
        "software",
        "numerical",
        "risk-management",
    ],

    "stochastic-processes": [
        "software",
        "numerical",
    ],

    "strategy": [
        "software",
        "data-causality",
        "accounting",
        "factor-research",
        "backtesting",
    ],

    "event-driven-analysis, data-processing": [
        "software",
        "data-causality",
        "factor-research",
    ],

    "extreme-value-theory": [
        "software",
        "numerical",
        "risk-management",
    ],
}

def infer_review_packs(
    spec: TaskSpecification,
) -> list[str]:
    """
    Route public and future category variants into the existing
    frozen skill-pack framework.

    Exact canonical categories remain unchanged. Known aliases
    map to their canonical family. Coarse categories use narrow,
    deterministic instruction-text signals where the category
    alone is insufficient.
    """

    raw_category = (
        spec.category
        or ""
    )

    category = (
        raw_category
        .strip()
        .lower()
    )

    canonical = (
        CATEGORY_ALIASES.get(
            category,
            category,
        )
    )

    if canonical in CATEGORY_REVIEW_PACKS:
        return list(
            CATEGORY_REVIEW_PACKS[
                canonical
            ]
        )

    if category in CATEGORY_ROUTE_OVERRIDES:
        return list(
            CATEGORY_ROUTE_OVERRIDES[
                category
            ]
        )

    text = (
        spec.instruction_text
        .lower()
    )

    # ---------------------------------------------------------
    # Ambiguous broad category: pricing
    # ---------------------------------------------------------
    if category == "pricing":
        rate_terms = (
            "zero coupon",
            "zero-coupon",
            "bootstrapp",
            "yield curve",
            "discount factor",
            "bond",
        )

        derivative_terms = (
            "option",
            "cap",
            "floor",
            "swaption",
            "black",
            "volatility",
        )

        has_rates = any(
            term in text
            for term in rate_terms
        )

        has_derivatives = any(
            term in text
            for term
            in derivative_terms
        )

        packs = [
            "software",
            "numerical",
        ]

        if has_rates:
            packs.append(
                "fixed-income"
            )

        if has_derivatives:
            packs.append(
                "derivatives"
            )

        return packs

    # ---------------------------------------------------------
    # Broad statistical tasks can still expose an obvious
    # fixed-income context such as yield-curve PCA.
    # ---------------------------------------------------------
    if category == "statistical-analysis":
        packs = [
            "software",
            "numerical",
        ]

        if any(
            term in text
            for term in (
                "yield curve",
                "zero rate",
                "discount factor",
                "term structure",
            )
        ):
            packs.append(
                "fixed-income"
            )

        return packs

    # ---------------------------------------------------------
    # Tool-oriented tasks vary by subject. Add only strong
    # finance signals rather than forcing a domain.
    # ---------------------------------------------------------
    if category == "tool-using":
        packs = [
            "software",
        ]

        if any(
            term in text
            for term in (
                "corporate action",
                "split adjustment",
                "dividend adjustment",
            )
        ):
            packs.extend(
                [
                    "data-causality",
                    "accounting",
                ]
            )

        elif any(
            term in text
            for term in (
                "earnings surprise",
                "actual eps",
                "consensus",
            )
        ):
            packs.extend(
                [
                    "data-causality",
                    "numerical",
                ]
            )

        return packs

    # Crypto is an asset class rather than one mathematical
    # domain. Route only when task semantics provide evidence.
    if category == "crypto":
        packs = [
            "software",
            "numerical",
        ]

        if any(
            term in text
            for term in (
                "strategy",
                "carry",
                "funding rate",
                "backtest",
            )
        ):
            packs.extend(
                [
                    "data-causality",
                    "accounting",
                    "backtesting",
                ]
            )

        return packs

    # Keep true software/debug tasks on the safe generic route.
    return [
        "software",
    ]

def infer_difficulty(spec: TaskSpecification) -> str:
    score = 0

    if spec.category == "cross-domain":
        score += 3

    if len(spec.required_output_paths) > 1:
        score += 1

    if len(spec.candidate_schema_fields) >= 8:
        score += 1

    text = spec.instruction_text.lower()

    hard_terms = [
        "monte carlo",
        "optimization",
        "calibration",
        "pde",
        "finite difference",
        "backtest",
        "portfolio",
        "stochastic",
        "simulation",
        "implied volatility",
    ]

    for term in hard_terms:
        if term in text:
            score += 1

    if score >= 4:
        return "hard"

    if score >= 2:
        return "medium"

    return "easy"


def infer_invariants(spec: TaskSpecification) -> list[str]:
    category = spec.category

    if category == "derivatives-pricing":
        return [
            "prices should satisfy basic arbitrage bounds",
            "call price should be nondecreasing in spot where applicable",
            "put price should be nonincreasing in spot where applicable",
            "gamma should generally be nonnegative for vanilla European options",
            "vega should generally be nonnegative for vanilla European options",
            "put-call parity should hold where assumptions match",
            "near-expiry values should approach payoff",
        ]

    if category == "backtesting":
        return [
            "information_time <= signal_time < execution_time",
            "positions must reconcile with signed fills",
            "cash must reconcile with trades, fees, and cashflows",
            "NAV must reconcile with cash and marked positions",
            "transaction costs must be applied consistently",
        ]

    if category == "risk-management":
        return [
            "risk measures should remain finite for valid inputs",
            "portfolio weights and exposures should reconcile",
            "scenario aggregation should preserve dimensions and units",
        ]

    if category == "fixed-income":
        return [
            "bond prices should reconcile with discounted cashflows",
            "discount factors should be positive under standard assumptions",
            "price should generally decrease as yield increases",
        ]

    if category == "credit":
        return [
            "default probabilities must remain within [0, 1]",
            "survival probabilities must remain within [0, 1]",
            "survival probabilities should be nonincreasing with horizon",
        ]

    return []


def infer_edge_cases(spec: TaskSpecification) -> list[str]:
    category = spec.category

    common = [
        "NaN and Inf inputs",
        "empty or very small datasets",
        "duplicate identifiers or timestamps",
        "near-zero denominators",
        "extreme numeric values",
    ]

    if category == "derivatives-pricing":
        return common + [
            "time to maturity approaching zero",
            "volatility approaching zero",
            "deep ITM and deep OTM contracts",
            "extreme spot-to-strike ratios",
        ]

    if category in {"factor-research", "backtesting", "microstructure"}:
        return common + [
            "look-ahead leakage",
            "same-bar signal and execution leakage",
            "timezone or timestamp misalignment",
            "full-sample normalization leakage",
        ]

    return common

def infer_numerical_risks(
    spec: TaskSpecification,
) -> list[str]:
    risks = [
        "overflow",
        "underflow",
        "catastrophic cancellation",
        "unstable near-zero denominators",
    ]

    text = spec.instruction_text.lower()

    convergence_patterns: tuple[
        tuple[str, tuple[str, ...]],
        ...
    ] = (
        (
            "monte-carlo sampling convergence",
            (
                r"\bmonte[\s-]+carlo\s+"
                r"(?:simulation|simulations|validation|verification)\b",
                r"\bn_simulations?\b",
                r"\bn_simulation_paths\b",
                r"\bmc_num_paths\b",
                r"\bsimulate\b[^\n]{0,100}\bpaths?\b",
                r"\bsimulated?\s+"
                r"(?:paths?|trials?|returns?|losses?|samples?)\b",
                r"\bgbm\s+paths?\b",
            ),
        ),
        (
            "iterative algorithm convergence",
            (
                r"\bbaum[\s-]+welch\b",
                r"\bem\s+algorithm\b",
                r"\bmax[_\s-]*iter(?:ations?)?\b",
                r"\bterminate\b[^\n]{0,100}"
                r"\b(?:tol|tolerance|converg)",
            ),
        ),
        (
            "root-finding convergence",
            (
                r"\bbrentq\b",
                r"\bbrent(?:'s)?\s+method\b",
                r"\bnewton(?:-raphson|'s\s+method)?\b",
                r"\broot[\s-]+find",
                r"\broot[\s-]+solv",
            ),
        ),
        (
            "optimisation convergence",
            (
                r"\bscipy\.optimize\b",
                r"\boptimize\.minimize\b",
                r"\bminimize\b",
                r"\bleast[\s-]+squares\s+optimi[sz]ation\b",
                r"\bconstrained\s+optimi[sz]ation\b",
                r"\bl-bfgs-b\b",
                r"\bslsqp\b",
            ),
        ),
        (
            "tree or grid convergence",
            (
                r"\btrinomial\s+tree\b",
                r"\bbinomial\s+tree\b",
                r"\bpde\b",
                r"\bgrid\s+convergence\b",
                r"\btimestep\s+convergence\b",
            ),
        ),
        (
            "numerical integration convergence",
            (
                r"\bquadrature\b",
                r"\bnumerical\s+integration\b",
            ),
        ),
    )

    for risk_name, patterns in convergence_patterns:
        if any(
            re.search(pattern, text)
            for pattern in patterns
        ):
            risks.append(risk_name)

    return risks

def compile_specification(
    spec: TaskSpecification,
) -> CompiledSpecification:
    review_packs = infer_review_packs(
        spec
    )

    deliverables = [
        {
            "path": path,
            "format": path.rsplit(".", 1)[-1]
            if "." in path
            else None,
        }
        for path in spec.required_output_paths
    ]

    return CompiledSpecification(
        task_id=spec.task_id,
        category=spec.category,
        difficulty=infer_difficulty(spec),

        deliverables=deliverables,
        required_columns=spec.required_output_columns,
        required_row_rules=[],
        ordering_rules=[],

        units={},
        conventions=[],

        edge_cases=infer_edge_cases(spec),
        invariants=infer_invariants(spec),

        review_packs=review_packs,
        numerical_risks=infer_numerical_risks(
            spec
        ),

        assumptions=[],
        unresolved_questions=[],
        required_dtypes=dict(
            spec.required_output_dtypes
    ),
    )