from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class SkillPack:
    name: str
    checks: list[str]
    guidance: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SKILL_PACKS: dict[str, SkillPack] = {
    "software": SkillPack(
        name="software",
        checks=[
            "detect NaN and Inf propagation",
            "check dtype and shape consistency",
            "check deterministic reproducibility",
            "check seed handling where randomness exists",
            "detect hidden global state",
            "detect accidental input mutation",
            "guard near-zero denominators",
        ],
        guidance=[
            "prefer explicit inputs and outputs",
            "avoid fragile implicit state",
            "fail clearly on invalid inputs",
        ],
    ),

    "numerical": SkillPack(
        name="numerical",
        checks=[
            "test extreme numeric inputs",
            "test limiting cases",
            "check numerical stability",
            "check convergence where applicable",
            "use perturbation or bump tests",
            "distinguish discretization error from sampling error",
            "for Monte Carlo or stochastic simulation, use an explicit seed and check estimate stability as path count increases",
            "report or estimate sampling uncertainty where simulation noise materially affects the requested result",
            "distinguish simulation bias, sampling error, and discretization error when stochastic paths are time-stepped",
        ],
        guidance=[
            "prefer bounded or safeguarded solvers when derivatives may vanish",
            "avoid false precision",
            "compare independent methods when disagreement matters",
            "use variance reduction only when it preserves the target estimator and produces measurable precision improvement",
        ],
    ),

    "derivatives": SkillPack(
        name="derivatives",
        checks=[
            "check basic arbitrage bounds",
            "check put-call parity where assumptions match",
            "check monotonicity in spot where applicable",
            "check near-expiry payoff limits",
            "check volatility approaching zero",
            "check deep ITM and deep OTM cases",
            "check gamma sign for vanilla European options",
            "check vega sign for vanilla European options",
            "validate Greeks with bump-and-reprice where practical",
            "validate Black-Scholes, Black-76, or Garman-Kohlhagen model choice against the instrument and carry assumptions",
            "for implied volatility calibration, solve the pricing residual with a bounded or bracketed root method and reject prices outside valid arbitrage bounds",
            "for binomial trees or finite difference PDE methods, verify terminal and boundary conditions and check grid or timestep convergence",
            "for Monte Carlo pricing, use an explicit seed, report or estimate sampling error, and check convergence as path count increases",
            "for discrete delta hedging, treat financing, dividends, rebalancing, transaction costs, and terminal settlement as an explicit ordered cashflow ledger",
            "for multi-asset exchange/spread options, validate the zero-strike spread limit against the corresponding exchange-option identity when assumptions match",
            "use variance reduction in Monte Carlo when it materially improves precision without changing the estimator target",
        ],
        guidance=[
            "preserve rate, dividend, maturity, and sign conventions",
            "be explicit about Greek units",
            "treat low-vega regimes carefully",
            "prefer a valid closed-form method when available; use trees or PDEs for exercise or boundary features and Monte Carlo for path-dependent or high-dimensional payoffs",
            "when practical, compare the primary price or Greek against an independent method such as bump-and-reprice, parity, or a converged numerical benchmark",
        ],
    ),

    "fixed-income": SkillPack(
        name="fixed-income",
        checks=[
            "reconcile price with discounted cashflows",
            "check yield-price monotonicity",
            "check discount factor validity",
            "check day-count and compounding conventions",
            "check maturity edge cases",
            "bootstrap the zero curve sequentially from quoted instruments and reprice the calibration instruments from the resulting curve",
            "check discount factor, zero rate, spot rate, and forward rate conversions under the declared compounding and day-count convention",
            "validate forward rates against adjacent discount factors or zero rates",
            "validate duration and DV01 with a small yield bump and compare with bump-and-reprice where practical",
            "check convexity or second-order effects when the requested yield move is too large for a first-order duration approximation",
        ],
        guidance=[
            "keep rate units and compounding explicit",
            "do not mix clean and dirty prices silently",
            "state the curve interpolation variable explicitly, such as discount factors, zero rates, or log discount factors",
            "use consistent settlement dates, accrual fractions, coupon frequency, and cashflow timing throughout pricing and risk calculations",
        ],
    ),

    "credit": SkillPack(
        name="credit",
        checks=[
            "probabilities remain within [0, 1]",
            "survival probabilities are nonincreasing with horizon",
            "hazard-related quantities remain valid",
            "reconcile expected loss components",
            "for CDS pricing, reconcile the discounted premium leg and protection leg under the stated recovery and accrual conventions",
            "when calibrating hazard rates from CDS spreads or risky prices, reprice the calibration instruments from the calibrated curve",
            "for a credit migration matrix, require nonnegative transition probabilities and rows that sum to one within tolerance",
            "when using multi-period migration, check that matrix powers preserve valid transition probabilities",
        ],
        guidance=[
            "keep recovery and loss-given-default conventions explicit",
            "preserve horizon and discounting conventions",
            "keep PD, LGD, EAD, hazard rate, and survival probability definitions distinct",
            "for piecewise-constant hazard models, keep interval boundaries and discounting conventions explicit",
        ],
    ),

    "data-causality": SkillPack(
        name="data-causality",
        checks=[
            "enforce information_time <= signal_time < execution_time",
            "detect negative shifts and future indexing",
            "detect same-bar signal and execution leakage",
            "detect full-sample normalization leakage",
            "check point-in-time joins",
            "check train-test preprocessing isolation",
            "check survivorship and universe construction",
            "check timestamps and timezones",
        ],
        guidance=[
            "prefer time-aware validation",
            "treat suspicious timing behavior as a warning unless conclusively invalid",
        ],
    ),

    "accounting": SkillPack(
        name="accounting",
        checks=[
            "reconcile positions with signed fills",
            "reconcile cash with trades, fees, and cashflows",
            "reconcile NAV with cash and marked positions",
            "reconcile realized and unrealized P&L",
            "check transaction costs",
            "check leverage and exposure constraints",
            "detect duplicate fills and overfills",
        ],
        guidance=[
            "treat accounting identities as hard invariants where applicable",
            "include multipliers and financing conventions explicitly",
        ],
    ),

    "factor-research": SkillPack(
        name="factor-research",
        checks=[
            "check factor timing",
            "check point-in-time universe membership",
            "check normalization leakage",
            "check cross-sectional ranking alignment",
            "check missing-data handling",
            "for information coefficient (IC), align factor values with future returns at the intended horizon and verify the correlation statistic and sign convention",
            "for long-short or neutral portfolios, reconcile long and short weights and check dollar-neutral, beta-neutral, or other declared neutrality constraints",
            "check portfolio weights, leverage, and concentration after ranking, winsorization, normalization, and missing-data filters",
            "for event studies, derive relative days from aligned trading-day indices, keep estimation and event windows non-overlapping, and exclude events with incomplete required windows",
            "for standardized event-study tests, preserve the stated event-specific prediction-error correction and any cross-sectional dependence adjustment rather than substituting a simpler t-test",
        ],
        guidance=[
            "avoid random cross-validation for time-dependent research",
            "preserve rebalance timing exactly",
            "use point-in-time data and form signals before the returns used to evaluate them",
            "state whether factor normalization is cross-sectional or time-series and apply it only with information available at that timestamp",
        ],
    ),

    "backtesting": SkillPack(
        name="backtesting",
        checks=[
            "check signal-to-execution timing",
            "check position updates",
            "check transaction cost application",
            "check turnover calculation",
            "check portfolio value reconciliation",
            "check delayed-signal robustness",
            "check the self-financing identity by reconciling position changes, trade cashflows, fees, financing, and resulting portfolio value",
            "recompute key performance statistics such as cumulative return, annualized return, volatility, Sharpe ratio, and maximum drawdown from the produced return or equity series",
            "check that performance metrics use the declared annualization frequency, risk-free convention, and return definition",
        ],
        guidance=[
            "separate suspicious performance behavior from conclusive leakage",
            "verify accounting identities before trusting performance metrics",
            "prefer event-ordering and accounting reconciliation over plausibility of headline Sharpe or return",
            "keep signal formation, order placement, execution, valuation, and performance measurement as distinct timeline steps",
        ],
    ),

    "risk-management": SkillPack(
        name="risk-management",
        checks=[
            "check exposure aggregation",
            "check risk measure dimensions",
            "check scenario consistency",
            "check tail-probability conventions",
            "check portfolio weight reconciliation",
            "for VaR, make the loss-versus-return sign convention, confidence level, horizon, and quantile definition explicit",
            "for expected shortfall or CVaR, average losses beyond the VaR threshold using the same tail convention and verify ES is not below VaR under a consistent loss convention",
            "check covariance and correlation matrices for symmetry, finite entries, valid diagonals, and positive-semidefinite behavior where required",
            "for linear portfolio risk, reconcile portfolio variance with w^T Sigma w under the stated exposure convention",
            "distinguish historical, parametric, and Monte Carlo risk estimates and validate horizon scaling assumptions rather than applying square-root-of-time blindly",
            "for FFT aggregate-loss methods, verify probability-mass normalization, tail-grid coverage, transform orientation, monotone CDF construction, and wrap-around aliasing control",
            "for compound-loss Monte Carlo, preserve the count/severity model, requested seed and simulation count, and use bounded-memory aggregation when total claim draws are large",
        ],
        guidance=[
            "keep confidence level and horizon explicit",
            "avoid mixing return and loss sign conventions",
            "keep covariance units, annualization, and exposure scaling consistent",
            "treat stress scenarios separately from probabilistic VaR or ES unless the task explicitly combines them",
        ],
    ),

    "microstructure": SkillPack(
        name="microstructure",
        checks=[
            "check event ordering",
            "check trade and quote timestamps",
            "check order-fill consistency",
            "check partial fills",
            "check spread and price constraints",
            "check duplicate events",
            "for order book data, require bid prices not to exceed ask prices and keep price levels, sizes, and depth ordering internally consistent",
            "recompute VWAP from executed price-times-quantity divided by executed quantity and distinguish it from TWAP",
            "reconcile implementation shortfall against the stated arrival or decision price, executed fills, quantities, fees, and any residual position",
            "for participation or execution scheduling, reconcile target quantity with scheduled and filled quantity and enforce participation-rate limits",
        ],
        guidance=[
            "preserve event-time ordering",
            "treat latency assumptions explicitly",
            "distinguish quote data, trade data, order submissions, and fills rather than merging their timestamps or semantics",
            "make the execution benchmark explicit, such as arrival price, VWAP, TWAP, or close",
        ],
    ),

    "fx": SkillPack(
        name="fx",
        checks=[
            "check currency quote direction",
            "check domestic-versus-foreign rate conventions",
            "check triangular consistency where applicable",
            "check forward-spot relationship",
            "check covered interest parity (CIP) using the declared quote orientation and compounding convention",
            "when spot is quoted as quote-currency units per one base-currency unit, verify the continuous-compounding forward rate as F = S * exp((r_quote - r_base) * T)",
            "check FX forward points and outright forwards against spot and the two currency interest-rate curves",
            "for FX options, validate Garman-Kohlhagen pricing and put-call parity using the foreign rate as the carry or dividend analogue",
        ],
        guidance=[
            "never assume quote orientation",
            "keep base and quote currency explicit",
            "keep domestic and foreign discounting consistent with the chosen quote convention",
            "for cross rates, derive the algebra from explicit currency units before multiplying or dividing rates",
        ],
    ),

    "nlp-finance": SkillPack(
        name="nlp-finance",
        checks=[
            "check timestamp alignment between text and market data",
            "check leakage from future filings or revisions",
            "check document duplication",
            "check label timing",
            "for sentiment or tone scores, enforce the declared score range and preserve the intended direction of positive versus negative sentiment",
            "check that text preprocessing, aggregation, and document-to-market joins do not use documents released after the prediction timestamp",
            "when multiple documents map to one event or date, make the aggregation rule explicit and deterministic",
        ],
        guidance=[
            "use only information available by the prediction timestamp",
            "preserve document chronology",
            "keep sentiment scale, missing-document handling, and aggregation conventions explicit",
            "treat revised filings or later transcripts as future information unless the task explicitly permits them",
        ],
    ),

    "cross-domain": SkillPack(
        name="cross-domain",
        checks=[
            "decompose the task into independent subproblems",
            "verify interfaces between subproblems",
            "check unit consistency across domains",
            "check timing consistency across domains",
            "before aggregation, convert constituent outputs to compatible currencies, units, horizons, and sign conventions",
            "recompute the final aggregate from validated component outputs rather than trusting an independently produced total",
            "require at least one relevant validation check from each constituent finance domain before accepting the integrated result",
        ],
        guidance=[
            "prefer modular solutions",
            "validate each domain component before integration",
            "make cross-domain conversion assumptions explicit before aggregation",
            "preserve intermediate component outputs when they are needed to explain or verify the final aggregate",
        ],
    ),
}


def load_skill_packs(
    names: list[str],
) -> list[SkillPack]:
    packs: list[SkillPack] = []

    for name in names:
        pack = SKILL_PACKS.get(name)

        if pack is not None:
            packs.append(pack)

    return packs